from pathlib import Path

import torch

from uncle_val.datasets import rubin_dp_catalog_single_band
from uncle_val.learning.losses import UncleLoss
from uncle_val.learning.models import LinearModel
from uncle_val.pipelines.splits import SurveyConfig
from uncle_val.pipelines.training_loop import training_loop


def run_rubin_dp_linear_flux_err(
    *,
    rubin_dp_root: str | Path,
    band: str,
    non_extended_only: bool,
    n_workers: int,
    n_src: int,
    n_lcs: int,
    train_batch_size: int,
    val_batch_size: int,
    output_root: str | Path,
    loss_fn: UncleLoss,
    val_losses: dict[str, UncleLoss] | None = None,
    start_tfboard: bool = False,
    log_activations: bool = False,
    device: str | torch.device = "cpu",
    survey_config: SurveyConfig,
) -> Path:
    """Run the training with the linear model on fluxes and errors

    Parameters
    ----------
    rubin_dp_root : str or Path
        The root directory of the Rubin DP HATS catalogs.
    band : str
        Passband to train the model on.
    non_extended_only : bool
        Whether to filter out extended sources.
    n_workers : int
        Number of Dask workers to use.
    n_src : int
        Number of sources to use per light curve.
    n_lcs : int
        Number of light curves to train on.
    train_batch_size : int
        Batch size for training.
    val_batch_size : int or None
        Batch size for validation.
    loss_fn : UncleLoss
        Loss function to use, by default soften Χ² is used.
    val_losses : dict[str, UncleLoss] or None
        Extra losses to compute on validation set and record, it maps name to
        loss function. If None, an empty dictionary is used.
    start_tfboard : bool
        Whether to start a TensorBoard session.
    log_activations : bool
        Whether to log validation activations with TensorBoard session.
    output_root : str or Path
        Where to save the intermediate results.
    device : str or torch.device, optional
        Torch device to use for training, default is "cpu".
    survey_config : SurveyConfig
        Train/val/test split boundaries.

    Returns
    -------
    Path
        Path to the output model.
    """
    catalog = rubin_dp_catalog_single_band(
        root=rubin_dp_root,
        band=band,
        obj="science",
        img="cal",
        phot="PSF",
        mode="forced",
    )

    if non_extended_only:
        catalog = catalog.query("extendedness == 0.0")
    catalog = catalog.map_partitions(
        lambda df: df.drop(columns=["r_psfMag", "coord_ra", "coord_dec", "extendedness"])
    )

    model = LinearModel(outputs_s=False).to(device=device)

    if val_losses is None:
        val_losses = {}

    return training_loop(
        catalog=catalog,
        columns=None,
        model=model,
        loss_fn=loss_fn,
        val_losses=val_losses,
        lr=3e-4,
        n_workers=n_workers,
        n_src=n_src,
        n_lcs=n_lcs,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        output_root=output_root,
        device=device,
        start_tfboard=start_tfboard,
        log_activations=log_activations,
        model_name="linear",
        survey_config=survey_config,
    )
