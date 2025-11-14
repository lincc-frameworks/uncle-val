from pathlib import Path

import torch

from uncle_val.datasets import dp1_catalog_single_band
from uncle_val.learning.losses import UncleLoss
from uncle_val.learning.models import LinearModel
from uncle_val.pipelines.training_loop import training_loop


def run_dp1_linear_flux_err(
    *,
    dp1_root: str | Path,
    band: str,
    non_extended_only: bool,
    n_workers: int,
    n_src: int,
    n_lcs: int,
    train_batch_size: int,
    val_batch_size: int,
    output_root: str | Path,
    loss_fn: UncleLoss,
    start_tfboard: bool = False,
    log_activations: bool = False,
    snapshot_every: int = 128,
    device: str | torch.device = "cpu",
) -> Path:
    """Run the training for DP1 with the linear model on fluxes and errors

    Parameters
    ----------
    dp1_root : str or Path
        The root directory of the DP1 HATS catalogs.
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
    snapshot_every : int
        Snapshot model and metrics every this much training batches.
    loss_fn : Callable or None
        Loss function to use, by default soften Χ² is used.
    start_tfboard : bool
        Whether to start a TensorBoard session.
    log_activations : bool
        Whether to log validation activations with TensorBoard session.
    output_root : str or Path
        Where to save the intermediate results.
    device : str or torch.device, optional
        Torch device to use for training, default is "cpu".

    Returns
    -------
    Path
        Path to the output model.
    """
    catalog = dp1_catalog_single_band(
        root=dp1_root,
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

    model = LinearModel(d_input=2, d_output=1).to(device=device)

    return training_loop(
        catalog=catalog,
        columns=None,
        model=model,
        loss_fn=loss_fn,
        lr=3e-4,
        n_workers=n_workers,
        n_src=n_src,
        n_lcs=n_lcs,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        snapshot_every=snapshot_every,
        output_root=output_root,
        device=device,
        start_tfboard=start_tfboard,
        log_activations=log_activations,
        model_name="linear",
    )
