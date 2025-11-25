from pathlib import Path

import torch

from uncle_val.datasets.dp1 import dp1_catalog_multi_band
from uncle_val.learning.losses import UncleLoss
from uncle_val.learning.models import MLPModel
from uncle_val.pipelines.training_loop import training_loop


def run_dp1_mlp(
    *,
    dp1_root: str | Path,
    n_workers: int,
    n_src: int,
    n_lcs: int,
    train_batch_size: int,
    val_batch_size: int,
    output_root: str | Path,
    loss_fn: UncleLoss,
    val_losses: dict[str, UncleLoss] | None = None,
    load_model_path: str | Path | None = None,
    lr: float = 1e-5,
    start_tfboard: bool = False,
    log_activations: bool = False,
    snapshot_every: int = 128,
    device: torch.device | str = "cpu",
) -> tuple[Path, list[str]]:
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
    output_root : str or Path
        Where to save the intermediate results.
    snapshot_every : int
        Snapshot model and metrics every this much training batches.
    loss_fn : UncleLoss
        Loss function to use, by default soften Χ² is used.
    val_losses : dict[str, UncleLoss] or None
        Extra losses to compute on validation set and record, it maps name to
        loss function. If None, an empty dictionary is used.
    load_model_path : str or Path or None
        Pre-trained model to continue training from.
    lr : float
        Learning rate.
    start_tfboard : bool
        Whether to start a TensorBoard session.
    log_activations : bool
        Whether to log validation activations with TensorBoard session.
    device : torch.device | str
        Torch device to use for training.

    Returns
    -------
    Path
        Path to the output model.
    list[str]
        List of columns to use as model inputs.
    """
    bands = "ugrizy"

    catalog = dp1_catalog_multi_band(
        root=dp1_root,
        bands=bands,
        obj="science",
        img="cal",
        phot="PSF",
        mode="forced",
    ).map_partitions(lambda df: df.drop(columns=["band", "object_mag", "coord_ra", "coord_dec"]))

    columns = [
        "lc.x",
        "lc.err",
        "extendedness",
        "lc.skyBg",
        "lc.seeing",
        "lc.expTime",
    ] + [f"is_{band}_band" for band in bands]
    columns_no_prefix = [col.removeprefix("lc.") for col in columns]

    if load_model_path is None:
        model = MLPModel(
            input_names=columns_no_prefix, d_middle=(300, 300, 500), dropout=None, outputs_s=False
        )
    else:
        model = torch.load(load_model_path, weights_only=False, map_location=device)

    if val_losses is None:
        val_losses = {}

    model_path = training_loop(
        catalog=catalog,
        columns=columns_no_prefix,
        model=model,
        loss_fn=loss_fn,
        val_losses=val_losses,
        lr=lr,
        n_workers=n_workers,
        n_src=n_src,
        n_lcs=n_lcs,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        snapshot_every=snapshot_every,
        output_root=output_root,
        start_tfboard=start_tfboard,
        log_activations=log_activations,
        device=device,
        model_name="mlp",
    )
    return model_path, columns
