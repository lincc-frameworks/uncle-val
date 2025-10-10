from collections.abc import Callable
from pathlib import Path

import torch

from uncle_val.datasets.dp1 import dp1_catalog_multi_band
from uncle_val.learning.models import MLPModel
from uncle_val.pipelines.training_loop import training_loop


def run_dp1_mlp(
    *,
    dp1_root: str | Path,
    n_workers: int,
    n_src: int,
    n_lcs: int,
    train_batch_size: int,
    output_root: str | Path,
    loss_fn: Callable | None = None,
    start_tfboard: bool = False,
    val_batch_over_train: int = 128,
    device: torch.device | str = "cpu",
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
    output_root : str or Path
        Where to save the intermediate results.
    val_batch_over_train : int
        Ratio of batch sizes for training and validation.
    loss_fn : Callable or None
        Loss function to use, by default soften Χ² is used.
    start_tfboard : bool
        Whether to start a TensorBoard session.
    device : torch.device | str
        Torch device to use for training.

    Returns
    -------
    Path
        Path to the output model.
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

    columns = ["x", "err", "extendedness"] + [f"is_{band}_band" for band in bands]

    model = MLPModel(d_input=len(columns), d_middle=(300, 300, 500, 1000, 500), dropout=0.1, d_output=1)

    return training_loop(
        catalog=catalog,
        columns=columns,
        model=model,
        loss_fn=loss_fn,
        lr=1e-5,
        n_workers=n_workers,
        n_src=n_src,
        n_lcs=n_lcs,
        train_batch_size=train_batch_size,
        val_batch_over_train=val_batch_over_train,
        output_root=output_root,
        start_tfboard=start_tfboard,
        device=device,
        model_name="mlp",
    )
