from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path
from warnings import catch_warnings, filterwarnings

import torch
from dask.distributed import Client
from tensorboard.program import TensorBoard
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from uncle_val.datasets import dp1_catalog_single_band
from uncle_val.learning.losses import minus_ln_chi2_prob
from uncle_val.learning.lsdb_dataset import LSDBIterableDataset
from uncle_val.learning.models import LinearModel
from uncle_val.learning.training import evaluate_loss, train_step


def _launch_tfboard(logdir: Path):
    with catch_warnings():
        filterwarnings("ignore", category=UserWarning, module="*")
        tb = TensorBoard()
        tb.configure(argv=[None, "--logdir", str(logdir)])
        url = tb.launch()
    print(f"Tensorboard Link: {url}")


def run_dp1_linear_flux_err(
    *,
    dp1_root: str | Path,
    band: str,
    non_extended_only: bool,
    n_workers: int,
    n_src: int,
    n_lcs: int,
    train_batch_size: int,
    output_root: str | Path,
    loss_fn: Callable | None = None,
    start_tfboard: bool = False,
    val_batch_over_train: int = 128,
):
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
    val_batch_over_train : int
        Ratio of batch sizes for training and validation.
    loss_fn : Callable or None
        Loss function to use, by default soften Χ² is used.
    start_tfboard : bool
        Whether to start a TensorBoard session.
    output_root : str or Path
        Where to save the intermediate results.
    """
    output_root = Path(output_root)
    output_dir = Path(output_root) / datetime.now().strftime("%Y-%m-%d_%H-%M")
    epoch_model_dir = output_dir / "models"
    epoch_model_dir.mkdir(parents=True, exist_ok=True)

    if start_tfboard:
        _launch_tfboard(output_root)
    summary_writer = SummaryWriter(log_dir=str(output_dir))

    validation_batch_size = val_batch_over_train * train_batch_size
    n_validation_batches = n_lcs // validation_batch_size

    if loss_fn is None:
        loss_fn = partial(minus_ln_chi2_prob, soft=20)

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

    model = LinearModel(d_input=2, d_output=1)
    model.train()

    optimizer = Adam(model.parameters(), lr=1e-3)

    with Client(n_workers=n_workers, memory_limit="8GB", threads_per_worker=1) as client:
        print(f"Dask Dashboard Link: {client.dashboard_link}")

        with catch_warnings():
            filterwarnings("ignore", category=RuntimeWarning, module="lsdb.*")

            training_dataset = iter(
                LSDBIterableDataset(
                    catalog=catalog,
                    columns=None,
                    client=client,
                    batch_lc=train_batch_size,
                    n_src=n_src,
                    partitions_per_chunk=n_workers * 2,
                    loop=True,
                    hash_range=(0.00, 0.70),
                    seed=0,
                )
            )
            validation_dataset = iter(
                LSDBIterableDataset(
                    catalog=catalog,
                    columns=None,
                    client=client,
                    batch_lc=validation_batch_size,
                    n_src=n_src,
                    partitions_per_chunk=n_workers * 4,
                    loop=True,
                    hash_range=(0.70, 0.85),
                    seed=0,
                )
            )

        val_tqdm = tqdm(range(n_validation_batches), desc="Validation batches")
        for epoch, val_batch in zip(val_tqdm, validation_dataset, strict=False):
            sum_train_loss = 0.0
            for _i_train_batch, train_batch in zip(
                range(val_batch_over_train), training_dataset, strict=False
            ):
                sum_train_loss += train_step(
                    model=model,
                    optimizer=optimizer,
                    loss=loss_fn,
                    batch=train_batch,
                )

            model.eval()
            validation_loss = evaluate_loss(
                model=model,
                loss=loss_fn,
                batch=val_batch,
            )
            model.train()

            summary_writer.add_scalar("Sum train loss", sum_train_loss, epoch)
            summary_writer.add_scalar("Validation loss", validation_loss, epoch)
            torch.save(model, epoch_model_dir / f"linear_model_{epoch:06d}.pt")

    model.eval()
    summary_writer.add_graph(model, train_batch[0])
    torch.save(model, output_dir / "linear_model.pt")
