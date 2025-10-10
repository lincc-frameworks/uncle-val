from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path
from warnings import catch_warnings, filterwarnings

import lsdb
import torch
from dask.distributed import Client
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from uncle_val.learning.losses import minus_ln_chi2_prob
from uncle_val.learning.lsdb_dataset import LSDBIterableDataset
from uncle_val.learning.models import UncleModel
from uncle_val.learning.training import evaluate_loss, train_step
from uncle_val.pipelines.utils import _launch_tfboard


def training_loop(
    *,
    catalog: lsdb.Catalog,
    columns: list[str] | None,
    model: UncleModel,
    n_workers: int,
    n_src: int,
    n_lcs: int,
    train_batch_size: int,
    val_batch_over_train: int,
    loss_fn: Callable | None,
    lr: float,
    start_tfboard: bool,
    output_root: str | Path,
    device: str | torch.device,
    model_name: str,
):
    """Run a training loop for a given model on a given catalog.

    Parameters
    ----------
    catalog : lsdb.Catalog
        Catalog to train on.
    columns : list[str]
        Columns to train on.
    model : UncleModel
        Model to train.
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
    lr : float
        Learning rate.
    start_tfboard : bool
        Whether to start a TensorBoard session.
    output_root : str or Path
        Where to save the intermediate results.
    device : str or torch.device
        Torch device to use for training.
    model_name : str
        Name of the model to use in the output Torch filename.

    Returns
    -------
    Path
        Path to the output model.
    """
    output_dir = Path(output_root) / datetime.now().strftime("%Y-%m-%d_%H-%M")
    epoch_model_dir = output_dir / "models"
    epoch_model_dir.mkdir(parents=True, exist_ok=True)

    validation_batch_size = val_batch_over_train * train_batch_size
    n_validation_batches = n_lcs // validation_batch_size

    output_root = Path(output_root)
    output_dir = Path(output_root) / datetime.now().strftime("%Y-%m-%d_%H-%M")
    epoch_model_dir = output_dir / "models"
    epoch_model_dir.mkdir(parents=True, exist_ok=True)

    if loss_fn is None:
        loss_fn = partial(minus_ln_chi2_prob, soft=20)

    if start_tfboard:
        _launch_tfboard(output_root)
    summary_writer = SummaryWriter(log_dir=str(output_dir))

    optimizer = Adam(model.parameters(), lr=lr)
    model = model.to(device)

    with Client(n_workers=n_workers, memory_limit="8GB", threads_per_worker=1) as client:
        print(f"Dask Dashboard Link: {client.dashboard_link}")

        with catch_warnings():
            # LSDB complains that we return a series, not a dataframe, from map_partitions
            filterwarnings("ignore", category=RuntimeWarning, module="lsdb.*")

            training_dataset = iter(
                LSDBIterableDataset(
                    catalog=catalog,
                    columns=columns,
                    client=client,
                    batch_lc=train_batch_size,
                    n_src=n_src,
                    partitions_per_chunk=n_workers * 2,
                    loop=True,
                    hash_range=(0.00, 0.70),
                    seed=0,
                    device=device,
                )
            )
            validation_dataset = iter(
                LSDBIterableDataset(
                    catalog=catalog,
                    columns=columns,
                    client=client,
                    batch_lc=validation_batch_size,
                    n_src=n_src,
                    partitions_per_chunk=n_workers * 8,
                    loop=True,
                    hash_range=(0.70, 0.85),
                    seed=0,
                    device=device,
                )
            )

        val_tqdm = tqdm(range(n_validation_batches), desc="Validation batches")
        for val_step, val_batch in zip(val_tqdm, validation_dataset, strict=False):
            sum_train_loss = torch.tensor(0.0).to(device)
            sum_grad_norm = torch.tensor(0.0).to(device)
            model.train()
            for _i_train_batch, train_batch in zip(
                range(val_batch_over_train), training_dataset, strict=False
            ):
                train_loss = train_step(
                    model=model,
                    optimizer=optimizer,
                    loss=loss_fn,
                    batch=train_batch,
                )
                sum_train_loss += train_loss.detach()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                sum_grad_norm += grad_norm

            model.eval()
            validation_loss = evaluate_loss(
                model=model,
                loss=loss_fn,
                batch=val_batch,
            )

            summary_writer.add_scalar("Sum train loss", sum_train_loss, val_step)
            summary_writer.add_scalar("Validation loss", validation_loss, val_step)
            summary_writer.add_scalar("Mean grad norm", sum_grad_norm / val_batch_over_train, val_step)
            torch.save(model, epoch_model_dir / f"{model_name}_{val_step:06d}.pt")

    model.eval()
    summary_writer.add_graph(model, train_batch[0])
    model_path = output_dir / f"{model_name}.pt"
    torch.save(model, model_path)

    return model_path
