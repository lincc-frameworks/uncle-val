import shutil
from datetime import datetime
from pathlib import Path

import lsdb
import numpy as np
import torch
from dask.distributed import Client, Future
from torch import tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from uncle_val.learning.losses import UncleLoss
from uncle_val.learning.lsdb_dataset import LSDBIterableDataset
from uncle_val.learning.models import UncleModel
from uncle_val.learning.training import train_step
from uncle_val.pipelines.splits import TRAIN_SPLIT, VALIDATION_SPLIT
from uncle_val.pipelines.utils import _launch_tfboard
from uncle_val.pipelines.validation_set_utils import ValidationDataLoaderContext, get_val_stats


def training_loop(
    *,
    catalog: lsdb.Catalog,
    columns: list[str] | None,
    model: UncleModel,
    n_workers: int,
    n_src: int,
    n_lcs: int,
    train_batch_size: int,
    val_batch_size: int,
    snapshot_every: int,
    loss_fn: UncleLoss,
    lr: float,
    start_tfboard: bool,
    log_activations: bool,
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
        Batch size for training set.
    val_batch_size : int
        Batch size for validation set.
    snapshot_every : int
        Snapshot model and metrics every this many training batches.
    loss_fn : Callable or None
        Loss function to use, by default soften Χ² is used.
    lr : float
        Learning rate.
    start_tfboard : bool
        Whether to start a TensorBoard session.
    log_activations : bool
        Whether to log validation activations with TensorBoard session.
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
    n_train_batches = int(np.ceil(n_lcs / train_batch_size))

    output_root = Path(output_root)
    output_dir = Path(output_root) / datetime.now().strftime("%Y-%m-%d_%H-%M")
    intermediate_model_dir = output_dir / "models"
    intermediate_model_dir.mkdir(parents=True, exist_ok=True)
    tmp_validation_dir = output_dir / "validation"

    if start_tfboard:
        _launch_tfboard(output_root)
    summary_writer = SummaryWriter(log_dir=str(output_dir))

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=10**-0.5, patience=16, cooldown=32)
    model = model.to(device)

    with Client(n_workers=n_workers, memory_limit="8GB", threads_per_worker=1) as client:
        print(f"Dask Dashboard Link: {client.dashboard_link}")

        validation_dataset_lsdb = LSDBIterableDataset(
            catalog=catalog,
            columns=columns,
            client=client,
            batch_lc=val_batch_size,
            n_src=n_src,
            partitions_per_chunk=n_workers * 8,
            loop=False,
            hash_range=VALIDATION_SPLIT,
            seed=1,
            device=device,
        )

        with ValidationDataLoaderContext(validation_dataset_lsdb, tmp_validation_dir) as val_dataloader:
            mean_val_loss_future: Future | None = None
            mean_val_loss_i = 0

            best_val_loss = tensor(float("inf"), device=device)
            best_model_path = None

            sum_train_loss = tensor(float("inf"), device=device)
            sum_grad_norm = tensor(float("inf"), device=device)
            max_abs_grad = tensor(float("inf"), device=device)

            activation_bins = np.linspace(-2, 2, 1001) if log_activations else None

            def snapshot(i):
                nonlocal \
                    mean_val_loss_future, \
                    mean_val_loss_i, \
                    best_val_loss, \
                    best_model_path, \
                    sum_train_loss, \
                    sum_grad_norm, \
                    max_abs_grad

                model.eval()

                current_model_path = intermediate_model_dir / f"{model_name}_{i:09d}.pt"
                torch.save(model, current_model_path)

                if mean_val_loss_future is None:
                    mean_val_loss = tensor(float("inf"))
                else:
                    if log_activations:
                        mean_val_loss, activation_hist = mean_val_loss_future.result()
                        summary_writer.add_histogram(
                            "Validation activations",
                            np.repeat(
                                0.5 * (activation_bins[:-1] + activation_bins[1:]),
                                np.ceil(activation_hist / activation_hist.max() * 1000).astype(int),
                            ),
                            mean_val_loss_i,
                        )
                    else:
                        mean_val_loss = mean_val_loss_future.result()
                    summary_writer.add_scalar("Mean validation loss", mean_val_loss, mean_val_loss_i)
                mean_val_loss_future = client.submit(
                    get_val_stats,
                    model_path=current_model_path,
                    loss=loss_fn,
                    data_loader=val_dataloader,
                    device=device,
                    activation_bins=activation_bins,
                )
                mean_val_loss_i = i

                n_train_batches_used = snapshot_every if i % snapshot_every == 0 else i % snapshot_every
                summary_writer.add_scalar("Mean train loss", sum_train_loss / n_train_batches_used, i)
                summary_writer.add_scalar("Mean grad norm", sum_grad_norm / n_train_batches_used, i)
                summary_writer.add_scalar("Max abs grad", max_abs_grad, i)
                sum_train_loss = tensor(0.0, device=device)
                sum_grad_norm = tensor(0.0, device=device)
                max_abs_grad = tensor(0.0, device=device)

                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    best_model_path = current_model_path

                scheduler.step(mean_val_loss)
                summary_writer.add_scalar("Learning rate", scheduler.get_last_lr()[0], i)

                model.train()

            training_dataset_iter = iter(
                LSDBIterableDataset(
                    catalog=catalog,
                    columns=columns,
                    client=client,
                    batch_lc=train_batch_size,
                    n_src=n_src,
                    partitions_per_chunk=n_workers * 2,
                    loop=True,
                    hash_range=TRAIN_SPLIT,
                    seed=0,
                    device=device,
                )
            )

            for i_train_batch, train_batch in zip(
                tqdm(range(n_train_batches), desc="Training batch"), training_dataset_iter, strict=False
            ):
                if i_train_batch == 0:
                    snapshot(i_train_batch)

                train_loss = train_step(
                    model=model,
                    optimizer=optimizer,
                    loss=loss_fn,
                    batch=train_batch,
                )
                sum_train_loss += train_loss.detach()
                sum_grad_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                max_abs_grad = torch.maximum(
                    max_abs_grad, torch.max(tensor([torch.max(p.grad) for p in model.parameters()]))
                )

                if i_train_batch % snapshot_every == 0 and i_train_batch > 0:
                    snapshot(i_train_batch)
            # Call twice to record the final validation loss
            snapshot(i_train_batch)
            snapshot(i_train_batch)

    model.eval()
    summary_writer.add_graph(model, train_batch[0])

    if best_model_path is None:
        raise RuntimeError("Model hasn't trained yet?")
    model_path = output_dir / f"{model_name}.pt"
    shutil.copy(best_model_path, model_path)

    return model_path
