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

from uncle_val.datasets.materialized import MaterializedDataLoaderContext
from uncle_val.learning.feature_importance import compute_shap_values, plot_shap_summary
from uncle_val.learning.losses import UncleLoss
from uncle_val.learning.lsdb_dataset import LSDBIterableDataset
from uncle_val.learning.models import BaseUncleModel
from uncle_val.learning.training import train_step
from uncle_val.pipelines.splits import DP1_CONFIG, SurveyConfig
from uncle_val.pipelines.utils import _launch_tfboard
from uncle_val.pipelines.validation_set_utils import get_val_stats


def get_val_workers(client: Client, device: torch.device) -> list[object] | None:
    """Get list of workers good to run the validation pipeline

    Parameters
    ----------
    client : dask.distributed.Client
        Dask client to be used for validation
    device : torch.device
        Torch device to be used for validation

    Return
    ------
    list or workers or None
        Value to pass into client.submit(..., workers=...)
    """
    # Run CUDA background tasks always on a single process to not allocate too much GPU memory
    if device.type == "cuda":
        all_workers = sorted(client.cluster.workers)
        return all_workers[:1]
    return None


def training_loop(
    *,
    catalog: lsdb.Catalog,
    columns: list[str] | None,
    model: BaseUncleModel,
    n_workers: int,
    n_src: int,
    n_lcs: int,
    train_batch_size: int,
    val_batch_size: int,
    snapshot_every: int,
    loss_fn: UncleLoss,
    val_losses: dict[str, UncleLoss],
    lr: float,
    start_tfboard: bool,
    log_activations: bool,
    output_root: str | Path,
    device: str | torch.device,
    model_name: str,
    survey_config: SurveyConfig = DP1_CONFIG,
):
    """Run a training loop for a given model on a given catalog.

    Parameters
    ----------
    catalog : lsdb.Catalog
        Catalog to train on.
    columns : list[str]
        Columns to train on.
    model : BaseUncleModel
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
    loss_fn : UncleLoss
        Loss function to use, by default soften Χ² is used.
    val_losses : dict[str, UncleLoss]
        Extra losses to compute on validation set and record, it maps name to
        loss function.
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
    survey_config : SurveyConfig, optional
        Train/val/test split boundaries. Defaults to ``DP1_CONFIG``.

    Returns
    -------
    Path
        Path to the output model.
    """
    n_train_batches = int(np.ceil(n_lcs / train_batch_size))

    device = torch.device(device)

    output_root = Path(output_root)
    output_dir = Path(output_root) / datetime.now().strftime("%Y-%m-%d_%H-%M")
    intermediate_model_dir = output_dir / "models"
    intermediate_model_dir.mkdir(parents=True, exist_ok=True)
    tmp_validation_dir = output_dir / "validation"

    if start_tfboard:
        _launch_tfboard(output_root)
    summary_writer = SummaryWriter(log_dir=str(output_dir))

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=10**-0.5, patience=16, cooldown=32, eps=1e-10)
    model = model.to(device)

    with Client(n_workers=n_workers, memory_limit="8GB", threads_per_worker=1) as client:
        try:
            print(f"Dask Dashboard Link: {client.dashboard_link}")
        except KeyError as e:
            print(f"Cannot get Dask Dashboard Link: {e}")

        validation_dataset_lsdb = LSDBIterableDataset(
            catalog=catalog,
            columns=columns,
            client=client,
            batch_lc=val_batch_size,
            n_src=n_src,
            partitions_per_chunk=n_workers * 8,
            loop=False,
            hash_range=survey_config.val_split,
            seed=1,
            device=device,
        )

        with MaterializedDataLoaderContext(
            validation_dataset_lsdb, tmp_validation_dir, cleanup=False
        ) as val_dataloader:
            val_stats_future: Future | None = None
            mean_val_loss_i = 0

            best_val_loss = tensor(float("inf"), device=device)
            best_model_path = None

            sum_train_loss = tensor(float("inf"), device=device)
            sum_grad_norm = tensor(float("inf"), device=device)
            max_abs_grad = tensor(float("inf"), device=device)

            activation_bins = np.linspace(-2, 2, 1001) if log_activations else None

            def snapshot(i):
                nonlocal \
                    val_stats_future, \
                    mean_val_loss_i, \
                    best_val_loss, \
                    best_model_path, \
                    sum_train_loss, \
                    sum_grad_norm, \
                    max_abs_grad

                model.eval()

                current_model_path = intermediate_model_dir / f"{model_name}_{i:09d}.pt"
                if not current_model_path.exists():
                    torch.save(model, current_model_path)

                if val_stats_future is None:
                    mean_val_loss = tensor(float("inf"))
                else:
                    val_stats = val_stats_future.result()
                    mean_val_loss = val_stats.pop("__training_loss__")
                    summary_writer.add_scalar("Mean validation loss", mean_val_loss, mean_val_loss_i)
                    if log_activations:
                        activation_hist = val_stats.pop("__activations_hist__")
                        summary_writer.add_histogram(
                            "Validation activations",
                            np.repeat(
                                0.5 * (activation_bins[:-1] + activation_bins[1:]),
                                np.ceil(activation_hist / activation_hist.max() * 1000).astype(int),
                            ),
                            mean_val_loss_i,
                        )
                    for mean_val_loss_name, mean_val_loss_value in val_stats.items():
                        summary_writer.add_scalar(
                            f"Mean validation loss: {mean_val_loss_name}",
                            mean_val_loss_value,
                            mean_val_loss_i,
                        )
                if mean_val_loss_i < n_train_batches - 1:
                    val_stats_future = client.submit(
                        get_val_stats,
                        model_path=current_model_path,
                        losses={"__training_loss__": loss_fn} | val_losses,
                        data_loader=val_dataloader,
                        device=device,
                        activation_bins=activation_bins,
                        workers=get_val_workers(client, device),
                    )
                else:
                    val_stats_future = None
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
                    hash_range=survey_config.train_split,
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

        if best_model_path is not None and model.input_names:
            test_dataset_lsdb = LSDBIterableDataset(
                catalog=catalog,
                columns=columns,
                client=client,
                batch_lc=val_batch_size,
                n_src=n_src,
                partitions_per_chunk=n_workers * 8,
                loop=False,
                hash_range=survey_config.test_split,
                seed=2,
                device=device,
            )
            tmp_test_dir = output_dir / "test_shap"
            with MaterializedDataLoaderContext(test_dataset_lsdb, tmp_test_dir) as test_dataloader:
                shap_values, feature_data = compute_shap_values(
                    model_path=best_model_path,
                    data_loader=test_dataloader,
                    device=device,
                )
            fig = plot_shap_summary(
                shap_values,
                feature_data,
                input_names=model.input_names,
                output_path=output_dir / "feature_importance.png",
            )
            summary_writer.add_figure("Feature importance", fig)

    model.eval()
    summary_writer.add_graph(model, train_batch[0])

    if best_model_path is None:
        raise RuntimeError("Model hasn't trained yet?")
    model_path = output_dir / f"{model_name}.pt"
    shutil.copy(best_model_path, model_path)

    return model_path
