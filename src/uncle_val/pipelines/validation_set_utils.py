import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from uncle_val.learning.losses import UncleLoss
from uncle_val.learning.lsdb_dataset import LSDBIterableDataset
from uncle_val.learning.training import evaluate_loss


class ValidationDataset(Dataset):
    """On-disk validation dataset.

    Parameters
    ----------
    data_dir : Path
        Directory with *.pt files
    device : torch.device
        PyTorch device to load data into.
    """

    def __init__(self, *, data_dir: Path, device: torch.device):
        self.paths = sorted(data_dir.glob("*.pt"))
        self.device = device

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return torch.load(self.paths[index], map_location=self.device)


class ValidationDataLoaderContext:
    """Context manager that gives a torch.DataLoader object

    Temporary directory is created for the validation dataset,
    and torch data loader fetches data from it.

    Parameters
    ----------
    input_dataset : LSDBIterableDataset
        Dataset which yields validation data.
    tmp_dir : Path or str
        Temporary directory to save data into. It will be created on
        the context entrance, and filled with the data.
        It will be deleted on the context exit.
    """

    def __init__(self, input_dataset: LSDBIterableDataset, tmp_dir: Path | str):
        self.input_dataset = input_dataset
        self.tmp_dir = Path(tmp_dir)

    def _serialize_data(self):
        for i, chunk in enumerate(self.input_dataset):
            if i >= 1e6:
                raise RuntimeError("Number of chunks is more than a million!!!")
            torch.save(chunk, self.tmp_dir / f"chunk_{i:05d}.pt")

    def __enter__(self) -> DataLoader:
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._serialize_data()
        dataset = ValidationDataset(data_dir=self.tmp_dir, device=self.input_dataset.device)
        return DataLoader(
            dataset,
            shuffle=False,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.tmp_dir)


class ActivationHistHook:
    """Torch model hook for making an activations histogram

    Parameters
    ----------
    activation_bins : torch.tensor, (n_bins+1,)
        Histogram bins

    Attributes
    ----------
    bins : torch.tensor, (n_bins+1,)
        Histogram bins
    counts : np.ndarray, (n_bins,)
        Current histogram counts, as float numbers
    """

    def __init__(self, activation_bins):
        self.bins = activation_bins
        self.counts = np.zeros(len(self.bins) - 1, dtype=np.float32)

    def __call__(self, module, input, output):
        """Hook function, it updates the histogram of the activations

        Parameters
        ----------
        module : torch.nn.Module
            PyTorch module
        input : torch.tensor
            Input tensor for the module
        output : torch.tensor
            Output tensor for the module
        """
        del module, input
        torch_counts, _torch_bins = torch.histogram(output, self.bins)
        self.counts += torch_counts.detach().cpu().numpy()


def get_val_stats(
    *,
    model_path: str | Path,
    loss: UncleLoss,
    data_loader: DataLoader,
    device: torch.device,
    activation_bins: np.ndarray | None,
) -> float:
    """Computes the loss for validation set with serialized model

    Parameters
    ----------
    model_path : Path or str
        Path to Uncle model
    loss : UncleLoss
        Loss function to use for validation
    data_loader : DataLoader
        DataLoader which yields validation data.
    device : torch.device
        PyTorch device to for the data and for the model.
    activation_bins : np.ndarray | None
        If specified, return the histogram of the activations.
    """
    model = torch.load(model_path, weights_only=False, map_location=device)

    if activation_bins is not None:
        hook = ActivationHistHook(torch.tensor(activation_bins.astype(np.float32), device=device))
        model.register_forward_hook(hook)

    sum_val_loss = 0.0
    n_val_batches_used = 0
    for val_batch in data_loader:
        sum_val_loss += (
            evaluate_loss(
                model=model,
                loss=loss,
                batch=val_batch,
            )
            .cpu()
            .detach()
            .numpy()
        )
        n_val_batches_used += 1
    mean_val_loss = sum_val_loss / n_val_batches_used

    if activation_bins is None:
        return mean_val_loss
    return mean_val_loss, hook.counts
