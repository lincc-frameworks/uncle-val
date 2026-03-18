import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from captum.attr import FeaturePermutation
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


def compute_permutation_importance(
    *,
    model_path: str | Path,
    data_loader: DataLoader,
    device: torch.device,
    n_repeats: int = 5,
    rng_seed: int = 0,
) -> dict[str, np.ndarray]:
    """Compute permutation feature importance on the validation set via Captum.

    For each input feature, uses :class:`captum.attr.FeaturePermutation` to
    permute that feature's values across all observations and measure the mean
    absolute change in the predicted uncertainty factor ``u``.  A larger value
    means the model relies more heavily on that feature.

    Parameters
    ----------
    model_path : str or Path
        Path to the saved model checkpoint.
    data_loader : DataLoader
        Validation data loader (yields batches of shape
        ``(1, batch_lc, n_src, n_features)``).
    device : torch.device
        Device for model and data.
    n_repeats : int
        Number of independent permutation repeats per feature, used to
        estimate variance.
    rng_seed : int
        Seed passed to :func:`torch.manual_seed` for reproducibility.

    Returns
    -------
    dict[str, np.ndarray]
        Maps each feature name to an array of shape ``(n_repeats,)`` with the
        mean absolute change in ``u`` across observations due to permuting that
        feature.
    """
    torch.manual_seed(rng_seed)
    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()
    input_names = model.input_names

    # Accumulate per-repeat importances across all batches: list of (n_repeats, n_features)
    batch_importances: list[np.ndarray] = []

    for batch in data_loader:
        # DataLoader wraps each saved chunk with an extra dim → (1, batch_lc, n_src, n_features)
        data = batch.squeeze(0)  # (batch_lc, n_src, n_features)
        batch_lc, n_src, n_features = data.shape
        flat = data.reshape(-1, n_features)  # (batch_lc * n_src, n_features)

        def forward_func(flat_inputs, _bl=batch_lc, _ns=n_src, _nf=n_features):
            b = flat_inputs.reshape(_bl, _ns, _nf)
            with torch.no_grad():
                output = model(b)  # (batch_lc, n_src, d_output)
            return output[..., 0].reshape(-1)  # u per observation: (batch_lc * n_src,)

        fp = FeaturePermutation(forward_func)

        repeat_attrs = []
        for _ in range(n_repeats):
            attrs = fp.attribute(flat)  # (batch_lc * n_src, n_features)
            repeat_attrs.append(attrs.abs().mean(dim=0).cpu().detach().numpy())

        batch_importances.append(np.stack(repeat_attrs, axis=0))  # (n_repeats, n_features)

    # Average over batches → (n_repeats, n_features)
    mean_importances = np.mean(batch_importances, axis=0)

    return {name: mean_importances[:, i] for i, name in enumerate(input_names)}


def get_val_stats(
    *,
    model_path: str | Path,
    losses: dict[str, UncleLoss],
    data_loader: DataLoader,
    device: torch.device,
    activation_bins: np.ndarray | None,
) -> dict[str, object]:
    """Computes the loss for validation set with serialized model

    Parameters
    ----------
    model_path : Path or str
        Path to Uncle model
    losses : dict[str, UncleLoss]
        Loss functions to use for validation
    data_loader : DataLoader
        DataLoader which yields validation data.
    device : torch.device
        PyTorch device to for the data and for the model.
    activation_bins : np.ndarray | None
        If specified, return the histogram of the activations.

    Returns
    -------
    dict[str, result]
        Result of calculation: float number for each loss and activations
        histogram if bins are given (under "__activations_hist__" key).
    """
    model = torch.load(model_path, weights_only=False, map_location=device)

    def loss_fn(*args, **kwargs):
        return {name: fn(*args, **kwargs).cpu().detach().numpy() for name, fn in losses.items()}

    if activation_bins is not None:
        hook = ActivationHistHook(torch.tensor(activation_bins.astype(np.float32), device=device))
        model.register_forward_hook(hook)

    sum_loss = defaultdict(float)
    n_batches_used = 0
    for val_batch in data_loader:
        batch_loss = evaluate_loss(
            model=model,
            loss=loss_fn,
            batch=val_batch,
        )
        for name, value in batch_loss.items():
            sum_loss[name] += value
        n_batches_used += 1
    result = {name: value / n_batches_used for name, value in sum_loss.items()}

    if activation_bins is not None:
        result["__activations_hist__"] = hook.counts

    return result
