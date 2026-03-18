import shutil
from collections import defaultdict
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


class _FlatWrapper(torch.nn.Module):
    """Wraps an UncleModel to accept flat ``(N, n_features)`` input.

    Treats N observations as a single light curve ``(1, N, n_features)``
    so that each observation is processed independently, and returns the
    per-observation uncertainty factor ``u`` as a 1-D tensor ``(N,)``.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, flat_inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning u per observation."""
        output = self.model(flat_inputs.unsqueeze(0))  # (1, N, d_output)
        return output[0, :, :1]  # (N, 1) — shap requires 2D output


def compute_shap_values(
    *,
    model_path: str | Path,
    data_loader: DataLoader,
    device: torch.device,
    n_background: int = 100,
    n_test: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values for the predicted uncertainty factor ``u``.

    Uses :class:`shap.GradientExplainer` with a flat ``(N, n_features)``
    wrapper around the model.  Background and test observations are drawn
    from the first batches of *data_loader*.

    Parameters
    ----------
    model_path : str or Path
        Path to the saved model checkpoint.
    data_loader : DataLoader
        Validation data loader (yields batches of shape
        ``(1, batch_lc, n_src, n_features)``).
    device : torch.device
        Device for model and data.
    n_background : int
        Number of background observations for the SHAP baseline.
    n_test : int
        Number of observations to explain.

    Returns
    -------
    shap_values : np.ndarray, shape ``(n_test, n_features)``
        SHAP values for each observation and feature.
    feature_data : np.ndarray, shape ``(n_test, n_features)``
        Raw feature values corresponding to *shap_values*, used for
        colouring the beeswarm plot.
    """
    import shap

    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()

    # Collect flat observations from validation batches
    obs: list[torch.Tensor] = []
    n_needed = n_background + n_test
    for batch in data_loader:
        flat = batch.squeeze(0).reshape(-1, batch.shape[-1])  # (batch_lc*n_src, n_features)
        obs.append(flat)
        if sum(t.shape[0] for t in obs) >= n_needed:
            break
    all_obs = torch.cat(obs, dim=0)[:n_needed]

    background = all_obs[:n_background]
    test_data = all_obs[n_background:n_needed]

    wrapped = _FlatWrapper(model)
    explainer = shap.GradientExplainer(wrapped, background)
    shap_values = np.array(explainer.shap_values(test_data))
    # GradientExplainer returns (n_test, n_features, n_outputs); drop the output dim
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 0]

    return shap_values, test_data.cpu().detach().numpy()


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
