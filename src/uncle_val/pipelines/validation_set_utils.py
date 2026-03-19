from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from uncle_val.learning.losses import UncleLoss
from uncle_val.learning.training import evaluate_loss


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
