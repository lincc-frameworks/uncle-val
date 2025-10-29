from torch import Tensor
from torch.optim import Optimizer

from uncle_val.learning.losses import UncleLoss
from uncle_val.learning.models import UncleModel


def train_step(
    *,
    optimizer: Optimizer,
    model: UncleModel,
    loss: UncleLoss,
    batch: Tensor,
) -> Tensor:
    """Training step on a single light curve.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to use for training
    model : UncleModel
        Model to train, input vector size is d_input.
    loss : callable, udf(flux, err) -> loss_value
        Loss function to call on corrected fluxes and errors.
    batch : torch.Tensor, (n_batch, n_obj, n_features]
        Data to train on. [..., 0] is assumed to be flux,
        [..., 1] is assumed to be error.

    Returns
    -------
    torch.Tensor
        Loss value
    """
    optimizer.zero_grad()

    loss_values_grads = evaluate_loss(
        model=model,
        loss=loss,
        batch=batch,
    )
    loss_values_grads.backward()

    optimizer.step()
    loss_values = loss_values_grads.detach()

    return loss_values


def evaluate_loss(
    *,
    model: UncleModel,
    loss: UncleLoss,
    batch: Tensor,
):
    """Evaluate and return loss"""
    flux, err = batch[..., 0], batch[..., 1]

    model_output = model(batch)

    u = model_output[..., 0]
    corrected_err = err * u

    if model.outputs_s:
        s = model_output[..., 1]
        corrected_flux = flux * (1.0 + s)
    else:
        corrected_flux = flux

    values = loss(corrected_flux, corrected_err, model_output)

    return values
