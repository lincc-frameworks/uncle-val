from collections.abc import Callable

from flax import nnx
from jax import numpy as jnp

from uncle_val.learning.models import UncleModel


def train_step(
    *,
    model: UncleModel,
    optimizer: nnx.Optimizer,
    loss: Callable[[UncleModel, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    theta,
    flux,
    err,
) -> None:
    """Training step on a single light curve.

    Parameters
    ----------
    model : UncleModel
        Model to train, input vector size is d_input.
    optimizer : flax.optim.Optimizer
        Optimizer to use for training
    loss : callable, func(model, theta, flux, err) -> loss_value
        Loss function
    theta : array-like
        Input parameter vector for the model, (n_obs, d_input).
    flux : array-like
        Flux vector, (n_obs,).
    err : array-like
        Error vector, (n_obs,).

    Returns
    -------
    None
    """

    vals, grads = nnx.value_and_grad(lambda model_: loss(model_, theta, flux, err))(model)
    optimizer.update(model, grads)

    return vals
