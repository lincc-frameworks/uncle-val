from collections.abc import Sequence

import jax.numpy as jnp
from flax import nnx
from jax.scipy import stats


class MLPModel(nnx.Module):
    """Multi-layer Perceptron (MLP) model for the u function

    Parameters
    ----------
    d_input : int
        Number of input parameters, e.g. length of theta
    d_middle : list of int
        Size of hidden layers, e.g. [64, 32, 16]
    d_output : int
        Number of output parameters, 1 for u, 2 for [u, l].
    dropout : float | None
        Dropout probability, do not use dropout layer if None.
    rngs : flax.nnx.Rangs
        Random number generator for parameter initialization.
    """

    def __init__(
        self,
        d_input: int,
        *,
        d_middle: Sequence[int] = (300, 300, 400),
        d_output: int = 1,
        dropout: None | float = None,
        rngs: nnx.Rngs,
    ):
        layers = []
        dims = [d_input] + list(d_middle) + [d_output]
        for i, (d1, d2) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
            layers.append(nnx.Linear(d1, d2, rngs=rngs, kernel_init=nnx.initializers.normal()))
            if i < len(dims) - 2:  # not the last layer
                layers.append(nnx.relu)
                if dropout is not None:
                    layers.append(nnx.Dropout(dropout, rngs=rngs))
        self.layers = nnx.Sequential(*layers)

    def __call__(self, x):
        """Compute the output of the model"""
        return jnp.exp(-self.layers(x))


def chi2_lc_train_step(model: nnx.Module, optimizer: nnx.Optimizer, theta, flux, err) -> None:
    """Training step on a single light curve, with chi2 probability based loss.

    This gets a single light curve, gets u=model(theta), computes chi-squared
    statistics for a constant-flux model using `flux` and `err`, and uses
    minus logarithm of chi-squared probability as the loss function.

    Parameters
    ----------
    model : flax.nnx.Module
        Model to train, input vector size is d_input.
    optimizer : flax.optim.Optimizer
        Optimizer to use for training
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

    def minus_lnprob_chi2(model):
        u = model(theta)[:, 0]
        total_err = u * err
        avg_flux = jnp.average(flux, weights=total_err**-2)
        chi2 = jnp.sum(jnp.square((flux - avg_flux) / total_err))
        lnprob = stats.chi2.logpdf(chi2, jnp.size(flux) - 1)
        return -lnprob

    loss, grads = nnx.value_and_grad(minus_lnprob_chi2)(model)
    optimizer.update(model, grads)

    return loss
