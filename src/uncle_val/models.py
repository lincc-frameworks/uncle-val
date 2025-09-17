from collections.abc import Callable, Sequence

import jax.numpy as jnp
from flax import nnx
from jax.scipy import stats


class UncleModel(nnx.Module):
    """Base class for u-function models

    You must re-implement __init__() to call super and to assign .module.
    The output is either 1-d or 2-d:
    - 0th index is -ln(uu), so u = exp(-output[0])
    - 1st index is ln(s - 1), so s = exp(output[1]) - 1
    The original residual is defined as: (flux - avg_flux) / err,
    where avg_flux is sum(flux / err^2) / sum(1 / err^2).
    The transformed uncertainty is defined is corrected_err = u*err,
    the transformed flux is corrected_flux = flux * (1 + s),
    so the transformed residual is (flux (1 + s) - avg_flux) / (u * err),
    where avg_flux is sum(flux / (u * err)^2) / sum(1 / (u * err)^2).

    Parameters
    ----------
    d_input : int
        Number of input parameters, e.g. length of theta
    d_output : int
        Number of output parameters, 1 for u, 2 for [u, l].
    rngs : flax.nnx.Rngs
        Random number generator for parameter initialization
    """

    module: Callable

    def __init__(self, *, d_input, d_output, rngs):
        self.d_input = d_input
        if d_output not in [1, 2]:
            raise ValueError("d_output must be 1 (for u) or 2 (for u and s)")
        self.d_output = d_output
        self.rngs = rngs
        self.outputs_s = d_output == 2

    def __call__(self, x):
        """Compute the output of the model"""
        output = self.module(x)
        u = jnp.exp(-output[..., 0])
        if not self.outputs_s:
            return u[..., None]
        s = jnp.expm1(output[..., 1])
        return jnp.stack([u, s], axis=-1)


class MLPModel(UncleModel):
    """Multi-layer Perceptron (MLP) model for the Uncle function

    Parameters
    ----------
    d_input : int
        Number of input parameters, e.g. length of theta
    d_middle : list of int
        Size of hidden module, e.g. [64, 32, 16]
    d_output : int
        Number of output parameters, 1 for u, 2 for [u, l].
    dropout : float | None
        Dropout probability, do not use dropout layer if None.
    rngs : flax.nnx.Rangs
        Random number generator for parameter initialization.
    """

    def __init__(
        self,
        *,
        d_input: int,
        d_middle: Sequence[int] = (300, 300, 400),
        d_output: int = 1,
        dropout: None | float = None,
        rngs: nnx.Rngs,
    ):
        super().__init__(d_input=d_input, d_output=d_output, rngs=rngs)
        self.d_middle = list(d_middle)
        self.dropout = dropout

        layers = []
        dims = [self.d_input] + self.d_middle + [self.d_output]
        for i, (d1, d2) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
            layers.append(nnx.Linear(d1, d2, rngs=self.rngs, kernel_init=nnx.initializers.normal()))
            if i < len(dims) - 2:  # not the last layer
                layers.append(nnx.relu)
                if self.dropout is not None:
                    layers.append(nnx.Dropout(self.dropout, rngs=self.rngs))
        self.module = nnx.Sequential(*layers)


class LinearModel(UncleModel):
    """A linear model for the Uncle function

    Parameters
    ----------
    d_input : int
        Number of input parameters, e.g. length of theta
    d_output : int
        Number of output parameters, 1 for u, 2 for [u, l].
    rngs : flax.nnx.Rngs
        Random number generator for parameter initialization
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module = nnx.Linear(
            self.d_input, self.d_output, rngs=self.rngs, kernel_init=nnx.initializers.normal()
        )


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
