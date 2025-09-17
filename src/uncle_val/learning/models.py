from collections.abc import Callable, Sequence

from flax import nnx
from jax import numpy as jnp


class UncleModel(nnx.Module):
    """Base class for u-function learning

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

    def corrections(self, x):
        """Outputs a tuple of u and s correction factors"""
        model_output = self(x)
        u = model_output[..., 0]
        s = model_output[..., 1] if self.outputs_s else 0.0
        return u, s


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
