# from collections.abc import Callable, Sequence
#
#
# class UncleModel(nnx.Module):
#     """Base class for u-function learning
#
#     You must re-implement __init__() to call super and to assign .module.
#     The output is either 1-d or 2-d:
#     - 0th index is -ln(uu), so u = exp(-output[0])
#     - 1st index is ln(s - 1), so s = exp(output[1]) - 1
#     The original residual is defined as: (flux - avg_flux) / err,
#     where avg_flux is sum(flux / err^2) / sum(1 / err^2).
#     The transformed uncertainty is defined is corrected_err = u*err,
#     the transformed flux is corrected_flux = flux * (1 + s),
#     so the transformed residual is (flux (1 + s) - avg_flux) / (u * err),
#     where avg_flux is sum(flux / (u * err)^2) / sum(1 / (u * err)^2).
#
#     Parameters
#     ----------
#     d_input : int
#         Number of input parameters, e.g. length of theta
#     d_output : int
#         Number of output parameters, 1 for u, 2 for [u, l].
#     rngs : flax.nnx.Rngs
#         Random number generator for parameter initialization
#     """
#
#     module: Callable
#
#     def __init__(self, *, d_input, d_output, rngs):
#         self.d_input = d_input
#         if d_output not in [1, 2]:
#             raise ValueError("d_output must be 1 (for u) or 2 (for u and s)")
#         self.d_output = d_output
#         self.rngs = rngs
#         self.outputs_s = d_output == 2
#
#     def __call__(self, x):
#         """Compute the output of the model"""
#         output = self.module(x)
#         u = jnp.exp(-output[..., 0])
#         if not self.outputs_s:
#             return u[..., None]
#         s = jnp.expm1(output[..., 1])
#         return jnp.stack([u, s], axis=-1)
#
#     def corrections(self, x):
#         """Outputs a tuple of u and s correction factors"""
#         model_output = self(x)
#         u = model_output[..., 0]
#         s = model_output[..., 1] if self.outputs_s else 0.0
#         return u, s
#
#
# class MLPModel(UncleModel):
#     """Multi-layer Perceptron (MLP) model for the Uncle function
#
#     Parameters
#     ----------
#     d_input : int
#         Number of input parameters, e.g. length of theta
#     d_middle : list of int
#         Size of hidden module, e.g. [64, 32, 16]
#     d_output : int
#         Number of output parameters, 1 for u, 2 for [u, l].
#     dropout : float | None
#         Dropout probability, do not use dropout layer if None.
#     rngs : flax.nnx.Rangs
#         Random number generator for parameter initialization.
#     """
#
#     def __init__(
#         self,
#         *,
#         d_input: int,
#         d_middle: Sequence[int] = (300, 300, 400),
#         d_output: int = 1,
#         dropout: None | float = None,
#         rngs: nnx.Rngs,
#     ):
#         super().__init__(d_input=d_input, d_output=d_output, rngs=rngs)
#         self.d_middle = list(d_middle)
#         self.dropout = dropout
#
#         layers = []
#         dims = [self.d_input] + self.d_middle + [self.d_output]
#         for i, (d1, d2) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
#             layers.append(nnx.Linear(d1, d2, rngs=self.rngs, kernel_init=nnx.initializers.normal()))
#             if i < len(dims) - 2:  # not the last layer
#                 layers.append(nnx.relu)
#                 if self.dropout is not None:
#                     layers.append(nnx.Dropout(self.dropout, rngs=self.rngs))
#         self.module = nnx.Sequential(*layers)
#
#
# class LinearModel(UncleModel):
#     """A linear model for the Uncle function
#
#     Parameters
#     ----------
#     d_input : int
#         Number of input parameters, e.g. length of theta
#     d_output : int
#         Number of output parameters, 1 for u, 2 for [u, l].
#     rngs : flax.nnx.Rngs
#         Random number generator for parameter initialization
#     """
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.module = nnx.Linear(
#             self.d_input, self.d_output, rngs=self.rngs, kernel_init=nnx.initializers.normal()
#         )
import numpy as np
import torch
from hyrax.models import hyrax_model
from torch import Tensor

from uncle_val.consts import MAG_AB_ZP_NJY
from uncle_val.utils import import_by_name
from uncle_val.utils.config_validation import validate_hyrax_batch_size


@hyrax_model
class LinearModel(torch.nn.Module):
    flux_scaler_scale_mag = 24
    flux_scaler_scale = 10**(-0.4 * (flux_scaler_scale_mag - MAG_AB_ZP_NJY))
    flux_scaler_norm_mag = 14
    flux_scaler_norm_flux = 10**(-0.4 * (flux_scaler_norm_mag - MAG_AB_ZP_NJY))
    flux_scaler_norm = np.arcsinh(flux_scaler_norm_flux / flux_scaler_scale).item()

    err_scaler_lg_lower = 0.0
    err_scaler_lg_upper = 5.0
    err_scaler_lg_interval = err_scaler_lg_upper - err_scaler_lg_lower

    def __init__(self, config, shape) -> None:
        super().__init__()
        self.layers = torch.nn.Linear(shape, 1)
        self.loss = import_by_name(config["model"]["loss_fn"])
        self.n_src = config["data_set"]["LSDBDataGenerator"]["n_src"]
        if not isinstance(self.n_src, int):
            raise ValueError(f"expected integer value for `config['data_set']['LSDBDataGenerator']['n_src']`, got {self.n_src}")

        hyrax_batch_size = config["data_loader"]["batch_size"]
        if not isinstance(hyrax_batch_size, int):
            raise ValueError(f"expected integer value for `config['data_loader']['batch_size']`, got {self.batch_size}")
        validate_hyrax_batch_size(config)

        self.obj_batch_size = hyrax_batch_size // self.n_src


    def scale_flux(self, flux: Tensor) -> Tensor:
        return torch.arcsinh(flux / self.flux_scaler_scale) / self.flux_scaler_norm

    def scale_err(self, err: Tensor) -> Tensor:
        lg_err = torch.log10(err)
        return (lg_err - self.err_scaler_lg_lower) / self.err_scaler_lg_interval

    def forward(self, x: Tensor | tuple[Tensor, Tensor]) -> Tensor:
        if isinstance(x, tuple):
            x, _ = x

        flux = x[..., 0]
        err = x[..., 1]
        theta = torch.stack([self.scale_flux(flux), self.scale_err(err)], dim=-1)

        return self.layers(theta)

    @staticmethod
    def to_tensor(data_dict: dict) -> tuple[Tensor, Tensor]:
        x = torch.tensor(list(data_dict.values()))
        y = torch.tensor([data_dict["flux"], data_dict["err"]])
        return x, y

    def train_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, float]:
        x, y = batch[0].reshape(self.obj_batch_size, self.n_src), batch[1].reshape(self.obj_batch_size, self.n_src)
        flux, err = y[..., 0], y[..., 1]

        self.optimizer.zero_grad()

        result = self.forward(x)
        u = result[..., 0]
        s = 1.0
        corrected_flux = flux * (1.0 + s)
        corrected_err = err * u

        loss = self.loss_fn(corrected_flux, corrected_err)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}
