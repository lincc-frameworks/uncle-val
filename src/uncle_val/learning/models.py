from collections.abc import Sequence
from functools import cached_property

import torch

from uncle_val.utils.mag_flux import mag2flux


class UncleModel(torch.nn.Module):
    """Base class for u-function learning

    You must re-implement __init__() to call super().__init__()
    and to assign `.module`.
    You may also want to override `.norm_inputs()` to account for
    extra input dimensions. The base class implementation assumes that
    the first two dimensions are for flux and error.

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
        Number of input dimensions, at least two (first parameters are flux and
        err).
    d_output : int
        Number of output parameters, 1 for u, 2 for [u, l].
    """

    module: torch.nn.Module

    norm_flux_scale_mag = 23.0
    norm_flux_max_mag = 14.0

    norm_err_lgmin = -1.0
    norm_err_lgmax = 4.0

    def __init__(self, *, d_input: int, d_output: int) -> None:
        super().__init__()

        if d_input < 2:
            raise ValueError("d_input must be >= 2")
        self.d_input = d_input
        if d_output not in {1, 2}:
            raise ValueError("d_output must be 1 (for u) or 2 (for u and s)")
        self.d_output = d_output
        self.outputs_s = d_output == 2

    @cached_property
    def norm_flux_scale(self) -> torch.Tensor:
        """Scale factor for flux normalization."""
        return torch.tensor(mag2flux(self.norm_flux_scale_mag))

    @cached_property
    def norm_flux_amplitude(self) -> torch.Tensor:
        """Amplitude factor for flux normalization."""
        norm_flux_max = torch.tensor(mag2flux(self.norm_flux_max_mag))
        return 1.0 / torch.arcsinh(norm_flux_max / self.norm_flux_scale)

    def norm_flux(self, flux: torch.Tensor) -> torch.Tensor:
        """Normalize flux."""
        return torch.arcsinh(flux / self.norm_flux_scale) * self.norm_flux_amplitude

    @cached_property
    def norm_err_lgrange(self) -> torch.Tensor:
        """Min-max range for decimal logarithm of error, for normalization."""
        return torch.tensor(self.norm_err_lgmax - self.norm_err_lgmin)

    def norm_err(self, err: torch.Tensor) -> torch.Tensor:
        """Normalize flux error."""
        lg_err = torch.log10(err)
        return (lg_err - self.norm_err_lgmin) / self.norm_flux_scale

    def norm_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes batch"""
        x = x.clone()
        x[..., 0] = self.norm_flux(x[..., 0])
        x[..., 1] = self.norm_err(x[..., 1])
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the output of the model"""
        output = self.module(self.norm_inputs(x))
        # u, uncertainty underestimation
        output[..., 0] = torch.exp(-output[..., 0])
        # s, flux shift
        if self.outputs_s:
            output[..., 1] = torch.expm1(output[..., 1])
        return output


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
    """

    def __init__(
        self,
        *,
        d_input: int,
        d_middle: Sequence[int] = (300, 300, 400),
        d_output: int = 1,
        dropout: None | float = None,
    ):
        super().__init__(d_input=d_input, d_output=d_output)
        self.d_middle = list(d_middle)
        self.dropout = dropout

        layers = []
        dims = [self.d_input] + self.d_middle + [self.d_output]
        for i, (d1, d2) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
            layers.append(torch.nn.Linear(d1, d2))
            if i < len(dims) - 2:  # not the last layer
                layers.append(torch.nn.ReLU())
                if self.dropout is not None:
                    layers.append(torch.nn.Dropout(self.dropout))
        self.module = torch.nn.Sequential(*layers)


class LinearModel(UncleModel):
    """A linear model for the Uncle function

    Parameters
    ----------
    d_input : int
        Number of input parameters, at least two.
    d_output : int
        Number of output parameters, 1 for u, 2 for [u, l].
    """

    def __init__(self, d_input: int, d_output: int) -> None:
        super().__init__(d_input=d_input, d_output=d_output)

        self.module = torch.nn.Linear(self.d_input, self.d_output, bias=True)


class ConstantModel(UncleModel):
    """Uncle function returns constant

    Parameters
    ----------
    d_input : int
        Number of input parameters, at least two.
    d_output : int
        Number of output parameters, 1 for u, 2 for [u, l].
    """

    def __init__(self, d_input: int, d_output: int) -> None:
        super().__init__(d_input=d_input, d_output=d_output)

        self.vector = torch.nn.Parameter(torch.zeros(self.d_output))

    def module(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the output of the model"""
        shape = x.shape[:-1]
        return self.vector.repeat(*shape, 1)
