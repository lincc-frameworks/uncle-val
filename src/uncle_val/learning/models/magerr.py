import math
from collections.abc import Sequence

import torch

from uncle_val.learning.models.base import BaseUncleModel
from uncle_val.utils.mag_flux import fluxerr2magerr, mag2flux, magerr2fluxerr


class MagErrModel(BaseUncleModel):
    """Base class for magnitude-error physics models

    Subclasses must assign ``self.module`` to a ``torch.nn.Module`` whose
    ``forward()`` returns the systematic magnitude error in centi-magnitudes
    (i.e. the value is multiplied by ``1e-2`` to get magnitudes).

    The corrected flux error is computed by adding the systematic magnitude
    error in quadrature to the photon-noise magnitude error:

        new_mag_err = hypot(mag_err, systematic_mag_err)
        u = magerr2fluxerr(new_mag_err) / flux_err

    Parameters
    ----------
    input_names : list of str
        Names of input dimensions. Must include a flux column (``'flux'`` or
        ``'x'``) and an error column (``'err'``).
    """

    flux_floor = mag2flux(30.0)
    ln10_0_4 = 0.4 * math.log(10.0)

    module: torch.nn.Module

    def __init__(self, *, input_names: Sequence[str]) -> None:
        super().__init__(input_names=input_names, outputs_s=False)

        if "flux" in input_names:
            self.flux_column = self.input_names.index("flux")
        elif "x" in input_names:
            self.flux_column = self.input_names.index("x")
        else:
            raise ValueError("input_names must include flux name, either 'flux' or 'x'")

        if "err" in input_names:
            self.err_column = self.input_names.index("err")
        else:
            raise ValueError("input_names must include 'err'")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the output of the model"""
        systematic_mag_err = 1e-2 * self.module(self.norm_inputs(inputs))

        flux = torch.maximum(
            inputs[..., self.flux_column], torch.tensor(self.flux_floor, device=inputs.device)
        )
        flux_err = inputs[..., self.err_column]
        mag_err = fluxerr2magerr(flux=flux, flux_err=flux_err)
        new_mag_err = torch.hypot(mag_err, systematic_mag_err)
        new_flux_err = magerr2fluxerr(flux=flux, mag_err=new_mag_err)
        u = new_flux_err / flux_err
        return u[..., None]


class ConstantMagErrModel(MagErrModel):
    """Uncle function adds a constant systematic magnitude error in quadrature

    Parameters
    ----------
    input_names : list of str
        Names of input dimensions. Must include a flux column (``'flux'`` or
        ``'x'``) and an error column (``'err'``).
    """

    def __init__(self, input_names: Sequence[str]) -> None:
        super().__init__(input_names=input_names)
        self.addition_centi_mag_err = torch.nn.Parameter(torch.ones(1))

    def module(self, inputs: torch.Tensor) -> torch.Tensor:
        """Trainable systematic magnitude error addition in centi-magnitudes."""
        return self.addition_centi_mag_err
