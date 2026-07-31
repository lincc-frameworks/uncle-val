import math
from collections.abc import Sequence

import torch

from uncle_val.learning.models.base import BaseUncleModel
from uncle_val.utils.mag_flux import mag2flux


class FluxErrModel(BaseUncleModel):
    r"""Analytical flux-space uncertainty correction.

    The corrected flux uncertainty is a quadrature sum of the reported error and
    two additive terms:

    .. math::
        \sigma_\mathrm{corr}^2 = (u_0\,\sigma)^2 + (b\,F_\mathrm{eff})^2 + a_0^2,
        \qquad u = \sigma_\mathrm{corr} / \sigma,

    where :math:`F` is the flux and :math:`\sigma` the reported flux error.
    :math:`b` is a fractional-flux term and :math:`a_0` a constant floor;
    :math:`u_0` a global scaling of the reported errors. The functional form is
    heuristic and does not assert a specific physical origin for the correction.

    Rather than the raw flux :math:`F`, the fractional term uses a smoothly
    floored version

    .. math::
        F_\mathrm{eff} = F_\mathrm{cut}\left(1 + \mathrm{softplus}
            \left(\frac{F}{F_\mathrm{cut}} - 1\right)\right),

    which is :math:`\approx F` for :math:`F \gg F_\mathrm{cut}` and saturates at
    :math:`F_\mathrm{cut}` for :math:`F \lesssim F_\mathrm{cut}` (including
    negative, noise-dominated flux). This keeps the fractional-flux term
    bounded at the faint end without an external magnitude cut on the
    training set, and without the dead gradients a hard clip would introduce
    at the floor.

    Coefficients are per-band scalars, kept non-negative through the exponential
    of the trainable log-parameters (:math:`F_\mathrm{cut}` through a
    trainable :math:`\log_{10}`, since flux spans many orders of magnitude).
    Inputs are used in raw physical units (no normalization).

    Parameters
    ----------
    input_names : sequence of str
        Must contain a flux column (``'x'`` or ``'flux'``) and ``'err'``.
    """

    def __init__(self, input_names: Sequence[str] = ("x", "err")) -> None:
        super().__init__(input_names=input_names, outputs_s=False)
        names = self.input_names
        if "x" in names:
            self.flux_column = names.index("x")
        elif "flux" in names:
            self.flux_column = names.index("flux")
        else:
            raise ValueError("input_names must include a flux column, either 'x' or 'flux'")
        if "err" not in names:
            raise ValueError("input_names must include 'err'")
        self.err_column = names.index("err")

        # Non-negative coefficients via exp(log-parameter). Initial values keep u
        # close to unity: u0 = 1, and the additive terms start small.
        self.log_u0 = torch.nn.Parameter(torch.zeros(1))
        self.log_b = torch.nn.Parameter(torch.full((1,), -5.0))
        self.log_a0 = torch.nn.Parameter(torch.full((1,), -3.0))
        # log10, not ln, since flux_cut is naturally thought of in decades of
        # flux; initialized at the flux of the old mag<23 training cut.
        self.log10_flux_cut = torch.nn.Parameter(torch.full((1,), math.log10(mag2flux(23.0))))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the uncertainty underestimation factor ``u``."""
        flux = inputs[..., self.flux_column]
        err = inputs[..., self.err_column]

        u0 = torch.exp(self.log_u0)
        b = torch.exp(self.log_b)
        a0 = torch.exp(self.log_a0)
        flux_cut = torch.exp(self.log10_flux_cut * math.log(10.0))

        flux_eff = flux_cut * (1.0 + torch.nn.functional.softplus(flux / flux_cut - 1.0))

        corr_var = torch.square(u0 * err) + torch.square(b * flux_eff) + torch.square(a0)
        u = torch.sqrt(corr_var) / err
        return u[..., None]
