import math
from collections.abc import Sequence

import torch

from uncle_val.learning.models.base import BaseUncleModel
from uncle_val.utils.mag_flux import mag2flux


class FluxErrPolyModel(BaseUncleModel):
    r"""Polynomial flux-space uncertainty correction.

    The uncertainty underestimation factor is directly parametrized as a
    total-degree-2 polynomial in the *normalized* reported error
    :math:`\hat\sigma` and normalized, flux-floored flux :math:`\hat F_\mathrm{eff}`
    (the same ``norm_err``/``norm_flux`` transforms
    :class:`~uncle_val.learning.models.base.UncleScaler` applies for the
    MLP-style models):

    .. math::
        u = \left| c_{00} + c_{10}\,\hat\sigma + c_{01}\,\hat F_\mathrm{eff}
            + c_{20}\,\hat\sigma^2 + c_{11}\,\hat\sigma \hat F_\mathrm{eff}
            + c_{02}\,\hat F_\mathrm{eff}^2 \right|,

    The six coefficients are unconstrained (free to be positive, negative, or
    zero), unlike :class:`~uncle_val.learning.models.flux_err.FluxErrModel`,
    whose terms are kept non-negative through an exponential
    reparametrization. Because this raw polynomial is not constrained to be
    positive, it can come out negative; only :math:`|u|` matters downstream
    (it enters the loss effectively squared), so the sign is a don't-care and
    :func:`torch.abs` disambiguates it into a positive factor.

    Normalizing the inputs (rather than using raw flux/error, as
    ``FluxErrModel`` does) keeps the quadratic terms well-scaled: raw DP1
    fluxes range over many orders of magnitude, so an unconstrained
    coefficient on a raw ``flux**2`` term can blow up within a handful of
    gradient steps. Normalized inputs are O(1), so the polynomial output
    stays well-behaved. The tradeoff is that the coefficients no longer have
    a direct raw-physical-unit interpretation the way ``FluxErrModel``'s do.

    As in ``FluxErrModel``, the flux entering the normalization is first
    smoothly floored (same ``flux_cut`` mechanism, same raw units, so the
    floor and the flux it acts on live in the same space before either gets
    normalized):

    .. math::
        F_\mathrm{eff} = F_\mathrm{cut}\left(1 + \mathrm{softplus}
            \left(\frac{F}{F_\mathrm{cut}} - 1\right)\right),

    which is :math:`\approx F` for :math:`F \gg F_\mathrm{cut}` and saturates
    at :math:`F_\mathrm{cut}` for :math:`F \lesssim F_\mathrm{cut}` (including
    negative, noise-dominated flux). This keeps the model from fitting noise
    at the faint end without an external magnitude cut on the training set.
    The functional form is heuristic, like its sibling model, and does not
    assert a specific physical origin for the correction.

    Coefficients are per-band scalars.

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

        # Free (unconstrained) polynomial coefficients, over normalized inputs.
        # Only the constant term has a physically-informed prior -- our best
        # estimate of the overall correction from the earlier raw-unit
        # FluxErrModel g-band fit (u0=1.38656). The linear/normalized-input
        # and curvature terms have no equivalent prior in this coordinate
        # system, so they start at 0.
        self.c_00 = torch.nn.Parameter(torch.full((1,), 1.38656))
        self.c_10 = torch.nn.Parameter(torch.zeros(1))
        self.c_01 = torch.nn.Parameter(torch.zeros(1))
        self.c_20 = torch.nn.Parameter(torch.zeros(1))
        self.c_11 = torch.nn.Parameter(torch.zeros(1))
        self.c_02 = torch.nn.Parameter(torch.zeros(1))
        # log10, not ln, since flux_cut is naturally thought of in decades of
        # flux; initialized at the flux of the old mag<23 training cut, same
        # as FluxErrModel's.
        self.log10_flux_cut = torch.nn.Parameter(torch.full((1,), math.log10(mag2flux(23.0))))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the uncertainty underestimation factor ``u``."""
        flux = inputs[..., self.flux_column]
        err = inputs[..., self.err_column]

        flux_cut = torch.exp(self.log10_flux_cut * math.log(10.0))
        flux_eff = flux_cut * (1.0 + torch.nn.functional.softplus(flux / flux_cut - 1.0))

        norm_flux = self.scaler.norm_flux(flux_eff)
        norm_err = self.scaler.norm_err(err)

        u_raw = (
            self.c_00
            + self.c_10 * norm_err
            + self.c_01 * norm_flux
            + self.c_20 * torch.square(norm_err)
            + self.c_11 * norm_err * norm_flux
            + self.c_02 * torch.square(norm_flux)
        )
        u = torch.abs(u_raw)
        return u[..., None]
