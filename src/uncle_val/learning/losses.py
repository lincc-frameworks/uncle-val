from abc import ABC, abstractmethod
from functools import partial
from typing import Literal

import torch
from torch import Tensor
from torch.distributions import Chi2

from uncle_val.stat_tests import epps_pulley_standard_norm
from uncle_val.whitening import whiten_data


class UncleLoss(ABC):
    """Base class for loss functions

    Parameters
    ----------
    lmbd : float | None
        Penalty term factor for Tikhonov regularization. None means no
        Tikhonov regularization.
    """

    def __init__(self, *, lmbd: float | None):
        self.lmbd = lmbd

    @abstractmethod
    def penalty_scale_factor(self, shape: torch.Size) -> Tensor:
        """Multiplication factor to use with penalty loss."""
        raise NotImplementedError

    def penalty_term(self, input_shape: torch.Size, model_outputs: Tensor) -> Tensor:
        """Compute loss term based on output vector"""
        if self.lmbd is None:
            return torch.zeros_like(model_outputs.flatten()[0])
        norm_outputs = torch.linalg.norm(model_outputs, dim=-1)
        mean_norm = torch.mean(norm_outputs)
        return self.lmbd * self.penalty_scale_factor(input_shape) * mean_norm

    @abstractmethod
    def lc_term(self, flux: Tensor, err: Tensor) -> Tensor:
        """Compute loss term based on light curve"""
        raise NotImplementedError

    def __call__(self, flux: Tensor, err: Tensor, model_outputs: Tensor) -> Tensor:
        """Compute loss"""
        if self.lmbd is None:
            return self.lc_term(flux, err)
        return self.lc_term(flux, err) + self.penalty_term(flux.shape, model_outputs)


class SoftenLoss(UncleLoss, ABC):
    """Base class for soften loss functions

    Parameters
    ----------
    lmbd : float | None
        Penalty term factor for Tikhonov regularization. None means no
        Tikhonov regularization.
    soft : float | None
         Softness factor for the light-curve loss term. None means no
         softening.
    """

    def __init__(self, *, lmbd: float | None, soft: float | None):
        super().__init__(lmbd=lmbd)
        self.soft = soft
        self.whiten_func = torch.vmap(partial(whiten_data, np=torch))

    def compute_z(self, flux: Tensor, err: Tensor) -> Tensor:
        """Compute the whiten values and soften them if needed

        Parameters
        ----------
        flux : torch.Tensor
            Corrected flux vector, (n_batch, n_src,)
        err : jnp.ndarray
            Corrected error vector, (n_batch, n_src,)

        Returns
        -------
        torch.Tensor, of shape (n_batch, n_src,)
            Soften whiten sources
        """
        orig_shape = flux.shape
        shape_2d = (torch.prod(torch.tensor(orig_shape[:-1]), dtype=int), orig_shape[-1])
        output_shape = *orig_shape[:-1], orig_shape[-1] - 1
        z = self.whiten_func(flux.reshape(shape_2d), err.reshape(shape_2d)).reshape(output_shape)

        if self.soft is None:
            return z
        return self.soft * torch.tanh(z / self.soft)


def _residuals_lc(flux: Tensor, err: Tensor) -> Tensor:
    """Residuals for a single light curve

    residuals = (flux - avg_flux) / err,
    avg_flux = sum(flux / err^2) / sum(1 / err^2)

    Parameters
    ----------
    flux : torch.Tensor
        Flux vector, (n_src,)
    err : torch.Tensor
        Error vector, (n_src,)

    Returns
    -------
    torch.Tensor
        Residuals vector, (n_src,)
    """
    weights = 1.0 / torch.square(err)
    avg_flux = torch.sum(weights * flux) / torch.sum(weights)

    residuals = (flux - avg_flux) / err
    return residuals


class Chi2BasedLoss(SoftenLoss, ABC):
    """Abstract class for chi2 log probability losses

    Parameters
    ----------
    lmbd : float | None
        Penalty term factor for Tikhonov regularization. None means no
        Tikhonov regularization.
    soft : float | None
         Softness factor for the light-curve loss term. None means no
         softening.
    """

    def __init__(self, *, lmbd: float | None, soft: float | None):
        super().__init__(lmbd=lmbd, soft=soft)
        self.chi2_func = torch.vmap(self._chi2_lc)

    def _chi2_lc(self, flux: Tensor, err: Tensor) -> Tensor:
        """chi2 for a single light curve

        Parameters
        ----------
        flux : torch.Tensor
            Flux vector, (n_src,)
        err : torch.Tensor
            Error vector, (n_src,)

        Returns
        -------
        jnp.ndarray, of shape ()
            chi2 value
        """
        residuals = _residuals_lc(flux, err)
        if self.soft is not None:
            residuals = self.soft * torch.tanh(residuals / self.soft)

        chi2 = torch.sum(torch.square(residuals))
        return chi2

    @abstractmethod
    def _degrees_of_freedom(self, shape) -> Tensor:
        raise NotImplementedError

    def _chi2_distr(self, shape) -> Chi2:
        dof = self._degrees_of_freedom(shape)
        distr = Chi2(dof)
        return distr

    def penalty_scale_factor(self, shape) -> Tensor:
        """Multiplication factor to use with penalty loss."""
        dof = self._degrees_of_freedom(shape)
        distr = self._chi2_distr(shape)
        return torch.abs(distr.log_prob(dof))


class MinusLnChi2ProbTotal(Chi2BasedLoss):
    """-ln(prob(chi2)) for chi2 computed for given light curves and model.

    This class combines all light curves together for a single Chi2 value.

    Parameters
    ----------
    lmbd : float | None
        Penalty term factor for Tikhonov regularization. None means no
        Tikhonov regularization.
    soft : float | None
         Softness factor for the light-curve loss term. None means no
         softening.
    """

    def __init__(self, *, lmbd: float | None, soft: float | None):
        super().__init__(lmbd=lmbd, soft=soft)

    def _degrees_of_freedom(self, shape) -> Tensor:
        n_light_curves = torch.prod(torch.tensor(shape[:-1]))
        n_obs_total = torch.prod(torch.tensor(shape))
        return n_obs_total - n_light_curves

    def lc_term(self, flux: Tensor, err: Tensor) -> Tensor:
        """Get the loss

        Parameters
        ----------
        flux : torch.Tensor
            Corrected flux vector, (n_batch, n_src,)
        err : torch.Tensor
            Corrected error vector, (n_batch, n_src,)

        Returns
        -------
        torch.Tensor, of shape ()
            Loss value
        """
        chi2_batch = self.chi2_func(flux, err)
        chi2 = torch.sum(chi2_batch)
        distr = self._chi2_distr(flux.shape)
        lnprob = distr.log_prob(chi2)
        return -lnprob


class MinusLnChi2ProbLc(Chi2BasedLoss):
    """-ln(prob(chi2)) for chi2 computed for given light curves and model.

    This class combines all light curves together for a single Chi2 value.

    Parameters
    ----------
    lmbd : float | None
        Penalty term factor for Tikhonov regularization. None means no
        Tikhonov regularization.
    soft : float | None
         Softness factor for the light-curve loss term. None means no
         softening.
    """

    def __init__(self, *, lmbd: float | None, soft: float | None):
        super().__init__(lmbd=lmbd, soft=soft)

    def _degrees_of_freedom(self, shape) -> Tensor:
        n_obs_per_lc = shape[-1]
        return n_obs_per_lc - 1

    def lc_term(self, flux: Tensor, err: Tensor) -> Tensor:
        """Get the loss

        Parameters
        ----------
        flux : torch.Tensor
            Corrected flux vector, (n_batch, n_src,)
        err : torch.Tensor
            Corrected error vector, (n_batch, n_src,)

        Returns
        -------
        torch.Tensor, of shape ()
            Loss value
        """
        chi2_batch = self.chi2_func(flux, err)
        distr = self._chi2_distr(flux.shape)
        lnprob = distr.log_prob(chi2_batch)
        mean_lnprob = torch.mean(lnprob)
        return -mean_lnprob


def minus_ln_chi2_prob_loss(
    *, lmbd: float | None, soft: float | None, kind: Literal["accum"] | Literal["mean"]
) -> Chi2BasedLoss:
    """Construct a -ln(prob(Chi2)) loss object.

    Parameters
    ----------
    lmbd : float | None
        Penalty term factor for Tikhonov regularization. None means no
        Tikhonov regularization.
    soft : float | None
        Softness parameter or no softness (`None`).
    kind : Literal['accum', 'mean']
        How to compute the loss for multiple light curves:
        - 'accum': sum per-light-curve Chi^2 values and compute a single
          log-probability value.
        - 'mean': compute log-prob per light curve and average them.
    """
    match kind:
        case "accum":
            return MinusLnChi2ProbTotal(lmbd=lmbd, soft=soft)
        case "mean":
            return MinusLnChi2ProbLc(lmbd=lmbd, soft=soft)
    raise ValueError(f"Unknown loss kind: {kind}")


class KLWhitenBasedLoss(SoftenLoss, ABC):
    """Base class for KL divergence based losses.

    Parameters
    ----------
    lmbd : float | None
        Penalty term factor for Tikhonov regularization. None means no
        Tikhonov regularization.
    soft : float | None
         Softness factor for the light-curve loss term. None means no
         softening.
    """

    def penalty_scale_factor(self, shape) -> Tensor:
        """Multiplication factor to use with penalty loss."""
        return torch.tensor(1)

    @staticmethod
    def compute_kl(mu: Tensor, var: Tensor) -> Tensor:
        """Compute KL divergence between N(mu, var) and N(0,1)

        # https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Multivariate_normal_distributions

        Parameters
        ----------
        mu : torch.Tensor
            Mean vector
        var : torch.Tensor
            Unbiased variance vector

        Returns
        -------
        torch.Tensor
            KL divergence between N(mu, var) and N(0,1)
        """
        return 0.5 * (torch.square(mu) + var - torch.log(var) - 1.0)


class KLWhitenTotal(KLWhitenBasedLoss):
    """KL divergence between all whiten sources and standard distribution

    KL(N(μ, σ²)|N(0,1)) where μ and σ are for the whiten light curves

    KL(N(μ, σ²)|N(0,1)) = 1/2 [μ² + σ² - ln σ² - 1]"""

    def lc_term(self, flux: Tensor, err: Tensor) -> Tensor:
        """Compute the loss

        Parameters
        ----------
        flux : torch.Tensor
            Corrected flux vector, (n_batch, n_src,)
        err : jnp.ndarray
            Corrected error vector, (n_batch, n_src,)

        Returns
        -------
        torch.Tensor, of shape ()
            Loss value
        """
        z = self.compute_z(flux, err)
        mu_z = torch.mean(z)
        # ddof is always 1 (not number of light curves), because "target" mu is shared for all of them
        var_z = torch.var(z, correction=1)
        # https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Multivariate_normal_distributions
        kl = self.compute_kl(mu_z, var_z)
        return kl


class KLWhitenLc(KLWhitenBasedLoss):
    """Mean of KL divergences between per-lc whiten sources and N(0, 1)

    KL(N(μ, σ²)|N(0,1)) where μ and σ are for the whiten light curves

    KL(N(μ, σ²)|N(0,1)) = 1/2 [μ² + σ² - ln σ² - 1]"""

    def lc_term(self, flux: Tensor, err: Tensor) -> Tensor:
        """Compute the loss

        Parameters
        ----------
        flux : torch.Tensor
            Corrected flux vector, (n_batch, n_src,)
        err : jnp.ndarray
            Corrected error vector, (n_batch, n_src,)

        Returns
        -------
        torch.Tensor, of shape ()
            Loss value
        """
        z = self.compute_z(flux, err)
        mu_z = torch.mean(z, dim=-1)
        var_z = torch.var(z, correction=1, dim=-1)
        kl = self.compute_kl(mu_z, var_z)
        mean_kl = torch.mean(kl)
        return mean_kl


def kl_divergence_whiten_loss(
    *, lmbd: float | None, soft: float | None, kind: Literal["accum"] | Literal["mean"]
) -> KLWhitenBasedLoss:
    """Construct an KL divergence loss object

    Parameters
    ----------
    lmbd : float | None
        Penalty term factor for Tikhonov regularization. None means no
        Tikhonov regularization.
    soft : float | None
        Softness parameter or no softness (`None`).
    kind : Literal['accum', 'mean']
        How to compute the loss for multiple light curves:
        - 'accum': use whiten sources from all light curves and compute the KL
          divergence for it.
        - 'mean': compute KL divergence per light curve and average them.
    """
    match kind:
        case "accum":
            return KLWhitenTotal(lmbd=lmbd, soft=soft)
        case "mean":
            return KLWhitenLc(lmbd=lmbd, soft=soft)
    raise ValueError(f"Unknown loss kind: {kind}")


class EPWhitenBasedLoss(SoftenLoss, ABC):
    """Base class for Epps-Pulley-based losses.

    Parameters
    ----------
    lmbd : float | None
        Penalty term factor for Tikhonov regularization. None means no
        Tikhonov regularization.
    soft : float | None
         Softness factor for the light-curve loss term. None means no
         softening.
    bins : torch.Tensor
        Frequency integration grid.
    """

    def __init__(self, lmbd: float | None, soft: float | None, bins: Tensor) -> None:
        super().__init__(lmbd=lmbd, soft=soft)
        self.bins = bins

    def penalty_scale_factor(self, shape) -> Tensor:
        """Multiplication factor to use with penalty loss."""
        return torch.tensor(1e-3)


class EBWhitenTotal(EPWhitenBasedLoss):
    """Epps-Pulley statistic computed for all whiten sources.

    Parameters
    ----------
    lmbd : float | None
        Penalty term factor for Tikhonov regularization. None means no
        Tikhonov regularization.
    soft : float | None
         Softness factor for the light-curve loss term. None means no
         softening.
    bins : torch.Tensor
        Frequency integration grid. If None, linspace(-5, 5, 17) is used.
    """

    def __init__(self, lmbd: float | None, soft: float | None, bins: Tensor | None) -> None:
        if bins is None:
            bins = torch.linspace(-5.0, 5.0, 17)
        super().__init__(lmbd=lmbd, soft=soft, bins=bins)

    def lc_term(self, flux: Tensor, err: Tensor) -> Tensor:
        """Compute the loss

        Parameters
        ----------
        flux : torch.Tensor
            Corrected flux vector, (n_batch, n_src,)
        err : jnp.ndarray
            Corrected error vector, (n_batch, n_src,)

        Returns
        -------
        torch.Tensor, of shape ()
            Loss value
        """
        z = self.compute_z(flux, err)
        return epps_pulley_standard_norm(z.flatten(), bins=self.bins.to(z.device))


class EBWhitenLc(EPWhitenBasedLoss):
    """Epps-Pulley statistic averaged over whiten light curves.

    Parameters
    ----------
    lmbd : float | None
        Penalty term factor for Tikhonov regularization. None means no
        Tikhonov regularization.
    soft : float | None
         Softness factor for the light-curve loss term. None means no
         softening.
    bins : torch.Tensor
        Frequency integration grid. If None, linspace(-3, 3, 9) is used.
    """

    def __init__(self, lmbd: float | None, soft: float | None, bins: Tensor | None) -> None:
        if bins is None:
            bins = torch.linspace(-3.0, 3.0, 9)
        super().__init__(lmbd=lmbd, soft=soft, bins=bins)
        self.ep_func = torch.vmap(lambda z: epps_pulley_standard_norm(z, bins=self.bins.to(z.device)))

    def lc_term(self, flux: Tensor, err: Tensor) -> Tensor:
        """Compute the loss

        Parameters
        ----------
        flux : torch.Tensor
            Corrected flux vector, (n_batch, n_src,)
        err : jnp.ndarray
            Corrected error vector, (n_batch, n_src,)

        Returns
        -------
        torch.Tensor, of shape ()
            Loss value
        """
        z = self.compute_z(flux, err)
        stat = self.ep_func(z)
        return torch.mean(stat)


def epps_pulley_whiten_loss(
    *,
    lmbd: float | None,
    soft: float | None,
    kind: Literal["accum"] | Literal["mean"],
    bins: Tensor | None = None,
) -> EPWhitenBasedLoss:
    """Construct an KL divergence loss object

    Parameters
    ----------
    lmbd : float | None
        Penalty term factor for Tikhonov regularization. None means no
        Tikhonov regularization.
    soft : float | None
        Softness parameter or no softness (`None`).
    kind : Literal['accum', 'mean']
        How to compute the loss for multiple light curves:
        - 'accum': use whiten sources from all light curves and compute the KL
          divergence for it.
        - 'mean': compute KL divergence per light curve and average them.
    bins : torch.Tensor or None
        Edges of integration bins in the frequency space. None uses pre-defined
        grid.
    """
    match kind:
        case "accum":
            return EBWhitenTotal(lmbd=lmbd, soft=soft, bins=bins)
        case "mean":
            return EBWhitenLc(lmbd=lmbd, soft=soft, bins=bins)
    raise ValueError(f"Unknown loss kind: {kind}")
