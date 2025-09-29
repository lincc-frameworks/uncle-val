from functools import partial

import torch
from torch import Tensor
from torch.distributions.chi2 import Chi2

from uncle_val.whitening import whiten_data


def _residuals_lc(flux: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
    """Residuals for a single light curve

    residuals = (flux * (1 + s) - avg_flux) / (u * err),
    avg_flux = sum(flux / u^2 / err^2) / sum(1 / u^2 / err^2)

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
    avg_flux = torch.average(flux, weights=err**-2)
    residuals = (flux - avg_flux) / err
    return residuals


def _chi2_lc(flux: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
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
    chi2 = torch.sum(torch.square(residuals))
    return chi2


def minus_ln_chi2_prob(flux: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
    """-ln(prob(chi2)) for chi2 computed for given light curves and model.

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
    chi2_func = torch.vmap(_chi2_lc, in_axes=(0, 0), out_axes=0)
    chi2_batch = chi2_func(flux, err)
    chi2 = torch.sum(chi2_batch)

    # "compile-time" values
    n_light_curves = torch.prod(torch.tensor(flux.shape[:-1]))
    n_obs_total = torch.prod(torch.tensor(flux.shape))
    degrees_of_freedom = n_obs_total - n_light_curves
    distr = Chi2(degrees_of_freedom)

    lnprob = distr.log_prob(chi2)
    return -lnprob


def kl_divergence_whiten(flux: Tensor, err: Tensor) -> Tensor:
    """KL(N(μ, σ²)|N(0,1)) where μ and σ are for the whiten light curves

    KL(N(μ, σ²)|N(0,1)) = 1/2 [μ² + σ² - ln σ² - 1]

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
    whiten_func = torch.vmap(partial(whiten_data, np=torch), in_axes=(0, 0), out_axes=0)
    z = whiten_func(flux, err)
    mu_z = torch.mean(z)
    # ddof is always 1 (not number of light curves), because "target" mu=0 for all whiten data points
    var_z = torch.var(z, correction=1)
    # https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Multivariate_normal_distributions
    kl = 0.5 * (mu_z**2 + var_z - torch.log(var_z) - 1.0)
    return kl
