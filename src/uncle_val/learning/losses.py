import jax
from jax import numpy as jnp
from jax.scipy import stats

from uncle_val.learning.models import UncleModel
from uncle_val.whitening import whiten_data


def _residuals_lc(model: UncleModel, theta: jnp.ndarray, flux: jnp.ndarray, err: jnp.ndarray) -> jnp.ndarray:
    """Light curve residuals

    residuals = (flux * (1 + s) - avg_flux) / (u * err),
    avg_flux = sum(flux / u^2 / err^2) / sum(1 / u^2 / err^2)

    Parameters
    ----------
    model : UncleModel
        Model to compute loss for, input dimensions is d_input
    theta : jnp.ndarray
        Parameter vector, (n_obs, d_input)
    flux : jnp.ndarray
        Flux vector, (n_obs,)
    err : jnp.ndarray
        Error vector, (n_obs,)

    Returns
    -------
    jnp.ndarray
        Residual vector, (n_obs,)
    """
    u, s = model.corrections(theta)
    corrected_flux = flux * (1.0 + s)
    corrected_err = u * err

    avg_flux = jnp.average(corrected_flux, weights=corrected_err**-2)
    residuals = (corrected_flux - avg_flux) / corrected_err
    return residuals


def _chi2_lc(model: UncleModel, theta: jnp.ndarray, flux: jnp.ndarray, err: jnp.ndarray) -> jnp.ndarray:
    """chi2 for a single light curve

    Parameters
    ----------
    model : UncleModel
        Model to compute loss for, input dimensions is d_input
    theta : jnp.ndarray
        Parameter vector, (n_obs, d_input)
    flux : jnp.ndarray
        Flux vector, (n_obs,)
    err : jnp.ndarray
        Error vector, (n_obs,)

    Returns
    -------
    jnp.ndarray, of shape ()
        chi2 value
    """
    residuals = _residuals_lc(model, theta, flux, err)
    chi2 = jnp.sum(jnp.square(residuals))
    return chi2


def minus_ln_chi2_prob(
    model: UncleModel, theta: jnp.ndarray, flux: jnp.ndarray, err: jnp.ndarray
) -> jnp.ndarray:
    """-ln(prob(chi2)) for chi2 computed for given light curves and model.

    Parameters
    ----------
    model : UncleModel
        Model to compute loss for, input dimensions is d_input
    theta : jnp.ndarray
        Parameter vector, (n_batch, n_obs, d_input)
    flux : jnp.ndarray
        Flux vector, (n_batch, n_obs,)
    err : jnp.ndarray
        Error vector, (n_batch, n_obs,)

    Returns
    -------
    jnp.ndarray, of shape ()
        Loss value
    """
    chi2_func = jax.vmap(_chi2_lc, in_axes=(None, 0, 0, 0), out_axes=0)
    chi2_batch = chi2_func(model, theta, flux, err)
    chi2 = jnp.sum(chi2_batch)

    # "compile-time" values
    n_light_curves = jnp.prod(jnp.asarray(flux.shape[:-1]))
    n_obs_total = jnp.prod(jnp.asarray(flux.shape))
    degrees_of_freedom = n_obs_total - n_light_curves

    lnprob = stats.chi2.logpdf(chi2, df=degrees_of_freedom)
    return -lnprob


def _whiten_light_curve(
    model: UncleModel, theta: jnp.ndarray, flux: jnp.ndarray, err: jnp.ndarray
) -> jnp.ndarray:
    """Whitening the light curve with the given model.

    Parameters
    ----------
    model : UncleModel
        Model to compute loss for, input dimensions is d_input
    theta : jnp.ndarray
        Parameter vector, (n_obs, d_input)
    flux : jnp.ndarray
        Flux vector, (n_obs,)
    err : jnp.ndarray
        Error vector, (n_obs,)

    Returns
    -------
    jnp.ndarray, (n_obs,)
        Whitened light curve vector
    """
    u, s = model.corrections(theta)
    corrected_flux = flux * (1.0 + s)
    corrected_err = u * err
    z = whiten_data(corrected_flux, corrected_err**2, np=jnp)
    return z


def kl_divergence_whiten(
    model: UncleModel, theta: jnp.ndarray, flux: jnp.ndarray, err: jnp.ndarray
) -> jnp.ndarray:
    """KL(N(μ, σ²)|N(0,1)) where μ and σ are for the whiten light curves

    KL(N(μ, σ²)|N(0,1)) = 1/2 [μ² + σ² - ln σ² - 1]

    Parameters
    ----------
    model : UncleModel
        Model to compute loss for, input dimensions is d_input
    theta : jnp.ndarray
        Parameter vector, (n_batch, n_obs, d_input)
    flux : jnp.ndarray
        Flux vector, (n_batch, n_obs,)
    err : jnp.ndarray
        Error vector, (n_batch, n_obs,)

    Returns
    -------
    jnp.ndarray
        Loss value
    """
    whiten_func = jax.vmap(_whiten_light_curve, in_axes=(None, 0, 0, 0), out_axes=0)
    z = whiten_func(model, theta, flux, err)
    mu_z = jnp.mean(z)
    # ddof is always 1 (not number of light curves), because "target" mu=0 for all whiten data points
    var_z = jnp.var(z, ddof=1)
    # https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Multivariate_normal_distributions
    kl = 0.5 * (mu_z**2 + var_z - jnp.log(var_z) - 1.0)
    return kl
