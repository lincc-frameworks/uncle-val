from jax import numpy as jnp
from jax.scipy import stats

from uncle_val.learning.models import UncleModel
from uncle_val.whitening import whiten_data


def minus_ln_chi2_prob(
    model: UncleModel, theta: jnp.ndarray, flux: jnp.ndarray, err: jnp.ndarray
) -> jnp.ndarray:
    """-ln(prob(chi2)) for chi2 computed for a given light curve and model.

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
        Loss value
    """
    n = jnp.size(flux)

    u, s = model.corrections(theta)

    corrected_flux = flux * (1.0 + s)
    corrected_err = u * err

    avg_flux = jnp.average(corrected_flux, weights=corrected_err**-2)
    chi2 = jnp.sum(jnp.square((corrected_flux - avg_flux) / corrected_err))
    lnprob = stats.chi2.logpdf(chi2, df=n - 1)

    return -lnprob


def _whiten_light_curve(
    model: UncleModel, theta: jnp.ndarray, flux: jnp.ndarray, err: jnp.ndarray
) -> jnp.ndarray:
    """Whitening the light curve with the given model."""
    u, s = model.corrections(theta)
    corrected_flux = flux * (1.0 + s)
    corrected_err = u * err
    z = whiten_data(corrected_flux, corrected_err**2, np=jnp)
    return z


def kl_divergence_whiten(
    model: UncleModel, theta: jnp.ndarray, flux: jnp.ndarray, err: jnp.ndarray
) -> jnp.ndarray:
    """KL(N(μ, σ²)|N(0,1)) where μ and σ are for the whiten light curve

    KL(N(μ, σ²)|N(0,1)) = 1/2 [μ² + σ² - ln σ² - 1]

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
        Loss value
    """
    z = _whiten_light_curve(model, theta, flux, err)
    mu_z = jnp.mean(z)
    var_z = jnp.var(z, ddof=1)
    kl = 0.5 * (mu_z**2 + var_z - jnp.log(var_z) - 1.0)
    return kl
