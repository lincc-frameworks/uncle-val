from jax import numpy as jnp
from jax.scipy import stats

from uncle_val.learning.models import UncleModel


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

    model_output = model(theta)
    u = model_output[..., 0]
    s = model_output[..., 1] if model.outputs_s else 0.0

    corrected_flux = flux * (1.0 + s)
    corrected_err = u * err

    avg_flux = jnp.average(corrected_flux, weights=corrected_err**-2)
    chi2 = jnp.sum(jnp.square((corrected_flux - avg_flux) / corrected_err))
    lnprob = stats.chi2.logpdf(chi2, n - 1)

    return -lnprob
