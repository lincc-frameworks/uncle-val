import numpy
from numba.extending import register_jitable


@register_jitable
def _whitening_matrix(sigma, np=None):
    """Returns matrix A such that [A @ (x - average_x)] ~ N(0, I_{n-1})"""
    if np is None:
        np = numpy

    ones = np.ones_like(sigma)
    eye = np.diag(ones)

    inv_sigma = np.reciprocal(sigma)
    weight = np.square(inv_sigma)
    sum_weight = np.sum(weight)

    weighted_mean_projector = eye - np.outer(ones, weight) / sum_weight

    # Householder reflection
    normalized_inv_sigma = inv_sigma / np.sqrt(sum_weight)
    householder_vector = np.concatenate((normalized_inv_sigma[:-1], normalized_inv_sigma[-1:] + 1.0))
    householder_vector_norm = householder_vector / np.linalg.norm(householder_vector)
    householder_matrix = eye - 2.0 * np.outer(householder_vector_norm, householder_vector_norm)

    orthonormal_rows = householder_matrix[:-1, :]

    whitening_transform = orthonormal_rows @ np.diag(inv_sigma) @ weighted_mean_projector
    return whitening_transform


def whiten_data(x, sigma, *, np=None):
    """Whitening of Gaussian time-independent data

    It is assumed that each x[i] ~ N(mu, v[i]),
    where the mean mu is unknown, but constant value,
    variance is known v, and all x[i] are independent.

    Parameters
    ----------
    x : array, (n_src,)
        Input signal
    sigma : array, (n_src,)
        Signal standard deviation
    np : module, optional
        Numpy-compatible module, numpy, jax.numpy and torch are supported

    Returns
    -------
    array, (n_src - 1,)
        Whiten values
    """
    if np is None:
        np = numpy

    transform = _whitening_matrix(sigma, np=np)

    # We can use np.average(x, weights=1/sigma**2), but it doesn't exist in torch
    weights = np.reciprocal(np.square(sigma))
    mu = np.sum(x * weights) / np.sum(weights)

    residual = x - mu
    z = transform @ residual

    return z
