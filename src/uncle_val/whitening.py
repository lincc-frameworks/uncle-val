import numpy
from numba.extending import register_jitable


@register_jitable
def whitening_operator(sigma, np=None):
    """Produces a whitening matrix for the linear squares problem.

    E.g. returns matrix A such that A @ x ~ N(0, I_{n-1}),
    where x_i ~ N(mu, sigma_i), mu is unknown but constant, and sigma_i
    are known.

    Parameters
    ----------
    sigma : array, (n_src,)
        Standard deviation vector of the observations
    np : module, optional
        Numpy-compatible module, numpy, jax.numpy and torch are supported

    Returns
    -------
    array, (n_src - 1, n_src)
        Whitening matrix
    """
    if np is None:
        np = numpy

    ones = np.ones_like(sigma)
    eye = np.diag(ones)

    inv_sigma = np.reciprocal(sigma)
    weight = np.square(inv_sigma)
    inv_sum_weight = np.reciprocal(np.sum(weight))

    ### Make Householder reflection matrix
    # Normal vector to the projection hyperplane, which is defined by the projection matrix representing
    # the relative residuals’ covariance.
    unit_projection_vector = inv_sigma * np.sqrt(inv_sum_weight)
    # Householder vector is normal to the reflection hyperplane, given by the Householder matrix.
    # The following is the same as: householder_vector = unit_projection_vector + [1, 0, 0, ..., 0]
    h_vector = np.concatenate((unit_projection_vector[:1] + 1.0, unit_projection_vector[1:]))
    householder_matrix = eye - 2.0 * np.outer(h_vector, h_vector) / (h_vector @ h_vector)

    to_residuals = eye - np.outer(ones, weight) * inv_sum_weight
    to_rel_residuals = np.diag(inv_sigma)
    to_whiten = householder_matrix[1:, :]

    return to_whiten @ to_rel_residuals @ to_residuals


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

    transform_operator = whitening_operator(sigma, np=np)
    z = transform_operator @ x

    return z
