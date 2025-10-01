import numpy
from numba.extending import register_jitable


@register_jitable
def _stable_decomposition(a, np=None):
    """Matrix decomposition for a constant-fit covariance matrix"""
    if np is None:
        np = numpy

    n = len(a)
    n_float = np.asarray(n, dtype=a.dtype)
    mat_d = np.diag(a)
    one_vec = np.ones((n, 1), dtype=a.dtype) / np.sqrt(n_float)

    inv_a_sum = np.sum(1.0 / a)
    u = np.ones((n, 1)) / np.sqrt(inv_a_sum)

    # Explicit eigenvalue along ones-vector:
    lambda_min = (one_vec.T @ (mat_d - u @ u.T) @ one_vec)[0]
    lambda_min = np.maximum(lambda_min, np.asarray(1e-8))

    # Construct orthonormal basis explicitly:
    mat_q, _ = np.linalg.qr(np.eye(n) - one_vec @ one_vec.T)
    mat_q = mat_q[:, : n - 1]  # explicitly enforce correct dimensions (n x n-1)

    # Project onto orthogonal subspace:
    m_orth = mat_q.T @ (mat_d - u @ u.T) @ mat_q
    eigvals_orth, eigvecs_orth = np.linalg.eigh(m_orth)

    # Combine eigenvectors/eigenvalues explicitly:
    # We could use np.insert, but it doesn't exist in torch
    eigvals_full = np.concatenate((lambda_min, eigvals_orth))
    eigvecs_full = np.hstack((one_vec, mat_q @ eigvecs_orth))

    # Cholesky-like decomposition
    mat_b = eigvecs_full @ np.diag(np.sqrt(eigvals_full))

    return mat_b


def whiten_data(x, v, *, np=None):
    """Whitening of Gaussian time-independent data

    It is assumed that each x[i] ~ N(mu, v[i]),
    where the mean mu is unknown, but constant value,
    variance is known v, and all x[i] are independent.

    Parameters
    ----------
    x : array, (n_src,)
        Input signal
    v : array, (n_src,)
        Signal variance
    np : module, optional
        Numpy-compatible module, numpy, jax.numpy and torch are supported
    """
    if np is None:
        np = numpy

    mean_v = np.mean(v)
    v = v / mean_v

    decomposed = _stable_decomposition(v, np=np)
    transform = np.linalg.inv(decomposed)

    # We can use np.average(x, weights=1/v), but it doesn't exist in torch
    weights = 1 / v
    mu = np.sum(x * weights) / np.sum(weights)

    residual = (x - mu) / np.sqrt(mean_v)
    z = transform @ residual

    return z
