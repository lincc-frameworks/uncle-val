import numpy as np


def stable_decomposition(a):
    """Matrix decomposition for constant-fit covariance matrix"""
    n = len(a)
    mat_d = np.diag(a)
    one_vec = np.ones((n, 1)) / np.sqrt(n)

    inv_a_sum = np.sum(1 / a)
    u = np.ones((n, 1)) / np.sqrt(inv_a_sum)

    # Explicit eigenvalue along ones-vector:
    lambda_min = float((one_vec.T @ (mat_d - u @ u.T) @ one_vec).item())
    assert lambda_min > 0

    # Construct orthonormal basis explicitly:
    mat_q, _ = np.linalg.qr(np.eye(n) - one_vec @ one_vec.T)
    mat_q = mat_q[:, : n - 1]  # explicitly enforce correct dimensions (n x n-1)

    # Project onto orthogonal subspace:
    m_orth = mat_q.T @ (mat_d - u @ u.T) @ mat_q
    eigvals_orth, eigvecs_orth = np.linalg.eigh(m_orth)
    assert np.all(eigvals_orth > 0)

    # Combine eigenvectors/eigenvalues explicitly:
    eigvals_full = np.concatenate(([lambda_min], eigvals_orth))
    eigvecs_full = np.hstack((one_vec, mat_q @ eigvecs_orth))

    # Cholesky-like decomposition
    mat_b = eigvecs_full @ np.diag(np.sqrt(eigvals_full))

    return mat_b


def whiten_data(x, v):
    """Whitening of Gaussian time-independent data

    It is assumed that each x[i] ~ N(mu, v[i]),
    where the mean mu is unknown, but constant value,
    variance is known v, and all x[i] are independent.
    """
    mu = np.average(x, weights=1 / v)

    decomposed = stable_decomposition(v)
    transform = np.linalg.inv(decomposed)

    residual = x - mu
    z = transform @ residual

    return z
