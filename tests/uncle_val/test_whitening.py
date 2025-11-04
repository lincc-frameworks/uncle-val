from functools import partial

import jax
import jax.numpy as jnp
import numba
import numpy as np
import torch
from numpy.testing import assert_allclose
from uncle_val.whitening import whiten_data


def test_whiten_data():
    """Test whiten_data() with err drawn from a narrow range"""
    n_obj = 1000
    n_src = 100
    rng = np.random.default_rng(42)

    true_flux = rng.exponential(size=n_obj)
    err = rng.uniform(low=0.5, high=2.0, size=(n_obj, n_src))
    flux = rng.normal(loc=true_flux[:, None], scale=err)

    z = []
    for f, e in zip(flux, err, strict=True):
        z.append(whiten_data(f, e))
    z = np.stack(z)

    assert z.shape == (n_obj, n_src - 1)
    assert_allclose(np.mean(z), 0.0, atol=3.0 / np.sqrt(n_src * n_obj))
    assert_allclose(np.std(z, ddof=1), 1.0, rtol=3.0 / np.sqrt(n_src * n_obj))

    corr = np.corrcoef(z, rowvar=False)
    corr_no_diag = corr - np.eye(n_src - 1)
    # Each individual correlation decreases as sqrt(n_obj), but the probability to find
    # any correlation coefficient larger than given value grow with the number of elements
    # we check.
    assert_allclose(corr_no_diag, 0.0, atol=3.0 * np.sqrt(n_src) / np.sqrt(n_obj))


def test_whiten_data_jax():
    """Test that whiten_data() doesn't fail when used with jax"""
    n_src = 1000
    rng = np.random.default_rng(42)
    err = rng.exponential(size=n_src)
    flux = rng.normal(loc=10, scale=err)
    func = jax.jit(partial(whiten_data, np=jnp))

    z = func(jnp.asarray(flux), jnp.asarray(err))
    assert_allclose(np.mean(z), 0.0, atol=3.0 / np.sqrt(n_src))
    assert_allclose(np.std(z), 1.0, rtol=3.0 / np.sqrt(n_src))


def test_whiten_data_numba():
    """Test that whiten_data is numba-compilable"""
    n_src = 1000
    rng = np.random.default_rng(42)
    err = rng.exponential(size=n_src)
    flux = rng.normal(loc=10, scale=err)
    func = partial(numba.njit(whiten_data), np=None)

    z = func(flux, err)
    assert_allclose(np.mean(z), 0.0, atol=3.0 / np.sqrt(n_src))
    assert_allclose(np.std(z), 1.0, rtol=3.0 / np.sqrt(n_src))


def test_whiten_data_torch():
    """Test that whiten_data() is torch-compilable"""
    n_src = 1000
    rng = np.random.default_rng(42)
    err = rng.exponential(size=n_src)
    flux = rng.normal(loc=10, scale=err)
    func = torch.compile(partial(whiten_data, np=torch))

    x = torch.tensor(flux, dtype=torch.float32)
    sigma = torch.tensor(err, dtype=torch.float32)

    # Check if no errors on a non-CPU device
    _ = func(x.to("meta"), sigma.to("meta"))

    # Check the results are correct
    z = func(x, sigma).cpu().numpy()

    assert_allclose(np.mean(z), 0.0, atol=3.0 / np.sqrt(n_src))
    assert_allclose(np.std(z), 1.0, rtol=3.0 / np.sqrt(n_src))
