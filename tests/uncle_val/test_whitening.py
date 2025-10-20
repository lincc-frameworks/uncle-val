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
    for f, v in zip(flux, err**2, strict=False):
        z.append(whiten_data(f, v))
    z = np.concatenate(z)

    assert_allclose(np.mean(z), 0.0, atol=3.0 / np.sqrt(n_src * n_obj))
    assert_allclose(np.std(z), 1.0, rtol=3.0 / np.sqrt(n_src * n_obj))


def test_whiten_data_jax():
    """Test that whiten_data() doesn't fail when used with jax"""
    n_src = 1000
    rng = np.random.default_rng(42)
    err = rng.exponential(size=n_src)
    flux = rng.normal(loc=10, scale=err)
    func = jax.jit(partial(whiten_data, np=jnp))

    z = func(jnp.asarray(flux), jnp.asarray(err) ** 2)
    assert_allclose(np.mean(z), 0.0, atol=3.0 / np.sqrt(n_src))
    assert_allclose(np.std(z), 1.0, rtol=3.0 / np.sqrt(n_src))


def test_whiten_data_numba():
    """Test that whiten_data is numba-compilable"""
    n_src = 1000
    rng = np.random.default_rng(42)
    err = rng.exponential(size=n_src)
    flux = rng.normal(loc=10, scale=err)
    func = partial(numba.njit(whiten_data), np=None)

    z = func(flux, err**2)
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
    v = torch.tensor(err**2, dtype=torch.float32)

    # Check if no errors on a non-CPU device
    _ = func(x.to("meta"), v.to("meta"))

    # Check the results are correct
    z = func(x, v).cpu().numpy()

    assert_allclose(np.mean(z), 0.0, atol=3.0 / np.sqrt(n_src))
    assert_allclose(np.std(z), 1.0, rtol=3.0 / np.sqrt(n_src))
