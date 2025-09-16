import numpy as np
from numpy.testing import assert_allclose
from uncle_val.whitening import whiten_data


def test_white_data():
    """Test white_data() with err drown from a narrow range"""
    n_obj = 1000
    n_obs = 100
    rng = np.random.default_rng(42)

    true_flux = rng.exponential(size=n_obj)
    err = rng.uniform(low=0.5, high=2.0, size=(n_obj, n_obs))
    flux = rng.normal(loc=true_flux[:, None], scale=err)

    z = []
    for f, e in zip(flux, err**2, strict=False):
        z.append(whiten_data(f, e))
    z = np.concatenate(z)

    assert_allclose(np.mean(z), 0.0, atol=3.0 / np.sqrt(n_obs * n_obj))
    assert_allclose(np.std(z), 1.0, rtol=3.0 / np.sqrt(n_obs * n_obj))
