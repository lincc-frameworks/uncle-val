import numpy as np
from numpy.testing import assert_allclose
from uncle_val.datasets import fake_non_variable_lcs
from uncle_val.whitening import whiten_data


def test_fake_non_variable_lcs():
    """Test fake_non_variable_lcs()"""
    rng = np.random.default_rng(42)
    n_obj = 1000
    n_src = 33
    u_value = 2.0
    nf = fake_non_variable_lcs(
        n_obj=n_obj,
        n_src=n_src,
        u=u_value,
        rng=rng,
    )
    assert len(nf) == n_obj
    assert len(nf["objectForcedSource.psfFlux"]) == n_obj * n_src

    whiten = nf.reduce(
        lambda flux, err: {"whiten.z": whiten_data(flux, err**2)},
        "objectForcedSource.psfFlux",
        "objectForcedSource.psfFluxErr",
    )
    z = whiten["whiten.z"]
    assert_allclose(np.std(z), u_value, rtol=3.0 / np.sqrt(n_obj * n_src))
