import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from uncle_val.datasets import fake_non_variable_lcs
from uncle_val.whitening import whiten_data


def test_fake_non_variable_lcs():
    """Test fake_non_variable_lcs()"""
    rng = np.random.default_rng(42)
    n_batches = 10
    n_obj = 100
    n_src = 33
    u_value = 2.0
    nf = pd.concat(
        list(
            fake_non_variable_lcs(
                n_batches=10,
                n_obj=n_obj,
                n_src=n_src,
                u=u_value,
                rng=rng,
            )
        )
    )
    assert len(nf) == n_batches * n_obj
    assert len(nf["lc.x"]) == n_batches * n_obj * n_src

    whiten = nf.reduce(
        lambda flux, err: {"whiten.z": whiten_data(flux, err**2)},
        "lc.x",
        "lc.err",
    )
    z = whiten["whiten.z"]
    assert_allclose(np.std(z), u_value, rtol=3.0 / np.sqrt(n_batches * n_obj * n_src))
