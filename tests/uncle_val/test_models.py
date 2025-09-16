import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from numpy.testing import assert_allclose
from uncle_val.datasets import fake_non_variable_lcs
from uncle_val.models import MLPModel, chi2_lc_train_step


@pytest.mark.long
def test_mlp_model():
    """Fit MLPModel for a constant u function"""
    np_rng = np.random.default_rng(42)
    nnx_rngs = nnx.Rngs(int(np_rng.integers(1 << 63)))

    train_steps = 2000
    n_obj = 1000
    n_src = np_rng.integers(30, 150, size=n_obj)
    u = 2.0

    nf = fake_non_variable_lcs(
        n_obj=n_obj,
        n_src=n_src,
        err=None,
        u=u,
        rng=np_rng,
    )
    ln_fluxes = np.log(nf["objectForcedSource.psfFlux"])
    nf["objectForcedSource.norm_flux"] = (ln_fluxes - np.mean(ln_fluxes)) / np.std(ln_fluxes)
    ln_errs = np.log(nf["objectForcedSource.psfFluxErr"])
    nf["objectForcedSource.norm_err"] = (ln_errs - np.mean(ln_errs)) / np.std(ln_errs)

    struct_array = nf["objectForcedSource"].array.struct_array.combine_chunks()
    flux_arr = struct_array.field("psfFlux")
    err_arr = struct_array.field("psfFluxErr")
    norm_flux_arr = struct_array.field("norm_flux")
    norm_err_arr = struct_array.field("norm_err")

    model = MLPModel(
        d_input=2,
        d_output=1,
        rngs=nnx_rngs,
    )
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    step = nnx.jit(chi2_lc_train_step)

    for idx in np_rng.choice(len(flux_arr), train_steps):
        flux = jnp.asarray(flux_arr[idx].values)
        err = jnp.asarray(err_arr[idx].values)
        norm_flux = jnp.asarray(norm_flux_arr[idx].values)
        norm_err = jnp.asarray(norm_err_arr[idx].values)

        theta = jnp.stack([norm_flux, norm_err], axis=-1)
        step(
            model=model,
            optimizer=optimizer,
            theta=theta,
            flux=flux,
            err=err,
        )

    assert_allclose(np.asarray(model(theta)), u, rtol=0.1)
