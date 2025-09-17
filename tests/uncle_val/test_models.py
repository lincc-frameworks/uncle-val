import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from numpy.testing import assert_allclose
from uncle_val.datasets import fake_non_variable_lcs
from uncle_val.learning.losses import minus_ln_chi2_prob
from uncle_val.learning.models import LinearModel, MLPModel, UncleModel
from uncle_val.learning.training import train_step


def run_model(*, train_steps: int, n_obj: int, model: UncleModel, rtol: float):
    """Run tests with MLP model

    Parameters
    ----------
    train_steps : int
        Number of training steps, e.g. number of light curves to train on.
    n_obj : int
        Number of unique objects to generate.
    dropout : float | None
        Dropout rate to use.
    model : UncleModel
        Model to use
    rtol : float
        Relative error tolerance for testing.
    """
    np_rng = np.random.default_rng(42)

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

    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    step = nnx.jit(lambda **kwargs: train_step(loss=minus_ln_chi2_prob, **kwargs))

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

    assert_allclose(np.mean(model(theta)), u, rtol=rtol)


@pytest.mark.long
def test_mlp_model_many_objects():
    """Fit MLPModel for a constant u function with many objects"""
    model = MLPModel(
        d_input=2,
        d_output=1,
        dropout=None,
        rngs=nnx.Rngs(0),
    )
    run_model(model=model, train_steps=2000, n_obj=1000, rtol=0.06)


def test_mlp_model_overfit_single_object():
    """Fit MLPModel for a constant u function with a single object"""
    model = MLPModel(
        d_input=2,
        d_output=1,
        dropout=0.2,
        rngs=nnx.Rngs(0),
    )
    run_model(model=model, train_steps=1, n_obj=1, rtol=0.5)
    run_model(model=model, train_steps=100, n_obj=1, rtol=0.3)


@pytest.mark.long
def test_linear_model_many_objects():
    """Fit MLPModel for a constant u function with many objects"""
    model = LinearModel(
        d_input=2,
        d_output=1,
        rngs=nnx.Rngs(0),
    )
    run_model(model=model, train_steps=2000, n_obj=1000, rtol=0.001)


def test_linear_model_overfit_single_object():
    """Fit MLPModel for a constant u function with a single object"""
    model = LinearModel(
        d_input=2,
        d_output=1,
        rngs=nnx.Rngs(0),
    )
    run_model(model=model, train_steps=1000, n_obj=1, rtol=0.3)
