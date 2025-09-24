from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from numpy.testing import assert_allclose
from uncle_val.datasets.fake import fake_non_variable_lcs
from uncle_val.learning.losses import kl_divergence_whiten, minus_ln_chi2_prob
from uncle_val.learning.models import LinearModel, MLPModel, UncleModel
from uncle_val.learning.training import train_step


def run_model(
    *, batch_size: int, train_steps: int, n_obj: int, model: UncleModel, loss: Callable, rtol: float
):
    """Run tests with MLP model

    Parameters
    ----------
    batch_size : int
        Batch size, e.g. number of light curves to average loss on.
    train_steps : int
        Number of training steps, e.g. number of batches to train on.
    n_obj : int
        Number of unique objects to generate.
    model : UncleModel
        Model to use
    loss : Callable
        Loss function to use
    rtol : float
        Relative error tolerance for testing.
    """
    np_rng = np.random.default_rng(42)

    n_src_training = 30
    n_src = np_rng.integers(n_src_training, 150, size=n_obj)
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

    step = nnx.jit(lambda **kwargs: train_step(loss=loss, **kwargs))

    for indexes in np_rng.choice(len(flux_arr), size=(train_steps, batch_size)):
        flux_batch = flux_arr.take(indexes)
        err_batch = err_arr.take(indexes)
        norm_flux_batch = norm_flux_arr.take(indexes)
        norm_err_batch = norm_err_arr.take(indexes)

        lengths = flux_batch.value_lengths()

        flux_ = []
        err_ = []
        norm_flux_ = []
        norm_err_ = []

        for length, flux, err, norm_flux, norm_err in zip(
            lengths, flux_batch, err_batch, norm_flux_batch, norm_err_batch, strict=False
        ):
            value_idx = np_rng.choice(length, n_src_training, replace=False)
            flux_.append(jnp.asarray(flux.values.take(value_idx)))
            err_.append(jnp.asarray(err.values.take(value_idx)))
            norm_flux_.append(jnp.asarray(norm_flux.values.take(value_idx)))
            norm_err_.append(jnp.asarray(norm_err.values.take(value_idx)))
        flux_ = jnp.stack(flux_)
        err_ = jnp.stack(err_)
        norm_flux_ = jnp.stack(norm_flux_)
        norm_err_ = jnp.stack(norm_err_)

        theta_ = jnp.stack([norm_flux_, norm_err_], axis=-1)
        step(
            model=model,
            optimizer=optimizer,
            theta=theta_,
            flux=flux_,
            err=err_,
        )

    assert_allclose(np.mean(model(theta_)), u, rtol=rtol)


@pytest.mark.parametrize("loss", [minus_ln_chi2_prob, kl_divergence_whiten])
@pytest.mark.long
def test_mlp_model_many_objects(loss):
    """Fit MLPModel for a constant u function with many objects"""
    model = MLPModel(
        d_input=2,
        d_output=1,
        dropout=None,
        rngs=nnx.Rngs(0),
    )
    run_model(model=model, loss=loss, batch_size=1, train_steps=2000, n_obj=1000, rtol=0.06)


@pytest.mark.parametrize("loss", [minus_ln_chi2_prob, kl_divergence_whiten])
def test_mlp_model_overfit_single_object(loss):
    """Fit MLPModel for a constant u function with a single object"""
    model = MLPModel(
        d_input=2,
        d_output=1,
        dropout=0.2,
        rngs=nnx.Rngs(0),
    )
    run_model(model=model, loss=loss, batch_size=1, train_steps=1, n_obj=1, rtol=0.5)
    run_model(model=model, loss=loss, batch_size=1, train_steps=100, n_obj=1, rtol=0.3)


@pytest.mark.parametrize("loss", [minus_ln_chi2_prob, kl_divergence_whiten])
@pytest.mark.long
def test_linear_model_many_objects(loss):
    """Fit MLPModel for a constant u function with many objects"""
    model = LinearModel(
        d_input=2,
        d_output=1,
        rngs=nnx.Rngs(0),
    )
    run_model(model=model, loss=loss, batch_size=2, train_steps=2000, n_obj=1000, rtol=0.01)


@pytest.mark.parametrize("loss", [minus_ln_chi2_prob, kl_divergence_whiten])
def test_linear_model_overfit_single_object(loss):
    """Fit MLPModel for a constant u function with a single object"""
    model = LinearModel(
        d_input=2,
        d_output=1,
        rngs=nnx.Rngs(0),
    )
    run_model(model=model, loss=loss, batch_size=1, train_steps=1000, n_obj=1, rtol=0.3)
