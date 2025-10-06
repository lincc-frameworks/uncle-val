import random
from collections.abc import Callable
from itertools import chain, count

import numpy as np
import pytest
import torch
import torch.optim
from numpy.testing import assert_allclose
from uncle_val.datasets.fake import fake_non_variable_lcs
from uncle_val.learning.losses import kl_divergence_whiten, minus_ln_chi2_prob
from uncle_val.learning.lsdb_dataset import LSDBIterableDataset
from uncle_val.learning.models import ConstantModel, LinearModel, MLPModel, UncleModel
from uncle_val.learning.training import train_step


def run_model(
    *, batch_size: int, train_batches: int, n_obj: int, model: UncleModel, loss: Callable, rtol: float
):
    """Run tests with MLP model

    Parameters
    ----------
    batch_size : int
        Batch size, e.g. number of light curves to average loss on.
    train_batches : int
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
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.use_deterministic_algorithms(True)

    rng = np.random.default_rng(42)

    n_src_training = 30
    n_src = rng.integers(n_src_training, 150, size=n_obj)
    u = 2.0

    catalog = fake_non_variable_lcs(
        n_obj=n_obj,
        n_src=n_src,
        err=None,
        u=u,
        rng=rng,
    )

    def dataset_fn():
        return LSDBIterableDataset(
            catalog=catalog,
            client=None,
            columns=["x", "err"],
            batch_lc=batch_size,
            n_src=n_src_training,
            partitions_per_chunk=12,
            seed=rng.integers(1 << 63),
        )

    infinite_dataset = chain.from_iterable(dataset_fn() for _ in count())

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _i_step, batch in zip(range(train_batches), infinite_dataset, strict=False):
        train_step(
            model=model,
            optimizer=optimizer,
            loss=loss,
            batch=batch,
        )

    model.eval()
    assert_allclose(np.mean(model(batch).detach().numpy()), u, rtol=rtol)


@pytest.mark.parametrize(
    "model",
    [
        MLPModel(
            d_input=2,
            d_middle=(3, 5, 6),
            d_output=2,
            dropout=0.2,
        ),
        MLPModel(
            d_input=4,
            d_output=1,
            dropout=None,
        ),
        LinearModel(
            d_input=5,
            d_output=1,
        ),
        ConstantModel(
            d_input=2,
            d_output=2,
        ),
    ],
)
def test_model(model):
    """Check that `UncleModel`s don't fail"""
    _ = model(torch.randn(model.d_input))


@pytest.mark.parametrize("loss", [minus_ln_chi2_prob, kl_divergence_whiten])
@pytest.mark.long
def test_mlp_model_many_objects(loss):
    """Fit MLPModel for a constant u function with many objects"""
    model = MLPModel(
        d_input=2,
        d_output=1,
        dropout=None,
    )
    run_model(model=model, loss=loss, batch_size=1, train_batches=2000, n_obj=1000, rtol=0.1)


@pytest.mark.parametrize("loss", [minus_ln_chi2_prob, kl_divergence_whiten])
@pytest.mark.long
def test_linear_model_many_objects(loss):
    """Fit MLPModel for a constant u function with many objects"""
    model = LinearModel(
        d_input=2,
        d_output=1,
    )
    run_model(model=model, loss=loss, batch_size=2, train_batches=2000, n_obj=1000, rtol=0.02)


@pytest.mark.parametrize("loss", [minus_ln_chi2_prob, kl_divergence_whiten])
@pytest.mark.long
def test_constant_model_many_objects(loss):
    """Fit MLPModel for a constant u function with many objects"""
    model = ConstantModel(
        d_input=2,
        d_output=1,
    )
    run_model(model=model, loss=loss, batch_size=2, train_batches=2000, n_obj=1000, rtol=0.01)
