import lsdb
import numpy as np
import pandas as pd
import pytest
import torch
from dask.distributed import Client
from numpy.testing import assert_array_equal
from uncle_val.datasets.fake import fake_non_variable_lcs
from uncle_val.learning.lsdb_dataset import LSDBIterableDataset, lsdb_nested_series_data_generator


def generate_fake_catalog(output_n_obj, output_n_src, rng):
    """Generate catalog of fake data for tests."""
    input_n_obj = output_n_obj * 2 + 123
    input_n_src = np.r_[[output_n_src // 2] * (input_n_obj - output_n_obj), [output_n_src] * output_n_obj]
    u = 1.0

    input_nf = next(
        fake_non_variable_lcs(
            n_batches=1,
            n_obj=input_n_obj,
            n_src=input_n_src,
            err=None,
            u=u,
            rng=rng,
        )
    )
    input_nf["ra"] = np.linspace(0.0, 360.0, input_n_obj)
    input_nf["dec"] = np.linspace(0.0, 90.0, input_n_obj)
    del input_nf["id"]
    catalog = lsdb.from_dataframe(input_nf)
    return catalog


@pytest.mark.parametrize("client", ("dask", None))
def test_lsdb_data_generator(client):
    """Test LSDBDataGenerator class."""
    if client == "dask":
        client = Client(n_workers=2)

    rng = np.random.default_rng(42)
    output_n_obj = 1234
    output_n_src = 100
    catalog = generate_fake_catalog(output_n_obj, output_n_src, rng)

    gen = lsdb_nested_series_data_generator(
        catalog=catalog,
        client=client,
        lc_col="lc",
        n_src=output_n_src,
        partitions_per_chunk=None,
        seed=rng.integers(1 << 63),
    )

    chunks = list(gen)
    assert len(chunks) > 1, "Expected more than 1 chunk"

    output_flat_nf = pd.concat(chunks)
    assert len(output_flat_nf) == output_n_obj
    assert_array_equal(output_flat_nf.nest.list_lengths, output_n_src)


@pytest.mark.parametrize("client", ("dask", None))
def test_lsdb_data_loader(client):
    """Test LSDBDataLoader class."""
    if client == "dask":
        client = Client(n_workers=2)

    rng = np.random.default_rng(42)
    batches = 10
    assert batches > 1
    output_n_obj = 123 * batches + 1
    output_n_src = 100
    catalog = generate_fake_catalog(output_n_obj, output_n_src, rng)

    dataset = LSDBIterableDataset(
        catalog=catalog,
        lc_col="lc",
        columns=["x", "err"],
        drop_columns=None,
        client=client,
        batch_lc=batches,
        n_src=output_n_src,
        # Large number, so we are fetching everything at once
        partitions_per_chunk=12,
        seed=rng.integers(1 << 63),
    )

    chunks = list(dataset)
    assert len(chunks) > 1, "Expected more than 1 chunk"

    tensor = torch.concatenate(chunks)
    # We have just two features: x and err
    assert tensor.shape == (output_n_obj // batches * batches, output_n_src, 2)
