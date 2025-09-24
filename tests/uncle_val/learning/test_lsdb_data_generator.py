import lsdb
import numpy as np
import pandas as pd
import pytest
from dask.distributed import Client
from nested_pandas import NestedFrame
from uncle_val.datasets.fake import fake_non_variable_lcs
from uncle_val.learning.lsdb_data_generator import LSDBDataGenerator


@pytest.mark.parametrize("client", ("dask", None))
def test_lsdb_data_generator(client):
    """Test LSDBDataGenerator class."""
    if client == "dask":
        client = Client(n_workers=2)

    rng = np.random.default_rng(42)
    input_n_obj = 2000
    output_n_obj = 1234
    output_n_src = 100
    input_n_src = np.r_[[output_n_src // 2] * (input_n_obj - output_n_obj), [output_n_src] * output_n_obj]
    u = 1.0

    input_nf = fake_non_variable_lcs(
        n_obj=input_n_obj,
        n_src=input_n_src,
        err=None,
        u=u,
        rng=rng,
    )
    input_nf["ra"] = np.linspace(0.0, 360.0, input_n_obj)
    input_nf["dec"] = np.linspace(0.0, 90.0, input_n_obj)
    del input_nf["objectId"]
    catalog = lsdb.from_dataframe(input_nf)

    gen = LSDBDataGenerator(
        catalog=catalog,
        client=client,
        n_src=output_n_src,
        partitions_per_chunk=None,
        seed=rng.integers(1 << 63),
    )

    chunks = list(gen)
    assert len(chunks) > 1, "Expected more than 1 chunk"

    output_flat_nf_chunks = [
        NestedFrame({col: np.asarray(value).flatten() for col, value in chunk.items()}) for chunk in chunks
    ]
    output_flat_nf = pd.concat(output_flat_nf_chunks)
    assert len(output_flat_nf) == output_n_obj * output_n_src
