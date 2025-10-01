from collections.abc import Generator

import dask
import numpy as np
import pandas as pd
import torch
from hats import HealpixPixel
from lsdb import Catalog
from nested_pandas import NestedFrame
from torch.utils.data import IterableDataset

from uncle_val.datasets.lsdb_generator import LSDBDataGenerator


def _reduce_all_columns_wrapper(*args, columns=None, udf, **kwargs):
    row_dict = dict(zip(columns, args, strict=False))
    return udf(row_dict, **kwargs)


def _process_lc(
    row: dict[str, object], *, n_src: int, lc_col: str, length_col: str, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    lc_length = row.pop(length_col)
    idx = rng.choice(lc_length, size=n_src, replace=False)

    result: dict[str, np.ndarray] = {}
    for col, value in row.items():
        if col.startswith(f"{lc_col}."):
            result[col] = value[idx]
        else:
            result[f"{lc_col}.{col}"] = np.full(n_src, value)
    return result


def _process_partition(
    nf: NestedFrame, pixel: HealpixPixel, *, n_src: int, lc_col: str, seed: int
) -> NestedFrame:
    length_col = f"__length_{lc_col}_"
    nf[length_col] = nf[lc_col].nest.list_lengths
    nf = nf.query(f"{length_col} >= {n_src}")

    rng = np.random.default_rng((pixel.order, pixel.pixel, seed))

    lc_subcolumns = [f"{lc_col}.{sub_column}" for sub_column in nf.all_columns[lc_col]]
    columns = list(nf.columns) + lc_subcolumns
    columns.remove(lc_col)

    if len(nf) == 0:
        fixed_length_series = nf[lc_col].copy()
        for col in nf.columns:
            if col == lc_col:
                continue
            fixed_length_series = fixed_length_series.nest.with_field(col, nf[col])
        return fixed_length_series

    fixed_length_nf = nf.reduce(
        _reduce_all_columns_wrapper,
        *columns,
        columns=columns,
        udf=_process_lc,
        n_src=n_src,
        lc_col=lc_col,
        length_col=length_col,
        rng=rng,
    )
    fixed_length_series = fixed_length_nf[lc_col]
    return fixed_length_series


def nested_series_data_generator(
    catalog: Catalog,
    *,
    lc_col: str = "lc",
    client: dask.distributed.Client | None,
    n_src: int,
    partitions_per_chunk: int | None,
    seed: int,
) -> Generator[pd.Series, None, None]:
    """Generator of nested_pandas.NestedSeries of sampled catalog data

    The data is pre-fetched on the background, 'n_workers' number
    of partitions per time (derived from `client` object).
    It filters out light curves with less than `n_src` observations,
    and selects `n_src` random observations per light curve.

    Parameters
    ----------
    catalog : Catalog
        LSDB catalog, should have the only nested column, `lc_col`.
    lc_col : str, optional
        LSDB light curve column name, default is "lc".
    client : dask.distributed.Client or None, optional
        Dask client to use, default is None, which would not lock on each next
        value. If Dask client is given, the data would be fetched on the
        background.
    n_src : int
        Number of random observations per light curve.
    partitions_per_chunk : int or None
        Number of `catalog` partitions per time, if None it is derived
        from the number of dask workers associated with `Client` (one if
        no workers or None `Client`).
        This changes the randomness.
    seed : int
        Random seed to use for shuffling.

    Yields
    ------
    nested_pandas.NestedSeries
        NestedSeries of sampled catalog data, one row per light curve.
    """
    dask_series = catalog.map_partitions(
        _process_partition, include_pixel=True, n_src=n_src, lc_col=lc_col, seed=seed
    )
    lsdb_generator = LSDBDataGenerator(
        catalog=dask_series,
        client=client,
        partitions_per_chunk=partitions_per_chunk,
        seed=seed,
    )
    yield from lsdb_generator


class LSDBIterableDataset(IterableDataset):
    """Iterable dataset fetching data through LSDB, use num_workers=0

    Torch iterable dataset fetching data using LSDB. When used with
    a torch DataLoader, num_workers=0 (default) must be used.

    It yields `torch.Tensor` objects of shape (batch_lc, n_src).

    Parameters
    ----------
    catalog : Catalog
        LSDB catalog, should have the only nested column, `lc_col`.
    lc_col : str, optional
        LSDB light curve column name, default is "lc".
    client : dask.distributed.Client or None, optional
        Dask client to use, default is None, which would not lock on each next
        value. If Dask client is given, the data would be fetched on the
        background.
    batch_lc : int
        Number of batches to yield.
    n_src : int
        Number of random observations per light curve.
    partitions_per_chunk : int or None
        Number of `catalog` partitions per time, if None it is derived
        from the number of dask workers associated with `Client` (one if
        no workers or None `Client`).
        This changes the randomness.
    seed : int
        Random seed to use for shuffling.

    See Also
    --------
    `nested_series_data_generator`, `LSDBDataGenerator`
    """
    def __init__(
        self,
        catalog: Catalog,
        *,
        lc_col: str = "lc",
        client: dask.distributed.Client | None,
        batch_lc: int,
        n_src: int,
        partitions_per_chunk: int | None,
        seed: int,
    ):
        self.nested_series_gen = nested_series_data_generator(
            catalog=catalog,
            lc_col=lc_col,
            client=client,
            partitions_per_chunk=partitions_per_chunk,
            n_src=n_src,
            seed=seed,
        )
        self.batch_lc = batch_lc
        self.n_src = n_src

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        for nested_series in self.nested_series_gen:
            for i in range(0, len(nested_series), self.batch_lc):
                batch_series = nested_series.iloc[i : i + self.batch_lc]
                batch_flat_df = batch_series.nest.to_flat()
                np_array = np.asarray(batch_flat_df).reshape(self.batch_lc, self.n_src)
                yield torch.tensor(np_array)
