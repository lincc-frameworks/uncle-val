from collections.abc import Generator
from itertools import chain

import dask
import numpy as np
import pandas as pd
import torch
from hats import HealpixPixel
from lsdb import Catalog
from nested_pandas import NestedFrame
from torch.utils.data import DataLoader, IterableDataset

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


def lsdb_nested_series_data_generator(
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

    It yields `torch.Tensor` objects of shape (batch_lc, n_src, n_features).

    Parameters
    ----------
    catalog : Catalog
        LSDB catalog, should have the only nested column, `lc_col`.
    lc_col : str, optional
        LSDB light curve column name, default is "lc".
    columns : list of str or None, optional
        List of column names (both base and nested into `lc_col`) to use.
        If None, all columns will be used, and it is assumed "x" and "err"
        are there. Note, that all columns should be castable to float32.
    drop_columns : list of str or None, optional
        List of column names (both base and nested into `lc_col`) to drop,
        if None nothing will be dropped.
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
        columns: list[str] | None,
        drop_columns: list[str] | None = None,
        client: dask.distributed.Client | None,
        batch_lc: int,
        n_src: int,
        partitions_per_chunk: int | None,
        seed: int,
    ):
        self.nested_series_gen = lsdb_nested_series_data_generator(
            catalog=catalog,
            lc_col=lc_col,
            client=client,
            partitions_per_chunk=partitions_per_chunk,
            n_src=n_src,
            seed=seed,
        )
        self.batch_lc = batch_lc
        self.n_src = n_src
        self.current_nested_series = next(self.nested_series_gen)

        if columns is None:
            orig_columns = self.current_nested_series.dtype.field_names
            self.columns = ["x", "err"] + [col for col in orig_columns if col not in {"x", "err"}]
        else:
            self.columns = columns
        if drop_columns is not None:
            if drop_columns is str:
                drop_columns = [drop_columns]
            drop_columns = set(drop_columns)
            self.columns = [col for col in self.columns if col not in drop_columns]
        self.n_columns = len(self.columns)

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        for nested_series in chain([self.current_nested_series], self.nested_series_gen):
            self.current_nested_series = nested_series
            for i in range(0, len(nested_series) - self.batch_lc, self.batch_lc):
                batch_series = nested_series.iloc[i : i + self.batch_lc]
                batch_flat_df = batch_series.nest.to_flat()[self.columns]
                np_array_2d = batch_flat_df.to_numpy(dtype=np.float32)
                np_array_3d = np_array_2d.reshape(self.batch_lc, self.n_src, self.n_columns)
                yield torch.tensor(np_array_3d)


def lsdb_data_loader(
    catalog: Catalog,
    *,
    lc_col: str = "lc",
    columns: list[str] | None,
    drop_columns: list[str] | None = None,
    client: dask.distributed.Client | None,
    batch_lc: int,
    n_src: int,
    partitions_per_chunk: int | None,
    seed: int,
    pin_memory: bool = False,
    pin_memory_device: str = "",
) -> DataLoader:
    """Make a torch DataLoader object from an LSDB catalog.

    Iterable dataset fetching data through LSDB, use num_workers=0

    Torch iterable dataset fetching data using LSDB. When used with
    a torch DataLoader, num_workers=0 (default) must be used.

    It yields a tuple of `torch.Tensor` objects of shape
    (batch_lc, n_src, n_features), and a list of feature names.

    Parameters
    ----------
    catalog : Catalog
        LSDB catalog, should have the only nested column, `lc_col`.
    lc_col : str, optional
        LSDB light curve column name, default is "lc".
    columns : list of str or None, optional
        List of column names (both base and nested into `lc_col`) to use.
        If None, all columns will be used, and it is assumed "x" and "err"
        are there. Note, that all columns should be castable to float32.
    drop_columns : list of str or None, optional
        List of column names (both base and nested into `lc_col`) to drop,
        if None nothing will be dropped.
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
    pin_memory : bool, optional
        Whether to pin memory, default is False.
    pin_memory_device : str, optional
        Device string to use for pin memory, passed to `DataLoader`.
    """
    dataset = LSDBIterableDataset(
        catalog=catalog,
        lc_col=lc_col,
        columns=columns,
        drop_columns=drop_columns,
        client=client,
        batch_lc=batch_lc,
        n_src=n_src,
        partitions_per_chunk=partitions_per_chunk,
        seed=seed,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=1,  # We batch in the dataset with batch_lc
        shuffle=False,  # We shuffle in the dataset
        num_workers=0,  # We use Dask workers already, no need to use parall processing with torch
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
    )
