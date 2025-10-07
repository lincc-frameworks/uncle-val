from collections.abc import Generator
from itertools import chain

import dask
import numpy as np
import pandas as pd
import torch
from hats import HealpixPixel
from lsdb import Catalog
from nested_pandas import NestedFrame, NestedSeries
from torch.utils.data import DataLoader, IterableDataset

from uncle_val.datasets.lsdb_generator import LSDBDataGenerator
from uncle_val.utils.hashing import uniform_hash


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
    nf: NestedFrame,
    pixel: HealpixPixel,
    *,
    n_src: int,
    lc_col: str,
    id_col: str,
    hash_range: tuple[int, int] | None,
    seed: int,
) -> NestedFrame:
    length_col = f"__length_{lc_col}_"
    nf[length_col] = nf[lc_col].nest.list_lengths
    nf = nf.query(f"{length_col} >= {n_src}")

    if hash_range is not None:
        hashes = uniform_hash(nf[id_col])
        mask = (hashes >= hash_range[0]) & (hashes < hash_range[1])
        nf = nf[mask]
    if id_col in nf.columns:
        nf = nf.drop(columns=[id_col])

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

    rng = np.random.default_rng((pixel.order, pixel.pixel, seed))

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
    id_col: str = "id",
    client: dask.distributed.Client | None,
    n_src: int,
    partitions_per_chunk: int | None,
    hash_range: tuple[int, int] | None = None,
    loop: bool = False,
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
    id_col : str, optional
        LSDB identifier column name, default is "id". Used for `hash_range` and
        it is removed after used.
    client : dask.distributed.Client or None, optional
        Dask client to use, default is None, which would not lock on each next
        value. If Dask client is given, the data would be fetched on the
        background.
    n_src : int
        Number of random observations per light curve.
    partitions_per_chunk : int
        Number of `catalog` partitions load in memory simultaneously.
        This changes the randomness.
    hash_range : (float, float) or None
        Compute hashes ∈ [0; 1) on `id_col` values and keeps only those in
        the specified [start; end) range. Turned off by default.
    loop : bool
        If `True` it runs infinitely selecting random partitions every time.
        If `False` it runs once.
    seed : int
        Random seed to use for shuffling.

    Yields
    ------
    nested_pandas.NestedSeries
        NestedSeries of sampled catalog data, one row per light curve.
    """
    if hash_range is not None and (hash_range[0] < 0 or hash_range[1] > 1):
        raise ValueError(f"`hash_range` must be between 0 and 1, {hash_range} is given")

    dask_series = catalog.map_partitions(
        _process_partition,
        include_pixel=True,
        n_src=n_src,
        lc_col=lc_col,
        id_col=id_col,
        hash_range=hash_range,
        seed=seed,
    )
    lsdb_generator = LSDBDataGenerator(
        catalog=dask_series,
        client=client,
        partitions_per_chunk=partitions_per_chunk,
        loop=loop,
        seed=seed,
    )
    yield from lsdb_generator


class LSDBIterableDataset(IterableDataset):
    """Iterable dataset fetching data through LSDB, use num_workers=0

    Torch iterable dataset fetching data using LSDB. When used with
    a torch DataLoader, num_workers=0 (default) must be used.

    It yields dict("subset": `torch.Tensor`).
    The first subset tensor shape is always (batch_lc, n_src, n_features);
    other batch sizes may vary.

    Parameters
    ----------
    catalog : Catalog
        LSDB catalog, it should have the only nested column, `lc_col`, and
        an `id` value if splits are specified.
    lc_col : str, optional
        LSDB light curve column name, default is "lc".
    id_col : str, optional
        LSDB ID column name, used for hash calculation when
        `hash_range` is specified. This column is always being dropped from
        the dataset.
    columns : list of str or None, optional
        List of column names (both base and nested into `lc_col`) to use.
        If None, all columns will be used, and it is assumed "x" and "err"
        are there. Note that all columns should be castable to float32.
    drop_columns : list of str or None, optional
        List of column names (both base and nested into `lc_col`) to drop,
        if None nothing is dropped.
    client : dask.distributed.Client or None, optional
        Dask client to use, default is None, which would not lock on each next
        value. If a Dask client is given, the data would be fetched on the
        background.
    batch_lc : int
        Number of batches to yield. If `splits` is used, it will be the size
        of the first subset.
    n_src : int
        Number of random observations per light curve.
    partitions_per_chunk : int or None
        Number of `catalog` partitions per time, if None it is derived
        from the number of dask workers associated with `Client` (one if
        no workers or None `Client`).
        This changes the randomness.
    hash_range : (float, float) or None
        Compute hashes ∈ [0; 1) on `id_col` values and keeps only those in
        the specified [start; end) range. Turned off by default.
    loop : bool
        If `True` it runs infinitely selecting random partitions every time.
        If `False` it runs once.
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
        id_col: str = "id",
        columns: list[str] | None,
        drop_columns: list[str] | None = None,
        client: dask.distributed.Client | None,
        batch_lc: int,
        n_src: int,
        partitions_per_chunk: int,
        hash_range: tuple[float, float] | None = None,
        loop: bool = False,
        seed: int,
    ):
        generator_kwargs = locals().copy()
        generator_kwargs.pop("self")
        generator_kwargs.pop("columns")
        generator_kwargs.pop("drop_columns")
        generator_kwargs.pop("batch_lc")
        self.nested_series_gen = lsdb_nested_series_data_generator(**generator_kwargs)

        self.id_col = id_col
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
        if self.id_col in self.columns:
            self.columns.remove(self.id_col)
        self.n_columns = len(self.columns)

    def _nested_series_to_3dtensor(self, series: NestedSeries) -> torch.Tensor:
        batch_flat_df = series.nest.to_flat()[self.columns]
        np_array_2d = batch_flat_df.to_numpy(dtype=np.float32)
        np_array_3d = np_array_2d.reshape(self.batch_lc, self.n_src, self.n_columns)
        return torch.tensor(np_array_3d)

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        for nested_series in chain([self.current_nested_series], self.nested_series_gen):
            self.current_nested_series = nested_series
            for i in range(0, len(nested_series) - self.batch_lc, self.batch_lc):
                batch_series = nested_series.iloc[i : i + self.batch_lc]
                tensor = self._nested_series_to_3dtensor(batch_series)
                yield tensor


def lsdb_data_loader(
    catalog: Catalog,
    *,
    lc_col: str = "lc",
    id_col: str = "id",
    columns: list[str] | None,
    drop_columns: list[str] | None = None,
    client: dask.distributed.Client | None,
    batch_lc: int,
    n_src: int,
    partitions_per_chunk: int | None,
    hash_range: tuple[float, float] | None = None,
    loop: bool = False,
    seed: int,
    pin_memory: bool = False,
    pin_memory_device: str = "",
) -> DataLoader:
    """Make a torch DataLoader object from an LSDB catalog.

    Torch data loader fetching data using LSDB.

    It yields dict("subset": `torch.Tensor`).
    The first subset tensor shape is always (batch_lc, n_src, n_features);
    other batch sizes may vary.

    Parameters
    ----------
    catalog : Catalog
        LSDB catalog, it should have the only nested column, `lc_col`, and
        an `id` value if splits are specified.
    lc_col : str, optional
        LSDB light curve column name, default is "lc".
    id_col : str, optional
        LSDB ID column name, used for hash calculation when
        `hash_range` is specified. This column is always being dropped from
        the dataset.
    columns : list of str or None, optional
        List of column names (both base and nested into `lc_col`) to use.
        If None, all columns will be used, and it is assumed "x" and "err"
        are there. Note that all columns should be castable to float32.
    drop_columns : list of str or None, optional
        List of column names (both base and nested into `lc_col`) to drop,
        if None nothing is dropped.
    client : dask.distributed.Client or None, optional
        Dask client to use, default is None, which would not lock on each next
        value. If a Dask client is given, the data would be fetched on the
        background.
    batch_lc : int
        Number of batches to yield. If `splits` is used, it will be the size
        of the first subset.
    n_src : int
        Number of random observations per light curve.
    partitions_per_chunk : int or None
        Number of `catalog` partitions per time, if None it is derived
        from the number of dask workers associated with `Client` (one if
        no workers or None `Client`).
        This changes the randomness.
    hash_range : (float, float) or None
        Compute hashes ∈ [0; 1) on `id_col` values and keeps only those in
        the specified [start; end) range. Turned off by default.
    loop : bool
        If `True` it runs infinitely selecting random partitions every time.
        If `False` it runs once.
    seed : int
        Random seed to use for shuffling.
    pin_memory : bool, optional
        Whether to pin memory, default is False.
    pin_memory_device : str, optional
        Device string to use for pin memory, passed to `DataLoader`.
    """
    kwargs = locals()
    kwargs.pop("pin_memory")
    kwargs.pop("pin_memory_device")
    dataset = LSDBIterableDataset(**kwargs)

    return DataLoader(
        dataset=dataset,
        batch_size=1,  # We batch in the dataset with batch_lc
        shuffle=False,  # We shuffle in the dataset
        num_workers=0,  # We use Dask workers already, no need to use parallel processing with torch
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
    )
