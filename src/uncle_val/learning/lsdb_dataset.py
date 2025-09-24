from collections.abc import Generator, Iterator
from typing import Self

import dask
import numpy as np
import pandas as pd
from dask.distributed import Client, Future
from hats import HealpixPixel
from hyrax.config_utils import ConfigDict
from hyrax.data_sets import HyraxDataset
from lsdb import Catalog
from nested_pandas import NestedFrame

from uncle_val.datasets import dp1_catalog_single_band


class _FakeFuture:
    def __init__(self, obj):
        self.obj = obj

    def result(self):
        return self.obj


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


class LSDBDataGenerator(Iterator[pd.Series]):
    """Generator yielding training data from an LSDB nested_series

    The data is pre-fetched on the background, 'n_workers' number
    of partitions per time (derived from `client` object).
    It filters out light curves with less than `n_src` observations,
    and selects `n_src` random observations per light curve.

    Catalog must have the only nested column, which is interpreted as a light
    curve data.

    Parameters
    ----------
    catalog : lsdb.Catalog
        LSDB nested_series object.
    client : dask.distributed.Client or None
        Dask client for distributed computation. None means running
        in a synced way with `dask.compute()` instead of asynced with
        `client.compute()`.
    n_src : int
        Number of observations per light curve.
    partitions_per_chunk : int or None
        Number of partitions to yield, default is None, which makes it equal
        to number of Dask workers being run with Dask client.
    seed : int
        Random seed to use for observation sampling.

    Methods
    -------
    __next__() -> pd.Series
        Provides light curves as a nested series.
    """

    def __init__(
        self,
        *,
        catalog: Catalog,
        client: Client | None,
        n_src: int,
        partitions_per_chunk: int | None = None,
        seed: int,
    ) -> None:
        self.client = client
        self.n_src = n_src
        self.seed = seed
        self.partitions_per_chunk = self._get_partitions_per_chunk(partitions_per_chunk, client)
        self.rng = np.random.default_rng((1 << 32, seed))

        self.partitions_left = self._shuffle_partitions(catalog, self.rng)
        if len(self.partitions_left) == 0:
            raise ValueError("Catalog must have at least one partition")
        self._empty = False

        self.lc_col = self._get_lc_col(catalog)

        self.nested_series = catalog.map_partitions(
            _process_partition,
            include_pixel=True,
            n_src=self.n_src,
            lc_col=self.lc_col,
            seed=self.seed,
        )

        self.future = self._submit_next_partitions()

    @classmethod
    def from_hyrax_config(cls, config: ConfigDict) -> Self:
        """Construct from a hyrax config

        These sections must be defined:
        [data_set.LSDBDataGenerator]
        n_src : int
        partitions_per_chunk : int or "none"
        seed : int

        [data_set.LSDBDataGenerator.nested_series.dp1]
        <Arguments to pass to dp1_catalog_single_band>

        [data_set.LSDBDataGenerator.dask.LocalCluster]
        <Arguments to pass to dask.distributed.Client>
        <If not defined, or "none" or `false`, client=None is used>

        Parameters
        ----------
        config : ConfigDict
            Hyrax config object.

        Returns
        -------
        LSDBDataGenerator
            Generator yielding training data from an LSDB nested_series
        """
        sub_config = config["data_set"]["LSDBDataGenerator"]

        dp1_catalog = dp1_catalog_single_band(**sub_config["catalog"]["dp1"])

        dask_client_config = sub_config["dask"].get("LocalCluster", "none")
        if dask_client_config == "none" or not dask_client_config:
            client = None
        else:
            client = dask.distributed.Client(**dask_client_config)

        n_src = sub_config["n_src"]
        if not isinstance(n_src, int):
            raise ValueError(
                f"Expected integer for `config['data_set']['LSDBDataGenerator']['n_src']`, but got {n_src}"
            )

        partitions_per_chunk = config["data_set"][""].get("partitions_per_chunk", "none")
        if partitions_per_chunk == "none":
            partitions_per_chunk = None
        if partitions_per_chunk is not None or isinstance(partitions_per_chunk, int):
            raise ValueError(
                f"Expected integer or 'none' for config['data_set']['LSDBDataGenerator']['partitions_per_chunk'], but got {partitions_per_chunk}"
            )

        seed = sub_config["seed"]
        if not isinstance(seed, int):
            raise ValueError(
                f"Expected integer for `config['data_set']['LSDBDataGenerator']['seed']`, but got {seed}"
            )

        return cls(
            catalog=dp1_catalog,
            client=client,
            n_src=n_src,
            partitions_per_chunk=partitions_per_chunk,
            seed=seed,
        )

    @staticmethod
    def _shuffle_partitions(catalog: Catalog, rng: np.random.Generator) -> list[int]:
        return rng.permutation(catalog.npartitions).tolist()

    @staticmethod
    def _get_lc_col(catalog: Catalog) -> str:
        nested_columns = catalog.nested_columns
        if len(nested_columns) == 0:
            raise ValueError("Catalog must have at least one nested column")
        if len(nested_columns) > 1:
            raise ValueError(
                "Catalog must have at most one nested column, these nested columns are found: "
                f"{nested_columns}"
            )
        nested_column = nested_columns[0]
        return nested_column

    @staticmethod
    def _get_partitions_per_chunk(input_chunk_size, client) -> int:
        """Number of chunk to yield, either given or number of Dask workers

        Returns one if no chunk size is given and workers are available yet
        (or no client is given).

        Returns
        -------
        int
            Number of partitions to yield.
        """
        if input_chunk_size is not None:
            return input_chunk_size
        if client is None:
            return 1
        n_workers = len(client.scheduler_info().get("workers", []))
        n_workers = n_workers if n_workers > 0 else 1
        return n_workers

    def _submit_next_partitions(self) -> Future | _FakeFuture:
        self.partitions_left, partitions = (
            self.partitions_left[: -self.partitions_per_chunk],
            self.partitions_left[-self.partitions_per_chunk :],
        )
        sliced_series = self.nested_series.partitions[partitions]

        if self.client is None:
            future = _FakeFuture(dask.compute(sliced_series)[0])
        else:
            future = self.client.compute(sliced_series)
        return future

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> pd.Series:
        if self._empty:
            raise StopIteration("All partitions have been processed")

        nested_series = self.future.result()
        result = nested_series.sample(frac=1, random_state=self.rng)

        if len(self.partitions_left) > 0:
            self.future = self._submit_next_partitions()
        else:
            self._empty = True
            self.future = None

        return result

    def __len__(self) -> int:
        return int(np.ceil(len(self.partitions_left) / self.partitions_per_chunk))


class LSDBIterableDataset(HyraxDataset):
    def __init__(self, config: ConfigDict, metadata_table=None):
        if metadata_table is not None:
            raise NotImplementedError("metadata_table=None is not supported")
        super().__init__(config, metadata_table=metadata_table)
        self.lsdb_data_generator = LSDBDataGenerator.from_hyrax_config(config)

    def __iter__(self) -> Generator[dict, None, None]:
        for chunk in self.lsdb_data_generator:
            flat_df = chunk.nest.to_flat()
            for record in flat_df.itertuples(index=False):
                yield {
                    "data": record._asdict(),
                }
