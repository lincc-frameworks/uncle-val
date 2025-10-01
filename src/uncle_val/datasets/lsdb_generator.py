from collections.abc import Iterator
from typing import Self

import dask
import numpy as np
import pandas as pd
from dask.distributed import Client, Future
from lsdb import Catalog


class _FakeFuture:
    """Duck-typed `Future` interface for a pre-computed value.

    Parameters
    ----------
    obj
        Value to hold
    """

    def __init__(self, obj):
        self.obj = obj

    def result(self):
        return self.obj


class LSDBDataGenerator(Iterator[pd.Series | pd.DataFrame]):
    """Generator yielding training data from an LSDB

    The data is pre-fetched on the background, 'n_workers' number
    of partitions per time (derived from `client` object).

    Parameters
    ----------
    catalog : lsdb.Catalog
        LSDB nested_series object.
    client : dask.distributed.Client or None
        Dask client for distributed computation. None means running
        in a synced way with `dask.compute()` instead of asynced with
        `client.compute()`.
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
        partitions_per_chunk: int | None = None,
        seed: int,
    ) -> None:
        self.client = client
        self.seed = seed
        self.partitions_per_chunk = self._get_partitions_per_chunk(partitions_per_chunk, client)
        self.rng = np.random.default_rng((1 << 32, seed))

        self.catalog = catalog
        self.partitions_left = self._shuffle_partitions(self.catalog, self.rng)
        if len(self.partitions_left) == 0:
            raise ValueError("Catalog must have at least one partition")
        self._empty = False

        self.future = self._submit_next_partitions()

    @staticmethod
    def _shuffle_partitions(catalog: Catalog, rng: np.random.Generator) -> np.ndarray:
        return rng.permutation(catalog.npartitions)

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
        sliced_catalog = self.catalog.partitions[partitions]
        futurable = sliced_catalog._ddf if hasattr(sliced_catalog, "_ddf") else sliced_catalog

        if self.client is None:
            future = _FakeFuture(dask.compute(futurable)[0])
        else:
            future = self.client.compute(futurable)
        return future

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> pd.Series:
        if self._empty:
            raise StopIteration("All partitions have been processed")

        result: pd.Series | pd.DataFrame = self.future.result()
        result = result.sample(frac=1, random_state=self.rng)

        if len(self.partitions_left) > 0:
            self.future = self._submit_next_partitions()
        else:
            self._empty = True
            self.future = None

        return result

    def __len__(self) -> int:
        return int(np.ceil(len(self.partitions_left) / self.partitions_per_chunk))
