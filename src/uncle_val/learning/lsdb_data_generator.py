from collections.abc import Iterator

import dask
import jax.numpy as jnp
import numpy as np
from dask.distributed import Client, Future
from hats import HealpixPixel
from lsdb import Catalog
from nested_pandas import NestedFrame


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
            result[f"{lc_col}.{col}"] = np.full(lc_length, value)
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

    if len(nf) > 0:
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
    else:
        fixed_length_nf = nf.copy()
        del fixed_length_nf[lc_col]
        del fixed_length_nf[length_col]
        fixed_length_nf = fixed_length_nf.join(nf[lc_col].nest.to_flat())
        return fixed_length_nf

    fixed_length_series = fixed_length_nf[lc_col]
    fixed_length_flat = fixed_length_series.nest.to_flat()
    return fixed_length_flat


class LSDBDataGenerator(Iterator[dict[str, jnp.ndarray]]):
    """Generator yielding training data from an LSDB catalog

    The data is pre-fetched on the background, 'n_workers' number
    of partitions per time (derived from `client` object).
    It filters out light curves with less than `n_src` observations,
    and selects `n_src` random observations per light curve.

    Catalog must have the only nested column, which is interpreted as a light
    curve data.

    Parameters
    ----------
    catalog : lsdb.Catalog
        LSDB catalog object.
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
    __next__() -> dict[str, jnp.ndarray]
        Provides light curves as a dictionaries, keys are
        sub-column names, and values are 2-D jax.numpy arrays, shape is
        (n_obj, n_src).
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
        self.catalog = catalog
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

        self.future = self._submit_next_partitions()

    @staticmethod
    def _shuffle_partitions(catalog: Catalog, rng: np.random.Generator) -> list[int]:
        return rng.choice(catalog.npartitions, size=catalog.npartitions, replace=False).tolist()

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
        sliced_catalog = self.catalog.partitions[partitions]

        mapped_catalog = sliced_catalog.map_partitions(
            _process_partition,
            include_pixel=True,
            n_src=self.n_src,
            lc_col=self.lc_col,
            seed=self.seed,
        )

        if self.client is None:
            future = _FakeFuture(dask.compute(mapped_catalog._ddf)[0])
        else:
            future = self.client.compute(mapped_catalog._ddf)
        return future

    def __iter__(self) -> Iterator[dict[str, jnp.ndarray]]:
        return self

    def __next__(self) -> dict[str, jnp.ndarray]:
        if self._empty:
            raise StopIteration("All partitions have been processed")

        flat_df = self.future.result()
        dict_1d = flat_df.to_dict(orient="series")
        dict_2d = {str(col): jnp.asarray(series).reshape(-1, self.n_src) for col, series in dict_1d.items()}
        n_obj_total = len(next(iter(dict_2d.values())))
        shuffle_idx = self.rng.permutation(n_obj_total)
        result = {col: value[shuffle_idx] for col, value in dict_2d.items()}

        if len(self.partitions_left) > 0:
            self.future = self._submit_next_partitions()
        else:
            self._empty = True
            self.future = None

        return result

    def __len__(self) -> int:
        return int(np.ceil(len(self.partitions_left) / self.partitions_per_chunk))
