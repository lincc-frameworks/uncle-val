from collections.abc import Callable, Generator, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dask.distributed import Client
from nested_pandas import NestedFrame

from uncle_val.datasets.rubin_dp import rubin_dp_catalog_multi_band
from uncle_val.learning.feature_importance import compute_shap_values, plot_shap_summary
from uncle_val.learning.lsdb_dataset import lsdb_nested_series_data_generator
from uncle_val.pipelines.splits import TEST_SPLIT


def _flat_obs_generator(
    catalog,
    *,
    columns_no_prefix: list[str],
    n_src: int,
    n_workers: int,
    client: Client,
    device: torch.device,
) -> Generator[torch.Tensor, None, None]:
    """Yield flat ``(n_obs, n_features)`` tensors from catalog test-split partitions.

    Uses ``subsample_src=False`` so every observation from every qualifying
    light curve is included — no subsampling is applied.
    """
    for chunk in lsdb_nested_series_data_generator(
        catalog,
        client=client,
        n_src=n_src,
        subsample_src=False,
        partitions_per_chunk=n_workers * 8,
        loop=False,
        hash_range=TEST_SPLIT,
        seed=0,
    ):
        flat_df = chunk.nest.to_flat()[columns_no_prefix]
        yield torch.tensor(flat_df.to_numpy(dtype=np.float32), device=device)


def run_rubin_dp_feature_importance(
    *,
    rubin_dp_root: str | Path,
    model_path: str | Path,
    model_columns: list[str],
    n_workers: int,
    n_src: int,
    bands: Sequence[str] = "ugrizy",
    pre_filter_partition: Callable[[NestedFrame], NestedFrame] | None = None,
    device: str | torch.device = "cpu",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Compute and plot SHAP feature importance on the Rubin DP test split.

    All observations from qualifying light curves are used — no subsampling
    is applied. Light curves with fewer than `n_src` observations are excluded.

    Parameters
    ----------
    rubin_dp_root : str or Path
        Root directory of the Rubin DP HATS catalogs.
    model_path : str or Path
        Path to the saved model checkpoint.
    model_columns : list of str
        Columns used as model inputs (with ``lc.`` prefix for nested columns),
        as returned by :func:`~uncle_val.pipelines.rubin_dp_mlp.run_rubin_dp_mlp`.
    n_workers : int
        Number of Dask workers.
    n_src : int, optional
        Minimum number of observations required per light curve. Light curves
        with fewer observations are excluded. Default is 1 (include all).
    bands : sequence of str
        Bands to include, subset of ``ugrizy``.
    pre_filter_partition : callable or None
        Optional function applied to each catalog partition before any other
        processing. Receives a ``NestedFrame`` and returns a filtered
        ``NestedFrame``.
    device : str or torch.device
        Torch device for model and data.
    output_path : str, Path, or None
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        SHAP beeswarm figure.
    """
    device = torch.device(device)
    columns_no_prefix = [col.removeprefix("lc.") for col in model_columns]

    catalog = rubin_dp_catalog_multi_band(
        root=rubin_dp_root,
        bands=bands,
        obj="science",
        img="cal",
        phot="PSF",
        mode="forced",
        pre_filter_partition=pre_filter_partition,
    ).map_partitions(lambda df: df.drop(columns=["band", "object_mag", "coord_ra", "coord_dec"]))

    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()

    with Client(n_workers=n_workers, memory_limit="8GB", threads_per_worker=1) as client:
        print(f"Dask Dashboard Link: {client.dashboard_link}")
        obs_iter = _flat_obs_generator(
            catalog,
            columns_no_prefix=columns_no_prefix,
            n_src=n_src,
            n_workers=n_workers,
            client=client,
            device=device,
        )
        shap_values, feature_data = compute_shap_values(
            model_path=model_path,
            data_loader=obs_iter,
            device=device,
        )

    return plot_shap_summary(
        shap_values,
        feature_data,
        input_names=model.input_names,
        output_path=output_path,
    )
