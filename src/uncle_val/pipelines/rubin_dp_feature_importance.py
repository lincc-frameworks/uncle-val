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
from uncle_val.pipelines.splits import SurveyConfig
from uncle_val.pipelines.training_config import ComputeConfig


def _flat_obs_generator(
    catalog,
    *,
    columns_no_prefix: list[str],
    client: Client,
    survey_config: SurveyConfig,
    compute_config: ComputeConfig,
) -> Generator[torch.Tensor, None, None]:
    """Yield flat ``(n_obs, n_features)`` tensors from catalog test-split partitions.

    Uses ``subsample_src=False`` so every observation from every qualifying
    light curve is included — no subsampling is applied.
    """
    for chunk in lsdb_nested_series_data_generator(
        catalog,
        client=client,
        n_src=survey_config.n_src,
        subsample_src=False,
        partitions_per_chunk=compute_config.n_workers * 8,
        loop=False,
        hash_range=survey_config.test_split,
        seed=0,
    ):
        flat_df = chunk.nest.to_flat()[columns_no_prefix]
        yield torch.tensor(flat_df.to_numpy(dtype=np.float32), device=compute_config.device)


def run_rubin_dp_feature_importance(
    *,
    model_path: str | Path,
    model_columns: list[str],
    bands: Sequence[str] | None = None,
    pre_filter_partition: Callable[[NestedFrame], NestedFrame] | None = None,
    output_path: str | Path | None = None,
    survey_config: SurveyConfig,
    compute_config: ComputeConfig,
) -> plt.Figure:
    """Compute and plot SHAP feature importance on the Rubin DP test split.

    All observations from qualifying light curves are used — no subsampling
    is applied. Light curves with fewer than ``survey_config.n_src`` observations
    are excluded.

    Parameters
    ----------
    model_path : str or Path
        Path to the saved model checkpoint.
    model_columns : list of str
        Columns used as model inputs (with ``lc.`` prefix for nested columns),
        as returned by :func:`~uncle_val.pipelines.rubin_dp_mlp.run_rubin_dp_mlp`.
    bands : sequence of str or None
        Bands to include, subset of ``ugrizy``. Defaults to ``survey_config.bands``.
    pre_filter_partition : callable or None
        Optional function applied to each catalog partition before any other
        processing. Receives a ``NestedFrame`` and returns a filtered
        ``NestedFrame``.
    output_path : str, Path, or None
        If given, save the figure to this path.
    survey_config : SurveyConfig
        Survey configuration including catalog root, split boundaries, and n_src.
    compute_config : ComputeConfig
        Compute/infrastructure parameters; ``n_workers`` and ``device`` are used.

    Returns
    -------
    matplotlib.figure.Figure
        SHAP beeswarm figure.
    """
    device = torch.device(compute_config.device)
    bands = survey_config.bands if bands is None else tuple(bands)
    columns_no_prefix = [col.removeprefix("lc.") for col in model_columns]

    catalog = rubin_dp_catalog_multi_band(
        root=survey_config.catalog_root,
        bands=bands,
        obj="science",
        img="cal",
        phot="PSF",
        mode="forced",
        pre_filter_partition=pre_filter_partition,
    ).map_partitions(lambda df: df.drop(columns=["band", "object_mag", "coord_ra", "coord_dec"]))

    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()

    with Client(n_workers=compute_config.n_workers, memory_limit="8GB", threads_per_worker=1) as client:
        print(f"Dask Dashboard Link: {client.dashboard_link}")
        obs_iter = _flat_obs_generator(
            catalog,
            columns_no_prefix=columns_no_prefix,
            client=client,
            survey_config=survey_config,
            compute_config=compute_config,
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
