from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from nested_pandas import NestedFrame

from uncle_val.datasets.rubin_dp import rubin_dp_catalog_multi_band
from uncle_val.learning.losses import UncleLoss
from uncle_val.learning.models import BaseUncleModel
from uncle_val.pipelines.splits import SurveyConfig
from uncle_val.pipelines.training_config import TrainingConfig
from uncle_val.pipelines.training_loop import training_loop


def train_on_rubin_dp(
    *,
    model: BaseUncleModel | str | Path,
    output_root: str | Path,
    loss_fn: UncleLoss,
    val_losses: dict[str, UncleLoss] | None = None,
    bands: Sequence[str] | None = None,
    pre_filter_partition: Callable[[NestedFrame], NestedFrame] | None = None,
    survey_config: SurveyConfig,
    training_config: TrainingConfig,
) -> tuple[Path, list[str]]:
    """Run the training on a multi-band Rubin DP catalog

    Parameters
    ----------
    model : BaseUncleModel or str or Path
        Model instance to train, or a path to a saved model to load and
        continue training from.
    output_root : str or Path
        Where to save the intermediate results.
    loss_fn : UncleLoss
        Loss function to use.
    val_losses : dict[str, UncleLoss] or None
        Extra losses to compute on validation set and record, it maps name to
        loss function. If None, an empty dictionary is used.
    bands : sequence of str or None
        Bands to include, subset of ``ugrizy``. Defaults to ``survey_config.bands``.
    pre_filter_partition : callable or None
        Optional function applied to each catalog partition before any other
        processing. Receives a ``NestedFrame`` and returns a filtered
        ``NestedFrame``.
    survey_config : SurveyConfig
        Survey configuration including catalog root, split boundaries, and n_src.
    training_config : TrainingConfig
        Training operational parameters (workers, batch sizes, lr, device, etc.).

    Returns
    -------
    Path
        Path to the output model.
    list[str]
        List of columns used as model inputs.
    """
    bands = survey_config.bands if bands is None else tuple(bands)
    catalog = rubin_dp_catalog_multi_band(
        root=survey_config.catalog_root,
        bands=bands,
        obj="science",
        img="cal",
        phot="PSF",
        mode="forced",
        pre_filter_partition=pre_filter_partition,
    ).map_partitions(lambda df: df.drop(columns=["band", "object_mag", "coord_ra", "coord_dec"]))

    columns = [
        "lc.x",
        "lc.err",
        "extendedness",
        "lc.skyBg",
        "lc.seeing",
        "lc.expTime",
        "lc.detector_rho",
        "lc.detector_cos_phi",
        "lc.detector_sin_phi",
    ] + [f"is_{band}_band" for band in bands]
    columns_no_prefix = [col.removeprefix("lc.") for col in columns]

    if isinstance(model, str | Path):
        model = torch.load(model, weights_only=False, map_location=training_config.compute_config.device)

    if val_losses is None:
        val_losses = {}

    model_path = training_loop(
        catalog=catalog,
        columns=columns_no_prefix,
        model=model,
        loss_fn=loss_fn,
        val_losses=val_losses,
        output_root=output_root,
        model_name=type(model).__name__,
        survey_config=survey_config,
        training_config=training_config,
    )
    return model_path, columns
