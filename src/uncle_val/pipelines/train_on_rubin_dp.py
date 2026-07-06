from collections.abc import Callable, Sequence
from pathlib import Path

import lsdb
import torch
from nested_pandas import NestedFrame

from uncle_val.datasets.rubin_dp import rubin_dp_catalog_multi_band
from uncle_val.learning.losses import UncleLoss
from uncle_val.learning.models import BaseUncleModel
from uncle_val.pipelines.splits import SurveyConfig
from uncle_val.pipelines.training_config import TrainingConfig
from uncle_val.pipelines.training_loop import training_loop


def rubin_dp_catalog_and_columns(
    *,
    model: BaseUncleModel,
    survey_config: SurveyConfig,
    bands: Sequence[str] | None = None,
    pre_filter_partition: Callable[[NestedFrame], NestedFrame] | None = None,
) -> tuple[lsdb.Catalog, list[str]]:
    """Build the training catalog and the catalog columns to feed the model.

    The model's ``input_names`` define which catalog columns are used and in
    which order; e.g. include "psfFlux" (science forced flux) for
    ``img="diff"`` catalogs so the model sees how direct and difference
    photometry differ.

    Parameters
    ----------
    model : BaseUncleModel
        Model to be trained on the returned catalog columns.
    survey_config : SurveyConfig
        Survey configuration including catalog root, split boundaries, and n_src.
    bands : sequence of str or None
        Bands to include, subset of ``ugrizy``. Defaults to ``survey_config.bands``.
    pre_filter_partition : callable or None
        Optional function applied to each catalog partition before any other
        processing. Receives a ``NestedFrame`` and returns a filtered
        ``NestedFrame``.

    Returns
    -------
    lsdb.Catalog
        Catalog to train on.
    list[str]
        Catalog columns matching ``model.input_names`` one-to-one; nested
        light-curve fields are prefixed with ``lc.``.

    Raises
    ------
    ValueError
        If some model input is not available in the catalog.
    """
    bands = survey_config.bands if bands is None else tuple(bands)
    catalog = rubin_dp_catalog_multi_band(
        root=survey_config.catalog_root,
        bands=bands,
        obj=survey_config.obj,
        img=survey_config.img,
        phot=survey_config.phot,
        mode=survey_config.mode,
        pre_filter_partition=pre_filter_partition,
    ).map_partitions(lambda df: df.drop(columns=["band", "object_mag", "coord_ra", "coord_dec"]))

    lc_fields = list(catalog.dtypes["lc"].column_dtypes)
    flat_columns = [col for col in catalog.columns if col != "lc"]

    columns = []
    for name in model.input_names:
        if name in lc_fields:
            columns.append(f"lc.{name}")
        elif name in flat_columns:
            columns.append(name)
        else:
            raise ValueError(
                f"Model input {name!r} is not available in the catalog: "
                f"nested 'lc' fields are {lc_fields}, other columns are {flat_columns}"
            )

    return catalog, columns


def train_on_rubin_dp(
    *,
    model: BaseUncleModel | str | Path,
    output_dir: str | Path,
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
        continue training from. The model's ``input_names`` define which
        catalog columns are used as inputs and in which order, see
        :func:`rubin_dp_catalog_and_columns`.
    output_dir : str or Path
        Run directory to save all the outputs to, see
        :func:`~uncle_val.pipelines.training_loop.training_loop` for details.
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
        Catalog columns used as model inputs, matching ``model.input_names``
        one-to-one; nested light-curve fields are prefixed with ``lc.``.
    """
    if isinstance(model, str | Path):
        model = torch.load(model, weights_only=False, map_location=training_config.compute_config.device)

    catalog, columns = rubin_dp_catalog_and_columns(
        model=model,
        survey_config=survey_config,
        bands=bands,
        pre_filter_partition=pre_filter_partition,
    )
    columns_no_prefix = [col.removeprefix("lc.") for col in columns]

    if val_losses is None:
        val_losses = {}

    model_path = training_loop(
        catalog=catalog,
        columns=columns_no_prefix,
        model=model,
        loss_fn=loss_fn,
        val_losses=val_losses,
        output_dir=output_dir,
        model_name=type(model).__name__,
        survey_config=survey_config,
        training_config=training_config,
    )
    return model_path, columns
