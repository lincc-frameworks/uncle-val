from pathlib import Path

from uncle_val.datasets.rubin_dp import rubin_dp_catalog_multi_band
from uncle_val.learning.losses import UncleLoss
from uncle_val.learning.models import PerBandConstantMagErrModel
from uncle_val.pipelines.splits import SurveyConfig
from uncle_val.pipelines.training_config import TrainingConfig
from uncle_val.pipelines.training_loop import training_loop


def run_rubin_dp_per_band_constant_magerr(
    *,
    non_extended_only: bool,
    max_mag: float | None = None,
    gr_color: tuple[float, float] | None = None,
    output_dir: str | Path,
    loss_fn: UncleLoss,
    val_losses: dict[str, UncleLoss] | None = None,
    survey_config: SurveyConfig,
    training_config: TrainingConfig,
) -> Path:
    """Run the training with the per-band constant mag-err model

    All bands of ``survey_config`` are trained together in a single run, the
    model having one systematic magnitude error per band, selected by the
    one-hot band columns of the multi-band catalog.

    Parameters
    ----------
    non_extended_only : bool
        Whether to filter out extended sources.
    max_mag : float or None
        Keep only objects brighter than this magnitude. None applies no cut.
    gr_color : (float, float) or None
        Keep only objects whose g-r colour lies in this half-open range.
        None applies no colour cut.
    output_dir : str or Path
        Run directory to save all the outputs to, see
        :func:`~uncle_val.pipelines.training_loop.training_loop` for details.
    loss_fn : UncleLoss
        Loss function to use.
    val_losses : dict[str, UncleLoss] or None
        Extra losses to compute on validation set and record, it maps name to
        loss function. If None, an empty dictionary is used.
    survey_config : SurveyConfig
        Survey configuration including catalog root, split boundaries, bands,
        and n_src.
    training_config : TrainingConfig
        Training operational parameters (workers, batch sizes, lr, device, etc.).

    Returns
    -------
    Path
        Path to the output model.
    """
    bands = list(survey_config.bands)

    catalog = rubin_dp_catalog_multi_band(
        root=survey_config.catalog_root,
        bands=bands,
        obj=survey_config.obj,
        img=survey_config.img,
        phot=survey_config.phot,
        mode=survey_config.mode,
        ccd_visit_cols=None,
    )

    if non_extended_only:
        catalog = catalog.query("extendedness == 0.0")
    if max_mag is not None:
        catalog = catalog.query(f"object_mag < {max_mag}")
    if gr_color is not None:
        low, high = gr_color
        catalog = catalog.query(f"{low} <= gr_color < {high}")

    # On difference images "x" is the difference flux, consistent with zero for a
    # non-variable object, so the systematic is referenced to the science flux.
    source_flux = "psfFlux" if survey_config.img == "diff" else None

    band_columns = [f"is_{band}_band" for band in bands]
    # The dataset emits nested columns first, so spell the order out rather
    # than relying on columns=None: the model indexes its band parameters by
    # position in input_names.
    columns = ["x", "err"] + ([source_flux] if source_flux is not None else [])
    keep_columns = ["id"] + [f"lc.{column}" for column in columns] + band_columns
    columns = columns + band_columns
    catalog = catalog.map_partitions(lambda df: df[keep_columns])

    model = PerBandConstantMagErrModel(columns, bands, source_flux=source_flux).to(
        device=training_config.compute_config.device
    )

    if val_losses is None:
        val_losses = {}

    return training_loop(
        catalog=catalog,
        columns=columns,
        model=model,
        loss_fn=loss_fn,
        val_losses=val_losses,
        output_dir=output_dir,
        model_name="per_band_constant_magerr",
        survey_config=survey_config,
        training_config=training_config,
    )
