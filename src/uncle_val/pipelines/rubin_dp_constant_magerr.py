from pathlib import Path

from uncle_val.datasets import rubin_dp_catalog_single_band
from uncle_val.learning.losses import UncleLoss
from uncle_val.learning.models import ConstantMagErrModel
from uncle_val.pipelines.splits import SurveyConfig
from uncle_val.pipelines.training_config import TrainingConfig
from uncle_val.pipelines.training_loop import training_loop


def run_rubin_dp_constant_magerr(
    *,
    band: str,
    non_extended_only: bool,
    output_root: str | Path,
    loss_fn: UncleLoss,
    val_losses: dict[str, UncleLoss] | None = None,
    survey_config: SurveyConfig,
    training_config: TrainingConfig,
) -> Path:
    """Run the training with the constant mag-err model on fluxes and errors

    Parameters
    ----------
    band : str
        Passband to train the model on.
    non_extended_only : bool
        Whether to filter out extended sources.
    output_root : str or Path
        Where to save the intermediate results.
    loss_fn : UncleLoss
        Loss function to use, by default soften Χ² is used.
    val_losses : dict[str, UncleLoss] or None
        Extra losses to compute on validation set and record, it maps name to
        loss function. If None, an empty dictionary is used.
    survey_config : SurveyConfig
        Survey configuration including catalog root, split boundaries, and n_src.
    training_config : TrainingConfig
        Training operational parameters (workers, batch sizes, lr, device, etc.).

    Returns
    -------
    Path
        Path to the output model.
    """
    catalog = rubin_dp_catalog_single_band(
        root=survey_config.catalog_root,
        band=band,
        obj="science",
        img="cal",
        phot="PSF",
        mode="forced",
    )

    if non_extended_only:
        catalog = catalog.query("extendedness == 0.0")
    catalog = catalog.map_partitions(lambda df: df[["id", "lc.x", "lc.err"]])

    model = ConstantMagErrModel(["x", "err"]).to(device=training_config.compute_config.device)

    if val_losses is None:
        val_losses = {}

    return training_loop(
        catalog=catalog,
        columns=None,
        model=model,
        loss_fn=loss_fn,
        val_losses=val_losses,
        output_root=output_root,
        model_name="constant_magerr",
        survey_config=survey_config,
        training_config=training_config,
    )
