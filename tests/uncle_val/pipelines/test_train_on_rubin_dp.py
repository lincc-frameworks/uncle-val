import pytest

from uncle_val.learning.lsdb_dataset import LSDBIterableDataset
from uncle_val.learning.models import MLPModel
from uncle_val.pipelines.splits import dp1_config
from uncle_val.pipelines.train_on_rubin_dp import rubin_dp_catalog_and_columns

BANDS = ("u", "g", "r", "i", "z", "y")

BASE_INPUT_NAMES = [
    "x",
    "err",
    "extendedness",
    "skyBg",
    "seeing",
    "expTime",
    "detector_rho",
    "detector_cos_phi",
    "detector_sin_phi",
] + [f"is_{band}_band" for band in BANDS]


def test_rubin_dp_catalog_and_columns_feed_model(rubin_dp_root):
    """Columns produced for img="diff" must feed a model not using psfFlux.

    Regression test: the pipeline used to append "lc.psfFlux" for img="diff"
    regardless of the model's inputs, feeding 16 features to a model built
    for 15 and crashing in the first linear layer (or silently dropping the
    last feature, depending on the model's input normalization details).
    """
    model = MLPModel(input_names=BASE_INPUT_NAMES, outputs_s=False, d_middle=(8,))
    catalog, columns = rubin_dp_catalog_and_columns(
        model=model,
        survey_config=dp1_config(rubin_dp_root, n_src=5, img="diff"),
    )
    assert [col.removeprefix("lc.") for col in columns] == model.input_names

    dataset = LSDBIterableDataset(
        catalog,
        columns=[col.removeprefix("lc.") for col in columns],
        client=None,
        batch_lc=1,
        n_src=5,
        partitions_per_chunk=1,
        seed=0,
    )
    batch = next(iter(dataset))
    output = model(batch)
    assert output.shape == (1, 5, 1)


def test_rubin_dp_catalog_and_columns_psf_flux(rubin_dp_root):
    """A model asking for psfFlux gets the nested lc.psfFlux column for img="diff"."""
    model = MLPModel(input_names=BASE_INPUT_NAMES + ["psfFlux"], outputs_s=False, d_middle=(8,))
    _catalog, columns = rubin_dp_catalog_and_columns(
        model=model,
        survey_config=dp1_config(rubin_dp_root, n_src=5, img="diff"),
    )
    assert [col.removeprefix("lc.") for col in columns] == model.input_names
    assert columns[-1] == "lc.psfFlux"


def test_rubin_dp_catalog_and_columns_unavailable_input_raises(rubin_dp_root):
    """psfFlux does not exist in img="cal" catalogs, so it must raise."""
    model = MLPModel(input_names=BASE_INPUT_NAMES + ["psfFlux"], outputs_s=False, d_middle=(8,))
    with pytest.raises(ValueError, match="psfFlux"):
        rubin_dp_catalog_and_columns(
            model=model,
            survey_config=dp1_config(rubin_dp_root, n_src=5, img="cal"),
        )
