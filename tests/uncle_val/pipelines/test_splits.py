import json
from pathlib import Path

from uncle_val.pipelines.splits import SurveyConfig, dp1_config


def test_survey_config_roundtrip(tmp_path):
    """SurveyConfig must survive a to_json/from_json roundtrip unchanged."""
    cfg = SurveyConfig(
        catalog_root="/data/dp1",
        val_start=0.6,
        test_start=0.85,
        n_src=10,
        bands=("g", "r", "i"),
    )
    path = tmp_path / "survey.json"
    cfg.to_json(path)
    assert path.exists()
    loaded = SurveyConfig.from_json(path)
    assert loaded == cfg


def test_survey_config_roundtrip_path_object(tmp_path):
    """catalog_root given as a Path should survive the roundtrip."""
    cfg = SurveyConfig(
        catalog_root=Path("/data/dp1"),
        val_start=0.6,
        test_start=0.85,
        n_src=10,
    )
    path = tmp_path / "survey.json"
    cfg.to_json(path)
    loaded = SurveyConfig.from_json(path)
    assert loaded.catalog_root == cfg.catalog_root


def test_survey_config_bands_stay_tuple(tmp_path):
    """bands must come back as a tuple, not a list."""
    cfg = dp1_config("/data/dp1", n_src=10)
    path = tmp_path / "survey.json"
    cfg.to_json(path)
    loaded = SurveyConfig.from_json(path)
    assert isinstance(loaded.bands, tuple)
    assert loaded.bands == cfg.bands


def test_survey_config_json_is_readable(tmp_path):
    """The JSON file must be valid and catalog_root stored as a string."""
    cfg = dp1_config("/data/catalogs", n_src=10)
    path = tmp_path / "survey.json"
    cfg.to_json(path)
    d = json.loads(path.read_text())
    assert d["catalog_root"] == "/data/catalogs"
    assert isinstance(d["bands"], list)
