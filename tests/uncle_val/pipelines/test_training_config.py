import json

import pytest
import torch
from uncle_val.pipelines.training_config import ComputeConfig, TrainingConfig

# --- ComputeConfig ---


def test_compute_config_roundtrip(tmp_path):
    """ComputeConfig must survive a to_json/from_json roundtrip unchanged."""
    cfg = ComputeConfig(n_workers=4, device="cuda:0")
    path = tmp_path / "compute.json"
    cfg.to_json(path)
    assert path.exists()
    loaded = ComputeConfig.from_json(path)
    assert loaded == cfg


def test_compute_config_roundtrip_device_object(tmp_path):
    """torch.device should be serialized as its string representation."""
    cfg = ComputeConfig(n_workers=2, device=torch.device("cpu"))
    path = tmp_path / "compute.json"
    cfg.to_json(path)
    loaded = ComputeConfig.from_json(path)
    assert loaded.n_workers == cfg.n_workers
    assert str(loaded.device) == str(cfg.device)


def test_compute_config_json_is_readable(tmp_path):
    """The JSON file must be valid and contain the expected keys."""
    cfg = ComputeConfig(n_workers=3)
    path = tmp_path / "compute.json"
    cfg.to_json(path)
    d = json.loads(path.read_text())
    assert d == {"n_workers": 3, "device": "cpu"}


# --- TrainingConfig ---


@pytest.fixture
def training_cfg():
    """TrainingConfig instance with non-default values for all fields."""
    return TrainingConfig(
        compute_config=ComputeConfig(n_workers=2),
        n_lcs=1000,
        train_batch_size=64,
        val_batch_size=128,
        lr=1e-3,
        max_val_size=512,
        snapshot_factor=0.5,
        start_tfboard=True,
        run_feature_importance=False,
    )


def test_training_config_roundtrip(tmp_path, training_cfg):
    """TrainingConfig must survive a to_json/from_json roundtrip unchanged."""
    path = tmp_path / "training.json"
    training_cfg.to_json(path)
    assert path.exists()
    loaded = TrainingConfig.from_json(path)
    assert loaded == training_cfg


def test_training_config_nested_compute_config(tmp_path, training_cfg):
    """Nested ComputeConfig must survive the roundtrip intact."""
    path = tmp_path / "training.json"
    training_cfg.to_json(path)
    loaded = TrainingConfig.from_json(path)
    assert loaded.compute_config == training_cfg.compute_config


def test_training_config_json_is_readable(tmp_path, training_cfg):
    """The JSON file must be valid and contain a nested compute_config dict."""
    path = tmp_path / "training.json"
    training_cfg.to_json(path)
    d = json.loads(path.read_text())
    assert isinstance(d["compute_config"], dict)
    assert d["compute_config"]["n_workers"] == 2
    assert d["lr"] == pytest.approx(1e-3)
