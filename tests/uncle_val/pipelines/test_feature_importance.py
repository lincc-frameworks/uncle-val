import matplotlib
import numpy as np
import pytest
import torch
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from uncle_val.learning.feature_importance import _FlatWrapper, compute_shap_values, plot_shap_summary
from uncle_val.learning.models import LinearModel

matplotlib.use("Agg")

N_FEATURES = 3
INPUT_NAMES = ["x", "err", "sky"]


class _FakeTensorDataset(torch.utils.data.Dataset):
    """Yields tensors shaped like a materialized validation set: (batch_lc, n_src, n_features)."""

    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx]


def _fake_val_dataloader(*, batch_lc=4, n_src=5, n_features=N_FEATURES, n_batches=5, seed=0):
    """DataLoader that mimics MaterializedDataLoaderContext output.

    Each batch has shape ``(1, batch_lc, n_src, n_features)``.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    tensors = [torch.randn(batch_lc, n_src, n_features, generator=rng) for _ in range(n_batches)]
    return DataLoader(_FakeTensorDataset(tensors), batch_size=1, shuffle=False)


@pytest.fixture
def linear_model():
    """Fixture providing a simple trained-free LinearModel for tests."""
    model = LinearModel(input_names=INPUT_NAMES, outputs_s=False)
    model.eval()
    return model


@pytest.fixture
def model_path(linear_model, tmp_path):
    """Fixture that saves a LinearModel to a temp file and returns its path."""
    path = tmp_path / "model.pt"
    torch.save(linear_model, path)
    return path


# --- _FlatWrapper ---


def test_flat_wrapper_output_shape(linear_model):
    """Output must be (N, 1) for shap's 2-D output requirement."""
    n_obs = 20
    flat = torch.randn(n_obs, N_FEATURES)
    out = _FlatWrapper(linear_model)(flat)
    assert out.shape == (n_obs, 1)


def test_flat_wrapper_gradients_flow_for_all_features(linear_model):
    """Gradients must reach every input feature — catches zero-gradient bugs early.

    Uses positive inputs because UncleScaler applies log10 to flux/err, which
    produces NaN gradients for negative values.
    """
    n_obs = 10
    flat = torch.rand(n_obs, N_FEATURES) * 100 + 1.0
    flat.requires_grad_(True)
    out = _FlatWrapper(linear_model)(flat)
    out.sum().backward()
    assert flat.grad is not None
    assert flat.grad.shape == (n_obs, N_FEATURES)
    assert torch.isfinite(flat.grad).all(), "Gradients contain NaN/Inf"
    assert (flat.grad.abs().sum(dim=0) > 0).all(), "Some features have zero gradient"


# --- compute_shap_values ---


def test_compute_shap_values_output_shapes(model_path):
    """shap_values must be (n_test, n_features), not (n_test, n_features, 1)."""
    n_background, n_test = 20, 30
    loader = _fake_val_dataloader()

    shap_values, feature_data = compute_shap_values(
        model_path=model_path,
        data_loader=loader,
        device=torch.device("cpu"),
        n_background=n_background,
        n_test=n_test,
    )

    assert shap_values.shape == (n_test, N_FEATURES), f"Got {shap_values.shape}"
    assert feature_data.shape == (n_test, N_FEATURES)
    assert isinstance(shap_values, np.ndarray)
    assert isinstance(feature_data, np.ndarray)


# --- plot_shap_summary ---


def test_plot_shap_summary_returns_figure():
    """plot_shap_summary must return a matplotlib Figure."""
    n_test = 50
    shap_values = np.random.randn(n_test, N_FEATURES)
    feature_data = np.random.randn(n_test, N_FEATURES)

    fig = plot_shap_summary(shap_values, feature_data, input_names=INPUT_NAMES)
    assert isinstance(fig, Figure)


def test_plot_shap_summary_saves_file(tmp_path):
    """plot_shap_summary must write a file to output_path when given."""
    n_test = 50
    shap_values = np.random.randn(n_test, N_FEATURES)
    feature_data = np.random.randn(n_test, N_FEATURES)
    out_path = tmp_path / "shap.png"

    plot_shap_summary(shap_values, feature_data, input_names=INPUT_NAMES, output_path=out_path)
    assert out_path.exists()
