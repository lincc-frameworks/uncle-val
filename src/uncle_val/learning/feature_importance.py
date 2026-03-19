from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch


class _FlatWrapper(torch.nn.Module):
    """Wraps an UncleModel to accept flat ``(N, n_features)`` input.

    Treats N observations as a single light curve ``(1, N, n_features)``
    so that each observation is processed independently, and returns the
    per-observation uncertainty factor ``u`` as a 2-D tensor ``(N, 1)``.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, flat_inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning u per observation."""
        output = self.model(flat_inputs.unsqueeze(0))  # (1, N, d_output)
        return output[0, :, :1]  # (N, 1) — shap requires 2D output


def compute_shap_values(
    *,
    model_path: str | Path,
    data_loader: Iterable,
    device: torch.device,
    n_background: int = 500,
    n_test: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values for the predicted uncertainty factor ``u``.

    Uses :class:`shap.GradientExplainer` with a flat ``(N, n_features)``
    wrapper around the model.  Background and test observations are drawn
    from the first batches of *data_loader*.

    Parameters
    ----------
    model_path : str or Path
        Path to the saved model checkpoint.
    data_loader : iterable
        Yields batches of shape ``(batch_lc, n_src, n_features)`` or
        ``(1, batch_lc, n_src, n_features)`` (e.g. a raw
        ``LSDBIterableDataset`` or a ``DataLoader`` wrapping one).
    device : torch.device
        Device for model and data.
    n_background : int
        Number of background observations for the SHAP baseline.
    n_test : int
        Number of observations to explain.

    Returns
    -------
    shap_values : np.ndarray, shape ``(n_test, n_features)``
        SHAP values for each observation and feature.
    feature_data : np.ndarray, shape ``(n_test, n_features)``
        Raw feature values corresponding to *shap_values*, used for
        colouring the beeswarm plot.
    """
    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()

    # Collect flat observations from batches
    obs: list[torch.Tensor] = []
    n_needed = n_background + n_test
    for batch in data_loader:
        flat = batch.squeeze(0).reshape(-1, batch.shape[-1])  # (batch_lc*n_src, n_features)
        obs.append(flat)
        if sum(t.shape[0] for t in obs) >= n_needed:
            break
    all_obs = torch.cat(obs, dim=0)[:n_needed]

    background = all_obs[:n_background]
    test_data = all_obs[n_background:n_needed]

    wrapped = _FlatWrapper(model)
    explainer = shap.GradientExplainer(wrapped, background)
    shap_values = np.array(explainer.shap_values(test_data))
    # GradientExplainer returns (n_test, n_features, n_outputs); drop the output dim
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 0]

    return shap_values, test_data.cpu().detach().numpy()


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_data: np.ndarray,
    input_names: list[str],
    *,
    output_path: str | Path | None = None,
    title: str = "SHAP Feature Importance",
) -> plt.Figure:
    """Plot a SHAP beeswarm summary for the predicted uncertainty factor ``u``.

    Each dot represents one observation, positioned by its SHAP value (impact
    on ``u``) and coloured by the raw feature value (red = high, blue = low).

    Parameters
    ----------
    shap_values : np.ndarray, shape ``(n_samples, n_features)``
        SHAP values as returned by :func:`compute_shap_values`.
    feature_data : np.ndarray, shape ``(n_samples, n_features)``
        Raw feature values corresponding to *shap_values*.
    input_names : list of str
        Feature names in the order of the last dimension.
    output_path : str, Path, or None
        If given, save the figure to this path.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    plt.close("all")
    explanation = shap.Explanation(
        values=shap_values,
        data=feature_data,
        feature_names=input_names,
    )
    shap.plots.beeswarm(explanation, show=False, max_display=len(input_names))
    fig = plt.gcf()
    fig.suptitle(title, y=1.01)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")

    return fig
