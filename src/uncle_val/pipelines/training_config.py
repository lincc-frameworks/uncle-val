from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import torch


def _config_encoder(obj):
    """JSON serializer for types not handled by the default encoder."""
    if isinstance(obj, Path | torch.device):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@dataclass(frozen=True)
class ComputeConfig:
    """Compute/infrastructure parameters shared by all pipeline functions.

    Parameters
    ----------
    n_workers : int
        Number of Dask workers.
    device : str or torch.device, optional
        Torch device to use. Defaults to ``"cpu"``.
    """

    n_workers: int
    device: str | torch.device = "cpu"

    def __post_init__(self):
        if self.n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {self.n_workers}")

    def to_json(self, path: str | Path) -> None:
        """Serialize to a JSON file.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        Path(path).write_text(json.dumps(dataclasses.asdict(self), default=_config_encoder, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> ComputeConfig:
        """Deserialize from a JSON file produced by :meth:`to_json`.

        Parameters
        ----------
        path : str or Path
            Source file path.

        Returns
        -------
        ComputeConfig
        """
        return cls(**json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class TrainingConfig:
    """Training-specific parameters.

    Parameters
    ----------
    compute_config : ComputeConfig
        Compute/infrastructure parameters (workers, device).
    n_lcs : int
        Total light curves to train on.
    train_batch_size : int
        Light curves per training batch.
    val_batch_size : int
        Light curves per validation batch.
    lr : float
        Learning rate.
    max_val_size : int, optional
        Maximum number of light curves to materialize for validation.
        Defaults to 2**20 = 1,048,576.
    snapshot_factor : float, optional
        Snapshot every ``factor × real_val_size`` training light curves.
        Computed after materialization so it scales with the actual val set.
        Defaults to 1.0.
    start_tfboard : bool, optional
        Whether to start a TensorBoard session. Defaults to ``False``.
    run_feature_importance : bool, optional
        Whether to compute SHAP feature importance at the end of training.
        Defaults to ``False``.
    """

    compute_config: ComputeConfig
    n_lcs: int
    train_batch_size: int
    val_batch_size: int
    lr: float
    max_val_size: int = 2**20
    snapshot_factor: float = 1.0
    start_tfboard: bool = False
    run_feature_importance: bool = False

    def __post_init__(self):
        if self.n_lcs < 1:
            raise ValueError(f"n_lcs must be >= 1, got {self.n_lcs}")
        if self.train_batch_size < 1:
            raise ValueError(f"train_batch_size must be >= 1, got {self.train_batch_size}")
        if self.val_batch_size < 1:
            raise ValueError(f"val_batch_size must be >= 1, got {self.val_batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.snapshot_factor <= 0.0:
            raise ValueError(f"snapshot_factor must be > 0, got {self.snapshot_factor}")

    def to_json(self, path: str | Path) -> None:
        """Serialize to a JSON file.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        Path(path).write_text(json.dumps(dataclasses.asdict(self), default=_config_encoder, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> TrainingConfig:
        """Deserialize from a JSON file produced by :meth:`to_json`.

        Parameters
        ----------
        path : str or Path
            Source file path.

        Returns
        -------
        TrainingConfig
        """
        d = json.loads(Path(path).read_text())
        d["compute_config"] = ComputeConfig(**d["compute_config"])
        return cls(**d)
