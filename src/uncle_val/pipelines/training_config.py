from dataclasses import dataclass

import torch


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
