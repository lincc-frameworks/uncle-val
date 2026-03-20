from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SurveyConfig:
    """Train/val/test split configuration for a survey.

    The hash range [0, 1) is partitioned as:
    - train: [0, val_start)
    - val:   [val_start, test_start)
    - test:  [test_start, 1)

    Parameters
    ----------
    catalog_root : str or Path
        Root directory of the survey's HATS catalogs.
    val_start : float
        Boundary between train and val sets. Must be in (0, 1).
    test_start : float
        Boundary between val and test sets. Must be in (val_start, 1).
    n_src : int
        Number of observations to subsample per light curve.
    bands : tuple of str, optional
        Survey filter bands. Defaults to all six LSST bands.
    max_val_size : int, optional
        Maximum number of light curves to materialize for validation.
        Defaults to 2**20 = 1,048,576.
    snapshot_factor : float, optional
        Snapshot every ``factor × real_val_size`` training light curves.
        Computed after materialization so it scales with the actual val set.
        Defaults to 1.0.

    Examples
    --------
    >>> cfg = dp1_config("/path/to/dp1")
    >>> cfg.train_split
    (0.0, 0.6)
    >>> cfg.val_split
    (0.6, 0.85)
    >>> cfg.test_split
    (0.85, 1.0)
    """

    catalog_root: Path
    val_start: float
    test_start: float
    n_src: int
    bands: tuple[str, ...] = ("u", "g", "r", "i", "z", "y")
    max_val_size: int = 2**20
    snapshot_factor: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "catalog_root", Path(self.catalog_root))
        if not (0.0 < self.val_start < self.test_start < 1.0):
            raise ValueError(
                f"Required: 0 < val_start < test_start < 1, "
                f"got val_start={self.val_start}, test_start={self.test_start}"
            )
        if self.n_src < 1:
            raise ValueError(f"n_src must be >= 1, got {self.n_src}")

    @property
    def train_split(self) -> tuple[float, float]:
        """Hash range for the training set."""
        return (0.0, self.val_start)

    @property
    def val_split(self) -> tuple[float, float]:
        """Hash range for the validation set."""
        return (self.val_start, self.test_start)

    @property
    def test_split(self) -> tuple[float, float]:
        """Hash range for the test set."""
        return (self.test_start, 1.0)


def dp1_config(
    catalog_root: str | Path,
    *,
    n_src: int = 10,
    val_start: float = 0.6,
    test_start: float = 0.85,
    bands: tuple[str, ...] = ("u", "g", "r", "i", "z", "y"),
    max_val_size: int = 2**20,
    snapshot_factor: float = 1.0,
) -> SurveyConfig:
    """SurveyConfig for Rubin Data Preview 1.

    Parameters
    ----------
    catalog_root : str or Path
        Root directory of the DP1 HATS catalogs.
    n_src : int, optional
        Observations to subsample per light curve. Defaults to 10.
    val_start : float, optional
        Train/val boundary. Defaults to 0.6.
    test_start : float, optional
        Val/test boundary. Defaults to 0.85.
    bands : tuple of str, optional
        Survey filter bands. Defaults to all six LSST bands.
    max_val_size : int, optional
        Maximum validation set size. Defaults to 2**20.
    snapshot_factor : float, optional
        Snapshot cadence factor. Defaults to 1.0.

    Returns
    -------
    SurveyConfig
    """
    return SurveyConfig(
        catalog_root=catalog_root,
        n_src=n_src,
        val_start=val_start,
        test_start=test_start,
        bands=bands,
        max_val_size=max_val_size,
        snapshot_factor=snapshot_factor,
    )


def dp2_config(
    catalog_root: str | Path,
    *,
    n_src: int = 50,
    val_start: float = 0.84,
    test_start: float = 0.85,
    bands: tuple[str, ...] = ("u", "g", "r", "i", "z", "y"),
    max_val_size: int = 2**20,
    snapshot_factor: float = 1.0,
) -> SurveyConfig:
    """SurveyConfig for Rubin Data Preview 2.

    Parameters
    ----------
    catalog_root : str or Path
        Root directory of the DP2 HATS catalogs.
    n_src : int, optional
        Observations to subsample per light curve. Defaults to 50.
    val_start : float, optional
        Train/val boundary. Defaults to 0.84.
    test_start : float, optional
        Val/test boundary. Defaults to 0.85.
    bands : tuple of str, optional
        Survey filter bands. Defaults to all six LSST bands.
    max_val_size : int, optional
        Maximum validation set size. Defaults to 2**20.
    snapshot_factor : float, optional
        Snapshot cadence factor. Defaults to 1.0.

    Returns
    -------
    SurveyConfig
    """
    return SurveyConfig(
        catalog_root=catalog_root,
        n_src=n_src,
        val_start=val_start,
        test_start=test_start,
        bands=bands,
        max_val_size=max_val_size,
        snapshot_factor=snapshot_factor,
    )
