from __future__ import annotations

import dataclasses
import json
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

    Examples
    --------
    >>> cfg = dp1_config("/path/to/dp1", n_src=10)
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

    def __post_init__(self):
        object.__setattr__(self, "catalog_root", Path(self.catalog_root))
        if not (0.0 < self.val_start < self.test_start < 1.0):
            raise ValueError(
                f"Required: 0 < val_start < test_start < 1, "
                f"got val_start={self.val_start}, test_start={self.test_start}"
            )
        if self.n_src < 1:
            raise ValueError(f"n_src must be >= 1, got {self.n_src}")

    def to_json(self, path: str | Path) -> None:
        """Serialize to a JSON file.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        Path(path).write_text(json.dumps(dataclasses.asdict(self), default=str, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> SurveyConfig:
        """Deserialize from a JSON file produced by :meth:`to_json`.

        Parameters
        ----------
        path : str or Path
            Source file path.

        Returns
        -------
        SurveyConfig
        """
        d = json.loads(Path(path).read_text())
        d["bands"] = tuple(d["bands"])
        return cls(**d)

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
    n_src: int,
    val_start: float = 0.6,
    test_start: float = 0.85,
    bands: tuple[str, ...] = ("u", "g", "r", "i", "z", "y"),
) -> SurveyConfig:
    """SurveyConfig for Rubin Data Preview 1.

    Parameters
    ----------
    catalog_root : str or Path
        Root directory of the DP1 HATS catalogs.
    n_src : int
        Observations to subsample per light curve.
    val_start : float, optional
        Train/val boundary. Defaults to 0.6.
    test_start : float, optional
        Val/test boundary. Defaults to 0.85.
    bands : tuple of str, optional
        Survey filter bands. Defaults to all six LSST bands.

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
    )


def dp2_config(
    catalog_root: str | Path,
    *,
    n_src: int,
    val_start: float = 0.84,
    test_start: float = 0.85,
    bands: tuple[str, ...] = ("u", "g", "r", "i", "z", "y"),
) -> SurveyConfig:
    """SurveyConfig for Rubin Data Preview 2.

    Parameters
    ----------
    catalog_root : str or Path
        Root directory of the DP2 HATS catalogs.
    n_src : int
        Observations to subsample per light curve.
    val_start : float, optional
        Train/val boundary. Defaults to 0.7.
    test_start : float, optional
        Val/test boundary. Defaults to 0.85.
    bands : tuple of str, optional
        Survey filter bands. Defaults to all six LSST bands.

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
    )
