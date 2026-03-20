from dataclasses import dataclass


@dataclass(frozen=True)
class SurveyConfig:
    """Train/val/test split configuration defined by two boundary points.

    The hash range [0, 1) is partitioned as:
    - train: [0, val_start)
    - val:   [val_start, test_start)
    - test:  [test_start, 1)

    Parameters
    ----------
    val_start : float
        Boundary between train and val sets. Must be in (0, 1).
    test_start : float
        Boundary between val and test sets. Must be in (val_start, 1).

    Examples
    --------
    >>> cfg = SurveyConfig(val_start=0.6, test_start=0.85)
    >>> cfg.train_split
    (0.0, 0.6)
    >>> cfg.val_split
    (0.6, 0.85)
    >>> cfg.test_split
    (0.85, 1.0)
    """

    val_start: float
    test_start: float

    def __post_init__(self):
        if not (0.0 < self.val_start < self.test_start < 1.0):
            raise ValueError(
                f"Required: 0 < val_start < test_start < 1, "
                f"got val_start={self.val_start}, test_start={self.test_start}"
            )

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


DP1_SURVEY_CONFIG = SurveyConfig(val_start=0.6, test_start=0.85)

DP2_SURVEY_CONFIG = SurveyConfig(val_start=0.6, test_start=0.65)

# Backward-compatible aliases
TRAIN_SPLIT = DP1_SURVEY_CONFIG.train_split
VALIDATION_SPLIT = DP1_SURVEY_CONFIG.val_split
TEST_SPLIT = DP1_SURVEY_CONFIG.test_split
