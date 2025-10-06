import numpy as np
from uncle_val.variability_detectors import get_combined_variability_detector


def test_combined_variability_detector():
    """Test the combined variability detector doesn't fail"""
    flux = np.linspace(1e3, 1e4, 100)
    err = 0.02 * flux

    f = get_combined_variability_detector()

    _ = f(flux, err)
