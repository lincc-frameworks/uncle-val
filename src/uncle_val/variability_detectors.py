import inspect
from collections.abc import Callable, Sequence

import numpy as np
from numba import njit


@njit
def detect_mag_amplitude(
    flux,
    err,
    max_mag_err: float | None = None,
    min_amplitude: float | None = None,
) -> bool:
    """Detects variability by large amplitude deviations.

    Returns `True` if there is a pair of detections for which both magnitude
    errors are smaller than `max_mag_err` and the magnitude difference is
    larger than `min_amplitude`.

    Parameters
    ----------
    flux : array, (n_src,)
        Flux array.
    err : array, (n_src,)
        Error array.
    max_mag_err : float, default: 0.05
        Maximum allowed magnitude error.
    min_amplitude : float, default: 1.0
        Minimum allowed magnitude amplitude.

    Returns
    -------
    bool
        `True` if the light curve is variable, `False` otherwise.
    """
    if max_mag_err is None:
        max_mag_err = 0.05
    if min_amplitude is None:
        min_amplitude = 1.0

    pos_flux_idx = flux > 0
    flux, err = flux[pos_flux_idx], err[pos_flux_idx]

    if len(flux) < 2:
        return False

    mag = 31.2 - 2.5 * np.log10(flux)
    mag_err = 2.5 / np.log(10) * err / flux

    small_err = mag_err < max_mag_err
    mag, mag_err = mag[small_err], mag_err[small_err]

    if len(mag) < 2:
        return False

    large_amplitude = np.ptp(mag) >= min_amplitude
    return large_amplitude


@njit
def detect_deviation_outlier(flux, err, n_sigma: float | None = None) -> bool:
    """Detects variability by outlier detection, in deviation space

    First, it converts a light curve to deviations,
    `g = (f - meadian(f)) / err`

    Then, it checks if any n_sigma-outliers are presented, where sigma is
    the standard deviation of `g` (NOT `err`).

    Parameters
    ----------
    flux : array, (n_src,)
        Flux array.
    err : array, (n_src,)
        Error array.
    n_sigma : float, default: 5.0
        Number of standard deviations to use as a threshold.

    Returns
    -------
    bool
        `True` if the light curve is variable, `False` otherwise.
    """
    if n_sigma is None:
        n_sigma = 5.0

    n = len(flux)
    if n < 2:
        return False

    deviations = (flux - np.median(flux)) / err

    # np.std(..., ddof) is not supported by numba
    std = np.std(deviations) * n / (n - 1)

    outliers = np.abs(deviations) > n_sigma * std
    any_outliers = np.any(outliers)
    return any_outliers


@njit
def detect_autocorrelation(
    flux, err=None, norm_threshold: float | None = None, min_median_deviation: float | None = None
) -> bool:
    r"""Detects variability by checking single-step autocorrelation

    We ignore time, the normalized autocorrelation value is
    `|norm l_1| = Nsrc * |\sum[(f_i - median_f)(f_i+1 - median_f)]| / \sum[(f_i - median_f)^2].

    For numerical stability we also check that
    `flux_amplitude / median_flux > min_median_derivation`

    Parameters
    ----------
    flux : array, (n_src,)
        Flux array.
    err : array or None
        Error array, not used.
    norm_threshold : float, default: 5.0
        Minimum allowed normalized autocorrelation value. We normalize by
        1) taking absolute value, 2) multiplying by the total number of
        sources.
    min_median_deviation : float, default: 1e-6
        Minimum ratio of flux amplitude to median flux.

    Returns
    -------
    bool
        `True` if the light curve is variable, `False` otherwise.
    """
    if norm_threshold is None:
        norm_threshold = 5.0
    if min_median_deviation is None:
        min_median_deviation = 1e-6

    n = len(flux)
    if n < 2:
        return False

    median_flux = np.median(flux)

    if np.ptp(flux) / median_flux <= min_median_deviation:
        return False

    numerator = np.sum((flux[:-1] - median_flux) * (flux[1:] - median_flux))
    denominator = np.sum(np.square(flux - median_flux))
    norm_l1 = n * np.abs(numerator) / denominator

    exceeded = norm_l1 > norm_threshold
    return exceeded


def get_combined_variability_detector(
    funcs: Sequence[Callable] | None = None,
) -> Callable[[np.ndarray, np.ndarray], bool]:
    """Multiple variability detectors called subsequently

    Parameters
    ----------
    funcs : sequence of f(flux, err, arg1=None, arg2=None, ...)
        Functions to detect variability. All optional parameters are set to None.
        If `None`, use all functions with default parameters.
    """
    if funcs is None:
        funcs = (detect_mag_amplitude, detect_deviation_outlier, detect_autocorrelation)

    none_args = []
    for func in funcs:
        n_optional_args = len(inspect.signature(func).parameters) - 2
        none_args.append((None,) * n_optional_args)

    def combined_variability_detector(flux: np.ndarray, err: np.ndarray) -> bool:
        return any(func(flux, err, *args) for args, func in zip(none_args, funcs, strict=True))

    return combined_variability_detector
