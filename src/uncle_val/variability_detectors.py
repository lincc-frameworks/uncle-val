import numpy as np
from numba import njit


@njit
def detect_mag_amplitude(
    flux,
    err,
    *,
    max_mag_err: float = 0.05,
    min_amplitude: float = 1.0,
) -> bool:
    """Detects variability by large amplitude deviations.

    Returns `True` if there is a pair of detections for which both  magnitude
    errors are smaller than `max_mag_err` and magnitude difference is larger
    than `min_amplitude`.

    Parameters
    ----------
    flux : array, (n_src,)
        Flux array.
    err : array, (n_src,)
        Error array.
    max_mag_err : float, optional
        Maximum allowed magnitude error.
    min_amplitude : float, optional
        Minimum allowed magnitude amplitude.

    Returns
    -------
    bool
        `True` if light curve is variable, `False` otherwise.
    """
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
def detect_deviation_outlier(flux, err, *, n_sigma: float = 5.0) -> bool:
    """Detects variability by outlier detection, in deviation space

    First, it converts light curve to deviations,
    `g = (f - meadian(f)) / err`

    Then, it checks if any n_sigma-outliers are presented, where sigma is
    the standard deviation of `g`.

    Parameters
    ----------
    flux : array, (n_src,)
        Flux array.
    err : array, (n_src,)
        Error array.
    n_sigma : float, optional
        Number of standard deviations to use as a threshold.

    Returns
    -------
    bool
        `True` if light curve is variable, `False` otherwise.
    """
    if len(flux) < 2:
        return False

    deviations = (flux - np.median(flux)) / err
    std = np.std(deviations, ddof=1)
    outliers = np.abs(deviations) > n_sigma * std
    any_outliers = np.any(outliers)
    return any_outliers


@njit
def detect_autocorrelation(
    flux, err=None, *, norm_threshold: float = 5.0, min_median_deviation: float = 1e-6
) -> bool:
    r"""Detects variability by checking single-step autocorrelation

    We ignore time, the target normalized autocorrelation value is
    `|norm l_1| = Nsrc * |\sum[(f_i - median_f)(f_i+1 - median_f)]| / \sum[(f_i - median_f)^2].

    For numerical stability we also check that
    `flux_amplitude / median_flux > min_median_derivation`

    Parameters
    ----------
    flux : array, (n_src,)
        Flux array.
    err : array or None
        Error array, not used.
    norm_threshold : float, optional
        Minimum allowed normalized autocorrelation value. We normalize by
        1) taking absolute value, 2) multiplying by the total number of
        sources.
    min_median_deviation : float, optional
        Minimum ratio of flux amplitude to median flux.

    Returns
    -------
    bool
        `True` if light curve is variable, `False` otherwise.
    """
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
