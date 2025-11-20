from typing import Literal

import torch
from torch import Tensor, tensor
from torch.distributions import Distribution


def _auto_bins(
    bins: Tensor | Literal["auto"], *, n_data: int, device: torch.device, half_width=5.0, n_bins=11
) -> Tensor:
    """Produce bins for auto-binning"""
    if isinstance(bins, torch.Tensor):
        return bins
    n_data = tensor(n_data, device=device)
    n_bins = tensor(n_bins, device=device, dtype=torch.int)
    half_width = tensor(half_width, device=device)
    n_bins = torch.maximum(n_bins, torch.asarray(torch.sqrt(n_data), dtype=torch.int))
    return torch.linspace(-half_width, half_width, n_bins, device=device)


def kolmogorov_smirnov(distr: Distribution, vector: Tensor) -> Tensor:
    """Performs one-sample Kolmogorov-Smirnov test

    Parameters
    ----------
    distr : Distribution
        The distribution to test versus
    vector : Tensor
        The data to test

    Returns
    -------
    Tensor
        Test statistic
    """
    n = vector.size(0)
    sorted_vector = vector.sort()[0]
    cdf_at_points = distr.cdf(sorted_vector)

    # Create indices 1/N, 2/N, ..., N/N
    indices = torch.arange(1, n + 1, device=vector.device, dtype=vector.dtype)
    i_over_n = indices / n

    # Create indices 0/N, 1/N, ..., (N-1)/N
    i_minus_1_over_n = (indices - 1) / n

    # D+ statistic: max | ECDF(x_i) - F(x_i) |
    d_plus = (i_over_n - cdf_at_points).abs().max()

    # D- statistic: max | F(x_i) - ECDF(x_{i-1}) |
    d_minus = (cdf_at_points - i_minus_1_over_n).abs().max()

    return torch.maximum(d_plus, d_minus)


def anderson_darling(distr: Distribution, vector: Tensor) -> Tensor:
    """Performa Anderson-Darling test

    Parameters
    ----------
    distr : Distribution
        The distribution to test versus
    vector : Tensor
        The data to test


    Returns
    -------
    Tensor
        Test statistic
    """
    n = vector.size(0)
    sorted_vector = vector.sort()[0]
    cdf_at_points = distr.cdf(sorted_vector)
    indices = torch.arange(1, n + 1, dtype=vector.dtype, device=vector.device)
    two_i_minus_1 = 2 * indices - 1
    log_f_i = torch.log(cdf_at_points)
    log_one_minus_f_n_minus_i_plus_1 = torch.log(1.0 - cdf_at_points.flip(dims=[0]))
    sum_term = (two_i_minus_1 * (log_f_i + log_one_minus_f_n_minus_i_plus_1)).sum()
    ad_statistic = -n - (1.0 / n) * sum_term
    return ad_statistic


def g_test(distr: Distribution, vector: Tensor, *, bins: Tensor | int = "auto") -> Tensor:
    """Performs G-test

    https://en.wikipedia.org/wiki/G-test

    Parameters
    ----------
    distr : Distribution
        The distribution to test versus
    vector : Tensor
        The data to test
    bins : int or 'auto'
        Edges of the bins to use, or 'auto' to use
        linspace(-5.0, 5.0, max(11, sqrt(n))), where n is the number of
        elements in vector.

    Returns
    -------
    Tensor
        Test statistic
    """
    n = vector.size(0)
    bins = _auto_bins(bins, n_data=n, device=vector.device)

    observed = torch.histogram(vector, bins)
    cdf_at_edges = distr.cdf(bins)
    expected = n * torch.diff(cdf_at_edges)

    return 2.0 * torch.xlogy(observed, observed / expected)


def extended_jarque_bera(vector: Tensor) -> Tensor:
    """Extended Jarque-Bera test of standard normality

    https://en.wikipedia.org/wiki/Jarque–Bera_test
    https://arxiv.org/abs/2511.08544v3 Section 4.2.1

    Parameters
    ----------
    vector : Tensor
        The data to test

    Returns
    -------
    Tensor
        Test statistic
    """
    n = vector.size(0)

    mu = torch.mean(vector)
    sigma = torch.std(vector, unbiased=False)
    skew = torch.mean(torch.power(vector - mu, 3)) / torch.pow(sigma, 3)
    kurtosis = torch.mean(torch.pow(vector - mu, 4)) / torch.pow(sigma, 4)

    jb_stat = n / 6.0 * (torch.square(skew) + torch.square(0.5 * (kurtosis - 3.0)))
    ejb_stat = n * torch.square(mu / sigma) + 0.5 * (n - 1.0) * torch.square(sigma - 1.0) + jb_stat
    return ejb_stat


def epps_pulley_standard_norm(vector: Tensor, *, bins: Tensor | Literal["auto"] = "auto") -> Tensor:
    """Epps-Pulley test statistic for standard normal distribution

    https://doi.org/10.1093/biomet/70.3.723
    https://arxiv.org/abs/2511.08544v3 Section 4.2.3

    Parameters
    ----------
    vector : Tensor
        The data to test
    bins : int or 'auto'
        Edges of the bins to use, or 'auto' to use
        linspace(-5.0, 5.0, max(11, sqrt(n))), where n is the number of
        elements in vector.

    Returns
    -------
    Tensor
        Test statistic
    """
    n = vector.size(0)
    # Frequency parameter
    t = _auto_bins(bins, n_data=n, device=vector.device)

    # Empirical characteristics function
    ecf = torch.mean(torch.exp(1j * vector[:, None] * t), dim=0)
    # We are also going to use it as an integration weight function, as in Algo 1. of arXiv:2511.08544
    standard_norm_cf = torch.exp(-0.5 * torch.square(t))

    diff_cf = ecf - standard_norm_cf
    diff_cf_abs_sq = torch.square(diff_cf.real) + torch.square(diff_cf.imag)
    integrand = diff_cf_abs_sq * standard_norm_cf
    integral = torch.trapz(integrand, x=t)
    return integral
