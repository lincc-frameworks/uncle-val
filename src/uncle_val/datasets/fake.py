from collections.abc import Callable

import numpy as np
from nested_pandas import NestedFrame
from numpy._typing import NDArray
from scipy import stats


def _fake_non_variable_lcs(
    *,
    n_src: NDArray[int],
    err: Callable[[float, np.random.Generator], float],
    u: Callable[[float, float], float],
    rng: np.random.Generator,
) -> NestedFrame:
    nf = NestedFrame({"n_src": n_src})

    def single_lc(n):
        true_mag = rng.uniform(low=14, high=24, size=1)
        true_flux = 10 ** (-0.4 * (true_mag - 31.4))
        reported_err = err(np.full(n, true_flux), rng)
        real_err = u(true_flux, reported_err) * reported_err
        flux = rng.normal(loc=true_flux, scale=real_err)
        return {
            "lc.x": flux,
            "lc.err": reported_err,
        }

    nf = nf.reduce(
        single_lc,
        "n_src",
        append_columns=True,
    )
    nf["id"] = np.arange(len(nf))

    return nf


def fake_non_variable_lcs(
    *,
    n_batches: int = 1,
    n_obj: int = 1000,
    n_src: int | NDArray[int] = 100,
    err: None | Callable[[float, np.random.Generator], float] = None,
    u: float | Callable[[float, float], float] = 1.0,
    rng: np.random.Generator,
):
    """Iterates over `NestedFrame`s of non-variable light curves

    Parameters
    ----------
    n_batches : int
        Number of batches to generate
    n_obj : int or None
        Number of light curves per batch.
    n_src : int or numpy.ndarray of int
        Number of observations per light curve to generate, either a single
        value or an array of values of `(n_obj,)` shape.
    err : None | udf(flux, rng) -> numpy.ndarray of float
        "Reported" uncertainties, e.g. underestimated ones. If None,
        `scipy.stats.uniform(low=0.5, high=2.0)` will be used.
        Otherwise, the user-provided function will be called with
        flux array and numpy random generator object. Default is
        `flux * scipy.stats.loguniform.rvs(low=0.01, high=0.1, size=len(flux), random_state=rng)`.
    u : float or udf(flux, err, rng) -> numpy.ndarray of float
        Uncertainty underestimation value, e.g. true_err / reported_err.
        Either a constant or a user-provided function of true flux,
        reported error and numpy random generator object.
    rng : numpy.random.Generator or None
        Random number generator used to generate light curves. If None,
        `np.random.default_rng()` will be used.
    """
    if np.ndim(n_src) == 0:
        n_src = np.full(n_obj, n_src, dtype=int)

    if err is None:
        rel_err_distr = stats.loguniform(a=0.01, b=0.1)

        def err_func(flux, rng_):
            return flux * rel_err_distr.rvs(size=len(flux), random_state=rng_)
    else:
        err_func = err

    if isinstance(u, float):

        def u_func(flux, _err):
            return np.full_like(flux, u)
    else:
        u_func = u

    rng = np.random.default_rng(rng)

    for _ in range(n_batches):
        yield _fake_non_variable_lcs(n_src=n_src, err=err_func, u=u_func, rng=rng)
