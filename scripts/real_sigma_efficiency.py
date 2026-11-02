#!/usr/bin/env python
"""Relative efficiencies of light curve transformations for the real uncertainties.

This is the real-data counterpart of the exponentially distributed uncertainties
example of the paper (Appendix "Statistical efficiency of light curve
transformations"). For each target light curve length N, every light curve
(object x band) with at least N good observations and object magnitude brighter
than --mag-max is subsampled to N random observations. For their reported
uncertainties we compute the exact variances of the mean-square statistic for
the whitening, pair differences, hold-out mean, and standardized residuals,
average the variances over the light curves, and report the relative
efficiencies e = <Var_whiten> / <Var_X>. The hold-out subset is the first n_H
observations of the random subsample, and n_H is chosen to maximize the
efficiency averaged over all light curves.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from uncle_val.datasets.rubin_dp import rubin_dp_catalog_multi_band

KAPPA_BINS = np.linspace(1.0, 2.0, 1001)

_SUMS_META = pd.DataFrame(
    {
        "n": pd.Series(dtype=np.int64),
        "quantity": pd.Series(dtype=str),
        "n_h": pd.Series(dtype=np.int64),
        "value": pd.Series(dtype=np.float64),
    }
)


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--catalog-root", type=Path, required=True, help="Root of the Rubin DP HATS catalogs."
    )
    parser.add_argument("--img", choices=["cal", "diff"], default="cal", help="Image type, default 'cal'.")
    parser.add_argument(
        "--mag-max", type=float, default=21.0, help="Keep objects brighter than this, default 21."
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[10, 30, 100],
        help="Target light curve lengths, default 10 30 100.",
    )
    parser.add_argument(
        "--n-partitions",
        type=int,
        default=None,
        help="Use this many randomly selected catalog partitions instead of the whole catalog.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed, default 0.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path for the summary table.")
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="If given, start a dask.distributed client with that many workers.",
    )
    return parser.parse_args(argv)


def mean_square_variance_decomp(c2: np.ndarray) -> np.ndarray:
    """Variance of the mean square for C_il = +-c_i c_l (i != l), row-wise.

    Parameters
    ----------
    c2 : np.ndarray
        Array of c_i^2 values, shape (n_lc, n).

    Returns
    -------
    np.ndarray
        Variance per light curve, shape (n_lc,).
    """
    n = c2.shape[1]
    return 2 / n**2 * (n + c2.sum(axis=1) ** 2 - (c2**2).sum(axis=1))


def standardized_residuals_variance(sigma: np.ndarray) -> np.ndarray:
    """Variance of the mean square of the standardized residuals, row-wise."""
    w = sigma**-2
    v2 = w / w.sum(axis=1, keepdims=True)
    return mean_square_variance_decomp(v2 / (1 - v2))


def hold_out_variance(sigma: np.ndarray, n_h: int) -> np.ndarray:
    """Variance of the mean square for the first n_h observations held out, row-wise."""
    w_h = (sigma[:, :n_h] ** -2).sum(axis=1, keepdims=True)
    return mean_square_variance_decomp(1 / (1 + sigma[:, n_h:] ** 2 * w_h))


def kappa(sigma: np.ndarray) -> np.ndarray:
    """kappa = N sum v_i^4, row-wise."""
    w = sigma**-2
    return sigma.shape[1] * (w**2).sum(axis=1) / w.sum(axis=1) ** 2


def _partition_sums(df, pixel, *, mag_max, ns, seed):
    """Sums of the exact variances over the light curves of one partition."""
    rng = np.random.default_rng([seed, pixel.order, pixel.pixel])
    df = df[df["object_mag"] < mag_max]
    errs = [np.asarray(err, dtype=np.float64) for err in df["lc"].nest.to_lists()["err"]]
    errs = [err[np.isfinite(err) & (err > 0)] for err in errs]

    rows = []
    for n in ns:
        selected = [err for err in errs if len(err) >= n]
        if not selected:
            continue
        sigma = np.stack([rng.permutation(err)[:n] for err in selected])
        rows.append((n, "n_lc", -1, float(len(selected))))
        rows.append((n, "var_std", -1, float(standardized_residuals_variance(sigma).sum())))
        k = kappa(sigma)
        rows.append((n, "kappa", -1, float(k.sum())))
        hist, _ = np.histogram(np.clip(k, KAPPA_BINS[0], KAPPA_BINS[-1]), bins=KAPPA_BINS)
        rows.extend((n, "kappa_hist", i, float(h)) for i, h in enumerate(hist) if h > 0)
        for n_h in range(1, n - 1):
            rows.append((n, "var_hold", n_h, float(hold_out_variance(sigma, n_h).sum())))
    return pd.DataFrame(rows, columns=_SUMS_META.columns).astype(_SUMS_META.dtypes)


def _hist_quantile(counts: np.ndarray, q: float) -> float:
    cdf = np.cumsum(counts) / counts.sum()
    return float(KAPPA_BINS[1:][np.searchsorted(cdf, q)])


def summarize(sums: pd.DataFrame) -> pd.DataFrame:
    """Turn the per-partition sums into the efficiency table."""
    sums = sums.groupby(["n", "quantity", "n_h"], as_index=False)["value"].sum()
    rows = []
    for n, group in sums.groupby("n"):
        by_q = group.set_index(["quantity", "n_h"])["value"]
        n_lc = by_q[("n_lc", -1)]
        var_whiten = 2 / (n - 1)
        hold = by_q.loc["var_hold"] / n_lc
        best_n_h = int(hold.idxmin())
        hist = np.zeros(len(KAPPA_BINS) - 1)
        kh = by_q.loc["kappa_hist"]
        hist[kh.index.to_numpy()] = kh.to_numpy()
        rows.append(
            {
                "N": n,
                "n_lc": int(n_lc),
                "kappa_mean": by_q[("kappa", -1)] / n_lc,
                "kappa_median": _hist_quantile(hist, 0.5),
                "kappa_99": _hist_quantile(hist, 0.99),
                "e_pairs": var_whiten / (2 / (n // 2)),
                "q_opt": best_n_h / n,
                "e_hold": var_whiten / hold.min(),
                "e_std": var_whiten / (by_q[("var_std", -1)] / n_lc),
            }
        )
    return pd.DataFrame(rows)


def main(argv=None):
    """Compute and print the efficiency table."""
    args = parse_args(argv)

    client = None
    if args.n_workers is not None:
        from distributed import Client

        client = Client(n_workers=args.n_workers, threads_per_worker=1)
        print(f"Dask dashboard: {client.dashboard_link}")

    try:
        catalog = rubin_dp_catalog_multi_band(
            root=args.catalog_root,
            obj="science",
            img=args.img,
            phot="PSF",
            mode="forced",
            ccd_visit_cols=None,
        )
        if args.n_partitions is not None and args.n_partitions < catalog.npartitions:
            pixels = catalog.get_healpix_pixels()
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(pixels), size=args.n_partitions, replace=False)
            catalog = catalog.pixel_search([pixels[i] for i in sorted(idx)])
        sums = catalog.map_partitions(
            _partition_sums,
            include_pixel=True,
            mag_max=args.mag_max,
            ns=sorted(args.n),
            seed=args.seed,
            meta=_SUMS_META,
        ).compute()
        table = summarize(sums)
        # Print before tearing down the cluster: on shared clusters
        # client.close() can hang and raise TimeoutError, which would
        # otherwise discard the already-computed result.
        print(
            f"Relative efficiencies for {args.catalog_root}, img={args.img}, mag < {args.mag_max}, "
            f"partitions: {args.n_partitions or 'all'}, seed: {args.seed}"
        )
        print(table.to_string(index=False))
        if args.output is not None:
            table.to_csv(args.output, index=False)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: ignoring error during Dask client shutdown: {exc!r}")


if __name__ == "__main__":
    main()
