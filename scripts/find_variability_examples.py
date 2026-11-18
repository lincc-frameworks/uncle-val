#!/usr/bin/env python
"""Find example light curves rejected by each variability criterion.

For each of the four variability detectors used to reject variable objects
(magnitude amplitude, flux outlier, deviation outlier, autocorrelation), find
single-band light curves that are rejected by this detector *only*, after the
same source quality cuts as the training pipeline. Save the examples (with
their full light curves) to a parquet file and plot each of them.
"""

import argparse
import re
from pathlib import Path

import matplotlib
import nested_pandas as npd
import numpy as np
import pandas as pd

from uncle_val.datasets.lsdb_generator import LSDBDataGenerator
from uncle_val.datasets.rubin_dp import rubin_dp_catalog_multi_band
from uncle_val.utils.mag_flux import flux2mag, fluxerr2magerr
from uncle_val.variability_detectors import (
    detect_autocorrelation,
    detect_deviation_outlier,
    detect_flux_outlier,
    detect_mag_amplitude,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

# Detector name -> (function, number of optional arguments to pass as None).
# numba-jitted functions need the optional arguments passed explicitly.
CRITERIA = {
    "mag_amplitude": (detect_mag_amplitude, 2),
    "flux_outlier": (detect_flux_outlier, 1),
    "deviation_outlier": (detect_deviation_outlier, 1),
    "autocorrelation": (detect_autocorrelation, 2),
}
# Exposure midpoint from the CCD visit table, joined to the sources by the catalog loader
TIME_COL = "expMidptMJD"

FLUX_LABELS = {"cal": "Science PSF flux, nJy", "diff": "Difference-image PSF flux, nJy"}
MAG_LABELS = {"cal": "Science PSF magnitude", "diff": "Difference-image PSF magnitude"}


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-root", type=Path, required=True, help="Root of the DP HATS catalogs.")
    parser.add_argument(
        "--img", choices=["cal", "diff"], default="cal", help="Direct or difference photometry."
    )
    parser.add_argument(
        "--min-n-src", type=int, default=20, help="Minimum number of sources per light curve."
    )
    parser.add_argument("--n-examples", type=int, default=10, help="Examples per criterion.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the order of partitions.")
    parser.add_argument(
        "--n-workers", type=int, default=None, help="Number of dask workers, no client if not given."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for parquet and figures.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip the catalog scan and replot from an existing examples.parquet in --output-dir.",
    )
    return parser.parse_args(argv)


def _single_criterion(flux, err) -> str:
    """Name of the only criterion rejecting the light curve, empty string otherwise."""
    fired = [name for name, (func, n_opt) in CRITERIA.items() if func(flux, err, *(None,) * n_opt)]
    return fired[0] if len(fired) == 1 else ""


def _partition_examples(df, *, min_n_src, n_examples):
    """Up to n_examples light curves per criterion rejected by this criterion only."""
    df = df.assign(n_src=df["lc"].nest.list_lengths)
    df = df.query(f"n_src >= {min_n_src}")
    if len(df) == 0:
        return df.assign(criterion=pd.Series(dtype=str))
    df = df.assign(criterion=df.reduce(_single_criterion, "lc.x", "lc.err")[0].to_numpy())
    df = df.query("criterion != ''")
    return df.groupby("criterion").head(n_examples)


def find_examples(catalog, *, client, n_workers, min_n_src, n_examples, seed):
    """Stream partitions until each criterion has n_examples light curves."""
    mapped = catalog.map_partitions(_partition_examples, min_n_src=min_n_src, n_examples=n_examples)
    generator = LSDBDataGenerator(
        catalog=mapped, client=client, partitions_per_chunk=n_workers or 1, seed=seed
    )
    chunks = []
    n_partitions = 0
    for chunk in generator:
        chunks.append(chunk)
        n_partitions += generator.partitions_per_chunk
        counts = pd.concat(chunks)["criterion"].value_counts()
        print(f"{n_partitions} partitions: {counts.to_dict()}")
        if all(counts.get(name, 0) >= n_examples for name in CRITERIA):
            break
    return pd.concat(chunks).groupby("criterion").head(n_examples)


def _sci_tick(value, _pos) -> str:
    """Tick label in scientific notation, e.g. 1.25x10^7; plain for small values."""
    if abs(value) < 1e4:
        return f"${value:g}$"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10**exponent
    return rf"${mantissa:g} \times 10^{{{exponent}}}$"


def _outlier_mask(name: str, flux: np.ndarray, err: np.ndarray) -> np.ndarray:
    """Sources that make the outlier criteria fire, with the detectors' default thresholds."""
    n = len(flux)
    if name == "flux_outlier":
        return np.abs(flux - np.median(flux)) > 100.0 * err
    if name == "deviation_outlier":
        deviations = (flux - np.median(flux)) / err
        # Same normalization as detect_deviation_outlier
        std = np.std(deviations) * n / (n - 1)
        return np.abs(deviations) > 5.0 * std
    return np.zeros(n, dtype=bool)


def designation_prefix(catalog_root: Path) -> str:
    """IAU-registered Rubin Object designation prefix, e.g. LSST-DP2-O for a dp2 catalog root."""
    name = catalog_root.name
    if re.fullmatch(r"dp\d+", name, flags=re.IGNORECASE) is None:
        raise ValueError(
            f"Cannot derive the data preview from the catalog root name {name!r}, expected e.g. 'dp2'"
        )
    return f"LSST-{name.upper()}-O"


def plot_light_curve(ax, lc: pd.DataFrame, *, criterion: str, img: str) -> None:
    """Plot one light curve, highlighting the sources that trigger an outlier criterion."""
    lc = lc.sort_values(TIME_COL)
    flux, err, mjd = (lc[col].to_numpy() for col in ["x", "err", TIME_COL])
    is_mag = criterion == "mag_amplitude"
    if is_mag:
        # The criterion works with magnitudes, defined for positive fluxes only
        positive = flux > 0
        flux, err, mjd = flux[positive], err[positive], mjd[positive]
        y, yerr = flux2mag(flux), fluxerr2magerr(flux=flux, flux_err=err)
    else:
        y, yerr = flux, err
    outliers = _outlier_mask(criterion, flux, err)
    for mask, marker, color, ecolor in [
        (~outliers, "o", "#1f3b73", "#7f8fb0"),
        (outliers, "D", "#d9541e", "#eba383"),
    ]:
        if mask.any():
            ax.errorbar(
                mjd[mask], y[mask], yerr=yerr[mask], fmt=marker, ms=3, lw=0.8, color=color, ecolor=ecolor
            )
    ax.set_xlabel("MJD", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    if is_mag:
        ax.set_ylabel(MAG_LABELS[img], fontsize=8)
        ax.invert_yaxis()
    else:
        ax.axhline(np.median(y), color="0.5", lw=0.8, ls="--")
        ax.set_ylabel(FLUX_LABELS[img], fontsize=8)
        ax.yaxis.set_major_formatter(FuncFormatter(_sci_tick))


def plot_examples(examples, output_dir: Path, *, img: str, designation_prefix: str) -> None:
    """One figure per example light curve, in a subdirectory per criterion."""
    for i in range(len(examples)):
        row = examples.iloc[i]
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        plot_light_curve(ax, row["lc"], criterion=row["criterion"], img=img)
        ax.set_title(f"{designation_prefix}-{row['id']} {row['band']}-band", fontsize=8)
        fig.tight_layout()
        criterion_dir = output_dir / row["criterion"]
        criterion_dir.mkdir(exist_ok=True)
        path = criterion_dir / f"{row['id']}_{row['band']}.pdf"
        fig.savefig(path)
        plt.close(fig)
        print(f"Saved {path}")


def main(argv=None):
    """Find, save, and plot example light curves."""
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output_dir / "examples.parquet"

    if not args.plot_only:
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
                variability_detectors=[],
                ccd_visit_cols=[TIME_COL],
            )
            examples = find_examples(
                catalog,
                client=client,
                n_workers=args.n_workers,
                min_n_src=args.min_n_src,
                n_examples=args.n_examples,
                seed=args.seed,
            )
            examples[["id", "band", "criterion", "n_src", "lc"]].to_parquet(parquet_path)
            print(f"Saved {parquet_path}")
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception as exc:  # noqa: BLE001
                    print(f"Warning: ignoring error during Dask client shutdown: {exc!r}")

    examples = npd.read_parquet(parquet_path)
    print(examples[["criterion", "id", "band", "n_src"]].to_string(index=False))
    plot_examples(
        examples, args.output_dir, img=args.img, designation_prefix=designation_prefix(args.catalog_root)
    )


if __name__ == "__main__":
    main()
