from collections.abc import Sequence
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from nested_pandas import NestedDtype, NestedFrame
from scipy.stats import chi2 as chi2_dist
from scipy.stats import norm

from uncle_val.datasets.rubin_dp import rubin_dp_catalog_multi_band
from uncle_val.learning.models import BaseUncleModel
from uncle_val.pipelines.splits import SurveyConfig
from uncle_val.pipelines.training_config import ComputeConfig
from uncle_val.utils.dask_client import Client
from uncle_val.utils.hashing import uniform_hash
from uncle_val.whitening import whiten_data

numba_whiten_data = partial(numba.njit(whiten_data), np=None)


def _whiten_flux_err(flux, err):
    return {"whiten.z": numba_whiten_data(flux.astype(float), err.astype(float))}


def _samples_in_bins(x, bins, sample_count, rng):
    bin_idx = np.digitize(x, bins)
    samples = []
    for i in range(len(bins) - 1):
        n_in_bin = np.count_nonzero(bin_idx == i)
        n_sample = min(sample_count, n_in_bin)
        idx = rng.choice(n_in_bin, n_sample, replace=False)
        samples.append(idx)
    return samples


def _build_hist_and_samples(monochrome, *, band, mag_bins, value_bins, value_column, n_samples, rng):
    nested_column, sub_column = value_column.split(".", maxsplit=1)

    mag_idx_grid, value_idx_grid = np.indices((len(mag_bins) - 1, len(value_bins) - 1))

    values_mag = monochrome[["object_mag", nested_column]].explode(nested_column)
    hist, _mag_bins, _value_bins = np.histogram2d(
        values_mag["object_mag"], values_mag[sub_column], bins=[mag_bins, value_bins]
    )

    flat_df = pd.DataFrame(
        {
            "band": band,
            "mag_bin_idx": mag_idx_grid.flatten(),
            "value_bin_idx": value_idx_grid.flatten(),
            "count": hist.flatten().astype(np.int64),
        }
    )
    nf = NestedFrame.from_flat(
        flat_df,
        name="hist",
        base_columns=["band"],
        nested_columns=["value_bin_idx", "count"],
        on="mag_bin_idx",
    ).reset_index(drop=False)

    sample_idx = _samples_in_bins(
        x=values_mag["object_mag"],
        bins=mag_bins,
        sample_count=n_samples,
        rng=rng,
    )
    nf[sub_column] = pd.Series(
        [np.asarray(values_mag[sub_column].iloc[idx]) for idx in sample_idx],
        dtype=pd.ArrowDtype(pa.list_(pa.float64())),
    )
    nf["object_mag"] = pd.Series(
        [np.asarray(values_mag["object_mag"].iloc[idx]) for idx in sample_idx],
        dtype=pd.ArrowDtype(pa.list_(pa.float64())),
    )
    nf = nf.nest_lists([sub_column, "object_mag"], name="samples")

    return nf


def _extract_hists_and_samples(
    df,
    pixel,
    *,
    hash_range,
    z_bins,
    mag_bins,
    add_mag_err_bins,
    bands,
    min_n_src,
    n_samples,
    non_extended_only,
    model_path,
    model_columns,
    device,
):
    rng = np.random.default_rng((pixel.order, pixel.pixel))

    df = df[df["lc"].nest.list_lengths >= min_n_src]

    if hash_range is not None:
        hashes = uniform_hash(df["id"])
        df = df[(hashes >= hash_range[0]) & (hashes < hash_range[1])]

    if non_extended_only:
        df = df.query("extendedness == 0.0")

    if len(df) == 0:
        return pd.DataFrame(
            {
                "mag_bin_idx": np.array([], dtype=int),
                "band": np.array([], dtype=str),
                "z_hist": NestedDtype(
                    pa.struct({"value_bin_idx": pa.list_(pa.int64()), "count": pa.list_(pa.int64())})
                ),
                "z_samples": NestedDtype(
                    pa.struct({"z": pa.list_(pa.float64()), "object_mag": pa.list_(pa.float64())})
                ),
                "add_mag_err_hist": NestedDtype(
                    pa.struct({"value_bin_idx": pa.list_(pa.int64()), "count": pa.list_(pa.int64())})
                ),
                "add_mag_err_samples": NestedDtype(
                    pa.struct({"add_mag_err": pa.list_(pa.float64()), "object_mag": pa.list_(pa.float64())})
                ),
            }
        )

    if model_path is not None:
        model = torch.load(model_path, weights_only=False).to(device)
        model.eval()

        def apply_model(x, err, *extras):
            inputs = np.stack(np.broadcast_arrays(x, err, *extras), axis=-1, dtype=np.float32)
            outputs = model(torch.tensor(inputs, device=device)).cpu().detach().numpy()

            uu = outputs[..., 0].flatten()
            corrected_err = uu * err
            if model.outputs_s:
                sf = outputs[..., 1].flatten()
                corrected_x = x + sf * corrected_err
            else:
                corrected_x = x

            orig_mag_err = 2.5 / np.log(10.0) * err / x
            corrected_mag_err = 2.5 / np.log(10.0) * corrected_err / corrected_x
            addition_mag_err = np.sqrt(np.maximum(corrected_mag_err**2 - orig_mag_err**2, 0))

            return {
                "corrected_lc.x": corrected_x,
                "corrected_lc.err": corrected_err,
                "corrected_lc.add_mag_err": addition_mag_err,
            }

        df["lc"] = df.reduce(apply_model, *model_columns)["corrected_lc"]
    else:
        df["lc.add_mag_err"] = 0.0

    whiten = df.reduce(
        _whiten_flux_err,
        "lc.x",
        "lc.err",
        append_columns=True,
    ).reset_index(
        drop=True,
    )

    result_nfs = []
    for band in bands:
        monochrome = whiten.query(f"band == {band!r}")
        z_nf = _build_hist_and_samples(
            monochrome,
            band=band,
            mag_bins=mag_bins,
            value_bins=z_bins,
            value_column="whiten.z",
            n_samples=n_samples,
            rng=rng,
        )
        addmagerr_nf = _build_hist_and_samples(
            monochrome,
            band=band,
            mag_bins=mag_bins,
            value_bins=add_mag_err_bins,
            value_column="lc.add_mag_err",
            n_samples=n_samples,
            rng=rng,
        )
        nf = z_nf.rename(columns={"hist": "z_hist", "samples": "z_samples"})
        nf["add_mag_err_hist"] = addmagerr_nf["hist"]
        nf["add_mag_err_samples"] = addmagerr_nf["samples"]
        result_nfs.append(nf)
    return pd.concat(result_nfs, ignore_index=True)


def _get_hists(
    rubin_dp_root: str | Path,
    *,
    hash_range: tuple[float, float] | None = None,
    bands: Sequence[str],
    obj: str,
    img: str,
    phot: str,
    mode: str,
    min_n_src: int,
    non_extended_only: bool,
    n_workers: int,
    model_path: str | Path | None,
    model_columns: Sequence[str],
    device: torch.device | str = "cpu",
    mag_bins: np.ndarray,
    z_bins: np.ndarray,
    add_mag_err_bins: np.ndarray,
    n_samples: int,
    subsample_partitions: float | None = None,
):
    catalog = rubin_dp_catalog_multi_band(
        rubin_dp_root,
        bands=bands,
        obj=obj,
        img=img,
        phot=phot,
        mode=mode,
    )
    if subsample_partitions is not None:
        n_partitions = max(1, int(round(catalog.npartitions * subsample_partitions)))
        rng = np.random.default_rng(0)
        partitions = rng.choice(catalog.npartitions, n_partitions, replace=True)
        catalog = catalog.partitions[partitions]

    hists = catalog.map_partitions(
        _extract_hists_and_samples,
        include_pixel=True,
        hash_range=hash_range,
        z_bins=z_bins,
        mag_bins=mag_bins,
        add_mag_err_bins=add_mag_err_bins,
        bands=bands,
        min_n_src=min_n_src,
        n_samples=n_samples,
        non_extended_only=non_extended_only,
        model_path=model_path,
        model_columns=model_columns,
        device=torch.device(device),
    )

    with Client(n_workers=n_workers, memory_limit="64GB") as client:
        print(f"Dask Dashboard Link: {client.dashboard_link}")
        hists_df = hists.compute()

    return hists_df


def _aggregate_hists(df, band, *, hist_column: str, value_centers: np.ndarray, value_width: float):
    df = (
        df.query(f"band == {band!r}")
        .explode(hist_column)
        .groupby(["value_bin_idx"])["count"]
        .sum()
        .sort_index()
        .reset_index(drop=False)
    )
    df["value_centers"] = value_centers
    df["prob"] = df["count"] / df["count"].sum()
    df["prob_dens"] = df["prob"] / value_width

    mean = np.sum(df["prob"] * df["value_centers"])
    std = np.sqrt(np.sum(df["prob"] * (df["value_centers"] - mean) ** 2))

    return df, mean, std


def _plot_hist(
    df, ax, band, *, if_model: bool, z_centers: np.ndarray, z_width: float, z_bins: np.ndarray, mag_bin
):
    df, mean, std = _aggregate_hists(
        df, band, hist_column="z_hist", value_centers=z_centers, value_width=z_width
    )

    title = f"Whiten Signal, {band}-band, μ={mean:.4f} σ={std:.4f}"
    if mag_bin is not None:
        title = f"{title}; for mag=[{mag_bin[0]}; {mag_bin[1]}]"
    ax.set_title(title)
    label = "corrected data" if if_model else "data"
    ax.bar(x=df["value_centers"], height=df["prob_dens"], width=z_width, label=label, color="blue", alpha=0.2)
    ax.plot(z_bins, norm(loc=mean, scale=std).pdf(z_bins), label="Normal PDF fit", color="blue", alpha=1.0)
    ax.plot(
        z_bins, norm(loc=0, scale=1).pdf(z_bins), label="Standard normal PDF", ls="--", color="red", alpha=0.6
    )
    ax.legend()


def _plot_magn_vs_uu(
    df,
    ax,
    band,
    *,
    mag_bins: np.ndarray,
    z_centers: np.ndarray,
    z_width: float,
    z_bins: np.ndarray,
    mag_centers: np.ndarray,
):
    means = []
    uu = []
    for mag_bin_idx in range(len(mag_bins) - 1):
        _df, mean, std = _aggregate_hists(
            df.query(f"mag_bin_idx == {mag_bin_idx}"),
            band,
            hist_column="z_hist",
            value_centers=z_centers,
            value_width=z_width,
        )
        means.append(mean)
        uu.append(std)
    means = np.array(means)
    uu = np.array(uu)

    mag_, means, uu = mag_centers[uu > 0], means[uu > 0], uu[uu > 0]

    samples_mag = df.query(f"band == {band!r}")["z_samples.object_mag"]
    samples_z = df.query(f"band == {band!r}")["z_samples.z"]

    ax.set_title(f"Whiten Sources, {band}-band")
    ax.scatter(samples_mag, samples_z, color="grey", marker=".", s=1, alpha=0.5, label="Samples")
    ax.hlines([1.0, -1.0], *ax.get_xlim(), color="red", ls="--", alpha=0.3, label="Perfect mean ± std")
    ax.plot(mag_, means + uu, color="blue", label="mean ± std (std is uncertainty underestimation)")
    ax.plot(mag_, means - uu, color="blue")
    ax.set_xlabel("Object PSF Magnitude")
    ax.set_ylabel("Whiten Signal")
    ax.set_ylim([z_bins[0], z_bins[-1]])
    ax.set_xlim([mag_bins[0], mag_bins[-1]])
    ax.legend()


def _plot_magn_vs_add_mag_err(
    df,
    ax,
    band,
    *,
    mag_bins: np.ndarray,
    add_mag_err_centers: np.ndarray,
    add_mag_err_width: float,
    add_mag_err_bins: np.ndarray,
    mag_centers: np.ndarray,
):
    means = []
    for mag_bin_idx in range(len(mag_bins) - 1):
        _df, mean, _std = _aggregate_hists(
            df.query(f"mag_bin_idx == {mag_bin_idx}"),
            band,
            hist_column="add_mag_err_hist",
            value_centers=add_mag_err_centers,
            value_width=add_mag_err_width,
        )
        means.append(mean)
    means = np.array(means)

    samples_mag = df.query(f"band == {band!r}")["add_mag_err_samples.object_mag"]
    samples_add_mag_err = df.query(f"band == {band!r}")["add_mag_err_samples.add_mag_err"]

    ax.set_title(f"Addition Mag Err, {band}-band")
    ax.scatter(samples_mag, samples_add_mag_err, color="grey", marker=".", s=1, alpha=0.5, label="Samples")
    ax.plot(mag_centers, means, color="blue", label="mean")
    ax.set_xlabel("Object PSF Magnitude")
    ax.set_ylabel("Addition Mag Err")
    ax.set_ylim([add_mag_err_bins[0], add_mag_err_bins[-1]])
    ax.set_xlim([mag_bins[0], mag_bins[-1]])
    ax.legend()


def _default_result_bins() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Default ``(z_bins, mag_bins, add_mag_err_bins)`` edges for the result plots."""
    z_bins = np.r_[-10:10:1001j]
    mag_bins = np.arange(13, 27, 0.5)
    add_mag_err_bins = np.r_[0:0.5:501j]
    return z_bins, mag_bins, add_mag_err_bins


def plot_whiten_distributions(
    hists_pre,
    hists_post,
    *,
    bands: Sequence[str] = "ugrizy",
    z_bins: np.ndarray,
    mag_bins: np.ndarray,
    mag_slice: float | None = None,
    xlim: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
):
    """Compact per-band figure of the whitened signal distribution.

    One panel per band overlays the uncorrected ("pre") and model-corrected
    ("post") whitened signal histograms against the standard normal, plus,
    optionally, the same pair restricted to a single object-magnitude bin.
    The fitted standard deviation (the uncertainty underestimation factor)
    is annotated per band.

    Parameters
    ----------
    hists_pre, hists_post : pd.DataFrame
        Histogram tables as returned by ``_get_hists`` for the uncorrected
        data (``model_path=None``) and the model-corrected data, respectively.
        Both must share the same ``z_bins`` and ``mag_bins``.
    bands : sequence of str
        Bands to plot, one panel each.
    z_bins : np.ndarray
        Bin edges of the whitened signal used to build ``hists_*``.
    mag_bins : np.ndarray
        Object-magnitude bin edges used to build ``hists_*``.
    mag_slice : float or None
        If given, also overlay the pre/post distributions for the magnitude
        bin containing this value.
    xlim : (float, float) or None
        Horizontal limits of every panel; defaults to the ``z_bins`` extent,
        matching the other result plots.
    output_path : path or None
        If given, save the figure there; otherwise return it.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Colorblind-safe categorical colors from matplotlib's built-in
    # "tableau-colorblind10" cycle, selected by index.
    cb = plt.style.library["tableau-colorblind10"]["axes.prop_cycle"].by_key()["color"]
    c_pre, c_post, c_pre_mag, c_post_mag, c_norm = cb[2], cb[0], cb[1], cb[4], "black"

    z_centers = 0.5 * (z_bins[1:] + z_bins[:-1])
    z_width = z_bins[1] - z_bins[0]
    slice_style = (0, (1, 1))
    if xlim is None:
        xlim = (z_bins[0], z_bins[-1])

    mag_slice_idx = None if mag_slice is None else int(np.searchsorted(mag_bins, mag_slice))

    def aggregate(hists, band):
        df, _mean, std = _aggregate_hists(
            hists, band, hist_column="z_hist", value_centers=z_centers, value_width=z_width
        )
        return df["prob_dens"].to_numpy(), std

    def panel(ax, band):
        pre, std_pre = aggregate(hists_pre, band)
        post, std_post = aggregate(hists_post, band)
        ax.stairs(pre, z_bins, fill=True, color=c_pre, alpha=0.45)
        ax.stairs(post, z_bins, color=c_post, lw=1.8)
        annotations = [rf"all: $\sigma$ {std_pre:.2f}$\to${std_post:.2f}"]
        if mag_slice_idx is not None:
            pre_mag, std_pre_mag = aggregate(hists_pre.query(f"mag_bin_idx == {mag_slice_idx}"), band)
            post_mag, std_post_mag = aggregate(hists_post.query(f"mag_bin_idx == {mag_slice_idx}"), band)
            ax.stairs(pre_mag, z_bins, color=c_pre_mag, lw=1.3, ls=slice_style)
            ax.stairs(post_mag, z_bins, color=c_post_mag, lw=1.3, ls=slice_style)
            annotations.append(rf"$m\approx${mag_slice:g}: $\sigma$ {std_pre_mag:.2f}$\to${std_post_mag:.2f}")
        ax.plot(z_centers, norm(loc=0, scale=1).pdf(z_centers), color=c_norm, ls="--", lw=1.1)
        ax.text(0.03, 0.96, f"${band}$", transform=ax.transAxes, va="top", fontsize=13, fontweight="bold")
        ax.text(
            0.97,
            0.96,
            "\n".join(annotations),
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=7.5,
            linespacing=1.3,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(bottom=0.0)

    ncols = 3
    nrows = int(np.ceil(len(bands) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, band in zip(axes, bands, strict=False):
        panel(ax, band)
    for ax in axes[len(bands) :]:
        ax.set_visible(False)
    for ax in axes[len(bands) - ncols : len(bands)]:
        ax.set_xlabel("whitened signal $z$")
    for ax in axes[::ncols]:
        ax.set_ylabel("probability density")

    handles = [
        Patch(facecolor=c_pre, alpha=0.45, label="uncorrected, all mag"),
        Line2D([], [], color=c_post, lw=1.8, label="corrected, all mag"),
    ]
    if mag_slice_idx is not None:
        mag_label = rf"$m\approx${mag_slice:g}"
        handles += [
            Line2D([], [], color=c_pre_mag, lw=1.3, ls=slice_style, label=f"uncorrected, {mag_label}"),
            Line2D([], [], color=c_post_mag, lw=1.3, ls=slice_style, label=f"corrected, {mag_label}"),
        ]
    handles.append(Line2D([], [], color=c_norm, lw=1.1, ls="--", label=r"$N(0,1)$"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def make_whiten_distribution_plot(
    *,
    survey_config: SurveyConfig,
    model_path: str | Path | BaseUncleModel,
    model_columns: Sequence[str] = ("lc.x", "lc.err"),
    compute_config: ComputeConfig,
    split: str | None = None,
    min_n_src: int | None = None,
    non_extended_only: bool = False,
    subsample_partitions: float | None = None,
    mag_slice: float | None = 21.0,
    bands: Sequence[str] = "ugrizy",
    output_path: str | Path | None = None,
):
    """Compute uncorrected/corrected whitened signal and plot them together.

    Runs the whitening pipeline twice for the same data and partitions, once
    without a model (uncorrected) and once with ``model_path`` (corrected),
    and renders the compact per-band figure of :func:`plot_whiten_distributions`.

    Parameters mirror :func:`make_plots`; ``mag_slice`` selects the object
    magnitude whose per-bin distribution is overlaid (``None`` to omit it).

    Returns
    -------
    matplotlib.figure.Figure
    """
    z_bins, mag_bins, add_mag_err_bins = _default_result_bins()
    hash_range = survey_config.hash_range(split)
    if min_n_src is None:
        min_n_src = survey_config.n_src

    common = dict(
        hash_range=hash_range,
        bands=bands,
        obj=survey_config.obj,
        img=survey_config.img,
        phot=survey_config.phot,
        mode=survey_config.mode,
        min_n_src=min_n_src,
        non_extended_only=non_extended_only,
        n_workers=compute_config.n_workers,
        model_columns=model_columns,
        device=compute_config.device,
        mag_bins=mag_bins,
        z_bins=z_bins,
        add_mag_err_bins=add_mag_err_bins,
        n_samples=1,
        subsample_partitions=subsample_partitions,
    )
    hists_pre = _get_hists(survey_config.catalog_root, model_path=None, **common)
    hists_post = _get_hists(survey_config.catalog_root, model_path=model_path, **common)

    return plot_whiten_distributions(
        hists_pre,
        hists_post,
        bands=bands,
        z_bins=z_bins,
        mag_bins=mag_bins,
        mag_slice=mag_slice,
        output_path=output_path,
    )


def _column_peak_density(hists, band, *, hist_column, value_bins, mag_bins):
    """Per-magnitude-column density and moments of a binned quantity.

    Aggregates the ``hist_column`` histogram of ``hists`` for ``band`` into a
    2D array ``[n_value, n_mag]`` where each object-magnitude column is
    normalized to its own peak, so the result encodes the *shape* of the
    conditional distribution independent of how many objects fall at each
    magnitude. Also returns the per-column ``mean`` and ``std`` of the value.

    Returns
    -------
    (numpy.ma.MaskedArray, np.ndarray, np.ndarray)
        The masked density ``[n_value, n_mag]`` (empty columns masked), and the
        per-magnitude ``mean`` and ``std`` (``NaN`` where undefined).
    """
    value_centers = 0.5 * (value_bins[1:] + value_bins[:-1])
    value_width = value_bins[1] - value_bins[0]
    n_mag = len(mag_bins) - 1
    dens = np.full((len(value_centers), n_mag), np.nan)
    mean = np.full(n_mag, np.nan)
    std = np.full(n_mag, np.nan)
    for mag_bin_idx in range(n_mag):
        df, mu, sigma = _aggregate_hists(
            hists.query(f"mag_bin_idx == {mag_bin_idx}"),
            band,
            hist_column=hist_column,
            value_centers=value_centers,
            value_width=value_width,
        )
        col = df["prob_dens"].to_numpy()
        if col.max() > 0:
            dens[:, mag_bin_idx] = col / col.max()
            mean[mag_bin_idx] = mu
            if sigma > 0:
                std[mag_bin_idx] = sigma
    return np.ma.masked_invalid(dens), mean, std


def plot_whiten_density(
    hists_pre,
    hists_post,
    *,
    bands: Sequence[str] = "ugrizy",
    z_bins: np.ndarray,
    mag_bins: np.ndarray,
    ylim: float = 8.0,
    xlim: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
):
    """Twin per-band heatmaps of the whitened-signal distribution vs magnitude.

    The top row shows the uncorrected ("pre") whitened signal ``z`` and the
    bottom row the model-corrected ("post") one, one column per band. Each
    panel is the 2D histogram of ``z`` against object magnitude with every
    magnitude column normalized to its own peak, so the color encodes the
    *shape and width* of the conditional distribution ``P(z | mag)``. A
    perfectly calibrated survey is a unit-width band between the ``±1``
    references; a wider band (``σ > 1``, top row) means the reported
    uncertainties are underestimated, and a good correction (bottom row) pulls
    the ``mean ± σ`` curves back onto ``±1``.

    Parameters
    ----------
    hists_pre, hists_post : pd.DataFrame
        Histogram tables as returned by ``_get_hists`` for the uncorrected
        data (``model_path=None``) and the model-corrected data, respectively.
        Both must share the same ``z_bins`` and ``mag_bins``.
    bands : sequence of str
        Bands to plot, one column each.
    z_bins : np.ndarray
        Bin edges of the whitened signal used to build ``hists_*``.
    mag_bins : np.ndarray
        Object-magnitude bin edges used to build ``hists_*``.
    ylim : float
        Vertical half-range of every panel, which spans ``[-ylim, ylim]``.
    xlim : (float, float) or None
        Object-magnitude limits of every panel; defaults to the ``mag_bins`` extent.
    output_path : path or None
        If given, save the figure there; otherwise return it.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Colorblind-safe warm line for the mean +/- sigma overlay on the blue map.
    c_line, c_ref = "#d95f02", "0.35"
    mag_centers = 0.5 * (mag_bins[1:] + mag_bins[:-1])

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    rows = (("uncorrected", hists_pre), ("corrected", hists_post))

    n_cols = len(bands)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(1.7 * n_cols + 0.6, 5.0),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    mesh = None
    for i, (row_label, hists) in enumerate(rows):
        for j, band in enumerate(bands):
            ax = axes[i, j]
            dens, mean, std = _column_peak_density(
                hists, band, hist_column="z_hist", value_bins=z_bins, mag_bins=mag_bins
            )
            mesh = ax.pcolormesh(
                mag_bins, z_bins, dens, cmap=cmap, vmin=0, vmax=1, shading="flat", rasterized=True
            )
            ax.plot(mag_centers, mean, color=c_line, lw=1.3)
            ax.plot(mag_centers, mean + std, color=c_line, lw=1.1, ls=(0, (4, 2)))
            ax.plot(mag_centers, mean - std, color=c_line, lw=1.1, ls=(0, (4, 2)))
            ax.axhline(1.0, color=c_ref, ls="--", lw=0.7)
            ax.axhline(-1.0, color=c_ref, ls="--", lw=0.7)
            ax.set_xlim(*(xlim if xlim is not None else (mag_bins[0], mag_bins[-1])))
            ax.set_ylim(-ylim, ylim)
            if i == 0:
                ax.set_title(f"${band}$", fontsize=13)
            else:
                ax.set_xlabel("object magnitude")
        axes[i, 0].set_ylabel(f"{row_label}\nwhitened signal $z$")

    handles = [
        Line2D([], [], color=c_line, lw=1.3, label="mean"),
        Line2D([], [], color=c_line, lw=1.1, ls=(0, (4, 2)), label=r"mean $\pm\,\sigma$"),
        Line2D([], [], color=c_ref, lw=0.7, ls="--", label=r"$\pm 1$ (calibrated)"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, fontsize=8.5)
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist())
    cbar.set_label(r"$P(z\,|\,\mathrm{mag})$, per-magnitude peak $=1$")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    return fig


def make_whiten_density_plot(
    *,
    survey_config: SurveyConfig,
    model_path: str | Path | BaseUncleModel,
    model_columns: Sequence[str] = ("lc.x", "lc.err"),
    compute_config: ComputeConfig,
    split: str | None = None,
    min_n_src: int | None = None,
    non_extended_only: bool = False,
    subsample_partitions: float | None = None,
    bands: Sequence[str] = "ugrizy",
    ylim: float = 8.0,
    xlim: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
):
    """Compute uncorrected/corrected whitened signal and plot its distribution vs magnitude.

    Runs the whitening pipeline twice for the same data and partitions, once
    without a model (uncorrected) and once with ``model_path`` (corrected),
    and renders the twin-heatmap figure of :func:`plot_whiten_density`.

    Parameters mirror :func:`make_whiten_distribution_plot`; ``ylim`` sets the
    vertical half-range of every panel.

    Returns
    -------
    matplotlib.figure.Figure
    """
    z_bins, mag_bins, add_mag_err_bins = _default_result_bins()
    hash_range = survey_config.hash_range(split)
    if min_n_src is None:
        min_n_src = survey_config.n_src

    common = dict(
        hash_range=hash_range,
        bands=bands,
        obj=survey_config.obj,
        img=survey_config.img,
        phot=survey_config.phot,
        mode=survey_config.mode,
        min_n_src=min_n_src,
        non_extended_only=non_extended_only,
        n_workers=compute_config.n_workers,
        model_columns=model_columns,
        device=compute_config.device,
        mag_bins=mag_bins,
        z_bins=z_bins,
        add_mag_err_bins=add_mag_err_bins,
        n_samples=1,
        subsample_partitions=subsample_partitions,
    )
    hists_pre = _get_hists(survey_config.catalog_root, model_path=None, **common)
    hists_post = _get_hists(survey_config.catalog_root, model_path=model_path, **common)

    return plot_whiten_density(
        hists_pre,
        hists_post,
        bands=bands,
        z_bins=z_bins,
        mag_bins=mag_bins,
        ylim=ylim,
        xlim=xlim,
        output_path=output_path,
    )


def plot_addmagerr_density(
    hists,
    *,
    bands: Sequence[str] = "ugrizy",
    add_mag_err_bins: np.ndarray,
    mag_bins: np.ndarray,
    ylim: float = 0.5,
    xlim: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
):
    """Per-band heatmaps of the model-added magnitude error vs object magnitude.

    One panel per band shows the 2D histogram of the additional magnitude
    error the model injects, ``Δm = sqrt(max(corrected_magerr² − magerr², 0))``,
    against object magnitude, with every magnitude column normalized to its own
    peak so the color encodes the *shape* of the conditional distribution
    ``P(Δm | mag)``. The per-magnitude ``mean`` is overlaid. Unlike the
    whitened signal, ``Δm`` is defined only for the model-corrected data (it is
    identically zero without a model), so this is a single-row figure.

    Parameters
    ----------
    hists : pd.DataFrame
        Histogram table as returned by ``_get_hists`` for the model-corrected
        data (``model_path`` set).
    bands : sequence of str
        Bands to plot, one column each.
    add_mag_err_bins : np.ndarray
        Bin edges of the added magnitude error used to build ``hists``.
    mag_bins : np.ndarray
        Object-magnitude bin edges used to build ``hists``.
    ylim : float
        Upper limit of every panel, which spans ``[0, ylim]``.
    xlim : (float, float) or None
        Object-magnitude limits of every panel; defaults to the ``mag_bins`` extent.
    output_path : path or None
        If given, save the figure there; otherwise return it.

    Returns
    -------
    matplotlib.figure.Figure
    """
    c_line = "#d95f02"  # colorblind-safe warm line for the mean overlay
    mag_centers = 0.5 * (mag_bins[1:] + mag_bins[:-1])

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")

    n_cols = len(bands)
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(1.7 * n_cols + 0.6, 2.9),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    mesh = None
    for j, band in enumerate(bands):
        ax = axes[0, j]
        dens, mean, _std = _column_peak_density(
            hists, band, hist_column="add_mag_err_hist", value_bins=add_mag_err_bins, mag_bins=mag_bins
        )
        mesh = ax.pcolormesh(
            mag_bins, add_mag_err_bins, dens, cmap=cmap, vmin=0, vmax=1, shading="flat", rasterized=True
        )
        ax.plot(mag_centers, mean, color=c_line, lw=1.3)
        ax.set_xlim(*(xlim if xlim is not None else (mag_bins[0], mag_bins[-1])))
        ax.set_ylim(0.0, ylim)
        ax.set_title(f"${band}$", fontsize=13)
        ax.set_xlabel("object magnitude")
    axes[0, 0].set_ylabel("added mag error")

    handles = [Line2D([], [], color=c_line, lw=1.3, label="mean")]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, fontsize=8.5)
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist())
    cbar.set_label(r"$P(\Delta m\,|\,\mathrm{mag})$, per-magnitude peak $=1$")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    return fig


def make_addmagerr_density_plot(
    *,
    survey_config: SurveyConfig,
    model_path: str | Path | BaseUncleModel,
    model_columns: Sequence[str] = ("lc.x", "lc.err"),
    compute_config: ComputeConfig,
    split: str | None = None,
    min_n_src: int | None = None,
    non_extended_only: bool = False,
    subsample_partitions: float | None = None,
    bands: Sequence[str] = "ugrizy",
    ylim: float = 0.5,
    xlim: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
):
    """Compute the model-added magnitude error and plot its distribution vs magnitude.

    Runs the whitening pipeline once with ``model_path`` (the added magnitude
    error is zero without a model) and renders :func:`plot_addmagerr_density`.

    Parameters mirror :func:`make_whiten_density_plot`; ``ylim`` sets the upper
    limit of every panel.

    Returns
    -------
    matplotlib.figure.Figure
    """
    z_bins, mag_bins, add_mag_err_bins = _default_result_bins()
    hash_range = survey_config.hash_range(split)
    if min_n_src is None:
        min_n_src = survey_config.n_src

    hists = _get_hists(
        survey_config.catalog_root,
        model_path=model_path,
        hash_range=hash_range,
        bands=bands,
        obj=survey_config.obj,
        img=survey_config.img,
        phot=survey_config.phot,
        mode=survey_config.mode,
        min_n_src=min_n_src,
        non_extended_only=non_extended_only,
        n_workers=compute_config.n_workers,
        model_columns=model_columns,
        device=compute_config.device,
        mag_bins=mag_bins,
        z_bins=z_bins,
        add_mag_err_bins=add_mag_err_bins,
        n_samples=1,
        subsample_partitions=subsample_partitions,
    )

    return plot_addmagerr_density(
        hists,
        bands=bands,
        add_mag_err_bins=add_mag_err_bins,
        mag_bins=mag_bins,
        ylim=ylim,
        xlim=xlim,
        output_path=output_path,
    )


def make_plots(
    *,
    split: str | None = None,
    survey_config: SurveyConfig,
    min_n_src: int | None = None,
    non_extended_only: bool,
    model_path: str | Path | BaseUncleModel | None,
    model_columns: Sequence[str] = ("lc.x", "lc.err"),
    n_samples: int,
    object_mags: Sequence[float] | float = (),
    output_dir: str | Path | None = None,
    compute_config: ComputeConfig,
    subsample_partitions: float | None = None,
):
    """Plot whiten signal for a Rubin DP catalog, optionally corrected with a model

    Parameters
    ----------
    split : ``'train'``, ``'val'``, ``'test'``, ``'all'``, or ``None``
        Which data split to use. ``None`` and ``'all'`` both use the full
        dataset with no hash filtering. ``'train'``, ``'val'``, and
        ``'test'`` use the corresponding hash range from ``survey_config``.
    survey_config : SurveyConfig
        Survey configuration including catalog root, split boundaries, and n_src.
    min_n_src : int or None
        Minimum number of sources per object. Defaults to ``survey_config.n_src``.
    non_extended_only : bool
        Whether to filter the data with `extendedness == 0.0`.
    model_path : path or None
        Path to a torch model file or None. If None, plot the original data.
    model_columns : Sequence[str], optional
        Columns to pass to the model, first two must be flux and error.
    n_samples : int
        Number of samples per magnitude bin to use for UU vs object mag
        plot.
    object_mags : list of float or float
        Object magnitude bins to use for histogram plots.
    output_dir : path or None
        If given, output PDF plots to a given directory. If not,
         show them.
    compute_config : ComputeConfig
        Compute/infrastructure parameters; ``n_workers`` and ``device`` are used.
    subsample_partitions : float | None
        Random fraction of partitions to take. At least one partition will be used.
    """
    hash_range = survey_config.hash_range(split)
    if min_n_src is None:
        min_n_src = survey_config.n_src

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if isinstance(object_mags, float):
        object_mags = [object_mags]

    if_model = model_path is not None

    filename_suffix = "_corrected" if if_model else ""

    bands = "ugrizy"

    z_bins, mag_bins, add_mag_err_bins = _default_result_bins()
    z_width = z_bins[1] - z_bins[0]
    z_centers = 0.5 * (z_bins[1:] + z_bins[:-1])

    mag_centers = 0.5 * (mag_bins[1:] + mag_bins[:-1])

    add_mag_err_width = add_mag_err_bins[1] - add_mag_err_bins[0]
    add_mag_err_centers = 0.5 * (add_mag_err_bins[1:] + add_mag_err_bins[:-1])

    hists = _get_hists(
        survey_config.catalog_root,
        hash_range=hash_range,
        bands=bands,
        obj=survey_config.obj,
        img=survey_config.img,
        phot=survey_config.phot,
        mode=survey_config.mode,
        min_n_src=min_n_src,
        non_extended_only=non_extended_only,
        n_workers=compute_config.n_workers,
        model_path=model_path,
        model_columns=model_columns,
        device=compute_config.device,
        mag_bins=mag_bins,
        z_bins=z_bins,
        add_mag_err_bins=add_mag_err_bins,
        n_samples=n_samples,
        subsample_partitions=subsample_partitions,
    )

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()
    for i, band in enumerate(bands):
        _plot_hist(
            hists,
            axes[i],
            band,
            if_model=if_model,
            z_centers=z_centers,
            z_width=z_width,
            z_bins=z_bins,
            mag_bin=None,
        )
    plt.tight_layout()
    if output_dir is None:
        plt.show()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"hists_all_mags{filename_suffix}.pdf")

    for obj_mag in object_mags:
        mag_bin_idx = np.searchsorted(mag_bins, obj_mag)

        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.flatten()

        for i, band in enumerate(bands):
            _plot_hist(
                hists.query(f"mag_bin_idx == {mag_bin_idx}"),
                axes[i],
                band,
                if_model=if_model,
                z_centers=z_centers,
                z_width=z_width,
                z_bins=z_bins,
                mag_bin=mag_bins[mag_bin_idx : mag_bin_idx + 2],
            )

        plt.tight_layout()
        if output_dir is None:
            plt.show()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / f"hists_{obj_mag:.1f}mag{filename_suffix}.pdf")

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()
    for i, band in enumerate(bands):
        _plot_magn_vs_uu(
            hists,
            axes[i],
            band,
            mag_bins=mag_bins,
            z_centers=z_centers,
            z_width=z_width,
            z_bins=z_bins,
            mag_centers=mag_centers,
        )
    plt.tight_layout()
    if output_dir is None:
        plt.show()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"uu_vs_mag{filename_suffix}.pdf")

    if if_model:
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.flatten()
        for i, band in enumerate(bands):
            _plot_magn_vs_add_mag_err(
                hists,
                axes[i],
                band,
                mag_bins=mag_bins,
                add_mag_err_centers=add_mag_err_centers,
                add_mag_err_width=add_mag_err_width,
                add_mag_err_bins=add_mag_err_bins,
                mag_centers=mag_centers,
            )
        plt.tight_layout()
        if output_dir is None:
            plt.show()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / f"add_mag_err_vs_mag{filename_suffix}.pdf")


# ---------------------------------------------------------------------------
# Reduced chi-squared distributions
# ---------------------------------------------------------------------------
#
# For a light curve of constant flux the whitened components z_j are, when the
# reported uncertainties are correct, n-1 i.i.d. N(0, 1) samples, so
#     chi2 = sum_j z_j**2 ~ chi2_{n-1}.
# The sum of squared whitened residuals is *identically* the weighted chi2 of
# the constant model,
#     sum_j z_j**2 == sum_i w_i (x_i - xbar_w)**2,  w_i = 1/err_i**2,
#     xbar_w = sum_i w_i x_i / sum_i w_i,
# (verified to machine precision), so we compute the reduced chi2 directly with
# the O(n) weighted formula instead of building the O(n**2) whitening matrix.

_CHI2_CATEGORIES = ("full_uncorrected", "full_corrected")
# Special category storing the per-band light-curve length distribution
# (bin_idx is the light-curve length n, count is the number of light curves),
# used to build the calibrated degrees-of-freedom-mixture reference.
_CHI2_LENGTH_CATEGORY = "full_length"


def _weighted_reduced_chi2(x, err):
    """Reduced chi2 of a constant model for one light curve.

    Equivalent to ``sum(whiten_data(x, err)**2) / (len(x) - 1)`` but computed
    with the O(n) weighted formula rather than the O(n**2) whitening matrix.
    """
    x = x.astype(float)
    weight = 1.0 / np.square(err.astype(float))
    xbar = np.sum(weight * x) / np.sum(weight)
    return float(np.sum(weight * np.square(x - xbar)) / (x.size - 1))


def _empty_chi2_counts(n_bins):
    return pd.DataFrame(
        {
            "band": pd.Series(dtype=str),
            "category": pd.Series(dtype=str),
            "bin_idx": pd.Series(dtype=np.int64),
            "count": pd.Series(dtype=np.int64),
        }
    )


def _extract_chi2(
    df,
    pixel,
    *,
    hash_range,
    lg_chi2_bins,
    bands,
    min_n_src,
    length_max,
    non_extended_only,
    model_path,
    model_columns,
    device,
):
    """Per-partition histograms of lg(reduced chi2) of full light curves.

    Returns a long-format frame with columns ``band``, ``category``,
    ``bin_idx``, ``count``. For each of :data:`_CHI2_CATEGORIES` the rows give
    the per-band light-curve counts in each ``lg_chi2_bins`` bin of
    ``lg(reduced chi2)``. Rows with category :data:`_CHI2_LENGTH_CATEGORY` give
    the per-band light-curve length distribution (``bin_idx`` is the length,
    clipped to ``length_max``), used to build the calibrated reference.
    """
    n_bins = len(lg_chi2_bins) - 1

    df = df[df["lc"].nest.list_lengths >= min_n_src]
    if hash_range is not None:
        hashes = uniform_hash(df["id"])
        df = df[(hashes >= hash_range[0]) & (hashes < hash_range[1])]
    if non_extended_only:
        df = df.query("extendedness == 0.0")
    if len(df) == 0:
        return _empty_chi2_counts(n_bins)

    model = None
    if model_path is not None:
        model = torch.load(model_path, weights_only=False).to(device)
        model.eval()

    def per_lc(x, err, *extras):
        x = x.astype(float)
        err = err.astype(float)
        if model is not None:
            inputs = np.stack(np.broadcast_arrays(x, err, *extras), axis=-1, dtype=np.float32)
            uu = model(torch.tensor(inputs, device=device)).cpu().detach().numpy()[..., 0].ravel()
            corr_err = uu * err
        else:
            corr_err = err
        return {
            "full_uncorrected": _weighted_reduced_chi2(x, err),
            "full_corrected": _weighted_reduced_chi2(x, corr_err),
        }

    chi2 = df.reduce(per_lc, *model_columns)
    chi2["band"] = df["band"].to_numpy()
    chi2["length"] = np.asarray(df["lc"].nest.list_lengths)

    rows = []
    for band in bands:
        monochrome = chi2[chi2["band"] == band]
        for category in _CHI2_CATEGORIES:
            lg_chi2 = np.log10(monochrome[category].to_numpy())
            counts, _ = np.histogram(lg_chi2, bins=lg_chi2_bins)
            rows.append(
                pd.DataFrame(
                    {
                        "band": band,
                        "category": category,
                        "bin_idx": np.arange(n_bins),
                        "count": counts.astype(np.int64),
                    }
                )
            )
        length_hist = np.bincount(
            np.clip(monochrome["length"].to_numpy(), min_n_src, length_max), minlength=length_max + 1
        )
        nz = np.nonzero(length_hist)[0]
        rows.append(
            pd.DataFrame(
                {
                    "band": band,
                    "category": _CHI2_LENGTH_CATEGORY,
                    "bin_idx": nz.astype(np.int64),
                    "count": length_hist[nz].astype(np.int64),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _get_chi2_hists(
    rubin_dp_root: str | Path,
    *,
    hash_range: tuple[float, float] | None,
    bands: Sequence[str],
    obj: str,
    img: str,
    phot: str,
    mode: str,
    min_n_src: int,
    length_max: int,
    non_extended_only: bool,
    n_workers: int,
    model_path: str | Path | None,
    model_columns: Sequence[str],
    device: torch.device | str,
    lg_chi2_bins: np.ndarray,
    subsample_partitions: float | None = None,
):
    catalog = rubin_dp_catalog_multi_band(rubin_dp_root, bands=bands, obj=obj, img=img, phot=phot, mode=mode)
    if subsample_partitions is not None:
        n_partitions = max(1, int(round(catalog.npartitions * subsample_partitions)))
        rng = np.random.default_rng(0)
        partitions = rng.choice(catalog.npartitions, n_partitions, replace=True)
        catalog = catalog.partitions[partitions]

    counts = catalog.map_partitions(
        _extract_chi2,
        include_pixel=True,
        hash_range=hash_range,
        lg_chi2_bins=lg_chi2_bins,
        bands=bands,
        min_n_src=min_n_src,
        length_max=length_max,
        non_extended_only=non_extended_only,
        model_path=model_path,
        model_columns=model_columns,
        device=torch.device(device),
        meta=_empty_chi2_counts(len(lg_chi2_bins) - 1),
    )

    with Client(n_workers=n_workers, memory_limit="64GB") as client:
        print(f"Dask Dashboard Link: {client.dashboard_link}")
        counts_df = counts.compute()

    return counts_df.groupby(["band", "category", "bin_idx"], as_index=False)["count"].sum()


def _lg_chi2_density(counts, band, category, *, lg_chi2_bins):
    """lg(chi2) probability density (and linear mean) of one category."""
    centers = 0.5 * (lg_chi2_bins[1:] + lg_chi2_bins[:-1])
    sub = counts.query(f"band == {band!r} and category == {category!r}").sort_values("bin_idx")
    count = sub["count"].to_numpy().astype(float)
    total = count.sum()
    if total == 0:
        return np.zeros_like(count), np.nan
    prob = count / total
    mean_lin = float(np.sum(prob * 10.0**centers))
    return prob / np.diff(lg_chi2_bins), mean_lin


def _calibrated_mixture_density(counts, band, *, centers):
    """Calibrated lg(reduced chi2) density for the real light-curve-length mix.

    The reduced chi2 of a well-calibrated light curve with ``n`` points follows
    ``chi2_{n-1} / (n-1)``. This averages that density (mapped onto the ``lg``
    axis) over the actual per-band distribution of light-curve lengths ``n``
    stored under :data:`_CHI2_LENGTH_CATEGORY`, giving the distribution a
    perfectly calibrated survey would produce.
    """
    lengths = counts.query(f"band == {band!r} and category == {_CHI2_LENGTH_CATEGORY!r}")
    n = lengths["bin_idx"].to_numpy()
    weight = lengths["count"].to_numpy().astype(float)
    if weight.sum() == 0:
        return np.zeros_like(centers)
    weight = weight / weight.sum()
    y = 10.0**centers
    dof = n - 1
    # density of lg(chi2_dof/dof): f_Y(y) * y * ln(10), averaged over dof.
    per_dof = dof[:, None] * chi2_dist.pdf(dof[:, None] * y[None, :], df=dof[:, None])
    return (weight @ per_dof) * y * np.log(10.0)


def plot_chi2_distributions(
    counts,
    *,
    bands: Sequence[str] = "ugrizy",
    lg_chi2_bins: np.ndarray,
    xlim: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
):
    """Per-band figure of the lg(reduced chi2) distribution of full light curves.

    One panel per band overlays, on the common ``lg`` axis, the reduced chi2 of
    the reported ("uncorrected", filled) and model-corrected ("corrected", line)
    uncertainties of full light curves. The chi2 uses the constant-flux weighted
    model and no signal correction is applied. On a logarithmic axis the model's
    multiplicative error rescaling is a horizontal shift and the calibrated peak
    is roughly symmetric. A calibrated reference is drawn for comparison: the
    exact degrees-of-freedom mixture of ``chi2_{n-1} / (n-1)`` over the real
    light-curve-length distribution, evaluated on a dense grid.

    The uncorrected/corrected colors match :func:`plot_whiten_distributions` so
    "before"/"after" reads consistently across figures.

    Parameters
    ----------
    counts : pd.DataFrame
        Long-format histogram table from :func:`_get_chi2_hists` with columns
        ``band``, ``category``, ``bin_idx``, ``count``, binned over
        ``lg(reduced chi2)``, including the light-curve-length rows.
    bands : sequence of str
        Bands to plot, one panel each.
    lg_chi2_bins : np.ndarray
        The ``lg(reduced chi2)`` bin edges used to build ``counts``.
    xlim : (float, float) or None
        Horizontal limits, in ``lg(reduced chi2)``; defaults to the bin extent.
    output_path : path or None
        If given, save the figure there; otherwise return it.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Match plot_whiten_distributions: cb[2] is the "before"/uncorrected color,
    # cb[0] the "after"/corrected color; references are black.
    cb = plt.style.library["tableau-colorblind10"]["axes.prop_cycle"].by_key()["color"]
    c_uncorr, c_corr, c_ref = cb[2], cb[0], "black"

    if xlim is None:
        xlim = (lg_chi2_bins[0], lg_chi2_bins[-1])

    # Dense grid for the smooth reference curve (independent of the histogram bins).
    ref_grid = np.linspace(lg_chi2_bins[0], lg_chi2_bins[-1], 400)

    ymax = 0.0

    def panel(ax, band):
        nonlocal ymax
        pre, mean_pre = _lg_chi2_density(counts, band, "full_uncorrected", lg_chi2_bins=lg_chi2_bins)
        post, mean_post = _lg_chi2_density(counts, band, "full_corrected", lg_chi2_bins=lg_chi2_bins)
        mixture_ref = _calibrated_mixture_density(counts, band, centers=ref_grid)
        ymax = max(ymax, pre.max(), post.max())

        ax.stairs(pre, lg_chi2_bins, fill=True, color=c_uncorr, alpha=0.45)
        ax.stairs(post, lg_chi2_bins, color=c_corr, lw=1.8)
        ax.plot(ref_grid, mixture_ref, color=c_ref, ls=(0, (4, 2)), lw=1.1)
        ax.axvline(0.0, color="0.6", lw=0.7, zorder=0)  # reduced chi2 = 1
        ax.text(
            0.97,
            0.96,
            f"${band}$",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            0.97,
            0.82,
            rf"red. $\chi^2$: {mean_pre:.2f}$\to${mean_post:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=8.0,
        )
        ax.set_xlim(*xlim)

    ncols = 3
    nrows = int(np.ceil(len(bands) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, band in zip(axes, bands, strict=False):
        panel(ax, band)
    axes[0].set_ylim(0.0, 1.12 * ymax)  # shared y; give the tall peaks headroom
    for ax in axes[len(bands) :]:
        ax.set_visible(False)
    for ax in axes[len(bands) - ncols : len(bands)]:
        ax.set_xlabel(r"$\lg(\mathrm{reduced}\ \chi^2)$")
    for ax in axes[::ncols]:
        ax.set_ylabel("probability density")

    handles = [
        Patch(facecolor=c_uncorr, alpha=0.45, label="uncorrected"),
        Line2D([], [], color=c_corr, lw=1.8, label="corrected"),
        Line2D([], [], color=c_ref, ls=(0, (4, 2)), lw=1.1, label="calibrated (dof mix)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def _default_lg_chi2_bins() -> np.ndarray:
    """Default ``lg(reduced chi2)`` bin edges."""
    return np.linspace(-1.0, 1.0, 31)


def make_chi2_distribution_plot(
    *,
    survey_config: SurveyConfig,
    model_path: str | Path | BaseUncleModel,
    model_columns: Sequence[str] = ("lc.x", "lc.err"),
    compute_config: ComputeConfig,
    split: str | None = None,
    min_n_src: int | None = None,
    length_max: int = 1000,
    non_extended_only: bool = False,
    subsample_partitions: float | None = None,
    bands: Sequence[str] = "ugrizy",
    lg_chi2_bins: np.ndarray | None = None,
    xlim: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
):
    """Compute and plot the lg(reduced chi2) distribution of full light curves.

    Runs the reduced-chi2 pipeline once, applying ``model_path`` to correct the
    uncertainties (error scaling only, no signal correction), and renders the
    per-band figure of :func:`plot_chi2_distributions`. Each full light curve
    contributes one reduced chi2 for the reported and one for the corrected
    uncertainties; the calibrated references are built from the light-curve
    length distribution collected during the same pass.

    Parameters mirror :func:`make_whiten_distribution_plot`. ``min_n_src``
    defaults to ``survey_config.n_src`` so only light curves the training
    pipeline would use are included; ``length_max`` caps the degrees of freedom
    tracked for the calibrated reference.

    Returns
    -------
    matplotlib.figure.Figure
    """
    hash_range = survey_config.hash_range(split)
    if min_n_src is None:
        min_n_src = survey_config.n_src
    if lg_chi2_bins is None:
        lg_chi2_bins = _default_lg_chi2_bins()

    counts = _get_chi2_hists(
        survey_config.catalog_root,
        hash_range=hash_range,
        bands=bands,
        obj=survey_config.obj,
        img=survey_config.img,
        phot=survey_config.phot,
        mode=survey_config.mode,
        min_n_src=min_n_src,
        length_max=length_max,
        non_extended_only=non_extended_only,
        n_workers=compute_config.n_workers,
        model_path=model_path,
        model_columns=model_columns,
        device=compute_config.device,
        lg_chi2_bins=lg_chi2_bins,
        subsample_partitions=subsample_partitions,
    )

    return plot_chi2_distributions(
        counts,
        bands=bands,
        lg_chi2_bins=lg_chi2_bins,
        xlim=xlim,
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# KL-statistic distributions
# ---------------------------------------------------------------------------
#
# For one light curve the whitened components z_j are, when the reported
# uncertainties are correct, n-1 i.i.d. N(0, 1) samples. The per-light-curve KL
# statistic is the KL divergence of the fitted N(mu, var) of those components
# from the target N(0, 1),
#     mu  = mean(z),  var = var(z, ddof=1),
#     KL  = 0.5 * (mu**2 + var - ln(var) - 1),
# identical to the per-LC term of ``KLWhitenLc`` in
# :mod:`uncle_val.learning.losses`. Unlike the reduced chi2 there is no
# whitening-free shortcut, so the actual whitened ``z`` is built per light curve
# via :data:`numba_whiten_data`. The KL statistic is >= 0 (0 for a perfectly
# calibrated fit), so it is plotted on an ``lg`` axis and the reference is the
# distribution a calibrated survey would produce, obtained by Monte-Carlo.

_KL_CATEGORIES = ("full_uncorrected", "full_corrected")
# Special category storing the per-band light-curve length distribution
# (bin_idx is the light-curve length n, count is the number of light curves),
# used to build the calibrated Monte-Carlo reference.
_KL_LENGTH_CATEGORY = "full_length"
# Floor applied to KL before log10 to guard tiny negatives from floating point
# (var ~ 1, mu ~ 0 can yield a KL just below 0).
_KL_FLOOR = 1e-12


def _lc_kl(x, err):
    """Per-light-curve KL statistic KL(N(mu, var) || N(0, 1)).

    Whitens ``x`` with ``err`` into ``z`` (n-1 values that are i.i.d. N(0, 1)
    when the uncertainties are correct), then returns
    ``0.5 * (mean(z)**2 + var(z, ddof=1) - ln(var(z, ddof=1)) - 1)`` --
    identical to the per-LC term of ``KLWhitenLc``.
    """
    z = numba_whiten_data(x.astype(float), err.astype(float))
    mu = float(np.mean(z))
    var = float(np.var(z, ddof=1))
    return 0.5 * (mu**2 + var - np.log(var) - 1.0)


def _empty_kl_counts(n_bins):
    return pd.DataFrame(
        {
            "band": pd.Series(dtype=str),
            "category": pd.Series(dtype=str),
            "bin_idx": pd.Series(dtype=np.int64),
            "count": pd.Series(dtype=np.int64),
        }
    )


def _extract_kl(
    df,
    pixel,
    *,
    hash_range,
    lg_kl_bins,
    bands,
    min_n_src,
    length_max,
    non_extended_only,
    model_path,
    model_columns,
    device,
):
    """Per-partition histograms of lg(KL statistic) of full light curves.

    Returns a long-format frame with columns ``band``, ``category``,
    ``bin_idx``, ``count``. For each of :data:`_KL_CATEGORIES` the rows give the
    per-band light-curve counts in each ``lg_kl_bins`` bin of
    ``lg(KL statistic)``. Rows with category :data:`_KL_LENGTH_CATEGORY` give the
    per-band light-curve length distribution (``bin_idx`` is the length, clipped
    to ``length_max``), used to build the calibrated Monte-Carlo reference.
    """
    n_bins = len(lg_kl_bins) - 1

    df = df[df["lc"].nest.list_lengths >= min_n_src]
    if hash_range is not None:
        hashes = uniform_hash(df["id"])
        df = df[(hashes >= hash_range[0]) & (hashes < hash_range[1])]
    if non_extended_only:
        df = df.query("extendedness == 0.0")
    if len(df) == 0:
        return _empty_kl_counts(n_bins)

    model = None
    if model_path is not None:
        model = torch.load(model_path, weights_only=False).to(device)
        model.eval()

    def per_lc(x, err, *extras):
        x = x.astype(float)
        err = err.astype(float)
        if model is not None:
            inputs = np.stack(np.broadcast_arrays(x, err, *extras), axis=-1, dtype=np.float32)
            uu = model(torch.tensor(inputs, device=device)).cpu().detach().numpy()[..., 0].ravel()
            corr_err = uu * err
        else:
            corr_err = err
        return {
            "full_uncorrected": _lc_kl(x, err),
            "full_corrected": _lc_kl(x, corr_err),
        }

    kl = df.reduce(per_lc, *model_columns)
    kl["band"] = df["band"].to_numpy()
    kl["length"] = np.asarray(df["lc"].nest.list_lengths)

    rows = []
    for band in bands:
        monochrome = kl[kl["band"] == band]
        for category in _KL_CATEGORIES:
            lg_kl = np.log10(np.clip(monochrome[category].to_numpy(), _KL_FLOOR, None))
            counts, _ = np.histogram(lg_kl, bins=lg_kl_bins)
            rows.append(
                pd.DataFrame(
                    {
                        "band": band,
                        "category": category,
                        "bin_idx": np.arange(n_bins),
                        "count": counts.astype(np.int64),
                    }
                )
            )
        length_hist = np.bincount(
            np.clip(monochrome["length"].to_numpy(), min_n_src, length_max), minlength=length_max + 1
        )
        nz = np.nonzero(length_hist)[0]
        rows.append(
            pd.DataFrame(
                {
                    "band": band,
                    "category": _KL_LENGTH_CATEGORY,
                    "bin_idx": nz.astype(np.int64),
                    "count": length_hist[nz].astype(np.int64),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _get_kl_hists(
    rubin_dp_root: str | Path,
    *,
    hash_range: tuple[float, float] | None,
    bands: Sequence[str],
    obj: str,
    img: str,
    phot: str,
    mode: str,
    min_n_src: int,
    length_max: int,
    non_extended_only: bool,
    n_workers: int,
    model_path: str | Path | None,
    model_columns: Sequence[str],
    device: torch.device | str,
    lg_kl_bins: np.ndarray,
    subsample_partitions: float | None = None,
):
    catalog = rubin_dp_catalog_multi_band(rubin_dp_root, bands=bands, obj=obj, img=img, phot=phot, mode=mode)
    if subsample_partitions is not None:
        n_partitions = max(1, int(round(catalog.npartitions * subsample_partitions)))
        rng = np.random.default_rng(0)
        partitions = rng.choice(catalog.npartitions, n_partitions, replace=True)
        catalog = catalog.partitions[partitions]

    counts = catalog.map_partitions(
        _extract_kl,
        include_pixel=True,
        hash_range=hash_range,
        lg_kl_bins=lg_kl_bins,
        bands=bands,
        min_n_src=min_n_src,
        length_max=length_max,
        non_extended_only=non_extended_only,
        model_path=model_path,
        model_columns=model_columns,
        device=torch.device(device),
        meta=_empty_kl_counts(len(lg_kl_bins) - 1),
    )

    with Client(n_workers=n_workers, memory_limit="64GB") as client:
        print(f"Dask Dashboard Link: {client.dashboard_link}")
        counts_df = counts.compute()

    return counts_df.groupby(["band", "category", "bin_idx"], as_index=False)["count"].sum()


def _lg_kl_density(counts, band, category, *, lg_kl_bins):
    """lg(KL) probability density (and linear mean) of one category."""
    centers = 0.5 * (lg_kl_bins[1:] + lg_kl_bins[:-1])
    sub = counts.query(f"band == {band!r} and category == {category!r}").sort_values("bin_idx")
    count = sub["count"].to_numpy().astype(float)
    total = count.sum()
    if total == 0:
        return np.zeros_like(count), np.nan
    prob = count / total
    mean_lin = float(np.sum(prob * 10.0**centers))
    return prob / np.diff(lg_kl_bins), mean_lin


def _calibrated_kl_density(counts, band, *, lg_kl_bins, n_mc=40000, seed=0):
    """Calibrated lg(KL statistic) density for the real light-curve-length mix.

    The KL statistic has no simple closed form, so for each light-curve length
    ``n`` present in the real per-band length distribution (stored under
    :data:`_KL_LENGTH_CATEGORY`) this draws ``n_mc`` samples of ``n-1`` i.i.d.
    N(0, 1) values, computes the KL statistic of each, and histograms
    ``lg(KL)``. The per-dof densities are averaged over the actual length
    distribution, giving the distribution a perfectly calibrated survey would
    produce. A fixed RNG seed makes the reference reproducible.
    """
    centers = 0.5 * (lg_kl_bins[1:] + lg_kl_bins[:-1])
    lengths = counts.query(f"band == {band!r} and category == {_KL_LENGTH_CATEGORY!r}")
    n = lengths["bin_idx"].to_numpy()
    weight = lengths["count"].to_numpy().astype(float)
    if weight.sum() == 0:
        return np.zeros_like(centers)
    weight = weight / weight.sum()
    width = np.diff(lg_kl_bins)
    rng = np.random.default_rng(seed)

    density = np.zeros_like(centers)
    for length, w in zip(n, weight, strict=True):
        dof = int(length) - 1
        z = rng.standard_normal((n_mc, dof))
        mu = z.mean(axis=1)
        var = z.var(axis=1, ddof=1)
        kl = 0.5 * (mu**2 + var - np.log(var) - 1.0)
        lg_kl = np.log10(np.clip(kl, _KL_FLOOR, None))
        hist, _ = np.histogram(lg_kl, bins=lg_kl_bins)
        density += w * hist / n_mc / width
    return density


def plot_kl_distributions(
    counts,
    *,
    bands: Sequence[str] = "ugrizy",
    lg_kl_bins: np.ndarray,
    xlim: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
):
    """Per-band figure of the lg(KL statistic) distribution of full light curves.

    One panel per band overlays, on the common ``lg`` axis, the per-light-curve
    KL statistic of the reported ("uncorrected", filled) and model-corrected
    ("corrected", line) uncertainties of full light curves. The KL statistic uses
    the actual whitened residuals and no signal correction is applied. A single
    calibrated reference is drawn: the KL distribution a perfectly calibrated
    survey would produce, obtained by Monte-Carlo over the real light-curve-length
    distribution (see :func:`_calibrated_kl_density`).

    The uncorrected/corrected colors match :func:`plot_whiten_distributions` so
    "before"/"after" reads consistently across figures.

    Parameters
    ----------
    counts : pd.DataFrame
        Long-format histogram table from :func:`_get_kl_hists` with columns
        ``band``, ``category``, ``bin_idx``, ``count``, binned over
        ``lg(KL statistic)``, including the light-curve-length rows.
    bands : sequence of str
        Bands to plot, one panel each.
    lg_kl_bins : np.ndarray
        The ``lg(KL statistic)`` bin edges used to build ``counts``.
    xlim : (float, float) or None
        Horizontal limits, in ``lg(KL statistic)``; defaults to the bin extent.
    output_path : path or None
        If given, save the figure there; otherwise return it.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Match plot_whiten_distributions: cb[2] is the "before"/uncorrected color,
    # cb[0] the "after"/corrected color; the reference is black.
    cb = plt.style.library["tableau-colorblind10"]["axes.prop_cycle"].by_key()["color"]
    c_uncorr, c_corr, c_ref = cb[2], cb[0], "black"

    centers = 0.5 * (lg_kl_bins[1:] + lg_kl_bins[:-1])
    if xlim is None:
        xlim = (lg_kl_bins[0], lg_kl_bins[-1])

    ymax = 0.0

    def panel(ax, band):
        nonlocal ymax
        pre, mean_pre = _lg_kl_density(counts, band, "full_uncorrected", lg_kl_bins=lg_kl_bins)
        post, mean_post = _lg_kl_density(counts, band, "full_corrected", lg_kl_bins=lg_kl_bins)
        calibrated_ref = _calibrated_kl_density(counts, band, lg_kl_bins=lg_kl_bins)
        ymax = max(ymax, pre.max(), post.max())

        ax.stairs(pre, lg_kl_bins, fill=True, color=c_uncorr, alpha=0.45)
        ax.stairs(post, lg_kl_bins, color=c_corr, lw=1.8)
        ax.plot(centers, calibrated_ref, color=c_ref, ls=(0, (4, 2)), lw=1.1)
        ax.text(
            0.97,
            0.96,
            f"${band}$",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            0.97,
            0.82,
            rf"$\langle\mathrm{{KL}}\rangle$ {mean_pre:.3f}$\to${mean_post:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=8.0,
        )
        ax.set_xlim(*xlim)

    ncols = 3
    nrows = int(np.ceil(len(bands) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, band in zip(axes, bands, strict=False):
        panel(ax, band)
    axes[0].set_ylim(0.0, 1.12 * ymax)  # shared y; give the tall peaks headroom
    for ax in axes[len(bands) :]:
        ax.set_visible(False)
    for ax in axes[len(bands) - ncols : len(bands)]:
        ax.set_xlabel(r"$\lg\,\mathrm{KL}$")
    for ax in axes[::ncols]:
        ax.set_ylabel("probability density")

    handles = [
        Patch(facecolor=c_uncorr, alpha=0.45, label="uncorrected"),
        Line2D([], [], color=c_corr, lw=1.8, label="corrected"),
        Line2D([], [], color=c_ref, ls=(0, (4, 2)), lw=1.1, label="calibrated"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
    return fig


def _default_lg_kl_bins() -> np.ndarray:
    """Default ``lg(KL statistic)`` bin edges."""
    return np.linspace(-4.0, 1.0, 31)


def make_kl_distribution_plot(
    *,
    survey_config: SurveyConfig,
    model_path: str | Path | BaseUncleModel,
    model_columns: Sequence[str] = ("lc.x", "lc.err"),
    compute_config: ComputeConfig,
    split: str | None = None,
    min_n_src: int | None = None,
    length_max: int = 1000,
    non_extended_only: bool = False,
    subsample_partitions: float | None = None,
    bands: Sequence[str] = "ugrizy",
    lg_kl_bins: np.ndarray | None = None,
    xlim: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
):
    """Compute and plot the lg(KL statistic) distribution of full light curves.

    Runs the KL pipeline once, applying ``model_path`` to correct the
    uncertainties (error scaling only, no signal correction), and renders the
    per-band figure of :func:`plot_kl_distributions`. Each full light curve
    contributes one KL statistic for the reported and one for the corrected
    uncertainties; the calibrated reference is built by Monte-Carlo from the
    light-curve length distribution collected during the same pass.

    Parameters mirror :func:`make_chi2_distribution_plot`. ``min_n_src`` defaults
    to ``survey_config.n_src`` so only light curves the training pipeline would
    use are included; ``length_max`` caps the degrees of freedom tracked for the
    calibrated reference.

    Returns
    -------
    matplotlib.figure.Figure
    """
    hash_range = survey_config.hash_range(split)
    if min_n_src is None:
        min_n_src = survey_config.n_src
    if lg_kl_bins is None:
        lg_kl_bins = _default_lg_kl_bins()

    counts = _get_kl_hists(
        survey_config.catalog_root,
        hash_range=hash_range,
        bands=bands,
        obj=survey_config.obj,
        img=survey_config.img,
        phot=survey_config.phot,
        mode=survey_config.mode,
        min_n_src=min_n_src,
        length_max=length_max,
        non_extended_only=non_extended_only,
        n_workers=compute_config.n_workers,
        model_path=model_path,
        model_columns=model_columns,
        device=compute_config.device,
        lg_kl_bins=lg_kl_bins,
        subsample_partitions=subsample_partitions,
    )

    return plot_kl_distributions(
        counts,
        bands=bands,
        lg_kl_bins=lg_kl_bins,
        xlim=xlim,
        output_path=output_path,
    )
