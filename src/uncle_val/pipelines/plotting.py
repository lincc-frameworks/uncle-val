from collections.abc import Sequence
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from dask.distributed import Client
from nested_pandas import NestedDtype, NestedFrame
from scipy.stats import norm

from uncle_val.datasets.dp1 import dp1_catalog_multi_band
from uncle_val.learning.models import UncleModel
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


def _extract_hists(
    df,
    pixel,
    *,
    hash_range,
    z_bins,
    mag_bins,
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
                "hist": NestedDtype(
                    pa.struct({"z_bin_idx": pa.list_(pa.int64()), "count": pa.list_(pa.int64())})
                ),
                "samples": NestedDtype(
                    pa.struct({"z": pa.list_(pa.float64()), "object_mag": pa.list_(pa.float64())})
                ),
            }
        )

    if model_path is not None:
        model = torch.load(model_path, weights_only=False).to(device)
        model.eval()

        # model_subcolumns = []
        # for col in model_columns:
        #     if col.startswith("lc."):
        #         model_subcolumns.append(col.removeprefix("lc."))
        #         continue
        #     model_subcolumns.append(col)
        #     df[f"lc.{col}"] = df[col]
        #
        # model_inputs = df["lc"].nest.to_flat(model_subcolumns).to_numpy(dtype=np.float32)
        # chunked_model = torch.vmap(model, chunk_size=128)
        # model_output = chunked_model(torch.tensor(model_inputs, device=device)).cpu().detach().numpy()
        #
        # uu = model_output[..., 0].flatten()
        # err_col = "lc.corrected_err"
        # df[err_col] = uu * df["lc.err"]
        #
        # if model.outputs_s:
        #     sf = model_output[..., 1].flatten()
        #     x_col = "lc.corrected_x"
        #     df[x_col] = (1.0 + sf) * df["lc.x"]

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

            return {"corrected_lc.x": corrected_x, "corrected_lc.err": corrected_err}

        df["lc"] = df.reduce(apply_model, *model_columns)["corrected_lc"]

    whiten = (
        df.reduce(
            _whiten_flux_err,
            "lc.x",
            "lc.err",
            append_columns=True,
        )
        .drop(
            columns=["lc"],
        )
        .reset_index(
            drop=True,
        )
    )

    mag_idx_grid, z_idx_grid = np.indices((len(mag_bins) - 1, len(z_bins) - 1))

    result_nfs = []
    for band in bands:
        monochrome = whiten.query(f"band=='{band}'")
        z_mag = monochrome[["object_mag"]].join(monochrome["whiten"].nest.to_flat())
        hist, _mag_bins, _z_bins = np.histogram2d(z_mag["object_mag"], z_mag["z"], bins=[mag_bins, z_bins])
        flat_df = pd.DataFrame(
            {
                "band": band,
                "mag_bin_idx": mag_idx_grid.flatten(),
                "z_bin_idx": z_idx_grid.flatten(),
                "count": hist.flatten().astype(np.int64),
            }
        )
        nf = NestedFrame.from_flat(
            flat_df,
            name="hist",
            base_columns=["band"],
            nested_columns=["z_bin_idx", "count"],
            on="mag_bin_idx",
        ).reset_index(drop=False)
        sample_idx = _samples_in_bins(
            x=z_mag["object_mag"],
            bins=mag_bins,
            sample_count=n_samples,
            rng=rng,
        )
        nf["z"] = pd.Series(
            [np.asarray(z_mag["z"].iloc[idx]) for idx in sample_idx],
            dtype=pd.ArrowDtype(pa.list_(pa.float64())),
        )
        nf["object_mag"] = pd.Series(
            [np.asarray(z_mag["object_mag"].iloc[idx]) for idx in sample_idx],
            dtype=pd.ArrowDtype(pa.list_(pa.float64())),
        )
        nf = nf.nest_lists(["z", "object_mag"], name="samples")

        result_nfs.append(nf)
    return pd.concat(result_nfs, ignore_index=True)


def _get_hists(
    dp1_root: str | Path,
    *,
    hash_range: tuple[float, float] | None = None,
    bands: Sequence[str],
    min_n_src: int,
    non_extended_only: bool,
    n_workers: int,
    model_path: str | Path | None,
    model_columns: Sequence[str],
    device: torch.device | str = "cpu",
    mag_bins: np.ndarray,
    z_bins: np.ndarray,
    n_samples: int,
):
    catalog = dp1_catalog_multi_band(
        dp1_root,
        bands=bands,
        obj="science",
        img="cal",
        phot="PSF",
        mode="forced",
    )

    hists = catalog.map_partitions(
        _extract_hists,
        include_pixel=True,
        hash_range=hash_range,
        z_bins=z_bins,
        mag_bins=mag_bins,
        bands=bands,
        min_n_src=min_n_src,
        n_samples=n_samples,
        non_extended_only=non_extended_only,
        model_path=model_path,
        model_columns=model_columns,
        device=torch.device(device),
    )

    with Client(n_workers=n_workers, threads_per_worker=1, memory_limit="8GB") as client:
        print(f"Dask Dashboard Link: {client.dashboard_link}")
        hists_df = hists.compute()

    return hists_df


def _aggregate_hists(df, band, *, z_centers: np.ndarray, z_width: float):
    df = (
        df.drop(columns=["samples"])
        .query(
            f"band == {band!r}",
        )
        .explode("hist")
        .groupby(
            ["z_bin_idx"],
        )["count"]
        .sum()
        .sort_index()
        .reset_index(
            drop=False,
        )
    )
    df["z_centers"] = z_centers
    df["prob"] = df["count"] / df["count"].sum()
    df["prob_dens"] = df["prob"] / z_width

    mean = np.sum(df["prob"] * df["z_centers"])
    std = np.sqrt(np.sum(df["prob"] * (df["z_centers"] - mean) ** 2))

    return df, mean, std


def _plot_hist(
    df, ax, band, *, if_model: bool, z_centers: np.ndarray, z_width: float, z_bins: np.ndarray, mag_bin
):
    df, mean, std = _aggregate_hists(df, band, z_centers=z_centers, z_width=z_width)

    title = f"Whiten Signal, {band}-band, μ={mean:.4f} σ={std:.4f}"
    if mag_bin is not None:
        title = f"{title}; for mag=[{mag_bin[0]}; {mag_bin[1]}]"
    ax.set_title(title)
    label = "corrected data" if if_model is None else "data"
    ax.bar(x=df["z_centers"], height=df["prob_dens"], width=z_width, label=label, color="blue", alpha=0.2)
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
            df.query(f"mag_bin_idx == {mag_bin_idx}"), band, z_centers=z_centers, z_width=z_width
        )
        means.append(mean)
        uu.append(std)
    means = np.array(means)
    uu = np.array(uu)

    mag_, means, uu = mag_centers[uu > 0], means[uu > 0], uu[uu > 0]

    samples_mag = df.query(f"band == {band!r}")["samples.object_mag"]
    samples_z = df.query(f"band == {band!r}")["samples.z"]

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


def make_plots(
    dp1_root: str | Path,
    *,
    hash_range: tuple[float, float] | None = None,
    min_n_src: int,
    non_extended_only: bool,
    n_workers: int,
    model_path: str | Path | UncleModel | None,
    model_columns: Sequence[str] = ("lc.x", "lc.err"),
    n_samples: int,
    device: torch.device | str = "cpu",
    object_mags: Sequence[float] | float = (),
    output_dir: str | Path | None = None,
):
    """Plot whiten signal for a DP1 catalog, optionally corrected with a model

    Parameters
    ----------
    dp1_root : str | Path
        The root directory of the DP1 HATS catalogs.
    hash_range : min and max hash value (between 0 and 1) or None
        If not None, filter by object's float hashes.
    min_n_src : int
        Minimum number of sources per object.
    non_extended_only : bool
        Whether to filter the data with `extendedness == 0.0`.
    n_workers : int
        Number of Dask workers to use.
    model_path : path or None
        Path to a torch model file or None. If None, plot the original data.
    model_columns : Sequence[str], optional
        Columns to pass to the model, first two must be flux and error.
    n_samples : int
        Number of samples per magnitude bin to use for UU vs object mag
        plot.
    device : torch.device | str
        Torch device to use with the model.
    object_mags : list of float or float
        Object magnitude bins to use for histogram plots.
    output_dir : path or None
        If given, output PDF plots to a given directory. If not,
         show them.
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if isinstance(object_mags, float):
        object_mags = [object_mags]

    if_model = model_path is not None

    filename_suffix = "" if if_model is None else "_corrected"

    bands = "ugrizy"

    z_bins = np.r_[-10:10:1001j]
    z_width = z_bins[1] - z_bins[0]
    z_centers = 0.5 * (z_bins[1:] + z_bins[:-1])

    mag_bins = np.arange(13, 27, 0.5)
    mag_centers = 0.5 * (mag_bins[1:] + mag_bins[:-1])

    hists = _get_hists(
        dp1_root,
        hash_range=hash_range,
        bands=bands,
        min_n_src=min_n_src,
        non_extended_only=non_extended_only,
        n_workers=n_workers,
        model_path=model_path,
        model_columns=model_columns,
        device=device,
        mag_bins=mag_bins,
        z_bins=z_bins,
        n_samples=n_samples,
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
        fig.savefig(output_dir / f"uu_vs_mag{filename_suffix}.pdf")
