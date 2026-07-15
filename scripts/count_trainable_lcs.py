#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from uncle_val.datasets.rubin_dp import rubin_dp_catalog_multi_band
from uncle_val.learning.lsdb_dataset import filter_trainable_lcs
from uncle_val.pipelines.splits import SurveyConfig, dp1_config, dp2_config

LC_COL = "lc"
ID_COL = "id"
SPLIT_NAMES = ("train", "val", "test")


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Count trainable light curves in a Rubin DP HATS catalog, "
        "using the same selection as the training pipeline: "
        "light curves with at least n_src observations, split into "
        "train/val/test by hashing the object ID. "
        "Use it to convert the number of light curves seen during training "
        "into the number of epochs (n_lcs_seen / train count)."
    )
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--survey-config",
        type=Path,
        help="Path to a survey_config.json saved with a training run.",
    )
    config_group.add_argument(
        "--preset",
        choices=["dp1", "dp2"],
        help="Use dp1_config/dp2_config defaults instead of a JSON file; "
        "requires --catalog-root and --n-src.",
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=None,
        help="Root directory of the HATS catalogs. Required with --preset; "
        "with --survey-config it overrides the (possibly relative) "
        "catalog_root stored in the JSON.",
    )
    parser.add_argument(
        "--n-src",
        type=int,
        default=None,
        help="Observations per light curve. Required with --preset; "
        "with --survey-config it overrides the stored value.",
    )
    parser.add_argument(
        "--img",
        choices=["cal", "diff"],
        default=None,
        help="Image type used for photometry, 'cal' (calibrated/direct) or "
        "'diff' (subtracted). Only used with --preset; defaults to the "
        "preset default ('cal').",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="If given, start a dask.distributed client with that many workers.",
    )
    return parser.parse_args(argv)


def make_survey_config(args) -> SurveyConfig:
    """Build the SurveyConfig from the command-line arguments."""
    if args.survey_config is not None:
        config = SurveyConfig.from_json(args.survey_config)
        overrides = {}
        if args.catalog_root is not None:
            overrides["catalog_root"] = args.catalog_root
        if args.n_src is not None:
            overrides["n_src"] = args.n_src
        if overrides:
            import dataclasses

            config = dataclasses.replace(config, **overrides)
        return config

    if args.catalog_root is None or args.n_src is None:
        raise ValueError("--preset requires both --catalog-root and --n-src")
    preset = {"dp1": dp1_config, "dp2": dp2_config}[args.preset]
    kwargs = {}
    if args.img is not None:
        kwargs["img"] = args.img
    return preset(args.catalog_root, n_src=args.n_src, **kwargs)


def _count_partition(nf, *, n_src: int, splits: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Per-band, per-split counts of trainable light curves in one partition."""
    parts = []
    for split_name, hash_range in splits.items():
        filtered, length_col = filter_trainable_lcs(
            nf, n_src=n_src, hash_range=hash_range, lc_col=LC_COL, id_col=ID_COL
        )
        counts = filtered.groupby("band", as_index=False, observed=True).agg(
            n_lc=("band", "size"),
            n_obs=(length_col, "sum"),
        )
        counts["split"] = split_name
        parts.append(counts)
    return pd.concat(parts, ignore_index=True)


_COUNT_META = pd.DataFrame(
    {
        "band": pd.Series(dtype=str),
        "n_lc": pd.Series(dtype=np.int64),
        "n_obs": pd.Series(dtype=np.int64),
        "split": pd.Series(dtype=str),
    }
)


def count_trainable_lcs(survey_config: SurveyConfig) -> pd.DataFrame:
    """Count trainable light curves per band and per split.

    A light curve is trainable if it survives the catalog quality and
    variability filters and has at least ``survey_config.n_src`` observations.

    Parameters
    ----------
    survey_config : SurveyConfig
        Survey configuration including catalog root, split boundaries, and n_src.

    Returns
    -------
    pd.DataFrame
        Long-format summary with columns "band" (including "all"), "split",
        "n_lc", and "n_obs".
    """
    catalog = rubin_dp_catalog_multi_band(
        root=survey_config.catalog_root,
        bands=survey_config.bands,
        obj=survey_config.obj,
        img=survey_config.img,
        phot=survey_config.phot,
        mode=survey_config.mode,
    )
    splits = {
        "train": survey_config.train_split,
        "val": survey_config.val_split,
        "test": survey_config.test_split,
    }
    partition_counts = catalog.map_partitions(
        _count_partition,
        n_src=survey_config.n_src,
        splits=splits,
        meta=_COUNT_META,
    ).compute()

    per_band = partition_counts.groupby(["band", "split"], as_index=False, observed=True).sum()
    per_split = per_band.groupby("split", as_index=False, observed=True).sum().assign(band="all")
    summary = pd.concat([per_band, per_split], ignore_index=True)
    summary["split"] = pd.Categorical(summary["split"], categories=SPLIT_NAMES, ordered=True)
    return summary.sort_values(["band", "split"], ignore_index=True)


def main(argv=None):
    """Count trainable light curves and print the summary."""
    args = parse_args(argv)
    survey_config = make_survey_config(args)

    client = None
    if args.n_workers is not None:
        from distributed import Client

        client = Client(n_workers=args.n_workers, threads_per_worker=1)
        print(f"Dask dashboard: {client.dashboard_link}")

    try:
        table = count_trainable_lcs(survey_config)
        # Print before tearing down the cluster: on shared clusters
        # client.close() can hang and raise TimeoutError, which would
        # otherwise discard the already-computed result.
        print(f"Trainable light curves for {survey_config.catalog_root}, n_src >= {survey_config.n_src}:")
        print(table.to_string(index=False))
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: ignoring error during Dask client shutdown: {exc!r}")


if __name__ == "__main__":
    main()
