#!/usr/bin/env python
"""Whitened-signal and added-magnitude-error density figures for a trained model.

Renders the per-band twin-heatmap whitened-signal-density figure (uncorrected
vs model-corrected) and the per-band added-magnitude-error-density figure
(model-corrected only), writing ``whiten_density.pdf`` and
``addmagerr_density.pdf``.
"""

import argparse
import dataclasses
import glob
from pathlib import Path

import torch

from uncle_val.pipelines import ComputeConfig
from uncle_val.pipelines.plotting import (
    make_addmagerr_density_plot,
    make_whiten_density_plot,
    selection_filter,
)
from uncle_val.pipelines.splits import SurveyConfig
from uncle_val.pipelines.train_on_rubin_dp import rubin_dp_catalog_and_columns


def _resolve_band_model_paths(pattern: str, model_file: str, bands: str) -> dict[str, Path]:
    """Resolve a ``--model-dir-pattern`` template into a per-band model-path dict.

    For each band in ``bands``, formats ``pattern`` with that band, glob-matches
    it, and picks the lexicographically-last (most recent) match, since
    timestamped run directories sort correctly lexicographically.
    """
    resolved = {}
    for band in bands:
        band_pattern = pattern.format(band=band)
        matches = sorted(Path(match) for match in glob.glob(band_pattern))
        if not matches:
            raise ValueError(f"No model directory found for band {band!r} matching pattern {band_pattern!r}")
        resolved[band] = matches[-1] / model_file.format(band=band)
    return resolved


def main():
    """Parse command-line arguments and render the density figures."""
    p = argparse.ArgumentParser(description=__doc__)
    model_dir_group = p.add_mutually_exclusive_group(required=True)
    model_dir_group.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Trained-model directory, one model shared across all bands.",
    )
    model_dir_group.add_argument(
        "--model-dir-pattern",
        default=None,
        help=(
            "Glob pattern with a literal '{band}' placeholder, resolved per band in "
            "--bands (picking the lexicographically-last match if several), e.g. "
            "'models/flux_err_{band}_cal_*'."
        ),
    )
    p.add_argument(
        "--model-file",
        default=None,
        help=(
            "Model file within --model-dir (or, with --model-dir-pattern, within each "
            "resolved band directory; '{band}' is substituted if present). Defaults to "
            "'MLPModel.pt' for --model-dir, 'flux_err_{band}.pt' for --model-dir-pattern."
        ),
    )
    p.add_argument("--catalog-root", type=Path, required=True, help="Override catalog_root (absolute).")
    p.add_argument("--split", default="test", help="'train', 'val', 'test', 'all', or 'none'.")
    p.add_argument("--obj", choices=["science", "dia"], default=None, help="Override object catalog type.")
    p.add_argument("--img", choices=["cal", "diff"], default=None, help="Override image type.")
    p.add_argument("--bands", default=None, help="Bands to plot, e.g. 'g' or 'ugrizy'. Defaults to all.")
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--subsample-partitions", type=float, default=None)
    p.add_argument(
        "--non-extended-only",
        action="store_true",
        help="Keep only point sources, extendedness == 0. Default: keep all objects.",
    )
    p.add_argument(
        "--max-mag", type=float, default=None, help="Keep objects brighter than this. Default: no cut."
    )
    p.add_argument(
        "--gr-color",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Keep objects with LOW <= g-r < HIGH. Default: no colour cut.",
    )
    p.add_argument(
        "--cone",
        type=float,
        nargs=3,
        metavar=("RA", "DEC", "RADIUS_ARCSEC"),
        default=None,
        help="Restrict to a cone on the sky: RA and DEC in degrees, radius in arcseconds.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the two PDFs; defaults into the model dir.",
    )
    args = p.parse_args()

    bands = args.bands or "ugrizy"
    split = None if args.split.lower() in ("none", "all") else args.split

    if args.model_dir is not None:
        model_file = args.model_file or "MLPModel.pt"
        model_path = args.model_dir / model_file
        model_dir_for_config = args.model_dir
    else:
        model_file = args.model_file or "flux_err_{band}.pt"
        model_path = _resolve_band_model_paths(args.model_dir_pattern, model_file, bands)
        # Any resolved band directory carries the same survey_config.json / serves as
        # the output-path anchor; arbitrarily use the first band's.
        model_dir_for_config = next(iter(model_path.values())).parent

    overrides = {"catalog_root": str(args.catalog_root)}
    if args.obj is not None:
        overrides["obj"] = args.obj
    if args.img is not None:
        overrides["img"] = args.img
    survey_config = dataclasses.replace(
        SurveyConfig.from_json(model_dir_for_config / "survey_config.json"),
        **overrides,
    )

    if isinstance(model_path, dict):
        # All FluxErrModel checkpoints share input_names=["x", "err"], so any one
        # band's model gives the same model_columns as the others.
        sample_model = torch.load(next(iter(model_path.values())), weights_only=False, map_location="cpu")
        model_columns = [f"lc.{name}" for name in sample_model.input_names]
    else:
        model = torch.load(model_path, weights_only=False, map_location="cpu")
        model.eval()
        _catalog, model_columns = rubin_dp_catalog_and_columns(
            model=model, survey_config=survey_config, bands=args.bands
        )

    output_dir = args.output_dir or model_dir_for_config / "plots" / f"model_{args.split}"

    pre_filter_partition, selection_label = selection_filter(
        non_extended_only=args.non_extended_only,
        max_mag=args.max_mag,
        gr_color=tuple(args.gr_color) if args.gr_color is not None else None,
    )
    cone = tuple(args.cone) if args.cone is not None else None

    compute_config = ComputeConfig(n_workers=args.n_workers, device=args.device)
    common = dict(
        survey_config=survey_config,
        model_path=model_path,
        model_columns=model_columns,
        compute_config=compute_config,
        split=split,
        bands=bands,
        subsample_partitions=args.subsample_partitions,
        pre_filter_partition=pre_filter_partition,
        selection_label=selection_label,
        cone=cone,
    )

    whiten_output = output_dir / "whiten_density.pdf"
    make_whiten_density_plot(output_path=whiten_output, **common)
    print(f"Wrote {whiten_output}")

    addmagerr_output = output_dir / "addmagerr_density.pdf"
    make_addmagerr_density_plot(output_path=addmagerr_output, **common)
    print(f"Wrote {addmagerr_output}")


if __name__ == "__main__":
    main()
