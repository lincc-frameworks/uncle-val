#!/usr/bin/env python
"""Whitened-signal histograms for a trained model, overall and in magnitude bins.

Renders the per-band histograms of the whitened signal ``z`` against the
``N(0, 1)`` reference: one figure over all magnitudes
(``hists_all_mags*.pdf``), one per requested object magnitude
(``hists_<mag>mag*.pdf``), and the whitened-signal and added-magnitude-error
scatter plots versus object magnitude. Uncorrected and model-corrected
versions are written to separate directories, so the pair can be compared.
"""

import argparse
import dataclasses
from pathlib import Path

import torch

from uncle_val.pipelines import ComputeConfig, make_plots
from uncle_val.pipelines.splits import SurveyConfig
from uncle_val.pipelines.train_on_rubin_dp import rubin_dp_catalog_and_columns


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Trained-model directory, holding the model file and survey_config.json.",
    )
    p.add_argument("--model-file", default="MLPModel.pt", help="Model file within --model-dir.")
    p.add_argument("--catalog-root", type=Path, required=True, help="Override catalog_root (absolute).")
    p.add_argument("--split", default="test", help="'train', 'val', 'test', 'all', or 'none'.")
    p.add_argument("--obj", choices=["science", "dia"], default=None, help="Override object catalog type.")
    p.add_argument("--img", choices=["cal", "diff"], default=None, help="Override image type.")
    p.add_argument(
        "--object-mags",
        type=float,
        nargs="*",
        default=[18.0, 21.0, 25.0],
        help="Object magnitudes to make a per-magnitude-bin histogram for.",
    )
    p.add_argument(
        "--n-samples", type=int, default=5, help="Light curves sampled per magnitude bin for the scatter."
    )
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--subsample-partitions", type=float, default=None)
    p.add_argument(
        "--non-extended-only",
        action="store_true",
        help="Keep only point sources, extendedness == 0. Default: keep all objects.",
    )
    p.add_argument(
        "--skip-uncorrected",
        action="store_true",
        help="Only render the model-corrected figures, skipping the uncorrected ones.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults into the model dir. Gets 'data' and 'model' subdirectories.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Render the whitened-signal histograms, uncorrected and model-corrected."""
    args = parse_args(argv)

    split = None if args.split.lower() in ("none", "all") else args.split

    overrides = {"catalog_root": str(args.catalog_root)}
    if args.obj is not None:
        overrides["obj"] = args.obj
    if args.img is not None:
        overrides["img"] = args.img
    survey_config = dataclasses.replace(
        SurveyConfig.from_json(args.model_dir / "survey_config.json"), **overrides
    )

    model_path = args.model_dir / args.model_file
    model = torch.load(model_path, weights_only=False, map_location="cpu")
    model.eval()
    _catalog, model_columns = rubin_dp_catalog_and_columns(
        model=model, survey_config=survey_config, bands=None
    )

    output_dir = args.output_dir or args.model_dir / "plots" / f"hists_{args.split}"

    common = dict(
        split=split,
        survey_config=survey_config,
        non_extended_only=args.non_extended_only,
        n_samples=args.n_samples,
        object_mags=list(args.object_mags),
        compute_config=ComputeConfig(n_workers=args.n_workers, device=args.device),
        subsample_partitions=args.subsample_partitions,
    )

    runs = [("model", model_path, model_columns)]
    if not args.skip_uncorrected:
        runs.insert(0, ("data", None, ("lc.x", "lc.err")))

    for name, path, columns in runs:
        make_plots(model_path=path, model_columns=columns, output_dir=output_dir / name, **common)
        print(f"Wrote {output_dir / name}")


if __name__ == "__main__":
    main()
