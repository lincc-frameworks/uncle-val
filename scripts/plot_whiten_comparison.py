#!/usr/bin/env python
"""Whitened-signal distributions of several trained models, overlaid.

Renders one figure with a panel over all object magnitudes and one per
requested object magnitude bin; each panel overlays the whitened-signal
histograms of all models in one band and the standard normal distribution.
Used to compare models trained with different loss functions on the same data.
"""

import argparse
import dataclasses
from pathlib import Path

import torch

from uncle_val.pipelines import ComputeConfig
from uncle_val.pipelines.plotting import make_whiten_comparison_plot
from uncle_val.pipelines.splits import SurveyConfig
from uncle_val.pipelines.train_on_rubin_dp import rubin_dp_catalog_and_columns


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-dir",
        type=Path,
        nargs="+",
        required=True,
        help="Trained-model directories, each holding the model file and survey_config.json. "
        "The data selection is taken from the first one.",
    )
    p.add_argument(
        "--label", nargs="+", required=True, help="Legend labels, one per --model-dir, in the same order."
    )
    p.add_argument("--model-file", default="MLPModel.pt", help="Model file within each --model-dir.")
    p.add_argument("--catalog-root", type=Path, required=True, help="Override catalog_root (absolute).")
    p.add_argument("--split", default="test", help="'train', 'val', 'test', 'all', or 'none'.")
    p.add_argument("--band", default="z", help="Band to plot.")
    p.add_argument(
        "--object-mags",
        type=float,
        nargs="*",
        default=[18.0, 21.0],
        help="Object magnitudes to add a magnitude-bin panel for.",
    )
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--subsample-partitions", type=float, default=None)
    p.add_argument("--output", type=Path, required=True, help="Output figure path.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Render the overlaid whitened-signal distributions."""
    args = parse_args(argv)
    if len(args.label) != len(args.model_dir):
        raise ValueError("--label must have one entry per --model-dir")

    split = None if args.split.lower() in ("none", "all") else args.split
    survey_config = dataclasses.replace(
        SurveyConfig.from_json(args.model_dir[0] / "survey_config.json"), catalog_root=str(args.catalog_root)
    )

    models = {}
    for label, model_dir in zip(args.label, args.model_dir, strict=True):
        model_path = model_dir / args.model_file
        model = torch.load(model_path, weights_only=False, map_location="cpu")
        model.eval()
        _catalog, model_columns = rubin_dp_catalog_and_columns(
            model=model, survey_config=survey_config, bands=None
        )
        models[label] = (model_path, model_columns)

    make_whiten_comparison_plot(
        models=models,
        survey_config=survey_config,
        split=split,
        band=args.band,
        object_mags=args.object_mags,
        compute_config=ComputeConfig(n_workers=args.n_workers, device=args.device),
        subsample_partitions=args.subsample_partitions,
        output_path=args.output,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
