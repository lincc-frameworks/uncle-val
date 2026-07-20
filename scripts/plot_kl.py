#!/usr/bin/env python
"""KL-statistic distributions of constant light curves for a trained model.

Renders the per-band figure (uncorrected / model-corrected uncertainties) of the
per-light-curve KL statistic KL(N(mu, var) || N(0, 1)) on shared bins, with a
Monte-Carlo calibrated reference. Uncertainties are corrected by the model's
error-scaling factor only (no signal correction).
"""

import argparse
import dataclasses
from pathlib import Path

import torch

from uncle_val.pipelines import ComputeConfig
from uncle_val.pipelines.plotting import make_kl_distribution_plot
from uncle_val.pipelines.splits import SurveyConfig
from uncle_val.pipelines.train_on_rubin_dp import rubin_dp_catalog_and_columns


def main():
    """Parse command-line arguments and render the KL-statistic figure."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True, help="Trained-model directory.")
    p.add_argument("--model-file", default="MLPModel.pt", help="Model file within --model-dir.")
    p.add_argument("--catalog-root", type=Path, required=True, help="Override catalog_root (absolute).")
    p.add_argument("--split", default="test", help="'train', 'val', 'test', 'all', or 'none'.")
    p.add_argument("--obj", choices=["science", "dia"], default=None, help="Override object catalog type.")
    p.add_argument("--img", choices=["cal", "diff"], default=None, help="Override image type.")
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--subsample-partitions", type=float, default=None)
    p.add_argument("--output", type=Path, default=None, help="Output PDF; defaults into the model dir.")
    args = p.parse_args()

    split = None if args.split.lower() in ("none", "all") else args.split
    overrides = {"catalog_root": str(args.catalog_root)}
    if args.obj is not None:
        overrides["obj"] = args.obj
    if args.img is not None:
        overrides["img"] = args.img
    survey_config = dataclasses.replace(
        SurveyConfig.from_json(args.model_dir / "survey_config.json"),
        **overrides,
    )
    model_path = args.model_dir / args.model_file
    model = torch.load(model_path, weights_only=False, map_location="cpu")
    model.eval()
    _catalog, model_columns = rubin_dp_catalog_and_columns(model=model, survey_config=survey_config)

    output = args.output or args.model_dir / "plots" / f"model_{args.split}" / "kl_statistic.pdf"

    make_kl_distribution_plot(
        survey_config=survey_config,
        model_path=model_path,
        model_columns=model_columns,
        compute_config=ComputeConfig(n_workers=args.n_workers, device=args.device),
        split=split,
        subsample_partitions=args.subsample_partitions,
        output_path=output,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
