#!/usr/bin/env python
"""Train ConstantMagErrModel on one band of a Rubin DP catalog.

The model has a single trainable parameter: a constant systematic magnitude
error added in quadrature to the reported photon-noise magnitude error,

    new_mag_err = hypot(mag_err, 1e-2 * addition_centi_mag_err)
    u = magerr2fluxerr(new_mag_err) / flux_err

Defaults target the DP2 science Object / forcedSource catalog, r band, on
gondor's second GPU; pass --non-extended-only to keep point sources only.
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch

from uncle_val.learning.losses import (
    epps_pulley_whiten_loss,
    kl_divergence_whiten_loss,
    minus_ln_chi2_prob_loss,
)
from uncle_val.pipelines import ComputeConfig, TrainingConfig, run_rubin_dp_constant_magerr
from uncle_val.pipelines.splits import SurveyConfig, dp1_config, dp2_config

SURVEY_CONFIGS = {"dp1": dp1_config, "dp2": dp2_config}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog-root", type=Path, required=True, help="Root of the DP HATS catalogs.")
    p.add_argument("--survey", choices=list(SURVEY_CONFIGS), default="dp2")
    p.add_argument("--band", default="r")
    p.add_argument("--img", choices=["cal", "diff"], default="cal")
    p.add_argument("--obj", choices=["science", "dia"], default="science")
    p.add_argument(
        "--non-extended-only",
        action="store_true",
        help="Keep only point sources, extendedness == 0. Default: keep all objects.",
    )
    p.add_argument("--val-start", type=float, default=0.7, help="Train/val hash boundary.")
    p.add_argument("--test-start", type=float, default=0.85, help="Val/test hash boundary.")
    p.add_argument("--n-src", type=int, default=10)
    p.add_argument("--n-lcs", type=int, default=30_000_000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--val-batch-size", type=int, default=16 << 10)
    p.add_argument("--max-val-size", type=int, default=128 << 10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args(argv)


def build_survey_config(args: argparse.Namespace) -> SurveyConfig:
    """Build the survey config for the requested survey and band."""
    return SURVEY_CONFIGS[args.survey](
        catalog_root=str(args.catalog_root),
        n_src=args.n_src,
        val_start=args.val_start,
        test_start=args.test_start,
        bands=(args.band,),
        obj=args.obj,
        img=args.img,
    )


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    """Build the training config, TensorBoard and feature importance are off."""
    return TrainingConfig(
        compute_config=ComputeConfig(n_workers=args.n_workers, device=args.device),
        n_lcs=args.n_lcs,
        train_batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        lr=args.lr,
        max_val_size=args.max_val_size,
        snapshot_factor=2.0,
        start_tfboard=False,
        run_feature_importance=False,
    )


def output_dir(args: argparse.Namespace) -> Path:
    """Run directory, either given by the user or a timestamped default."""
    if args.output_dir is not None:
        return args.output_dir
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"constant_magerr_{args.survey}_{args.band}_{args.obj}_{args.img}_{now}"


def print_fitted_params(model_path: Path) -> None:
    """Load the trained model and print its single fitted parameter."""
    model = torch.load(model_path, weights_only=False, map_location="cpu")
    centi_mag = float(model.addition_centi_mag_err.item())
    print("### fitted parameters:")
    print(f"    addition_centi_mag_err = {centi_mag:.6g}  ({1e-2 * centi_mag:.6g} mag)")


def main(argv: list[str] | None = None) -> None:
    """Train ConstantMagErrModel and report the fitted magnitude error."""
    args = parse_args(argv)

    model_path = run_rubin_dp_constant_magerr(
        band=args.band,
        non_extended_only=args.non_extended_only,
        output_dir=output_dir(args),
        loss_fn=epps_pulley_whiten_loss(lmbd=2.0, soft=20.0, kind="accum"),
        val_losses={
            "Total Soften KL": kl_divergence_whiten_loss(soft=20.0, kind="accum", lmbd=None),
            "Total Soften -ln(p_chi2)": minus_ln_chi2_prob_loss(soft=20.0, kind="accum", lmbd=None),
        },
        survey_config=build_survey_config(args),
        training_config=build_training_config(args),
    )
    print(f"### Trained model saved to {model_path}")
    print_fitted_params(model_path)
    print("### DONE")


if __name__ == "__main__":
    main()
