#!/usr/bin/env python
"""Train a constant-magnitude-error model on a Rubin DP catalog.

A constant systematic magnitude error is added in quadrature to the reported
photon-noise magnitude error,

    new_mag_err = hypot(mag_err, 1e-2 * addition_centi_mag_err)
    u = magerr2fluxerr(new_mag_err) / flux_err

By default a single band is trained, giving one trainable parameter. With
--per-band, all of --bands are trained together in one run, giving one
parameter per band, selected by the catalog's one-hot band columns.

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
from uncle_val.pipelines import (
    ComputeConfig,
    TrainingConfig,
    run_rubin_dp_constant_magerr,
    run_rubin_dp_per_band_constant_magerr,
)
from uncle_val.pipelines.splits import SurveyConfig, dp1_config, dp2_config

SURVEY_CONFIGS = {"dp1": dp1_config, "dp2": dp2_config}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog-root", type=Path, required=True, help="Root of the DP HATS catalogs.")
    p.add_argument("--survey", choices=list(SURVEY_CONFIGS), default="dp2")
    p.add_argument("--band", default=None, help="Band to train on, single-band mode. Default: r.")
    p.add_argument(
        "--per-band",
        action="store_true",
        help="Train one systematic magnitude error per band, over --bands, in a single run.",
    )
    p.add_argument(
        "--bands",
        default="ugrizy",
        help="Bands to fit with --per-band, e.g. 'gri'. Ignored without --per-band.",
    )
    p.add_argument("--img", choices=["cal", "diff"], default="cal")
    p.add_argument("--obj", choices=["science", "dia"], default="science")
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
    args = p.parse_args(argv)

    if args.per_band and args.band is not None:
        p.error("--band is single-band mode only; with --per-band use --bands")
    if args.band is None:
        args.band = "r"
    return args


def build_survey_config(args: argparse.Namespace) -> SurveyConfig:
    """Build the survey config for the requested survey and band(s)."""
    bands = tuple(args.bands) if args.per_band else (args.band,)
    return SURVEY_CONFIGS[args.survey](
        catalog_root=str(args.catalog_root),
        n_src=args.n_src,
        val_start=args.val_start,
        test_start=args.test_start,
        bands=bands,
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
    bands = args.bands if args.per_band else args.band
    return Path("runs") / f"constant_magerr_{args.survey}_{bands}_{args.obj}_{args.img}_{now}"


def print_fitted_params(model_path: Path) -> None:
    """Load the trained model and print its fitted magnitude error(s)."""
    model = torch.load(model_path, weights_only=False, map_location="cpu")
    centi_mags = model.addition_centi_mag_err.detach().flatten().tolist()
    labels = getattr(model, "bands", None) or [""] * len(centi_mags)
    print("### fitted parameters:")
    for label, centi_mag in zip(labels, centi_mags, strict=True):
        name = f"addition_centi_mag_err[{label}]" if label else "addition_centi_mag_err"
        print(f"    {name} = {centi_mag:.6g}  ({1e-2 * centi_mag:.6g} mag)")


def main(argv: list[str] | None = None) -> None:
    """Train the constant mag-err model and report the fitted magnitude error(s)."""
    args = parse_args(argv)

    common = {
        "non_extended_only": args.non_extended_only,
        "max_mag": args.max_mag,
        "gr_color": tuple(args.gr_color) if args.gr_color is not None else None,
        "cone": tuple(args.cone) if args.cone is not None else None,
        "output_dir": output_dir(args),
        "loss_fn": epps_pulley_whiten_loss(lmbd=2.0, soft=20.0, kind="accum"),
        "val_losses": {
            "Total Soften KL": kl_divergence_whiten_loss(soft=20.0, kind="accum", lmbd=None),
            "Total Soften -ln(p_chi2)": minus_ln_chi2_prob_loss(soft=20.0, kind="accum", lmbd=None),
        },
        "survey_config": build_survey_config(args),
        "training_config": build_training_config(args),
    }
    if args.per_band:
        model_path = run_rubin_dp_per_band_constant_magerr(**common)
    else:
        model_path = run_rubin_dp_constant_magerr(band=args.band, **common)
    print(f"### Trained model saved to {model_path}")
    print_fitted_params(model_path)
    print("### DONE")


if __name__ == "__main__":
    main()
