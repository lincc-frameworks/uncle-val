#!/usr/bin/env python
"""Train the analytical FluxErrModel on one band of a Rubin DP catalog.

Corrected flux error is a quadrature sum of the reported error, a
fractional-flux term (b * flux_eff, smoothly floored at flux_cut), and a
constant floor (a0). Trained per band, on the full magnitude range by default
(the model's own flux floor regularizes the faint end, so an external
magnitude cut is no longer needed; pass --max-mag to still apply one). The
functional forms are heuristic and do not assert a physical origin.
"""

import argparse
from datetime import datetime
from pathlib import Path


def main():
    """Parse command-line arguments and train the FluxErrModel."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog-root", type=Path, required=True, help="Root of the DP HATS catalogs.")
    p.add_argument("--model", choices=["flux_err", "flux_err_poly"], default="flux_err")
    p.add_argument("--band", default="g")
    p.add_argument("--img", choices=["cal", "diff"], default="cal")
    p.add_argument("--obj", choices=["science", "dia"], default="science")
    p.add_argument(
        "--max-mag", type=float, default=None, help="Keep objects brighter than this. Default: no cut."
    )
    p.add_argument("--n-src", type=int, default=10)
    p.add_argument("--n-lcs", type=int, default=2_000_000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--val-batch-size", type=int, default=2048)
    p.add_argument("--max-val-size", type=int, default=65536)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--device", default="mps")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    from uncle_val.datasets.rubin_dp import rubin_dp_catalog_multi_band
    from uncle_val.learning.losses import (
        epps_pulley_whiten_loss,
        kl_divergence_whiten_loss,
        minus_ln_chi2_prob_loss,
    )
    from uncle_val.learning.models import FluxErrModel, FluxErrPolyModel
    from uncle_val.pipelines import ComputeConfig, TrainingConfig
    from uncle_val.pipelines.splits import dp1_config
    from uncle_val.pipelines.training_loop import training_loop

    survey_config = dp1_config(
        catalog_root=str(args.catalog_root), n_src=args.n_src, obj=args.obj, img=args.img
    )
    compute_config = ComputeConfig(n_workers=args.n_workers, device=args.device)
    training_config = TrainingConfig(
        compute_config=compute_config,
        n_lcs=args.n_lcs,
        train_batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        lr=args.lr,
        max_val_size=args.max_val_size,
        snapshot_factor=2.0,
        start_tfboard=False,
        run_feature_importance=False,
    )

    catalog = rubin_dp_catalog_multi_band(
        root=survey_config.catalog_root,
        bands=[args.band],
        obj=survey_config.obj,
        img=survey_config.img,
        phot=survey_config.phot,
        mode=survey_config.mode,
        ccd_visit_cols=None,
    )
    if args.max_mag is not None:
        catalog = catalog.query(f"object_mag < {args.max_mag}")
    catalog = catalog.map_partitions(lambda df: df[["id", "lc.x", "lc.err"]])

    model_cls = {"flux_err": FluxErrModel, "flux_err_poly": FluxErrPolyModel}[args.model]
    model = model_cls(["x", "err"]).to(device=compute_config.device)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("runs") / f"{args.model}_{args.band}_{args.img}_{now}"

    model_path = training_loop(
        catalog=catalog,
        columns=None,
        model=model,
        loss_fn=epps_pulley_whiten_loss(lmbd=2.0, soft=20.0, kind="accum"),
        val_losses={
            "Total Soften KL": kl_divergence_whiten_loss(soft=20.0, kind="accum", lmbd=None),
            "Total Soften -ln(p_chi2)": minus_ln_chi2_prob_loss(soft=20.0, kind="accum", lmbd=None),
        },
        output_dir=output_dir,
        model_name=f"{args.model}_{args.band}",
        survey_config=survey_config,
        training_config=training_config,
    )
    print(f"### Trained model saved to {model_path}")
    print("### fitted parameters:")

    if args.model == "flux_err":
        import torch

        for name in ("log_u0", "log_b", "log_a0"):
            val = float(torch.exp(getattr(model, name)).item())
            print(f"    {name.removeprefix('log_')} = {val:.6g}")
        print(f"    flux_cut = {10 ** float(model.log10_flux_cut.item()):.6g}")
    else:
        for name in ("c_00", "c_10", "c_01", "c_20", "c_11", "c_02"):
            val = float(getattr(model, name).item())
            print(f"    {name} = {val:.6g}")
        print(f"    flux_cut = {10 ** float(model.log10_flux_cut.item()):.6g}")
    print("### DONE")


if __name__ == "__main__":
    main()
