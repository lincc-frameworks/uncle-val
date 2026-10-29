#!/usr/bin/env python
"""Train the multi-band MLP model on DP1 Object ForcedSource with a chosen loss function.

Reproduces the fiducial DP1 training setup of ``docs/pre_executed/dp1_object_forcedSource.ipynb``
(same model inputs, penalty, softening, batch sizes, learning rate, and validation losses),
with the training loss replaced by the chi2 or KL divergence loss, accumulated over the batch
("accum") or averaged over light curves ("mean"). Used to compare alternative loss setups with
the fiducial Epps-Pulley loss.
"""

import argparse
from datetime import datetime
from pathlib import Path


def main():
    """Parse command-line arguments and train the MLP model."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog-root", type=Path, required=True, help="Root of the DP1 HATS catalogs.")
    p.add_argument("--loss", choices=["chi2", "kl", "ep"], required=True)
    p.add_argument("--kind", choices=["accum", "mean"], default="accum")
    p.add_argument("--lmbd", type=float, default=2.0, help="Penalty strength.")
    p.add_argument("--soft", type=float, default=20.0, help="Softening scale.")
    p.add_argument("--n-src", type=int, default=10)
    p.add_argument("--n-lcs", type=int, default=300_000_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--val-batch-size", type=int, default=16 << 10)
    p.add_argument("--max-val-size", type=int, default=128 << 10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    from uncle_val.learning.losses import (
        epps_pulley_whiten_loss,
        kl_divergence_whiten_loss,
        minus_ln_chi2_prob_loss,
    )
    from uncle_val.learning.models import MLPModel
    from uncle_val.pipelines import ComputeConfig, TrainingConfig, train_on_rubin_dp
    from uncle_val.pipelines.splits import dp1_config

    survey_config = dp1_config(catalog_root=str(args.catalog_root), n_src=args.n_src)
    compute_config = ComputeConfig(n_workers=args.n_workers, device=args.device)
    training_config = TrainingConfig(
        compute_config=compute_config,
        n_lcs=args.n_lcs,
        train_batch_size=args.batch_size,
        max_val_size=args.max_val_size,
        snapshot_factor=2.0,
        val_batch_size=args.val_batch_size,
        lr=args.lr,
        start_tfboard=False,
    )

    loss_factory = {
        "chi2": minus_ln_chi2_prob_loss,
        "kl": kl_divergence_whiten_loss,
        "ep": epps_pulley_whiten_loss,
    }[args.loss]
    loss_fn = loss_factory(lmbd=args.lmbd, soft=args.soft, kind=args.kind)

    model = MLPModel(
        input_names=[
            "x",
            "err",
            "extendedness",
            "skyBg",
            "seeing",
            "expTime",
            "detector_rho",
            "detector_cos_phi",
            "detector_sin_phi",
        ]
        + [f"is_{band}_band" for band in survey_config.bands],
        outputs_s=False,
    )

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("runs") / f"dp1_mlp_{args.loss}_{args.kind}_{now}"

    model_path, model_columns = train_on_rubin_dp(
        model=model,
        loss_fn=loss_fn,
        val_losses={
            "Total Soften KL": kl_divergence_whiten_loss(soft=args.soft, kind="accum", lmbd=None),
            "Total Soften -ln(p_chi2)": minus_ln_chi2_prob_loss(soft=args.soft, kind="accum", lmbd=None),
        },
        output_dir=output_dir,
        survey_config=survey_config,
        training_config=training_config,
    )
    print(f"### Trained model saved to {model_path}")
    print(f"### Model columns: {model_columns}")
    print("### DONE")


if __name__ == "__main__":
    main()
