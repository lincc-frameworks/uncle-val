#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.parquet as pq


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-compute CCD visit feather file with detector polar coordinates."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to ccd_visit.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .feather path (default: same dir as input, ccd_visit.feather)",
    )
    parser.add_argument(
        "--detector-centers",
        type=Path,
        default=None,
        help="Path to detector_center.parquet for LSSTCam mode. "
        "Absent -> ComCam mode (9 detectors, 3x3 grid). "
        "Present -> LSSTCam mode (actual focal plane positions).",
    )
    return parser.parse_args(argv)


def _polar_from_xy(x, y):
    rho = np.hypot(x, y)
    angle = np.arctan2(y, x)
    cos_phi = np.cos(angle)
    sin_phi = np.sin(angle)
    return rho, cos_phi, sin_phi


def main(argv=None):
    """Pre-compute CCD visit feather file with detector polar coordinates."""
    from uncle_val.datasets.rubin_dp import _xy_encode_detector

    args = parse_args(argv)

    output = args.output
    if output is None:
        output = args.input.parent / "ccd_visit_prepared.feather"

    table = pq.read_table(args.input)

    # Patch seeing NaN -> column mean
    seeing = table.column("seeing")
    is_nan = pc.is_nan(seeing)
    mean_val = pc.mean(pc.if_else(is_nan, None, seeing)).as_py()
    seeing_patched = pc.if_else(is_nan, mean_val, seeing)
    table = table.set_column(table.schema.get_field_index("seeing"), "seeing", seeing_patched)

    if args.detector_centers is None:
        # ComCam mode: 9 detectors on a 3x3 grid
        detector_ids = table.column("detectorId").to_numpy(zero_copy_only=False)
        x, y = _xy_encode_detector(detector_ids)
    else:
        # LSSTCam mode: actual focal plane positions
        centers = pq.read_table(
            args.detector_centers, columns=["detector_id", "focal_plane_x", "focal_plane_y"]
        )
        table = table.join(
            centers, keys="detectorId", right_keys="detector_id", join_type="left outer", coalesce_keys=False
        )
        x = table.column("focal_plane_x").to_numpy(zero_copy_only=False)
        y = table.column("focal_plane_y").to_numpy(zero_copy_only=False)

    rho, cos_phi, sin_phi = _polar_from_xy(x, y)
    table = table.append_column("detector_rho", pa.array(rho))
    table = table.append_column("detector_cos_phi", pa.array(cos_phi))
    table = table.append_column("detector_sin_phi", pa.array(sin_phi))

    feather.write_feather(table, output, compression="uncompressed")
    print(f"Written to {output}")


if __name__ == "__main__":
    main()
