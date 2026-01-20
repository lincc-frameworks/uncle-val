#!/usr/bin/env python

import argparse
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Make a parquet file with LSSTCam detector center coordinates. "
        "Requires LSST stack environment."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "dp2" / "public_parquet" / "detector_center.parquet",
        help="The local path to save the parquet file "
        "(default: 'data/dp2/public_parquet/detector_center.parquet').",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Produce detector centers parquet file"""
    from lsst.afw import cameraGeom
    from lsst.obs.lsst import LsstCam

    arg = parse_args(argv)

    camera = LsstCam().getCamera()
    data = defaultdict(list)
    for det in camera:
        data["detector_id"].append(det.getId())

        field_angle = det.getCenter(cameraGeom.FIELD_ANGLE)
        data["field_angle_x"].append(field_angle.x)
        data["field_angle_y"].append(field_angle.y)

        focal_plane = det.getCenter(cameraGeom.FOCAL_PLANE)
        data["focal_plane_x"].append(focal_plane.x)
        data["focal_plane_y"].append(focal_plane.y)

    table = pa.table(data)
    arg.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, arg.output)


if __name__ == "__main__":
    main()
