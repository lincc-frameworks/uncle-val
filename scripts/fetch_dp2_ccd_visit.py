#!/usr/bin/env python

import argparse
from pathlib import Path
from shutil import copyfileobj


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch DP2 CCD visit parquet file from the Butler repository. "
        "Requires LSST stack environment."
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="The run identifier for the data (e.g., '20250417_20250921').",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="The version (e.g., for weekly, 'w_2025_49').",
    )
    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="The collection name (e.g., 'DM-53545').",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="dp2_prep",
        help="The Butler configuration to use (default: 'dp2_prep').",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "dp2" / "public_parquet" / "ccd_visit.parquet",
        help="The local path to save the parquet file "
        "(default: 'data/dp2/public_parquet/ccd_visit.parquet').",
    )
    return parser.parse_args(argv)


def get_uri(*, run, version, collection, config, **_kwargs):
    """Get LSST URI object for the visit_detector_table parquet file."""
    from lsst.daf.butler import Butler

    collections = f"LSSTCam/runs/DRP/{run}/{version}/{collection}"
    butler = Butler(config, collections=collections)
    uri = butler.getURI("visit_detector_table", dataId={"instrument": "LSSTCam"})
    return uri


def copy_to_local(uri, local_path):
    """Copy file from LSST URI to local path."""
    with uri.open("rb") as src, local_path.open("wb") as dest:
        copyfileobj(src, dest)


def main(argv=None):
    """Fetch DP2 CCD visit parquet file and save it locally."""
    arg = parse_args(argv)
    uri = get_uri(**vars(arg))
    copy_to_local(uri, arg.destination)


if __name__ == "__main__":
    main()
