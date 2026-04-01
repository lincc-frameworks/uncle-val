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
        "--collection",
        type=str,
        required=True,
        help="The full collection name (e.g., 'LSSTCam/runs/DRP/v30_0_4/DM-54249').",
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


def get_uri(*, collection, config, **_kwargs):
    """Get LSST URI object for the visit_detector_table parquet file."""
    from lsst.daf.butler import Butler

    butler = Butler(config, collections=collection)
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
