#!/usr/bin/env python3

import argparse
from pathlib import Path
from shutil import copyfile

import lsdb
import numpy as np
from dask.distributed import Client
from farmhash import Fingerprint32

PARENT_DIR = Path(__file__).resolve().parent


def get_dp1_data_dir():
    """Default location of DP1 data"""
    path = PARENT_DIR.parent.parent / "data" / "dp1"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def get_test_data_dir():
    """Default destination of test data"""
    path = PARENT_DIR.parent / "data" / "dp1"
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args(argv):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser("Create test data from DP1 catalogs")
    parser.add_argument("-i", "--input-root", type=Path, default=get_dp1_data_dir())
    parser.add_argument("-o", "--output-root", type=Path, default=get_test_data_dir())
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of Dask workers")
    return parser.parse_args(argv)


def prepare_hats_catalog(args, collection):
    """Create test catalogs from original catalogs"""
    print(f"Preparing {collection}")

    rng = np.random.default_rng(Fingerprint32(str(collection)))
    output_path = args.output_root / collection

    catalog = lsdb.open_catalog(args.input_root / collection, columns="all")
    subset = catalog.partitions[rng.integers(catalog.npartitions, size=8)]
    heads = subset.map_partitions(lambda df: df.head(10))
    heads.write_catalog(output_path, overwrite=True)

    # Delete giant FITS files
    for fits in output_path.glob("*/*.fits"):
        fits.unlink()


def prepare_ccd_visit(args):
    """Copy CCD Visit table to test directory"""
    source = args.input_root / "public_parquet" / "ccd_visit.parquet"
    dest = args.output_root / "public_parquet" / "ccd_visit.parquet"
    print(f"Copying {source} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    copyfile(source, dest)


def main(argv=None):
    """Run test data preparation"""
    args = parse_args(argv)

    with Client(n_workers=args.num_workers) as client:
        print(f"Dask dashboard: {client.dashboard_link}")
        for collection in ["dia_object_collection", "object_collection"]:
            prepare_hats_catalog(args, collection)

    prepare_ccd_visit(args)


if __name__ == "__main__":
    main()
