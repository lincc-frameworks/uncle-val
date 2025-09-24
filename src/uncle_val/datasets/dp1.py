from pathlib import Path
from typing import Literal

import lsdb
from upath import UPath


def _filter_bad_obs(df, *, lc_col, flag_cols):
    cols = [f"{lc_col}.{flag_col}" for flag_col in flag_cols]
    expr = " and ".join(f"~{col}" for col in cols)

    filtered = df.query(expr)
    wo_flag_cols = filtered.drop(labels=cols, axis=1)
    return wo_flag_cols


def _process_partition(
    df,
    *,
    lc_col,
    source_flag_cols,
    band,
):
    # Filter by band
    df = df.query(f"{lc_col}.band == '{band}'")
    df = df.drop(labels=[f"{lc_col}.band"], axis=1)

    # Filter by bad observation flags
    df = _filter_bad_obs(df, lc_col=lc_col, flag_cols=source_flag_cols)

    # Drop empty light curves
    df = df.dropna(subset=[lc_col])

    return df


def dp1_catalog_single_band(
    root: Path | str,
    *,
    band: str,
    obj: Literal["science"] | Literal["dia"],
    img: Literal["cal"] | Literal["diff"],
    phot: Literal["PSF"],
    mode: Literal["forced"],
) -> lsdb.Catalog:
    """LSDB Catalog filtered for single-band DP1 sources

    Parameters
    ----------
    root : Path or str
        Path to the root folder of dp1 HATS catalogs, e.g. one having
        object_collection and dia_object_collection subfolders.
    band : str
        Band to use, one of ugrizy
    obj : str
        Type of object catalog, "science" or "dia".
    img : str
        Type of image used for photometry, "cal" (calibrated) or
        "diff" (subtracted).
    phot : str
        Type of photometry, "PSF".
    mode : str
        Type of source coordinate mode, "forced".

    Returns
    -------
    lsdb.Catalog
        Filtered LSDB Catalog object.
    """
    root = UPath(root)

    match obj:
        case "science":
            catalog_name = "object_collection"
            id_col = "objectId"
        case _:
            raise NotImplementedError(f"obg '{obj}' not implemented")

    match mode, obj:
        case "forced", "science":
            source_col = "objectForcedSource"
        case _:
            raise NotImplementedError(f"mode '{mode}' and obj '{obj}' are not supported")

    match img:
        case "cal":
            phot_col_prefix = "psf"
        case "diff":
            phot_col_prefix = "psfDiff"
        case _:
            raise NotImplementedError(f"img '{img}' is not supported")

    if phot != "PSF":
        raise NotImplementedError(f"img '{img}' is not supported")

    flux_col = f"{phot_col_prefix}Flux"
    flux_err_col = f"{flux_col}Err"
    flux_flag_col = f"{flux_col}_flag"

    obj_mag_col = f"{band}_psfMag"

    other_flag_cols = [
        "pixelFlags_suspect",
        "pixelFlags_saturated",
        "pixelFlags_cr",
        "pixelFlags_bad",
    ]

    catalog = lsdb.open_catalog(
        root / catalog_name,
        columns=[
            id_col,
            f"{source_col}.band",
            f"{source_col}.{flux_col}",
            f"{source_col}.{flux_err_col}",
            f"{source_col}.{flux_flag_col}",
        ]
        + [f"{source_col}.{flag_col}" for flag_col in other_flag_cols]
        + [obj_mag_col],
    )
    mapped_catalog = catalog.map_partitions(
        _process_partition,
        lc_col=source_col,
        source_flag_cols=[flux_flag_col] + other_flag_cols,
        band=band,
    )
    return mapped_catalog
