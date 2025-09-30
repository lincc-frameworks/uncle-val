from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import lsdb
import numpy as np
import pandas as pd
from upath import UPath

LSDB_BANDS = "ugrizy"


def _one_hot_encode_band(df, dtype: type = bool):
    for lsdb_band in LSDB_BANDS:
        col_name = f"is_{lsdb_band}_band"
        df[col_name] = np.asarray(df["band"] == lsdb_band, dtype=dtype)
    return df


def _filter_bad_obs(df, *, lc_col, flag_cols):
    cols = [f"{lc_col}.{flag_col}" for flag_col in flag_cols]
    expr = " and ".join(f"~{col}" for col in cols)

    filtered = df.query(expr)
    wo_flag_cols = filtered.drop(labels=cols, axis=1)
    return wo_flag_cols


def _rename_columns(df, *, lc_col, id_col, flux_col, flux_err_col):
    df = df.rename(columns={lc_col: "lc", id_col: "id"})
    for orig, new in {flux_col: "x", flux_err_col: "err"}.items():
        df[f"lc.{new}"] = df[f"lc.{orig}"]
        df["lc"] = df["lc"].nest.without_field(orig)
    return df


def _process_partition_single_band(
    df,
    *,
    lc_col,
    id_col,
    flux_col,
    flux_err_col,
    source_flag_cols,
    band,
):
    # Normalize column names
    df = _rename_columns(df, lc_col=lc_col, id_col=id_col, flux_col=flux_col, flux_err_col=flux_err_col)
    df = df.rename(columns={f"{band}_psgMag": "object_mag", f"{band}_extendedness": "extendedness"})

    # Filter by band
    df = df.query(f"lc.band == {band!r}")
    df = df.drop(labels=["lc.band"], axis=1)

    # Filter by bad observation flags
    df = _filter_bad_obs(df, lc_col="lc", flag_cols=source_flag_cols)

    # Drop empty light curves
    df = df.dropna(subset=["lc"])

    return df


def _split_light_curves_by_band(
    df,
    *,
    bands,
    lc_col,
):
    single_band_dfs = []

    for band in bands:
        single_band = df.query(f"{lc_col}.band == {band!r}")

        single_band = single_band.drop(labels=[f"{lc_col}.band"], axis=1)
        single_band["band"] = band

        single_band["object_mag"] = single_band[f"{band}_psfMag"]
        single_band = single_band.drop(columns=[f"{band}_psfMag" for band in LSDB_BANDS])

        single_band["extendedness"] = single_band[f"{band}_extendedness"]
        single_band = single_band.drop(columns=[f"{band}_extendedness" for band in LSDB_BANDS])

        single_band_dfs.append(single_band)

    if all(sb.empty for sb in single_band_dfs):
        return single_band_dfs[0]

    return pd.concat(single_band_dfs)


def _process_partition_multi_band(
    df,
    *,
    lc_col,
    id_col,
    flux_col,
    flux_err_col,
    source_flag_cols,
    bands,
):
    df = _rename_columns(df, lc_col=lc_col, id_col=id_col, flux_col=flux_col, flux_err_col=flux_err_col)

    # Filter by bad observation flags
    df = _filter_bad_obs(df, lc_col="lc", flag_cols=source_flag_cols)
    df = df.dropna(subset=["lc"])

    # Filter bands, and split light curves (rows) by band
    df = _split_light_curves_by_band(df, bands=bands, lc_col="lc")
    df = df.dropna(subset=["lc", "object_mag", "extendedness"])

    # Encode band names
    df = _one_hot_encode_band(df, dtype=bool)

    return df


def _open_catalog(
    root: Path | str,
    *,
    bands: Sequence[str],
    obj: Literal["science"] | Literal["dia"],
    img: Literal["cal"] | Literal["diff"],
    phot: Literal["PSF"],
    mode: Literal["forced"],
) -> tuple[lsdb.Catalog, dict[str, str | list[str]]]:
    """Open right catalog with right columns, no filtering.
    Also returns column specs dict.
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

    obj_mag_cols = [f"{band}_psfMag" for band in bands]

    obj_extendedness_cols = [f"{band}_extendedness" for band in bands]

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
        + obj_mag_cols
        + obj_extendedness_cols,
    )
    return catalog, {
        "id_col": id_col,
        "flux_col": flux_col,
        "flux_err_col": flux_err_col,
        "source_col": source_col,
        "flux_flag_col": flux_flag_col,
        "other_flag_cols": other_flag_cols,
    }


def dp1_catalog_single_band(
    root: Path | str,
    *,
    band: str,
    obj: Literal["science"] | Literal["dia"],
    img: Literal["cal"] | Literal["diff"],
    phot: Literal["PSF"],
    mode: Literal["forced"],
) -> lsdb.Catalog:
    """Rubin DP1 LSDB Catalog filtered for single-band DP1 sources

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
    catalog, col_names = _open_catalog(
        root,
        bands=[band],
        obj=obj,
        img=img,
        phot=phot,
        mode=mode,
    )

    mapped_catalog = catalog.map_partitions(
        _process_partition_single_band,
        lc_col=col_names["source_col"],
        source_flag_cols=[col_names["flux_flag_col"]] + col_names["other_flag_cols"],
        id_col=col_names["id_col"],
        flux_col=col_names["flux_col"],
        flux_err_col=col_names["flux_err_col"],
        band=band,
    )
    return mapped_catalog


def dp1_catalog_multi_band(
    root: Path | str,
    *,
    bands: Sequence[str] = LSDB_BANDS,
    obj: Literal["science"] | Literal["dia"],
    img: Literal["cal"] | Literal["diff"],
    phot: Literal["PSF"],
    mode: Literal["forced"],
):
    """Rubin DP1 LSDB catalog, bands are one-hot encoded.

    The function brakes light curves by passband, and adding
    "is_u_band", "is_g_band", etc. columns. It replaces "u_psfMag", etc
    columns with a single "object_mag" column. Rows with Null "object_mag"
    values are dropped.
    Similarly for "u_extendedness", etc, we rename it to "extendedness", and
    drop Null values.

    It filters by source's photometric flags, u_psfFlux_flag, etc.,
    and u_extendedness_flag, etc.

    It also "normalizes"
    column names:
    - "source" nested columns is "lc"
    - source flux is "x"
    - source flux error is "err"
    - object ID is "id"

    Parameters
    ----------
    root : Path or str
        Path to the root folder of dp1 HATS catalogs, e.g. one having
        object_collection and dia_object_collection subfolders.
    bands : str or list of str
        Bands to use, should be subset of ugrizy
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
    input_bands = frozenset(bands)
    if not input_bands.issubset(LSDB_BANDS):
        raise ValueError(f"Some of the given bands ({bands}) are not in {LSDB_BANDS}")
    bands = [band for band in LSDB_BANDS if band in input_bands]

    catalog, col_names = _open_catalog(
        root,
        bands=bands,
        obj=obj,
        img=img,
        phot=phot,
        mode=mode,
    )

    mapped_catalog = catalog.map_partitions(
        _process_partition_multi_band,
        lc_col=col_names["source_col"],
        source_flag_cols=[col_names["flux_flag_col"]] + col_names["other_flag_cols"],
        id_col=col_names["id_col"],
        flux_col=col_names["flux_col"],
        flux_err_col=col_names["flux_err_col"],
        bands=bands,
    )
    return mapped_catalog
