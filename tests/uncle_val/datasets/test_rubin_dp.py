import numpy as np
import pytest

from uncle_val.datasets.rubin_dp import rubin_dp_catalog_multi_band, rubin_dp_catalog_single_band


@pytest.mark.parametrize("img, n_obj, n_src", [("cal", 106, 631), ("diff", 112, 874)])
def test_rubin_dp_catalog_multi_band(rubin_dp_root, img, n_obj, n_src):
    """Test rubin_dp_catalog_multi_band()"""
    catalog = rubin_dp_catalog_multi_band(
        rubin_dp_root,
        obj="science",
        img=img,
        phot="PSF",
        mode="forced",
    )
    df = catalog.compute()
    assert df.shape == (n_obj, 14)
    assert list(df.columns) == [
        "id",
        "coord_ra",
        "coord_dec",
        "gr_color",
        "band",
        "object_mag",
        "extendedness",
        "is_u_band",
        "is_g_band",
        "is_r_band",
        "is_i_band",
        "is_z_band",
        "is_y_band",
        "lc",
    ]
    extra_source_cols = ["psfFlux"] if img == "diff" else []
    assert df["lc"].dtype.field_names == [
        "expTime",
        "seeing",
        "skyBg",
        "detector_rho",
        "detector_cos_phi",
        "detector_sin_phi",
        *extra_source_cols,
        "x",
        "err",
    ]
    flat_lc = df["lc"].nest.to_flat()
    assert len(flat_lc) == n_src
    if img == "diff":
        assert not flat_lc["psfFlux"].isna().any()


def test_rubin_dp_catalog_single_band_has_object_mag(rubin_dp_root):
    """The single-band catalog exposes object_mag, so magnitude cuts can be applied"""
    catalog = rubin_dp_catalog_single_band(
        rubin_dp_root,
        band="r",
        obj="science",
        img="cal",
        phot="PSF",
        mode="forced",
    )
    df = catalog.compute()
    assert "object_mag" in df.columns
    assert "r_psfMag" not in df.columns, "the per-band magnitude should have been renamed"
    assert df["object_mag"].notna().any()


def test_rubin_dp_catalog_gr_color_available_for_any_band(rubin_dp_root):
    """g and r magnitudes are read whatever band is trained on, so gr_color exists"""
    catalog = rubin_dp_catalog_single_band(
        rubin_dp_root,
        band="i",
        obj="science",
        img="cal",
        phot="PSF",
        mode="forced",
    )
    df = catalog.compute()
    assert "gr_color" in df.columns
    assert df["gr_color"].notna().any()
    # the per-band magnitudes are consumed, not left lying around
    assert not [col for col in df.columns if col.endswith("_psfMag")]


def test_rubin_dp_catalog_cone_search(rubin_dp_root):
    """A cone restricts the catalog to objects within the given radius"""
    kwargs = dict(obj="science", img="cal", phot="PSF", mode="forced")
    everything = rubin_dp_catalog_multi_band(rubin_dp_root, **kwargs).compute()

    # centre on a real object, so the cone is not empty
    center = everything.iloc[0]
    ra, dec = float(center["coord_ra"]), float(center["coord_dec"])

    radius_arcsec = 1.0
    coned = rubin_dp_catalog_multi_band(rubin_dp_root, cone=(ra, dec, radius_arcsec), **kwargs).compute()

    assert 0 < len(coned) < len(everything)

    separation = np.hypot(
        (coned["coord_ra"] - ra) * np.cos(np.radians(dec)),
        coned["coord_dec"] - dec,
    )
    assert (separation <= radius_arcsec / 3600.0).all()
