import pytest

from uncle_val.datasets.rubin_dp import rubin_dp_catalog_multi_band


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
    assert df.shape == (n_obj, 13)
    assert list(df.columns) == [
        "id",
        "coord_ra",
        "coord_dec",
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
