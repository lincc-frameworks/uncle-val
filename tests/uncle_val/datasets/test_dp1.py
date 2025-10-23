import pytest
from uncle_val.datasets.dp1 import dp1_catalog_multi_band


@pytest.mark.parametrize("img, n_obj", [("cal", 106), ("diff", 115)])
def test_dp1_catalog_multi_band(dp1_root, img, n_obj):
    """Test dp1_catalog_multi_band()"""
    catalog = dp1_catalog_multi_band(
        dp1_root,
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
    assert df["lc"].dtype.field_names == [
        "expTime",
        "seeing",
        "skyBg",
        "detector",
        "x",
        "err",
    ]
