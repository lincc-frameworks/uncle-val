import uncle_val


def test_version():
    """Check to see that we can get the package version"""
    assert uncle_val.__version__ is not None
