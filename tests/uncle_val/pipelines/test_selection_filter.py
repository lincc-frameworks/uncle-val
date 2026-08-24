import pandas as pd
import pytest

from uncle_val.pipelines.plotting import selection_filter


@pytest.fixture
def objects():
    """Four objects spanning the extendedness, magnitude and colour cuts."""
    return pd.DataFrame(
        {
            "extendedness": [0.0, 1.0, 0.0, 0.0],
            "object_mag": [20.0, 20.0, 22.0, 20.0],
            "gr_color": [1.7, 1.7, 1.7, 0.2],
        }
    )


def test_selection_filter_without_cuts_is_none():
    """No cuts means no filter and nothing to label"""
    assert selection_filter() == (None, None)


def test_selection_filter_single_cut(objects):
    """A single cut keeps only the matching rows and labels itself"""
    filt, label = selection_filter(non_extended_only=True)
    assert label == "extendedness == 0.0"
    assert len(filt(objects)) == 3


def test_selection_filter_combines_cuts(objects):
    """Cuts combine, and the label states every one of them"""
    filt, label = selection_filter(non_extended_only=True, max_mag=21.0, gr_color=(1.5, 2.0))
    assert label == "extendedness == 0.0 and object_mag < 21.0 and 1.5 <= gr_color < 2.0"
    kept = filt(objects)
    # only the first object passes all three
    assert len(kept) == 1
    assert kept.index.tolist() == [0]
