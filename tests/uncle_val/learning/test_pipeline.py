from hyrax import Hyrax

import uncle_val.learning.lsdb_dataset as _  # noqa: F401


def test_run_pipeline():
    h = Hyrax()
    h.train()
