# uncle-val

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/lincc-frameworks/uncle-val/smoke-test.yml)](https://github.com/lincc-frameworks/uncle-val/actions/workflows/smoke-test.yml)
[![DOI](https://zenodo.org/badge/1057502122.svg)](https://doi.org/10.5281/zenodo.23924873)

Code for the paper "Photometric Uncertainty Correction for Time-Domain Surveys: Vera C. Rubin Data Preview 2 Case Study", Malanchev et al., in prep.

## Repository structure

- `src/uncle_val/` — the Python package.
- `scripts/` — scripts for data preparation, model training, and the paper figures.
- `models/` — trained models.
- `docs/` — documentation and notebooks.
- `tests/` — unit tests.
- `data/` — input data, not included in the repository. Expected structure:

```
data/
├── dp1/
│   ├── object_collection/          # HATS collection of DP1 objects with forced sources
│   ├── dia_object_collection/      # HATS collection of DP1 DIA objects with forced sources
│   └── public_parquet/
│       ├── ccd_visit.parquet
│       └── ccd_visit_prepared.feather   # made by scripts/prepare_ccd_visit_feather.py
└── dp2/
    ├── object_collection/
    ├── dia_object_collection/
    └── public_parquet/
        ├── ccd_visit.parquet            # fetched by scripts/fetch_dp2_ccd_visit.py
        ├── ccd_visit_prepared.feather   # made by scripts/prepare_ccd_visit_feather.py
        └── detector_center.parquet
```

## Acknowledgements

This project is supported by Schmidt Sciences.
