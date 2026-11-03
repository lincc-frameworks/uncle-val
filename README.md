# uncle-val

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/lincc-frameworks/uncle-val/smoke-test.yml)](https://github.com/lincc-frameworks/uncle-val/actions/workflows/smoke-test.yml)

Code for the paper "Photometric Uncertainty Correction for Time-Domain Surveys: Vera C. Rubin Data Preview 2 Case Study", Malanchev et al., in prep.

## Repository structure

- `src/uncle_val/` — the Python package:
  - `datasets/` — light curve datasets: Rubin DP catalogs streamed with LSDB, in-memory and synthetic ones;
  - `learning/` — uncertainty correction models, loss functions, training, and feature importance;
  - `pipelines/` — end-to-end training and validation pipelines for Rubin DP1 and DP2, and plotting;
  - `utils/` — magnitude–flux conversions, hashing for the train/validation/test split, run provenance, Dask client;
  - `stat_tests.py` — normality tests (Kolmogorov–Smirnov, Anderson–Darling, Epps–Pulley, etc.) of the whitened signal.
- `scripts/` — command-line scripts for training the models and making the paper figures.
- `models/` — trained models, with their configurations and diagnostic plots, including the DP2 models used for the corrected uncertainties in the DP2 HATS catalogs.
- `docs/` — documentation and notebooks, including pre-executed DP1 and DP2 analysis notebooks in `docs/pre_executed/`.
- `tests/` — unit tests and test data.

## Acknowledgements

This project is supported by Schmidt Sciences.
