# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

`uncle-val` is a scientific Python library for photometric uncertainty calibration for the Vera Rubin Observatory / LSST. It learns an "Uncle function" correction factor `u` such that `corrected_err = u * reported_err`, ensuring whitened photometric residuals of constant stars follow `N(0, 1)`.

## Commands

**Development setup:**
```bash
./.setup_dev.sh          # Install package + pre-commit hooks
pip install -e '.[dev]'  # Manual dev install (use single quotes on macOS)
```

**Tests:**
```bash
python -m pytest                             # All tests (includes doctests in src/ and docs/)
python -m pytest -m 'not long'              # Skip slow tests (used by pre-commit)
python -m pytest tests/uncle_val/path/test_file.py::test_name  # Single test
python -m pytest --cov=uncle_val --cov-report=xml  # With coverage (CI mode)
```

**Lint/format:**
```bash
ruff check .    # Lint
ruff format .   # Format
```

**Docs:**
```bash
sphinx-build -M html ./docs ./_readthedocs
```

## Architecture

The codebase has four layers:

### 1. Core Math (`src/uncle_val/`)
- **`whitening.py`**: Householder-reflection-based whitening transform. Converts `n` i.i.d. normal observations with known variances and unknown shared mean into `n-1` independent `N(0,1)` samples. Supports numpy, JAX, and PyTorch backends via a `np` parameter.
- **`variability_detectors.py`**: Four numba-JIT detectors (magnitude amplitude, flux outlier, deviation outlier, autocorrelation) combined via `get_combined_variability_detector()`. Used to exclude variable stars from training.
- **`stat_tests.py`**: PyTorch goodness-of-fit tests (KS, Anderson-Darling, G-test, Jarque-Bera, Epps-Pulley) to verify whitened residuals are `N(0,1)`.
- **`utils/mag_flux.py`**: AB mag ↔ nJy flux conversions and error propagation.
- **`utils/hashing.py`**: FarmHash-based `uniform_hash()` for deterministic train/val/test splits by object ID.
- **`utils/variable.py`**: Dask-distributed-safe mutable wrapper (`Variable`) that works with or without a Dask client.

### 2. Data (`src/uncle_val/datasets/`)
- **`dp1.py`**: Loads Rubin DP1 HATS catalogs via LSDB. `dp1_catalog_single_band()` filters by band/quality/variability; `dp1_catalog_multi_band()` joins CCD visit info and one-hot-encodes bands. All lazy/Dask-backed.
- **`fake.py`**: Generates synthetic non-variable light curves with controllable `u` for testing.
- **`lsdb_generator.py`**: `LSDBDataGenerator` — asynchronously pre-fetches LSDB partitions via Dask futures for streaming training data.

### 3. ML Models and Losses (`src/uncle_val/learning/`)
All models inherit from `UncleModel(torch.nn.Module)`. The base class applies `UncleScaler` normalization (arcsinh for flux, log10 for error/exposure/sky/seeing), runs the submodule, then applies `exp(-output[0])` to produce `u`.

Concrete models:
- `ConstantMagErrModel`: Physics-based; one trainable parameter (`addition_centi_mag_err`) adds a constant error in quadrature.
- `LinearModel`: Single linear layer.
- `MLPModel`: MLP with GELU, configurable hidden dims and dropout.

Loss hierarchy rooted at `UncleLoss` (ABC). All losses apply the whitening transform and support Tikhonov regularization:
- `Chi2BasedLoss`: `-log P(chi²)` under chi-squared distribution.
- `KLWhitenBasedLoss`: KL divergence of whitened residuals vs. `N(0,1)`.
- `EPWhitenBasedLoss`: Epps-Pulley characteristic-function test as a differentiable loss.

Each has `*Total` (accumulate all light curves) and `*Lc` (mean across per-LC values) variants.

`lsdb_dataset.py` / `LSDBIterableDataset`: Bridges LSDB/Dask to PyTorch. Handles hash-based train/val/test splits, random subsampling of `n_src` observations per light curve, batching, and device placement.

### 4. Training Pipelines (`src/uncle_val/pipelines/`)
- **`training_loop.py`**: `training_loop()` — orchestrates everything: creates Dask client, materializes validation set to disk (`.pt` files), runs Adam + `ReduceLROnPlateau`, logs to TensorBoard, saves best model checkpoint.
- **`validation_set_utils.py`**: `ValidationDataLoaderContext` (saves validation tensors), `get_val_stats()` (evaluates multiple losses + histogram hooks on validation set).
- **`plotting.py`**: `make_plots()` — per-band histograms of whitened residuals vs. `N(0,1)`, and whitened signal vs. magnitude scatter plots.
- **`splits.py`**: Hash-based split boundaries: train `[0, 0.75)`, val `[0.75, 0.85)`, test `[0.85, 1.0)`.
- Concrete pipelines: `dp1_constant_magerr.py`, `dp1_linear_flux_err.py`, `dp1_mlp.py`.

### Data Flow
```
DP1 HATS Catalog (LSDB/HATS on disk)
  → datasets/dp1.py       (filter variables, add CCD visit features)
  → learning/lsdb_dataset.py  (subsample n_src obs/LC, batch, to tensor)
  → pipelines/training_loop.py  (Dask + Adam + TensorBoard)
      ↓                   ↓
  learning/models.py   learning/losses.py
  → Output: best model .pt + TensorBoard logs
  → pipelines/plotting.py (diagnostic whitened residual plots)
```

## Code Style
- Use `isinstance(x, A | B)` not `isinstance(x, (A, B))` — ruff UP038 enforces this.

## Testing Notes
- Doctests are collected from `src/` and `docs/*.rst` (configured in `pyproject.toml`).
- Mark slow tests with `@pytest.mark.long`; these are skipped in pre-commit hooks but run in CI.