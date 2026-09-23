# MulCas Ethereum Ponzi Detection

Reproducing and improving the MulCas multi-view cascade ensemble for Ethereum
smart Ponzi scheme detection through feature engineering, model fine-tuning, and
performance evaluation.

This repository is currently in Phase 0: local development scaffolding for a
research-reproducible Python project. Dataset processing, feature engineering,
model implementations, and experiments are intentionally not implemented yet.

## Local Setup

From Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation scripts, allow scripts for the current process
and activate again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

## Verify Environment

Run the test suite from the repository root:

```powershell
python -m pytest
```

The initial smoke tests verify that the `src` package imports correctly, common
repository paths resolve from the project root, and the deterministic seed helper
is reusable.

## Repository Layout

- `configs/`: configuration files for future experiments.
- `data/`: local datasets and derived data. Raw and processed data are ignored.
- `docs/`: research notes, protocol documents, and references.
- `notebooks/`: exploratory notebooks.
- `src/`: Python package code.
- `tests/`: pytest tests.
- `results/`: local experiment outputs and temporary artifacts.

## Reproducibility Notes

Use `src.utils.paths` for repository-relative paths and
`src.utils.seed.set_random_seed` before any future randomized processing or model
training. For fully deterministic Python hash behavior, set `PYTHONHASHSEED`
before starting Python.
