# MulCas Ethereum Ponzi Detection

Reproducing and improving the MulCas multi-view cascade ensemble for Ethereum
smart Ponzi scheme detection through feature engineering, model fine-tuning, and
performance evaluation.

The repository currently implements the research workflow through Phase 7C:
the frozen paper-aligned baseline, controlled Phase 6 improvement experiments,
temporal robustness analysis, viewer ablation, and creator-feature ablation.

See [Project Progress Report](docs/PROJECT_PROGRESS_REPORT.md) for the complete,
presentation-ready account of the methodology, results through Phase 7C, and
current run/test commands.

See [Running and Testing Guide](docs/RUN_AND_TEST_GUIDE.md) for the complete
setup, phase-by-phase execution, notebook, testing, reproducibility, and do/don't
instructions.

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

The tests cover the environment, data validation, temporal and feature leakage,
evaluation metrics, baseline models, and MulCas cascade routing and voting.

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
