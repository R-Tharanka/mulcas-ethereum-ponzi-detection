# MulCas Ethereum Ponzi Detection

Reproducing and improving the MulCas multi-view cascade ensemble for Ethereum
smart Ponzi scheme detection through feature engineering, model fine-tuning, and
performance evaluation.

The repository implements the research workflow through the formal Phase 8
extension closure. The retained reference remains the frozen Phase 5.5
paper-aligned MulCas; no Phase 8 candidate met its predefined temporal
acceptance criterion.

See [Full Current Research Progress Report](docs/FULL_CURRENT_RESEARCH_PROGRESS_REPORT.md)
for the complete cross-phase account and [Phase 8 Research Extension Closure](docs/phase8_summary.md)
for the frozen extension decision. The next recommended research phase is
Phase 8R6A, an external/later dataset provenance and overlap audit; it has not
started.

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
