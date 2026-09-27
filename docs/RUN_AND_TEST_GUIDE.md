# Running and Testing Guide

This guide describes how to reproduce and verify the frozen baseline. The
project now continues through Phase 7C; see
[Project Progress Report](PROJECT_PROGRESS_REPORT.md) for current Phase 6/7
results and commands. Run all commands from the repository root in Windows
PowerShell.

## 1. Scope

The implemented workflow covers:

1. Dataset integrity auditing.
2. Leakage-safe chronological train, validation, and test partitions.
3. Independent reproduction of the static feature views.
4. Ridge, SVM, and XGBoost individual baselines.
5. The independently reconstructed MulCas cascade and voting baseline.

Phase 6 improvement experiments are closed and Phase 7A-7C analyses are
implemented. Notebook `08` remains reserved for future final evaluation.

## 2. Prerequisites

- Windows with PowerShell.
- Python 3.11 or newer available as `python`.
- Enough free disk space for the 307 MB raw CSV and locally generated model and
  feature artifacts.
- The released dataset at:
  `data/raw/PonziDataset_20221114.csv`

The raw dataset is intentionally excluded from Git. Do not rename, edit,
reformat, sort, or overwrite it.

## 3. Create the Environment

Create and activate the project-local virtual environment:

```powershell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The execution-policy command affects only the current PowerShell process. It is
needed only when PowerShell blocks the activation script.

Confirm that commands resolve to the virtual environment:

```powershell
python -c "import sys; print(sys.executable); print(sys.version)"
python -c "import numpy, pandas, sklearn, xgboost, gensim; print('Imports OK')"
```

`sys.executable` should end in `.venv\Scripts\python.exe`.

Activation is optional if every command explicitly uses
`.\.venv\Scripts\python.exe` in place of `python`.

## 4. Verify the Dataset

Confirm the file exists and compute its SHA-256 without modifying it:

```powershell
Test-Path -LiteralPath data\raw\PonziDataset_20221114.csv
Get-FileHash -Algorithm SHA256 data\raw\PonziDataset_20221114.csv
```

The audit records the measured digest and dataset properties in
`results/baseline/dataset_audit.json`. The released file is the source of truth;
do not force it to match the row or class counts reported by the paper.

## 5. Run the Pipeline

Run these commands in order. Downstream phases depend on outputs from earlier
phases.

### Phase 1: dataset audit

```powershell
python -m src.data.audit
```

Expected output:

- `results/baseline/dataset_audit.json`

The audit measures schema, row and class counts, missing values, duplicates,
block and timestamp ranges, creator statistics, opcode integrity, memory use,
and the raw-file SHA-256.

### Phase 2: chronological split

```powershell
python -m src.data.split --config configs/data_split.yaml
```

Expected outputs:

- `results/baseline/temporal_split_membership.csv`
- `results/baseline/temporal_split_metadata.json`

Only row identifiers, addresses, ordering values, labels, and split membership
are saved. The large source rows are not copied. The split must satisfy:

```text
max(train block) <= min(validation block)
max(validation block) <= min(test block)
```

### Phase 3: feature reproduction

```powershell
python -m src.features.reproduce --config configs/feature_reproduction.yaml
```

Expected tracked output:

- `results/baseline/feature_view_summary.json`

Fitted vectorizers and other reusable transformers are written under
`results/artifacts/features/`. That directory is local and ignored by Git.
Training data alone is used to fit vocabularies, IDF statistics, and Word2Vec.

### Phase 4: individual baselines

```powershell
python -m src.models.run_baselines --config configs/baseline_models.yaml
```

Expected tracked outputs:

- `results/baseline/individual_baseline_results.json`
- `results/baseline/individual_baseline_results.csv`

Fitted model files are written under `results/artifacts/baseline_models/` and
remain ignored by Git.

### Phase 5: MulCas reproduction

```powershell
python -m src.models.run_mulcas --config configs/mulcas_reproduction.yaml
```

Expected tracked outputs:

- `results/baseline/mulcas_reproduction_results.json`
- `results/baseline/mulcas_traces/validation_trace.csv`
- `results/baseline/mulcas_traces/test_trace.csv`
- `docs/mulcas_baseline_report.md`

Stage and voting models are written under `results/artifacts/mulcas/`. The JSON
result embeds the exact configuration and its SHA-256. Traces preserve sample
indices and show cascade decisions, voting-viewer predictions, and final routes.

### Phase 5.5: paper-aligned protocol correction

Phase 5.5 is a frozen correction, not an improvement experiment. It writes to a
separate namespace and does not replace any Phase 1-5 result:

```powershell
python -m src.data.paper_aligned_split --config configs/paper_aligned_split.yaml
python -m src.features.paper_aligned --config configs/paper_aligned_features.yaml
python -m src.models.run_baselines --config configs/paper_aligned_baseline_models.yaml
python -m src.models.run_mulcas --config configs/paper_aligned_mulcas.yaml
```

Machine-readable outputs are stored under
`results/baseline/paper_aligned/`. Large fitted artifacts are stored under
`results/artifacts/paper_aligned/` and ignored by Git. The corrected report is
`docs/paper_aligned_baseline_report.md`.

## 6. Run Automated Tests

Run the complete test suite:

```powershell
python -m pytest -q
```

The current suite contains 100 tests. It covers imports, deterministic seeds,
data and feature leakage controls, frozen manifests, model construction,
MulCas routing/voting, Phase 6 experiments, and Phase 7 temporal ablations.

Run focused suites while changing one area:

```powershell
python -m pytest tests/test_data_validation.py tests/test_temporal_split.py -q
python -m pytest tests/test_feature_leakage.py -q
python -m pytest tests/test_baseline_models.py tests/test_evaluation_metrics.py -q
python -m pytest tests/test_mulcas.py -q
```

A pytest cache permission warning may appear on some Windows installations. It
does not indicate a failed test; rely on the final passed/failed summary.

## 7. Run the Notebooks

Start Jupyter from the activated environment:

```powershell
python -m jupyter notebook
```

Open notebooks `00` through `07` in numerical order for the implemented work.
They are presentation layers over reusable code in `src`; research logic
should not be added only to notebook cells. Notebook `08` remains reserved for
future final evaluation.

To verify a notebook non-interactively without replacing the source notebook:

```powershell
New-Item -ItemType Directory -Force results\temp | Out-Null
python -m jupyter nbconvert --execute --to notebook `
  --output-dir results\temp `
  --output mulcas_reproduction_executed.ipynb `
  notebooks\05_mulcas_reproduction.ipynb
```

`results/temp/` is ignored by Git and may be removed after inspection.

## 8. Reproducibility Checks

Before accepting regenerated results:

```powershell
python -m pytest -q
git diff -- data/raw/PonziDataset_20221114.csv
git status --short
```

The raw-data diff must be empty. Review changes to configuration and tracked
baseline results together. A changed configuration, dependency version, raw
file digest, split membership, or random seed can legitimately change metrics
and must be documented in the experiment log.

The primary reproducibility controls are:

- `createdBlockNumber` determines chronological ordering.
- Train, validation, and final test partitions remain separate.
- Text transformers and model estimators fit on training data only.
- Creator features use labels from strictly earlier contracts only.
- Validation may guide model selection; test labels never guide parameters,
  viewer order, confidence thresholds, voting thresholds, or decision cutoffs.
- Randomized components use the configured seed and single-worker Word2Vec and
  XGBoost settings where configured.

## 9. What You Should Do

- Run commands from the repository root with the `.venv` active.
- Keep the raw CSV byte-for-byte unchanged.
- Change behavior through versioned YAML configuration where possible.
- Run focused tests during development and the full suite before reporting.
- Fit every learned preprocessing object on training rows only.
- Keep validation and final-test metrics clearly labeled.
- Record configuration, seed, software changes, and observations in
  `docs/experiment_log.md` for new experiments.
- Store compact summaries, metrics, and traces under `results/baseline/`.
- Store large matrices, fitted transformers, and models under ignored
  `results/artifacts/`.
- Compare published metrics and experimental metrics as distinct results.

## 10. What You Should Not Do

- Do not edit, sort, clean, deduplicate, or resave the raw CSV.
- Do not shuffle rows before chronological partition assignment.
- Do not combine train, validation, and test data before fitting transforms.
- Do not fit vocabularies, IDF values, Word2Vec, scalers, or models on validation
  or test rows.
- Do not let future labels affect earlier creator/developer features.
- Do not tune hyperparameters, thresholds, cascade order, or voting rules using
  final-test labels.
- Do not overwrite the frozen Phase 5 baseline with improvement experiments;
  use separate configurations and result paths.
- Do not commit raw datasets, secrets, virtual environments, caches, large
  feature matrices, or fitted model artifacts.
- Do not place reusable pipeline logic exclusively in notebooks.
- Do not claim that independently reconstructed metrics are the paper's
  published results.

## 11. Troubleshooting

**`ModuleNotFoundError`**

Run from the repository root, activate `.venv`, and reinstall requirements.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**Missing feature artifacts**

Run Phase 3 before either model phase:

```powershell
python -m src.features.reproduce
```

**Missing split metadata**

Run Phase 2 before feature reproduction:

```powershell
python -m src.data.split
```

**Unexpected metric changes**

Check the raw CSV hash, active Python executable, installed package versions,
YAML diffs, split metadata, and configured seed:

```powershell
Get-FileHash -Algorithm SHA256 data\raw\PonziDataset_20221114.csv
python -c "import sys; print(sys.executable)"
python -m pip freeze
git diff -- configs results/baseline
```

Do not resolve unexpected changes by editing result files manually. Regenerate
them through the corresponding module after identifying the cause.
