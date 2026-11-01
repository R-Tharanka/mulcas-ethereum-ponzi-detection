# Running and Testing Guide

This guide describes how to reproduce and verify the frozen baseline, run the
separate random-split three-viewer diagnostic, and read the completed Phase 8
closure. See
[Full Current Research Progress Report](FULL_CURRENT_RESEARCH_PROGRESS_REPORT.md)
for the complete project account and [Phase 8 Research Extension Closure](phase8_summary.md)
for the current research decision. Run all commands from the repository root
in Windows PowerShell.

## 1. Scope

The implemented workflow covers:

1. Dataset integrity auditing.
2. Leakage-safe chronological train, validation, and test partitions.
3. Independent reproduction of the static feature views.
4. Ridge, SVM, and XGBoost individual baselines.
5. The independently reconstructed MulCas cascade and voting baseline.
6. The optional collaborator-authored random-split three-viewer diagnostic.

Phases 6, 7, and 8 are closed. The retained reference remains Phase 5.5 MulCas;
Phase 8R6A has not started, and notebook `08` remains reserved for read-only
final synthesis.

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
python -c "import numpy, pandas, sklearn, xgboost, gensim, imblearn; print('Imports OK')"
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

### Optional diagnostic: random-split three-viewer method

This standalone experiment was introduced by commit
`5ed8509613838f8e43268e075b58753f77c60e80`. It is not the frozen MulCas
cascade and is not part of the chronological Phase 1-5.5 dependency chain.
Read `docs/RANDOM_SPLIT_THREE_VIEWER_FLOW.md` before interpreting its metrics.

Required files:

- `scripts/static_random_mulcas.py`
- `data/raw/PonziDataset_20221114.csv`
- `results/baseline/paper_aligned/mulcas_results.json`
- `requirements.txt`

The recommended command writes to an ignored temporary location and leaves the
tracked result unchanged:

```powershell
New-Item -ItemType Directory -Force results\temp | Out-Null
python scripts\static_random_mulcas.py `
  --output results\temp\static_random_mulcas_results.json
```

To reproduce the collaborator's default output path exactly, use:

```powershell
python scripts\static_random_mulcas.py
```

This second command creates or overwrites:

- `results/experiments/static_random_mulcas_results.json`

The default run uses seed 42, four XGBoost workers, and at most 20,000 text
features. It performs the following operations:

1. Randomly stratifies the 6,498 rows into development and test partitions.
2. Randomly splits development rows into fit and validation partitions.
3. Tunes unigram-count XGBoost and 1-3-gram TF-IDF XGBoost viewers with
   shuffled three-fold cross-validation.
4. Fits a calibrated Ridge viewer using TF-IDF and 64-dimensional mean
   Word2Vec features.
5. Applies SMOTE only inside fitting and cross-validation pipelines.
6. Averages the three viewer probabilities.
7. Selects the ensemble threshold that maximizes validation F1.
8. Applies the frozen validation-selected threshold to the random test rows.

The recorded commit result used threshold `0.39088033297408625` and reported
test precision 0.8833, recall 0.8413, F1 0.8618, and average precision 0.8833.
Minor numerical differences can occur when dependency versions differ.

To display the already saved comparison without training models:

```powershell
python scripts\static_random_mulcas.py --show-comparison-only
```

To inspect a temporary result without changing it:

```powershell
$result = Get-Content `
  results\temp\static_random_mulcas_results.json -Raw | ConvertFrom-Json
$result.split_counts
$result.test | Select-Object threshold, precision, recall, f1, average_precision
$result.train_test_shared_addresses
```

The method reports train/test address overlap but does not prevent it. The
recorded run contains three shared addresses, and exact duplicate opcode or
contract-family groups are not kept together. Random splitting also mixes
deployment periods. Consequently, its higher random-holdout F1 must not be
reported as an equivalent chronological improvement over published MulCas or
the frozen Phase 5.5 reference.

The script does not use `src/models/mulcas.py`, does not save fitted model
objects, and has no dedicated automated test module. Run the full repository
suite to detect broader integrity regressions, but do not describe that suite
as direct behavioral coverage of this standalone script.

## 6. Run Automated Tests

Run the complete test suite:

```powershell
python -m pytest -q
```

The suite covers imports, deterministic seeds, data and feature leakage
controls, frozen manifests, model construction, MulCas routing/voting, Phase 6
experiments, Phase 7 analyses, Phase 8 representations, and formal closure
consistency.

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

## 8. Read or Rebuild the Phase 8 Closure

The tracked closure artifacts are under `results/extension/phase8_closure/`.
Reading them does not require model retraining. The main files are:

- `docs/phase8_summary.md`
- `results/extension/phase8_closure/phase8_closure_results.json`
- `results/extension/phase8_closure/phase8_closure_report.md`

To reverify frozen inputs and deterministically regenerate only the closure:

```powershell
python -m src.extension.phase8_closure
python -m pytest tests/test_phase8_closure.py -q
```

This command reads completed artifacts only. It does not train a model,
evaluate P5, change a threshold, or start Phase 8R6A.

`results/temp/` is ignored by Git and may be removed after inspection.

## 9. Reproducibility Checks

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

## 10. What You Should Do

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

## 11. What You Should Not Do

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

## 12. Troubleshooting

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
