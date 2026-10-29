# Running and Testing the Random-Split Flow

This guide covers the repository's active workflow only:

```text
opcode CSV
-> random stratified split
-> three model viewers
-> equal probability average
-> validation-selected threshold
-> random test metrics
```

Historical chronological, Phase 6-9, and extension commands are preserved under
`archive/history/` and indexed by `archive/CHRONOLOGY.md`. They are not active
from their archived paths.

Run all commands from the repository root in Windows PowerShell.

## 1. Active Files

The executable flow uses:

- `data/raw/PonziDataset_20221114.csv`
- `scripts/static_random_mulcas.py`
- `results/baseline/paper_aligned/mulcas_results.json`
- `results/experiments/static_random_mulcas_results.json`
- `requirements.txt`

Supporting integrity and documentation files are:

- `results/baseline/dataset_audit.json`
- `tests/test_static_random_mulcas.py`
- `docs/RANDOM_SPLIT_THREE_VIEWER_FLOW.md`
- `docs/LAST_COMMIT_PROGRESS_REPORT.md`
- `pytest.ini`

## 2. Create the Environment

```powershell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm that the active dependencies import successfully:

```powershell
python -c "import numpy, pandas, sklearn, xgboost, gensim, imblearn; print('Imports OK')"
```

Activation is optional when commands explicitly use
`.\.venv\Scripts\python.exe` instead of `python`.

## 3. Verify the Raw Dataset

The raw CSV is local and intentionally ignored by Git:

```powershell
Test-Path -LiteralPath data\raw\PonziDataset_20221114.csv
Get-FileHash -Algorithm SHA256 data\raw\PonziDataset_20221114.csv
```

Expected SHA-256:

```text
48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf
```

Do not edit, sort, clean, deduplicate, or resave the CSV.

## 4. Inspect the Existing Result Without Training

This command reads the saved random-split result and comparison baseline. It
does not load the raw dataset or train a model:

```powershell
python scripts\static_random_mulcas.py --show-comparison-only
```

The recorded random-split test result is:

| Precision | Recall | F1 | Average precision |
| ---: | ---: | ---: | ---: |
| 0.8833 | 0.8413 | 0.8618 | 0.8833 |

## 5. Run a Non-Destructive Retraining Experiment

Use an ignored temporary output so the tracked evidence remains unchanged:

```powershell
New-Item -ItemType Directory -Force results\temp | Out-Null
python scripts\static_random_mulcas.py `
  --output results\temp\static_random_mulcas_results.json
```

The default experiment settings are:

- seed: `42`;
- XGBoost workers: `4`;
- maximum text features: `20,000`;
- outer test fraction: `0.20`;
- validation fraction within development: `0.20`;
- SMOTE neighbors: `5`;
- XGBoost selection: shuffled three-fold F1 CV; and
- Ridge probability calibration: sigmoid, three-fold CV.

The run performs:

1. unigram count -> SMOTE -> XGBoost;
2. 1-3-gram TF-IDF -> SMOTE -> XGBoost;
3. TF-IDF + 64-dimensional mean Word2Vec -> SMOTE -> calibrated Ridge;
4. equal averaging of all three probabilities;
5. validation-F1 threshold selection; and
6. one evaluation on the random test partition.

## 6. Reproduce the Default Result Path

Only use this command when intentionally replacing the tracked result:

```powershell
python scripts\static_random_mulcas.py
```

It writes:

```text
results/experiments/static_random_mulcas_results.json
```

The script does not save trained models, fitted vectorizers, Word2Vec weights,
or synthetic SMOTE rows. Those objects exist only while the process runs.

## 7. Inspect a Temporary Result

```powershell
$result = Get-Content `
  results\temp\static_random_mulcas_results.json -Raw | ConvertFrom-Json

$result.split_counts
$result.train_test_shared_addresses
$result.xgboost_count
$result.xgboost_tfidf
$result.test | Select-Object `
  threshold, precision, recall, f1, average_precision
```

## 8. Run Tests

Run the focused active suite:

```powershell
python -m pytest -q
```

The tests verify:

- deterministic and disjoint random row membership;
- expected 80/20 and inner 80/20 split sizes;
- equal-weight probability averaging;
- validation-threshold behavior;
- fixed-threshold metric calculation;
- recorded-result schema and headline values; and
- the local raw dataset hash when the ignored CSV is present.

The tests do not rerun the expensive XGBoost grid search.

## 9. Result Interpretation

The result is a random row-level holdout diagnostic. It is not an equivalent
chronological comparison with published MulCas because:

- deployment periods are mixed across partitions;
- three addresses cross the development/test boundary;
- duplicate opcode/code-family groups are not kept together;
- SMOTE creates synthetic minority-class feature vectors; and
- the operating threshold is selected for validation F1.

Use this wording:

> The three-viewer ensemble obtained F1 0.8618 on a stratified random holdout.
> Because the protocol differs from published chronological MulCas and does not
> isolate duplicate identities or code families, it does not establish superior
> forward temporal generalization.

## 10. Historical Archive

Historical tracked files are stored at:

```text
archive/history/<original path>
```

Use these indexes:

- `archive/README.md`
- `archive/CHRONOLOGY.md`
- `archive/history/TRACKED_FILE_MANIFEST.csv`

Large local-only historical datasets and generated artifacts are under ignored
`archive/local-only/`. Do not force-add that directory to Git.
