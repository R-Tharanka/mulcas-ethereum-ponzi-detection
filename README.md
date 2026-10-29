# Random-Split Static Ponzi Detection

This repository's active workflow is the collaborator-authored random-split
three-viewer experiment introduced by commit
`5ed8509613838f8e43268e075b58753f77c60e80`.

The active method uses static Ethereum opcode documents and combines:

1. unigram opcode counts, SMOTE, and XGBoost;
2. 1-3-gram opcode TF-IDF, SMOTE, and XGBoost; and
3. 1-3-gram TF-IDF plus mean Word2Vec, SMOTE, calibrated Ridge.

The three positive-class probabilities are averaged. A validation-selected
threshold is then applied to a random-stratified test partition.

## Important Interpretation

The recorded random-split result is:

| Precision | Recall | F1 | Average precision |
| ---: | ---: | ---: | ---: |
| 0.8833 | 0.8413 | 0.8618 | 0.8833 |

This is not an equivalent chronological comparison with published MulCas. The
split mixes deployment periods, contains three addresses shared across the
development/test boundary, and does not keep duplicate code families together.
The result is therefore a random-holdout diagnostic, not proof of forward
temporal generalization.

## Active Files

```text
data/raw/PonziDataset_20221114.csv
scripts/static_random_mulcas.py
results/baseline/dataset_audit.json
results/baseline/paper_aligned/mulcas_results.json
results/experiments/static_random_mulcas_results.json
tests/test_static_random_mulcas.py
docs/RANDOM_SPLIT_THREE_VIEWER_FLOW.md
docs/RUN_AND_TEST_GUIDE.md
docs/LAST_COMMIT_PROGRESS_REPORT.md
requirements.txt
pytest.ini
```

The raw CSV is intentionally ignored by Git and must remain byte-for-byte
unchanged. Its expected SHA-256 is:

```text
48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf
```

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python scripts\static_random_mulcas.py --show-comparison-only
```

For a non-destructive retraining run:

```powershell
New-Item -ItemType Directory -Force results\temp | Out-Null
python scripts\static_random_mulcas.py `
  --output results\temp\static_random_mulcas_results.json
```

See `docs/RUN_AND_TEST_GUIDE.md` for exact commands and
`docs/RANDOM_SPLIT_THREE_VIEWER_FLOW.md` for the full model flow.

## Historical Research Archive

All previous chronological baseline, Phase 6-9, extension, notebook, test, and
report files are preserved under `archive/history/` using their original
relative paths. `archive/CHRONOLOGY.md` explains the research timeline, and
`archive/history/TRACKED_FILE_MANIFEST.csv` records every moved tracked file and
its original Git blob ID.

Large ignored datasets, generated artifacts, caches, and private local material
are preserved locally under ignored `archive/local-only/` paths.
