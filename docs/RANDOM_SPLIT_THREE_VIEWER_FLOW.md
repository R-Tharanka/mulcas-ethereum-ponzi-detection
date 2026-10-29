# Random-Split Three-Viewer Method

## 1. Purpose

This document explains the static random-split experiment introduced by commit
`5ed8509613838f8e43268e075b58753f77c60e80`. The implementation is in
`scripts/static_random_mulcas.py`, and its recorded output is in
`results/experiments/static_random_mulcas_results.json`.

The method obtained a higher F1 score than the published MulCas Table 5 result,
but it uses a different and easier evaluation protocol. It should therefore be
described as a random-split diagnostic, not as evidence that the published
chronological MulCas result was beaten under equivalent conditions.

## 2. Complete Prediction Flow

```text
PonziDataset CSV
    |
    |-- Read opcode, label, and address
    |
    |-- Stratified random split, seed 42
    |     |-- 80% development: 5,198 rows
    |     |     |-- Fit: 4,158 rows
    |     |     `-- Validation: 1,040 rows
    |     `-- Test: 1,300 rows
    |
    |-- Train three viewers on the fit partition
    |     |
    |     |-- Viewer 1: unigram counts -> SMOTE -> XGBoost
    |     |
    |     |-- Viewer 2: 1-3-gram TF-IDF -> SMOTE -> XGBoost
    |     |
    |     `-- Viewer 3:
    |           1-3-gram TF-IDF + mean Word2Vec
    |           -> SMOTE -> Ridge
    |           -> sigmoid probability calibration
    |
    |-- Every viewer scores every validation contract
    |
    |-- Average the three viewer probabilities
    |
    |-- Select the threshold with the highest validation F1
    |     selected threshold = 0.39088033297408625
    |
    |-- Every viewer scores every test contract
    |
    |-- Average the three test probabilities
    |
    |-- score >= threshold -> Ponzi
    `-- score < threshold  -> Benign
```

## 3. Input Data

The script reads three columns from `data/raw/PonziDataset_20221114.csv`:

- `opcode`: the static opcode sequence used by the models;
- `label`: `0` for benign and `1` for Ponzi; and
- `address`: used to audit address overlap, not as a model feature.

Empty opcode sequences and labels outside `{0, 1}` are rejected.

## 4. Random Data Splitting

The complete 6,498-contract dataset is divided using stratified random
splitting. Stratification keeps approximately the same positive-class
proportion in each partition.

| Partition | Rows | Ponzi | Purpose |
| --- | ---: | ---: | --- |
| Fit | 4,158 | 201 | Feature fitting, model fitting, and cross-validation |
| Validation | 1,040 | 50 | Ensemble threshold selection |
| Test | 1,300 | 63 | Final random-split evaluation |

The first split assigns 80% of rows to development and 20% to testing. A second
split assigns 80% of the development rows to fitting and 20% to validation.
Both splits are label-stratified and use fixed random seeds.

Unlike the project's chronological protocols, this split can place old and
newer contracts in every partition. It therefore measures random holdout
performance rather than forward temporal generalization.

## 5. Viewer 1: Term-Count XGBoost

```text
Opcode sequence
    -> individual opcode counts
    -> at most 20,000 features
    -> SMOTE on fitting/CV data
    -> XGBoost classifier
    -> Ponzi probability
```

This viewer uses unigram opcode counts. XGBoost hyperparameters are selected by
shuffled three-fold cross-validation on the fit partition using F1 as the
selection metric.

The searched values are:

- maximum depth: `3`, `5`, or `7`;
- learning rate: `0.01` or `0.1`; and
- number of estimators: `100` or `200`.

The recorded selected configuration is maximum depth 5, learning rate 0.1, and
100 estimators.

## 6. Viewer 2: N-Gram TF-IDF XGBoost

```text
Opcode sequence
    -> unigram, bigram, and trigram TF-IDF
    -> at most 20,000 features
    -> SMOTE on fitting/CV data
    -> XGBoost classifier
    -> Ponzi probability
```

This viewer uses opcode 1-3-gram TF-IDF. It uses the same cross-validation grid
and selection process as Viewer 1. Its recorded selected configuration is
maximum depth 3, learning rate 0.1, and 100 estimators.

## 7. Viewer 3: Calibrated Ridge

```text
                 -> 1-3-gram TF-IDF --------\
Opcode sequence                              +--> SMOTE --> Ridge
                 -> 64-d mean Word2Vec -----/                |
                                                              v
                                                sigmoid calibration
                                                              |
                                                      Ponzi probability
```

Word2Vec is fitted on the relevant fitting data only. Each contract receives a
64-dimensional vector formed by averaging the embeddings of its known opcode
tokens. This vector is joined with the TF-IDF representation.

The combined representation is processed by SMOTE and a Ridge classifier with
`alpha=1.0`. Sigmoid calibration converts Ridge decision values into
probabilities so that this viewer can participate in probability averaging.

## 8. Ensemble Decision

This method is not a cascade. Every contract is scored by all three viewers.
Their probabilities receive equal weight:

```text
Count XGBoost probability ------\
TF-IDF XGBoost probability ------+--> arithmetic mean -> final score
Calibrated Ridge probability ----/
```

For example:

```text
Viewer 1 probability = 0.72
Viewer 2 probability = 0.61
Viewer 3 probability = 0.50

Final score = (0.72 + 0.61 + 0.50) / 3
            = 0.61
```

There is no early confident acceptance, second-stage routing, or MulCas voting
lambda in this implementation.

## 9. Threshold Selection

The ensemble first scores the validation partition. The implementation checks
the available precision-recall thresholds and selects the threshold with the
highest validation F1.

The recorded threshold is:

```text
0.39088033297408625
```

This threshold is lower than 0.5. Lowering the decision threshold generally
increases positive predictions and recall, although it can reduce precision.
The threshold is then frozen and applied to the random test partition.

## 10. Recorded Test Result

| Metric | Random-split result |
| --- | ---: |
| Accuracy | 0.9869 |
| Balanced accuracy | 0.9178 |
| Precision | 0.8833 |
| Recall | 0.8413 |
| F1 | 0.8618 |
| PR-AUC / average precision | 0.8833 |
| True negatives | 1,230 |
| False positives | 7 |
| False negatives | 10 |
| True positives | 53 |

## 11. Comparison With Published MulCas

| Metric | Published MulCas Table 5 | Random-split method | Difference |
| --- | ---: | ---: | ---: |
| Precision | 0.9510 | 0.8833 | -0.0677 |
| Recall | 0.6740 | 0.8413 | +0.1673 |
| F1 | 0.7890 | 0.8618 | +0.0728 |

The random-split method has higher recall and F1 but lower precision. These
numbers are not a controlled head-to-head comparison because the evaluation
protocols and model structures differ.

## 12. Why the Random-Split Result Is Higher

Several factors can contribute to the stronger recorded result:

1. Random splitting mixes deployment periods across fit, validation, and test.
2. Related contract families can occur on both sides of the split.
3. Three normalized addresses cross the recorded train/test boundary.
4. Duplicate opcode sequences are not assigned as indivisible groups.
5. SMOTE balances positive examples inside fitting and cross-validation.
6. XGBoost hyperparameters are selected with shuffled cross-validation.
7. The operating threshold is selected specifically for validation F1.
8. All three viewers contribute to every prediction.

The first four points can make the test partition easier, especially when code
or contract identities are duplicated or closely related. They do not prove
that all of the performance gain is leakage, but they prevent the result from
supporting a strong forward-generalization claim.

## 13. Difference From Frozen MulCas

| Topic | Frozen MulCas | Random-split three-viewer method |
| --- | --- | --- |
| Split | Chronological | Stratified random rows |
| Temporal direction | Historical to future | Mixed periods |
| Architecture | Two-stage cascade plus voting | Three-viewer probability average |
| Routing | Confident early decisions and escalation | Every model scores every row |
| Creator feature | Chronological creator indicator | Not used |
| Imbalance treatment | Frozen baseline settings | SMOTE |
| Model tuning | Frozen settings/protocol | Shuffled grid search for XGBoost |
| Threshold | Frozen cascade thresholds | Validation-F1 optimized threshold |
| Primary interpretation | Forward temporal evidence | Random holdout diagnostic |

## 14. Correct Research Interpretation

The defensible conclusion is:

> The three-viewer method obtained strong performance on a stratified random
> row-level holdout, including higher recall and F1 than the published MulCas
> Table 5 values. However, it used a different ensemble, mixed deployment
> periods across partitions, optimized its threshold on a random validation
> set, and did not keep duplicate identities or code families together. It
> therefore does not establish that the published MulCas system was surpassed
> under an equivalent or leakage-safe chronological protocol.

The result remains useful as a diagnostic showing that the available static
opcode representations can separate many contracts when training and testing
come from a mixed-time distribution.

## 15. Source Artifacts

- Implementation: `scripts/static_random_mulcas.py`
- Recorded metrics: `results/experiments/static_random_mulcas_results.json`
- Dataset: `data/raw/PonziDataset_20221114.csv`
- Introducing commit: `5ed8509613838f8e43268e075b58753f77c60e80`

## 16. Complete Related-File Inventory

### 16.1 Direct Runtime Files

| File | Role | Required to train? |
| --- | --- | --- |
| `scripts/static_random_mulcas.py` | Complete experiment implementation: loading, random splitting, feature extraction, SMOTE, model tuning, probability averaging, threshold selection, evaluation, and result writing | Yes |
| `data/raw/PonziDataset_20221114.csv` | Input dataset containing `opcode`, `label`, and `address` | Yes |
| `results/baseline/paper_aligned/mulcas_results.json` | Supplies the published MulCas and frozen paper-aligned values shown in the comparison table | Yes for the default comparison step; not used to fit the random-split models |
| `results/baseline/dataset_audit.json` | Preserves source-dataset integrity measurements and SHA-256 | Supporting evidence; not read by the training script |
| `requirements.txt` | Declares the Python packages used by the script, including NumPy, pandas, scikit-learn, XGBoost, imbalanced-learn, and gensim | Environment dependency |

The raw CSV exists locally but is ignored by Git. Creating another branch does
not create another copy of this ignored dataset in the same worktree.

### 16.2 Generated Experiment Output

| File | Role | Created or overwritten by a full run? |
| --- | --- | --- |
| `results/experiments/static_random_mulcas_results.json` | Stores split counts, address-overlap count, selected XGBoost parameters, validation F1, selected threshold, test metrics, confusion matrix, classification report, and baseline comparison | Yes |

The script does not persist fitted vectorizers, Word2Vec weights, SMOTE output,
or trained model files. Those objects exist only in memory during execution.

### 16.3 Documentation Files

| File | Role |
| --- | --- |
| `docs/RANDOM_SPLIT_THREE_VIEWER_FLOW.md` | Focused explanation of this method, its flow, results, and interpretation |
| `docs/LAST_COMMIT_PROGRESS_REPORT.md` | Wider audit of commit `5ed8509`, including implementation details, limitations, commands, and affected files |
| `docs/RUN_AND_TEST_GUIDE.md` | Commands for a non-destructive temporary run, an exact default-path run, and comparison-only inspection |

### 16.4 Repository Storage Policy

| File | Role |
| --- | --- |
| `.gitignore` | Excludes `data/raw/`, `.venv/`, `results/temp/`, and large artifact directories from Git tracking |

The default result under `results/experiments/` is tracked. The recommended
temporary output under `results/temp/` is ignored.

### 16.5 Files Changed in the Same Commit but Not Used by the Script

Commit `5ed8509` also modified the following tracked metadata artifacts:

| File | Relationship to the random-split run |
| --- | --- |
| `results/baseline/dataset_audit.json` | Retained as active source-integrity evidence; the random-split script does not read it |
| `archive/history/results/baseline/paper_aligned/feature_summary.json` | Paper-aligned feature metadata changed in the same commit; now archived and not read by the random-split script |
| `archive/history/results/baseline/paper_aligned/split_metadata.json` | Paper-aligned split metadata changed in the same commit; now archived and not read by the random-split script |

These files should not be described as inputs to the random-split model merely
because they appear in the same commit.

### 16.6 Files Not Involved

The random-split implementation does not import or call:

- `src/models/mulcas.py`;
- the paper-aligned feature-generation modules under `src/features/`;
- the chronological split modules under `src/data/`;
- Phase 8 or Phase 9 experiment modules;
- Track B data or sealed labels; or
- any notebook.

There is no dedicated automated test module for
`scripts/static_random_mulcas.py`. The script is a standalone experiment rather
than an integration with the repository's frozen MulCas pipeline.

## 17. Dataset-to-Result File Lineage

The complete repository-level lineage is:

```text
requirements.txt
    -> installs the required Python packages into local .venv/

data/raw/PonziDataset_20221114.csv
    -> supplies opcode, label, and address
    |
    v
scripts/static_random_mulcas.py
    -> random stratified splitting
    -> feature fitting
    -> SMOTE inside fit/CV pipelines
    -> three viewer fits
    -> validation threshold selection
    -> test scoring and metrics
    |
    | reads comparison values from
    v
results/baseline/paper_aligned/mulcas_results.json
    |
    v
results/experiments/static_random_mulcas_results.json
    -> default tracked result

or, for a non-destructive run:

results/temp/static_random_mulcas_results.json
    -> ignored temporary result
```

### 17.1 Files Read During a Default Full Run

1. `scripts/static_random_mulcas.py`
2. `data/raw/PonziDataset_20221114.csv`
3. `results/baseline/paper_aligned/mulcas_results.json`

`requirements.txt` is used during environment installation, not opened by the
experiment process itself.

### 17.2 File Written During a Default Full Run

1. `results/experiments/static_random_mulcas_results.json`

No other dataset, split-membership file, configuration file, trace CSV, fitted
model, vectorizer, embedding model, or prediction table is written by the
script.

### 17.3 Files Read in Comparison-Only Mode

Running with `--show-comparison-only` does not read the raw dataset and does not
fit a model. It reads:

1. `results/experiments/static_random_mulcas_results.json`
2. `results/baseline/paper_aligned/mulcas_results.json`

It prints the comparison to the terminal and does not write a new file.

### 17.4 Complete Classification of Related Repository Files

| Classification | Files |
| --- | --- |
| Environment declaration | `requirements.txt` |
| Storage/ignore policy | `.gitignore` |
| Raw model input | `data/raw/PonziDataset_20221114.csv` |
| Executable implementation | `scripts/static_random_mulcas.py` |
| Dataset-integrity evidence | `results/baseline/dataset_audit.json` |
| Comparison-only input | `results/baseline/paper_aligned/mulcas_results.json` |
| Primary generated result | `results/experiments/static_random_mulcas_results.json` |
| Optional temporary result | `results/temp/static_random_mulcas_results.json` |
| Focused method documentation | `docs/RANDOM_SPLIT_THREE_VIEWER_FLOW.md` |
| Run instructions | `docs/RUN_AND_TEST_GUIDE.md` |
| Commit-level audit | `docs/LAST_COMMIT_PROGRESS_REPORT.md` |
| Archived same-commit metadata, not runtime input | `archive/history/results/baseline/paper_aligned/feature_summary.json`; `archive/history/results/baseline/paper_aligned/split_metadata.json` |

This table covers every repository file directly involved in executing,
recording, governing, or documenting the current random-split flow. Files under
`.venv/` are installed third-party package internals rather than project flow
files and are intentionally not enumerated individually.
