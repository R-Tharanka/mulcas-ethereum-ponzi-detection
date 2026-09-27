# MulCas Ethereum Ponzi Detection: Project Progress Report

## 1. Purpose and Current Status

This project independently reproduces and evaluates MulCas, a multi-view
cascade system for detecting Ethereum Ponzi smart contracts from information
available at contract creation time. The implementation is based on the
released dataset and the paper *Securing the Ethereum from Smart Ponzi Schemes*.

Work is complete through **Phase 7C**:

- Phases 0-5 built the independent data, feature, model, and MulCas pipeline.
- Phase 5.5 corrected the protocol to align more closely with the paper and
  established the frozen reference baseline.
- Phases 6A-6E tested predefined improvements. None received reliable support
  under the forward-only temporal selection protocol, so Phase 6F was not run.
- Phase 7A measured model aging, Phase 7B ablated MulCas viewers, and Phase 7C
  ablated the chronological creator feature.
- Phase 7D, Phase 7E, and final evaluation notebook 08 have not been performed.

The main conclusion so far is not that improvement is impossible. It is that
the tested interventions did not demonstrate a reliable improvement under the
predefined temporal protocol, and the frozen model's effectiveness varies
substantially across historical periods.

## 2. How to Read the Metrics

- **Precision:** among contracts predicted as Ponzi, the fraction that are
  actually Ponzi. High precision means few false alarms.
- **Recall:** among actual Ponzi contracts, the fraction detected. High recall
  means few missed Ponzi contracts.
- **F1:** harmonic mean of precision and recall. It rewards a useful balance.
- **PR-AUC:** ranking quality on the positive class across operating thresholds;
  particularly useful for imbalanced data.
- **Balanced accuracy:** average of positive-class recall and negative-class
  recall, preventing the majority benign class from dominating accuracy.
- **TN / FP / FN / TP:** true negatives, false positives, false negatives, and
  true positives.

The dataset is highly imbalanced. Ordinary accuracy can therefore look strong
even when most Ponzi contracts are missed, so precision, recall, F1, PR-AUC,
balanced accuracy, and the confusion matrix must be considered together.

## 3. Data and Reproducibility Foundation

The immutable input is `data/raw/PonziDataset_20221114.csv`.

| Property | Measured result |
| --- | ---: |
| File size | 307,176,676 bytes |
| SHA-256 | `48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf` |
| Contracts | 6,498 |
| Ponzi contracts | 314 |
| Benign contracts | 6,184 |
| Earliest block | 49,880 |
| Latest block | 7,136,486 |
| Earliest timestamp | 2015-08-07 20:48:27 |
| Latest timestamp | 2019-01-27 23:52:51 |
| Unique creators | 3,360 |

The released file matches the paper's headline total of 6,498 contracts and
314 Ponzi contracts. The audit found no null columns and no missing or empty
opcode strings. It also recorded 30 duplicate-address rows, 197 duplicate
transaction-hash rows, 30 duplicate block/address rows, and 195 groups of exact
duplicate opcode strings covering 712 rows. These observations were reported,
not silently cleaned or removed.

Reproducibility controls established in the early phases include:

- repository-relative paths and deterministic random seeding;
- an isolated `.venv` and explicit dependencies;
- chronological ordering by `createdBlockNumber`;
- train-only fitting of vocabularies, IDF statistics, Word2Vec, scalers, and
  estimators;
- creator features based only on labels of strictly earlier contracts;
- compact split-membership files instead of copied 307 MB datasets;
- machine-readable configurations, metrics, traces, hashes, and frozen-phase
  manifests;
- large generated matrices and fitted models kept under ignored artifact paths.

Detailed audit: `results/baseline/dataset_audit.json` and
`docs/dataset_description.md`.

## 4. Work Completed Through Phase 5.5

### Phase 0 - Local Research Environment

The repository was prepared as an importable Python project. Package
initializers were added under `src`, reusable path and seed helpers were added,
pytest smoke coverage was established, `.gitignore` was expanded, and Windows
PowerShell setup instructions were documented.

### Phase 1 - Dataset Integrity Audit

Reusable loading and validation code was implemented without modifying the raw
CSV. The audit measures schema, types, labels, missing data, duplicates,
chronology, block and timestamp ranges, creator reuse, opcode integrity, memory
use, and the raw-file hash. The first-pass audit avoids loading unnecessary
large text columns.

### Phase 2 - Initial 70/15/15 Chronological Split

The first independent reconstruction used deterministic chronological
partitions and never shuffled rows.

| Split | Rows | Ponzi | Benign | Benign:Ponzi | Block range |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 4,548 | 140 | 4,408 | 31.49:1 | 49,880-4,572,153 |
| Validation | 974 | 8 | 966 | 120.75:1 | 4,572,291-4,955,839 |
| Test | 976 | 166 | 810 | 4.88:1 | 4,957,244-7,136,486 |

The split is chronological and disjoint by row/address. Duplicate static code
does cross boundaries: 6 opcode values, 10 `code` values, 5 `contractCode`
values, and 1 `creationCode` value occur in more than one split. This was
detected and recorded for transparency.

### Phase 3 - Independent Feature Reproduction

The original implementation produced these training-fitted feature views:

| Feature view | Dimension | Meaning |
| --- | ---: | --- |
| Opcode term count | 142 | Counts of individual opcodes |
| Opcode TF-IDF | 142 | Term counts reweighted by training IDF |
| Opcode 1-3-gram count | 70,867 | Counts of opcode sequences of length 1-3 |
| Opcode 1-3-gram TF-IDF | 70,867 | N-gram counts reweighted by training IDF |
| Mean Word2Vec | 100 | Mean learned opcode embedding per contract |
| Creator feature | 1 | Leakage-safe prior-Ponzi-creator indicator |

All vocabulary and embedding learning used training rows only. Validation and
test rows were transform-only.

### Phase 4 - Individual Baseline Models

Ridge, linear SVM, and XGBoost were evaluated on the feature views using fixed,
documented baseline parameters. The original split's validation partition had
only eight positives, making selection unstable. The table shows the
validation-F1-selected representation per family and its test result.

| Model | Selected original view | Val P | Val R | Val F1 | Test P | Test R | Test F1 | Test PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge | N-gram TF-IDF | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6160 |
| SVM | N-gram count | 0.2500 | 0.5000 | 0.3333 | 0.5897 | 0.1386 | 0.2244 | 0.3509 |
| XGBoost | Term count | 1.0000 | 0.3750 | 0.5455 | 0.0000 | 0.0000 | 0.0000 | 0.4540 |

This result is one reason the paper-aligned protocol correction was necessary.
The complete model-by-feature-by-split table is in
`results/baseline/individual_baseline_results.csv`.

### Phase 5 - Independent MulCas Reconstruction

Reusable cascade and voting code was implemented with preserved sample order,
explicit confidence thresholds, a voting threshold, and per-contract routing
traces. The independently reconstructed 70/15/15 MulCas produced:

| Split | Precision | Recall | F1 | PR-AUC | Balanced acc. | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.5714 | 0.5000 | 0.5333 | 0.4160 | 0.7484 | 963 | 3 | 4 | 4 |
| Test | 0.6000 | 0.1084 | 0.1837 | 0.5222 | 0.5468 | 798 | 12 | 148 | 18 |

The model had reasonable precision but missed most test Ponzi contracts. This
was an independent reconstruction, not an exact paper replication.

### Phase 5.5 - Paper-Aligned Correction and Frozen Baseline

Phase 5.5 preserved all earlier outputs and introduced a corrected protocol:

- train through the 200th chronological Ponzi contract;
- validate after that through the 250th Ponzi contract;
- test after the 250th through the 314th Ponzi contract;
- assign benign contracts using the same creation-block boundaries;
- keep whole creation blocks together and never shuffle.

| Split | Rows | Ponzi | Benign | Benign:Ponzi | Block range |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 5,839 | 200 | 5,639 | 28.20:1 | 49,880-5,021,809 |
| Validation | 150 | 50 | 100 | 2.00:1 | 5,021,917-5,038,334 |
| Test | 509 | 64 | 445 | 6.95:1 | 5,038,337-7,136,486 |

The paper reports 5,990 rows through P4 and 508 in P5; this reconstruction has
5,989 and 509. The one-row difference is retained and documented rather than
manually forcing membership.

Paper-aligned features preserve the old 70,867-dimensional N-grams as an
alternative while adding document-frequency-filtered N-grams and IDF-weighted
Word2Vec. The creator feature is appended to every static view.

| Paper-aligned representation | Final dimension including creator |
| --- | ---: |
| Term count | 143 |
| TF-IDF | 143 |
| Filtered 1-3-gram count | 44,672 |
| Filtered 1-3-gram TF-IDF | 44,672 |
| Mean Word2Vec | 101 |
| IDF-weighted Word2Vec | 101 |

Individual selected model results on the final test were:

| Model | Selected representation | Validation F1 | Test P | Test R | Test F1 | Test PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ridge | Term count | 0.8519 | 0.9310 | 0.4219 | 0.5806 | 0.6365 |
| SVM | Filtered N-gram TF-IDF | 0.8519 | 0.8824 | 0.2344 | 0.3704 | 0.6717 |
| XGBoost | Filtered N-gram TF-IDF | 0.8519 | 0.8333 | 0.2344 | 0.3659 | 0.5943 |

The frozen reconstructed MulCas sequence is:

1. Cascade viewer 1: XGBoost with opcode term count, confidence interval
   thresholds 0.05 and 0.95.
2. Cascade viewer 2: XGBoost with filtered N-gram count, thresholds 0.05 and
   0.51.
3. Voting stage: filtered N-gram TF-IDF XGBoost and opcode TF-IDF Ridge, with
   voting threshold `lambda = 0`.
4. The chronological creator feature is appended to every view.

Frozen MulCas results:

| Split | Accuracy | Balanced acc. | Precision | Recall | F1 | PR-AUC | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.8933 | 0.9000 | 0.7931 | 0.9200 | 0.8519 | 0.8706 | 88 | 12 | 4 | 46 |
| Final test | 0.9253 | 0.7232 | 0.9063 | 0.4531 | 0.6042 | 0.6624 | 442 | 3 | 35 | 29 |

Comparison with the publication:

| Result source | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Published paper, Table 5 | 0.9510 | 0.6740 | 0.7890 |
| Original 70/15/15 reconstruction | 0.6000 | 0.1084 | 0.1837 |
| Corrected Phase 5.5 reconstruction | 0.9063 | 0.4531 | 0.6042 |
| Phase 5.5 minus published | -0.0448 | -0.2209 | -0.1848 |

The correction greatly improved the reconstruction but did not reproduce the
published recall or F1. Table 5 reports F1 = 0.789, while nearby prose reports
F1 = 0.743; the project records this source inconsistency without silently
choosing one value. Exact publication details for every preprocessing and model
setting are unavailable, so exact replication cannot be claimed.

Primary Phase 5.5 artifacts are in `results/baseline/paper_aligned/`, with the
human report at `docs/paper_aligned_baseline_report.md`.

## 5. Work Completed After the Frozen Baseline

### Shared Phase 6 Experimental Protocol

Phase 6 used three expanding, forward-only folds inside the Phase 5.5 training
partition. Learned preprocessing was refitted within each historical fold.

| Fold | Historical train | Later CV interval |
| --- | --- | --- |
| 1 | 104 rows, 50 Ponzi, blocks 49,880-1,287,252 | 305 rows, 50 Ponzi, blocks 1,287,353-2,000,473 |
| 2 | 409 rows, 100 Ponzi, blocks 49,880-2,000,473 | 5,141 rows, 50 Ponzi, blocks 2,008,140-4,963,650 |
| 3 | 5,550 rows, 150 Ponzi, blocks 49,880-4,963,650 | 289 rows, 50 Ponzi, blocks 4,964,082-5,021,809 |

The predeclared eligibility rule was mean temporal-CV precision at least 0.80,
then F1, recall, balanced accuracy, and simplicity tie-breaks. External
validation results did not choose candidates, and final-test data was not used.

### Phase 6A - Cost-Sensitive Learning

Class weighting was varied for Ridge/SVM and `scale_pos_weight` for XGBoost.
No selected family variant met the CV precision floor; all selections were
diagnostic fallbacks. On validation all three retained P = 0.7931, R = 0.9200,
and F1 = 0.8519, but PR-AUC was 0.7162 (Ridge), 0.7883 (SVM), and 0.8669
(XGBoost), all below the frozen 0.8706. **H6A was not supported.**

### Phase 6B - Training-Data Sampling

Controlled random under-sampling and over-sampling tested natural, 1:10, 1:5,
1:2, and 1:1 positive:negative targets. Only the selected SVM candidate met the
CV precision rule.

| Selected diagnostic | Validation P | R | F1 | PR-AUC | Main observation |
| --- | ---: | ---: | ---: | ---: | --- |
| Ridge, under 1:10 | 0.7931 | 0.9200 | 0.8519 | 0.7175 | Same classifications, worse ranking |
| SVM, over 1:10 | 0.7931 | 0.9200 | 0.8519 | 0.8074 | Same classifications, worse ranking |
| XGBoost, under 1:1 | 0.7302 | 0.9200 | 0.8142 | 0.8796 | More false positives, lower F1 |

**H6B was not supported.**

### Phase 6C - Decision-Threshold Optimization

Only native classifier thresholds changed; MulCas cascade thresholds and voting
were untouched. Ridge and SVM used decision scores with default threshold 0;
XGBoost used probabilities with default threshold 0.5.

| Model | CV-selected threshold | Eligible | Selected validation P | R | F1 |
| --- | ---: | --- | ---: | ---: | ---: |
| Ridge | -0.9730 | No, diagnostic fallback | 0.5000 | 0.9800 | 0.6622 |
| SVM | 1.6765 | Yes | 0.0000 | 0.0000 | 0.0000 |
| XGBoost | 0.9936 | Yes | 0.0000 | 0.0000 | 0.0000 |

The eligible SVM/XGBoost thresholds were too conservative to detect any
validation positives; the Ridge fallback raised recall but seriously reduced
precision and F1. **H6C was not supported.**

### Phase 6D1 - Structural Opcode Features

Seventy-four fit-free, contract-local features were implemented: opcode size
and diversity, canonical category counts/ratios, selected behavior-related
operations, PUSH/stack structure, and entropy/run complexity. No operand values
were inspected. No candidate met the CV precision floor. Ridge and SVM matched
the frozen validation classifications but lowered PR-AUC; XGBoost selected the
unchanged control. **H6D1 was not supported.**

### Phase 6D2 - Normalized Operand Features

Operand extraction was feasible from `contractCode` deployed runtime bytecode.
Thirty-seven normalized features described presence, volume, constant
structure, width/magnitude, and size-pattern indicators without exposing exact
hex constants, selectors, or addresses. The audit parsed 6,498 contracts and
recorded 3,425 truncated PUSH instructions and 42,382 unsupported opcode bytes;
affected contracts were handled deterministically rather than dropped.

| Selected diagnostic | Validation P | R | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Ridge, operand only | 0.6667 | 0.0800 | 0.1429 | 0.7632 |
| SVM, baseline + width group | 0.7931 | 0.9200 | 0.8519 | 0.8354 |
| XGBoost, baseline + width group | 0.7931 | 0.9200 | 0.8519 | 0.8681 |

No candidate met the CV precision floor. **H6D2 was not supported.**

### Phase 6E - Alternative Classifiers

Logistic Regression and LightGBM 4.7.0 were evaluated only on frozen Phase 5.5
views. Neither family produced an eligible temporal-CV candidate.

| Selected diagnostic | Validation P | R | F1 | PR-AUC | Balanced acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression, term count, C=0.1 | 0.7931 | 0.9200 | 0.8519 | 0.7635 | 0.9000 |
| LightGBM, term count | 0.8036 | 0.9000 | 0.8491 | 0.8386 | 0.8950 |

Logistic Regression matched classifications but ranked cases less well.
LightGBM gained a little precision while losing recall, F1, PR-AUC, and balanced
accuracy. **H6E was not supported.**

### Phase 6 Closure

**None of the tested Phase 6 interventions demonstrated a reliable improvement
under the predefined temporal evaluation protocol.** This does not mean that
improvement is impossible. Phase 6F was not executed because no independently
supported intervention existed to combine. The frozen Phase 5.5 MulCas model
was carried into Phase 7.

See `docs/phase6_summary.md` and
`results/experiments/phase6_summary.json` for the closure record.

### Phase 7A - Temporal Robustness and Model Aging

The full dataset was divided into whole-block periods based on chronological
Ponzi occurrences.

| Period | Rows | Ponzi | Benign | Block range |
| --- | ---: | ---: | ---: | --- |
| P0 | 104 | 50 | 54 | 49,880-1,287,252 |
| P1 | 305 | 50 | 255 | 1,287,353-2,000,473 |
| P2 | 5,141 | 50 | 5,091 | 2,008,140-4,963,650 |
| P3 | 289 | 50 | 239 | 4,964,082-5,021,809 |
| P4 | 150 | 50 | 100 | 5,021,917-5,038,334 |
| P5 | 509 | 64 | 445 | 5,038,337-7,136,486 |

The frozen model was retrained on expanding history and used only to predict
the immediately following period: P0+P1 -> P2, through P0-P4 -> P5.

| Predicted period | Precision | Recall | F1 | PR-AUC | Balanced acc. | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | 0.0492 | 0.2600 | 0.0828 | 0.0462 | 0.6053 | 4,840 | 251 | 37 | 13 |
| P3 | 0.5385 | 0.1400 | 0.2222 | 0.4786 | 0.5574 | 233 | 6 | 43 | 7 |
| P4 | 0.7931 | 0.9200 | 0.8519 | 0.8706 | 0.9000 | 88 | 12 | 4 | 46 |
| P5 | 0.8710 | 0.4219 | 0.5684 | 0.6674 | 0.7064 | 441 | 4 | 37 | 27 |

Published and reconstructed temporal precision/recall/F1 are:

| Period | Published P / R / F1 | Reconstruction P / R / F1 |
| --- | --- | --- |
| P2 | 0.8800 / 0.3800 / 0.5300 | 0.0492 / 0.2600 / 0.0828 |
| P3 | 0.9600 / 0.3200 / 0.4800 | 0.5385 / 0.1400 / 0.2222 |
| P4 | 0.8100 / 0.9400 / 0.8700 | 0.7931 / 0.9200 / 0.8519 |
| P5 | 0.9500 / 0.6700 / 0.7900 | 0.8710 / 0.4219 / 0.5684 |

Routing through the frozen system also changes by period:

| Period | Accepted by viewer 1 | Accepted by viewer 2 | Reached voting |
| --- | ---: | ---: | ---: |
| P2 | 4,859 | 263 | 19 |
| P3 | 271 | 13 | 5 |
| P4 | 134 | 15 | 1 |
| P5 | 454 | 29 | 26 |

Performance is highly period-dependent. P2 has many false positives, P3 has
very low recall, P4 is exceptionally strong, and P5 preserves high precision
but misses more than half of the Ponzi contracts. These are temporal performance
variations consistent with distribution change; they do not by themselves
prove concept drift. They also show why a single favorable split is not enough
to characterize long-term robustness.

### Phase 7B - Frozen MulCas Component Ablation

Only actual frozen components were removed; no threshold or model was retuned.
The table reports F1 by period.

| Ablation | P2 | P3 | P4 | P5 |
| --- | ---: | ---: | ---: | ---: |
| Full frozen MulCas | 0.0828 | 0.2222 | 0.8519 | 0.5684 |
| Remove first cascade viewer | 0.0755 | 0.1786 | 0.8519 | 0.3704 |
| Remove second cascade viewer | 0.0984 | 0.1455 | 0.8519 | 0.4944 |
| Remove voting stage | 0.0823 | 0.2222 | 0.8519 | 0.5532 |
| First viewer only | 0.0984 | 0.0000 | 0.8462 | 0.4944 |
| Second viewer only | 0.0755 | 0.0000 | 0.8381 | 0.3704 |

The first cascade viewer is especially important in P5. The second viewer has
mixed effects: removing it slightly improves P2 F1 but hurts P3 and P5. Voting
has a small nonnegative contribution. Neither standalone viewer is temporally
reliable, and no ablation is treated as a replacement model.

### Phase 7C - Chronological Creator Feature Ablation

The creator column was removed from every view without a constant replacement;
all other features, models, thresholds, routing, and periods remained fixed.

| Period | With creator F1 | Without creator F1 | Delta | Recall delta | FP delta | FN delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | 0.0828 | 0.1048 | +0.0220 | -0.0400 | -102 | +2 |
| P3 | 0.2222 | 0.3099 | +0.0876 | +0.0800 | +4 | -4 |
| P4 | 0.8519 | 0.8519 | +0.0000 | +0.0000 | 0 | 0 |
| P5 | 0.5684 | 0.5979 | +0.0295 | +0.0312 | 0 | -2 |

Creator-feature prevalence was sparse:

| Period | Previously observed creator | Creator flag = 1 | Ponzi flag = 1 | Benign flag = 1 |
| --- | ---: | ---: | ---: | ---: |
| P2 | 148 | 1 | 0 | 1 |
| P3 | 49 | 2 | 1 | 1 |
| P4 | 20 | 6 | 4 | 2 |
| P5 | 85 | 11 | 4 | 7 |

The feature does not improve recall consistently. P4 classifications are
identical with and without it. Removing it improves P3 and P5 F1, while in P2
it removes 102 false positives but also loses two true positives. In P5, 36 of
37 reference false negatives come from creators without prior positive history.
The creator flag is therefore a sparse historical predictive signal, not a
causal statement about creator identity.

## 6. Overall Findings So Far

1. **Protocol design matters.** The paper-aligned split and representations
   improved test F1 from 0.1837 to 0.6042 compared with the initial independent
   reconstruction.
2. **The paper result was not exactly reproduced.** Phase 5.5 approached its
   precision but remained substantially lower in recall and F1.
3. **Phase 6 did not identify a reliable improvement.** Cost weighting,
   sampling, thresholds, structural features, operand features, Logistic
   Regression, and LightGBM all failed the predefined temporal evidence rule.
4. **Performance is not temporally stable.** F1 ranges from 0.0828 in P2 to
   0.8519 in P4, then falls to 0.5684 in P5.
5. **MulCas components are complementary but period-dependent.** The first
   viewer matters strongly in P5; the second viewer and voting stage have more
   modest or mixed effects.
6. **Creator history is not uniformly beneficial.** Its effect changes by
   period, and most late false negatives involve creators with no prior Ponzi
   history.
7. **The correct reference remains Phase 5.5.** Phase 6 diagnostics and Phase 7
   ablations are analyses, not replacement production models.

## 7. Repository Map

- `configs/`: frozen baseline and experiment YAML files.
- `data/raw/`: immutable local source CSV; ignored by Git.
- `data/processed/`: intentionally unused for copied datasets or matrices.
- `docs/`: reports, research notes, protocol, and this progress report.
- `notebooks/`: thin presentation notebooks `00` through `07`; notebook `08`
  is reserved for future final evaluation.
- `src/data/`: loading, validation, auditing, and temporal splitting.
- `src/features/`: baseline, paper-aligned, structural, and operand extraction.
- `src/models/`: baseline estimators, evaluation, and MulCas implementation.
- `src/experiments/`: Phase 6 and Phase 7 experiment runners.
- `src/evaluation/`: metrics and temporal evaluation helpers.
- `tests/`: unit, leakage, determinism, frozen-manifest, and experiment tests.
- `results/baseline/`: compact baseline outputs.
- `results/experiments/`: Phase 6 and Phase 7 machine-readable outputs.
- `results/artifacts/`: ignored fitted models/transformers and large artifacts.

`data/processed/` can remain empty because the project deliberately stores
split membership and compact summaries instead of duplicating the raw CSV or
committing large feature matrices. Trained objects belong under
`results/artifacts/`, not `data/processed/`.

## 8. Local Setup on Windows PowerShell

Run all commands from the repository root.

```powershell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm the interpreter and imports:

```powershell
python -c "import sys; print(sys.executable); print(sys.version)"
python -c "import numpy, pandas, sklearn, xgboost, lightgbm, gensim; print('Imports OK')"
```

`sys.executable` should point to `.venv\Scripts\python.exe`.

Verify the raw dataset without editing it:

```powershell
Test-Path -LiteralPath data\raw\PonziDataset_20221114.csv
Get-FileHash -Algorithm SHA256 data\raw\PonziDataset_20221114.csv
```

The expected SHA-256 is the value recorded in Section 3.

## 9. How to Test the Current Project

Run the complete suite:

```powershell
python -m pytest -q
```

At the time of this report, the suite contains **100 passing tests**. A pytest
cache permission warning can occur on Windows without representing test
failure; use the final pass/fail summary.

Useful focused checks:

```powershell
python -m pytest tests/test_data_validation.py tests/test_temporal_split.py -q
python -m pytest tests/test_feature_leakage.py tests/test_paper_aligned_protocol.py -q
python -m pytest tests/test_mulcas.py tests/test_evaluation_metrics.py -q
python -m pytest tests/test_phase6a.py tests/test_phase6b.py tests/test_phase6c.py -q
python -m pytest tests/test_phase6d1.py tests/test_phase6d2.py tests/test_phase6e.py -q
python -m pytest tests/test_phase7a.py tests/test_phase7b.py tests/test_phase7c.py -q
```

Check that the raw data and frozen artifacts were not accidentally changed:

```powershell
git diff -- data/raw/PonziDataset_20221114.csv
git status --short
```

The raw-data diff should be empty. Review any baseline or frozen-manifest
changes before accepting regenerated outputs.

## 10. How to Reproduce the Pipeline

### Baseline through Phase 5.5

```powershell
python -m src.data.audit
python -m src.data.split --config configs/data_split.yaml
python -m src.features.reproduce --config configs/feature_reproduction.yaml
python -m src.models.run_baselines --config configs/baseline_models.yaml
python -m src.models.run_mulcas --config configs/mulcas_reproduction.yaml

python -m src.data.paper_aligned_split --config configs/paper_aligned_split.yaml
python -m src.features.paper_aligned --config configs/paper_aligned_features.yaml
python -m src.models.run_baselines --config configs/paper_aligned_baseline_models.yaml
python -m src.models.run_mulcas --config configs/paper_aligned_mulcas.yaml
```

These commands retrain models and regenerate baseline outputs. Do not run them
merely to inspect existing results; open the saved JSON/CSV/report files instead.

### Archived Phase 6 Experiments

Each command can be run independently after Phase 5.5 artifacts exist:

```powershell
python -m src.experiments.phase6a --config configs/experiments/phase6a/cost_sensitive.yaml
python -m src.experiments.phase6b --config configs/experiments/phase6b/training_sampling.yaml
python -m src.experiments.phase6c --config configs/experiments/phase6c/threshold_optimization.yaml
python -m src.experiments.phase6d1 --config configs/experiments/phase6d1/structural_features.yaml
python -m src.experiments.phase6d2 --config configs/experiments/phase6d2/operand_features.yaml
python -m src.experiments.phase6e --config configs/experiments/phase6e/alternative_classifiers.yaml
```

These are archived negative experiments. Rerunning them is for reproducibility,
not an invitation to select candidates after seeing validation results.

### Phase 7 Analyses

Run in order because 7B and 7C reuse frozen prior outputs:

```powershell
python -m src.experiments.phase7a --config configs/phase7/temporal_robustness.yaml
python -m src.experiments.phase7b --config configs/phase7/view_ablation.yaml
python -m src.experiments.phase7c --config configs/phase7/creator_ablation.yaml
```

### Notebooks

Start Jupyter from the activated environment:

```powershell
python -m jupyter notebook
```

Notebooks `00_environment_check.ipynb` through `07_ablation_study.ipynb` are
presentation layers that read generated artifacts. Core logic belongs under
`src/`. Notebook `08_final_evaluation.ipynb` is not yet an executed final study.

Non-interactive execution example:

```powershell
New-Item -ItemType Directory -Force results\temp | Out-Null
python -m jupyter nbconvert --execute --to notebook `
  --output-dir results\temp `
  --output ablation_study_executed.ipynb `
  notebooks\07_ablation_study.ipynb
```

## 11. What Researchers Should Do

- Keep the raw CSV byte-for-byte unchanged and verify its SHA-256.
- Activate `.venv` and run commands from the repository root.
- Treat Phase 5.5 as the frozen reference system.
- Use chronological, forward-only evaluation for future work.
- Fit every learned transform only on the applicable historical training rows.
- Preserve creator-history semantics: only strictly earlier labels are known.
- Keep validation, final test, and published metrics clearly labeled.
- Add new work under new config/result namespaces and record hashes and seeds.
- Run focused tests while developing and all tests before reporting results.
- Store compact metrics and traces in Git; keep large fitted artifacts under
  ignored `results/artifacts/` paths.
- Interpret temporal changes descriptively unless stronger drift evidence is
  established by an appropriate statistical analysis.

## 12. What Researchers Should Not Do

- Do not edit, sort, deduplicate, clean, or resave the raw CSV.
- Do not shuffle before chronological partition assignment.
- Do not fit vocabularies, IDF, embeddings, scalers, or models on future rows.
- Do not use validation or test labels to tune after viewing their results.
- Do not evaluate the final test repeatedly while developing improvements.
- Do not overwrite Phase 5.5 or archived Phase 6/7 outputs.
- Do not combine failed Phase 6 interventions post hoc and call the result a
  prespecified experiment.
- Do not treat a diagnostic fallback as an eligible winner.
- Do not claim the reconstructed values are the paper's published values.
- Do not infer that creator identity causes fraud from a predictive flag.
- Do not claim concept drift is proven solely from changing period metrics.
- Do not commit `.venv`, raw/local datasets, secrets, caches, large matrices, or
  fitted model files.
- Do not place reusable research logic only in notebooks.

## 13. Key Result Locations

- Dataset audit: `results/baseline/dataset_audit.json`
- Initial split: `results/baseline/temporal_split_metadata.json`
- Initial features: `results/baseline/feature_view_summary.json`
- Initial model results: `results/baseline/individual_baseline_results.csv`
- Initial MulCas: `results/baseline/mulcas_reproduction_results.json`
- Phase 5.5 baseline: `results/baseline/paper_aligned/mulcas_results.json`
- Phase 5.5 report: `docs/paper_aligned_baseline_report.md`
- Phase 6 closure: `results/experiments/phase6_summary.json`
- Phase 7A: `results/experiments/phase7a/phase7a_results.json`
- Phase 7B: `results/experiments/phase7b/phase7b_results.json`
- Phase 7C: `results/experiments/phase7c/phase7c_results.json`
- Experiment chronology: `docs/experiment_log.md`

The JSON and CSV files are the machine-readable source of truth. Markdown
reports summarize them for explanation and presentation.

