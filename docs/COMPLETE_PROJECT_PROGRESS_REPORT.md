# Complete Project Progress Report

## MulCas Ethereum Ponzi Smart-Contract Detection

## 1. Report Purpose

This document is a complete, current account of the project through the formal
closure of Phase 7. It explains what was implemented, why each phase was
performed, what results were obtained, what conclusions are supported, what
limitations remain, and what work is still pending.

This is a new report. It does not replace or modify the earlier progress,
baseline, Phase 6, or Phase 7 reports. Machine-readable JSON and CSV artifacts
remain the source of truth for exact experimental values.

## 2. Executive Summary

The project independently reconstructs MulCas, a multi-view cascade system for
detecting Ethereum Ponzi smart contracts from static information available at
or near contract creation time. The work uses the released
`PonziDataset_20221114.csv` dataset and the paper *Securing the Ethereum from
Smart Ponzi Schemes*.

The project has completed the following major objectives:

1. Established a local, reproducible Python research environment.
2. Audited and fingerprinted the released dataset without modifying it.
3. Implemented leakage-safe chronological splitting and feature fitting.
4. Independently reproduced the paper's principal static feature families.
5. Evaluated Ridge, SVM, and XGBoost baseline classifiers.
6. Independently reconstructed the MulCas cascade and voting system.
7. Corrected the baseline protocol to align more closely with the publication.
8. Tested controlled improvement hypotheses for weighting, sampling,
   thresholds, features, and alternative classifiers.
9. Closed Phase 6 after none of those interventions demonstrated reliable
   improvement under the predefined temporal protocol.
10. Evaluated temporal robustness, viewer contribution, creator-history
    contribution, error patterns, and historical code novelty.
11. Closed Phase 7 with conservative answers to RQ4 and RQ5.

The frozen Phase 5.5 MulCas final-test result is:

| Precision | Recall | F1 | PR-AUC | Balanced accuracy | TN | FP | FN | TP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9063 | 0.4531 | 0.6042 | 0.6624 | 0.7232 | 442 | 3 | 35 | 29 |

This means the reconstructed system is usually correct when it predicts Ponzi,
but it misses more than half of the Ponzi contracts in the final Phase 5.5 test
partition.

The central result after Phase 7 is:

> The reconstructed MulCas model exhibits substantial temporal performance
> variation. Component and creator-feature contributions are period-dependent.
> Later-period errors are concentrated among contracts with unseen creators
> and historically novel code, and many are confidently misclassified. Exact
> historical code reuse is associated with easier detection where cloned Ponzi
> contracts occur, but clone prevalence does not fully explain temporal
> performance variation.

The project does not claim that concept drift has been proven, that novel code
causes errors, or that creator identity causes Ponzi behavior.

## 3. Current Status at a Glance

| Phase | Purpose | Status | Main result |
| --- | --- | --- | --- |
| 0 | Local environment and repository setup | Complete | Reproducible `.venv`, package structure, path/seed helpers, tests |
| 1 | Dataset integrity audit | Complete | Released file verified: 6,498 contracts, 314 Ponzi |
| 2 | Initial temporal split | Complete | Chronological 70/15/15 split created without shuffling |
| 3 | Feature reproduction | Complete | Count, TF-IDF, N-gram, Word2Vec, and creator views implemented |
| 4 | Individual baselines | Complete | Ridge, SVM, and XGBoost evaluated consistently |
| 5 | Independent MulCas reconstruction | Complete | Cascade, voting, traces, and baseline report implemented |
| 5.5 | Paper-aligned correction | Complete and frozen | Corrected split/features; frozen reference established |
| 6A | Cost-sensitive learning | Complete | H6A not supported |
| 6B | Training-data sampling | Complete | H6B not supported |
| 6C | Decision-threshold optimization | Complete | H6C not supported |
| 6D1 | Structural opcode features | Complete | H6D1 not supported |
| 6D2 | Normalized operand features | Complete | H6D2 not supported |
| 6E | Alternative classifiers | Complete | H6E not supported |
| 6F | Combine supported improvements | Not executed by design | No independently supported intervention existed |
| 6 closure | Improvement research closure | Complete | Frozen Phase 5.5 system retained |
| 7A | Temporal robustness | Complete | Strong P2-P5 performance variation |
| 7B | Viewer/component ablation | Complete | Contributions are period-dependent |
| 7C | Creator-feature ablation | Complete | Creator signal is sparse and not uniformly beneficial |
| 7D | Systematic error analysis | Complete | Errors often involve unseen creators and confident routing mistakes |
| 7E | Clone/novelty sensitivity | Complete | Clones are easier where present, but do not fully explain variation |
| 7 closure | Robustness and ablation closure | Complete | RQ4/RQ5 evidence consolidated |
| 8 | Final synthesis/evaluation notebook and research packaging | Pending | `08_final_evaluation.ipynb` is empty |
| Optional prototype | New-contract warning interface | Not started | Outside the completed research scope |

## 4. Research Objective and Questions

The practical objective is to reproduce MulCas faithfully enough to establish
a defensible benchmark, then test whether carefully controlled changes improve
Ponzi detection without temporal leakage.

The completed work provides evidence for the following questions:

- **RQ1:** Can the published static feature views and individual classifiers be
  independently reproduced?
- **RQ2:** Can the MulCas cascade and voting process be independently rebuilt?
- **RQ3:** Do controlled changes improve the frozen reconstruction under
  forward-only temporal evaluation?
- **RQ4:** Which feature groups and MulCas components contribute to its
  predictions?
- **RQ5:** Does the reconstruction remain effective on contracts created later
  than its historical training data?

The evidence supports independent reconstruction for RQ1/RQ2, but not exact
replication of the paper's scores. None of the Phase 6 changes reliably
improved the model for RQ3. For RQ4, contribution is period-dependent. For RQ5,
performance is useful in some periods but not temporally stable.

## 5. Dataset and Integrity Findings

The immutable source is:

`data/raw/PonziDataset_20221114.csv`

| Property | Measured value |
| --- | ---: |
| File size | 307,176,676 bytes |
| SHA-256 | `48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf` |
| Total contracts | 6,498 |
| Ponzi contracts | 314 |
| Benign contracts | 6,184 |
| Earliest block | 49,880 |
| Latest block | 7,136,486 |
| Timestamp range | 2015-08-07 to 2019-01-27 |
| Unique creators | 3,360 |

The row and class totals match the paper's headline dataset description. The
audit found no null values and no empty opcode strings. It also documented:

- 30 duplicate-address rows;
- 197 duplicate transaction-hash rows;
- 30 duplicate block/address rows;
- 1,093 creators associated with multiple contracts;
- 55 creators associated with multiple positive contracts; and
- 195 groups of exact duplicate opcode strings covering 712 rows.

These rows were not removed. The project reports the source as released and
does not silently clean it to obtain a preferred result.

Primary artifacts:

- `results/baseline/dataset_audit.json`
- `docs/dataset_description.md`

## 6. Reproducibility and Leakage Controls

The project uses the following controls throughout:

- contracts are ordered by `createdBlockNumber`;
- chronological partitions are assigned without shuffling;
- train, validation, and test memberships are stored explicitly;
- vocabulary, IDF, Word2Vec, scaler, and model fitting use historical training
  rows only;
- later intervals are transform-only;
- creator features use labels from strictly earlier historical contracts;
- temporal CV always trains on earlier contracts and predicts later contracts;
- random components use the configured seed, normally 42;
- exact dataset, configuration, and frozen-artifact hashes are recorded;
- large generated matrices and fitted models are kept out of Git;
- notebooks present saved outputs rather than containing the core algorithms;
- final-test labels were prohibited from Phase 6 candidate selection.

The current automated test suite contains **118 tests** covering imports,
validation, leakage, deterministic behavior, feature extraction, estimators,
MulCas routing, frozen manifests, Phase 6 experiments, and Phase 7 analyses.

## 7. Phase-by-Phase Work Through the Frozen Baseline

### 7.1 Phase 0: Repository and Environment

Phase 0 prepared the project for local Windows development:

- created the Python package structure under `src/`;
- added missing `__init__.py` files;
- added repository-relative path helpers;
- added deterministic seed handling;
- established pytest environment/import checks;
- expanded `.gitignore` for `.venv`, caches, data, secrets, model files, and
  notebook checkpoints; and
- documented PowerShell setup and testing.

No data transformation or ML training occurred in this phase.

### 7.2 Phase 1: Dataset Audit

Reusable loading and validation functions were implemented under `src/data`.
The first audit avoided unnecessary loading of the largest text columns. It
measured schema, types, labels, missing values, duplicates, block/timestamp
ranges, chronology, creator reuse, opcode availability, memory use, and SHA-256.

The raw CSV remained byte-for-byte unchanged.

### 7.3 Phase 2: Initial 70/15/15 Chronological Split

| Partition | Rows | Ponzi | Benign | Benign:Ponzi | Block range |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 4,548 | 140 | 4,408 | 31.49:1 | 49,880-4,572,153 |
| Validation | 974 | 8 | 966 | 120.75:1 | 4,572,291-4,955,839 |
| Test | 976 | 166 | 810 | 4.88:1 | 4,957,244-7,136,486 |

The split was chronological and disjoint by row/address. Exact static code did
cross some split boundaries, and those duplicates were recorded. The key
weakness was class allocation: validation had only eight positive examples,
while test had 166, making validation-based decisions unstable.

### 7.4 Phase 3: Feature Reproduction

| Original feature view | Dimension | Description |
| --- | ---: | --- |
| Opcode term count | 142 | Individual opcode frequencies |
| Opcode TF-IDF | 142 | Training-IDF-weighted opcode frequencies |
| Opcode 1-3-gram count | 70,867 | Counts of opcode sequences of length 1-3 |
| Opcode 1-3-gram TF-IDF | 70,867 | Training-IDF-weighted N-grams |
| Mean Word2Vec | 100 | Mean training-learned opcode embedding |
| Creator feature | 1 | Prior-Ponzi-creator historical indicator |

All learned transforms were fitted on training data only.

### 7.5 Phase 4: Individual Baseline Models

Ridge, linear SVM, and XGBoost were evaluated with fixed baseline settings. On
the initial split, the validation-F1-selected representations gave:

| Model | Selected view | Validation F1 | Test precision | Test recall | Test F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Ridge | N-gram TF-IDF | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SVM | N-gram count | 0.3333 | 0.5897 | 0.1386 | 0.2244 |
| XGBoost | Term count | 0.5455 | 0.0000 | 0.0000 | 0.0000 |

These weak and unstable results helped motivate the paper-aligned correction.

### 7.6 Phase 5: Independent MulCas Reconstruction

MulCas was implemented as reusable cascade/voting code with configurable viewer
order and thresholds, preserved sample identity, and per-contract traces.

| Split | Precision | Recall | F1 | PR-AUC | Balanced accuracy | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.5714 | 0.5000 | 0.5333 | 0.4160 | 0.7484 | 963 | 3 | 4 | 4 |
| Test | 0.6000 | 0.1084 | 0.1837 | 0.5222 | 0.5468 | 798 | 12 | 148 | 18 |

The reconstruction missed most test positives and was clearly below the
published system.

### 7.7 Phase 5.5: Paper-Aligned Correction

Phase 5.5 did not overwrite Phases 2-5. It created a separate, corrected
baseline using positive-occurrence boundaries:

- training through the 200th chronological Ponzi;
- validation after that through the 250th Ponzi; and
- final test after the 250th through the 314th Ponzi.

Benign rows follow the corresponding whole-block boundaries.

| Partition | Rows | Ponzi | Benign | Benign:Ponzi | Block range |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 5,839 | 200 | 5,639 | 28.20:1 | 49,880-5,021,809 |
| Validation | 150 | 50 | 100 | 2.00:1 | 5,021,917-5,038,334 |
| Test | 509 | 64 | 445 | 6.95:1 | 5,038,337-7,136,486 |

The paper reports 5,990 rows through P4 and 508 in P5; the released data and
whole-block policy produce 5,989 and 509. This one-row difference remains
documented rather than manually corrected.

Paper-aligned representations include:

| Representation | Dimension including creator |
| --- | ---: |
| Term count | 143 |
| TF-IDF | 143 |
| DF-filtered 1-3-gram count | 44,672 |
| DF-filtered 1-3-gram TF-IDF | 44,672 |
| Mean Word2Vec | 101 |
| IDF-weighted Word2Vec | 101 |

Individual selected final-test results:

| Model | Representation | Precision | Recall | F1 | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Ridge | Term count | 0.9310 | 0.4219 | 0.5806 | 0.6365 |
| SVM | Filtered N-gram TF-IDF | 0.8824 | 0.2344 | 0.3704 | 0.6717 |
| XGBoost | Filtered N-gram TF-IDF | 0.8333 | 0.2344 | 0.3659 | 0.5943 |

The frozen MulCas architecture is:

1. **Viewer 1:** XGBoost with opcode term counts. It accepts scores below 0.05
   as benign and above 0.95 as Ponzi.
2. **Viewer 2:** XGBoost with filtered N-gram counts. It accepts scores below
   0.05 as benign and above 0.51 as Ponzi.
3. **Voting:** unresolved rows receive votes from term-count XGBoost,
   N-gram-count XGBoost, N-gram-TF-IDF XGBoost, and TF-IDF Ridge.
4. **Voting rule:** `lambda = 0`, so at least one positive vote produces a
   positive result.
5. **Creator feature:** the leakage-safe creator indicator is appended to every
   static view.

Frozen MulCas results:

| Split | Accuracy | Balanced accuracy | Precision | Recall | F1 | PR-AUC | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.8933 | 0.9000 | 0.7931 | 0.9200 | 0.8519 | 0.8706 | 88 | 12 | 4 | 46 |
| Final test | 0.9253 | 0.7232 | 0.9063 | 0.4531 | 0.6042 | 0.6624 | 442 | 3 | 35 | 29 |

Comparison with the paper:

| Result | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Published Table 5 | 0.9510 | 0.6740 | 0.7890 |
| Initial independent reconstruction | 0.6000 | 0.1084 | 0.1837 |
| Corrected Phase 5.5 reconstruction | 0.9063 | 0.4531 | 0.6042 |

The paper's nearby prose states F1 = 0.743 while Table 5 states F1 = 0.789.
This inconsistency is recorded rather than silently resolved. Exact replication
is not claimed because the publication omits several implementation details,
including the exact N-gram cutoff, complete model parameters, library versions,
and row membership.

## 8. Controlled Improvement Research: Phase 6

### 8.1 Shared Selection Protocol

Phase 6 used three expanding, forward-only temporal folds inside the Phase 5.5
training partition:

| Fold | Historical training | Later CV interval |
| --- | --- | --- |
| 1 | 104 rows, 50 Ponzi | 305 rows, 50 Ponzi |
| 2 | 409 rows, 100 Ponzi | 5,141 rows, 50 Ponzi |
| 3 | 5,550 rows, 150 Ponzi | 289 rows, 50 Ponzi |

Candidates required mean temporal-CV precision of at least 0.80. Eligible
candidates were ranked by F1, recall, balanced accuracy, and simplicity. The
Phase 5.5 validation partition was evaluated only after temporal-CV selection.
The Phase 5.5 final-test labels were not used for Phase 6 selection.

### 8.2 Phase 6A: Cost-Sensitive Learning

Ridge and SVM class weights and XGBoost `scale_pos_weight` were tested in small,
predefined ranges. No selected family variant passed the CV precision rule.
Validation classification remained similar while PR-AUC was lower than the
frozen reference. **H6A was not supported.**

### 8.3 Phase 6B: Sampling

Random under-sampling and over-sampling tested natural, 1:10, 1:5, 1:2, and
1:1 positive:negative targets. Sampling occurred only after fold-local feature
construction and only on training rows.

| Selected diagnostic | Validation precision | Recall | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Ridge, under 1:10 | 0.7931 | 0.9200 | 0.8519 | 0.7175 |
| SVM, over 1:10 | 0.7931 | 0.9200 | 0.8519 | 0.8074 |
| XGBoost, under 1:1 | 0.7302 | 0.9200 | 0.8142 | 0.8796 |

Only the SVM selection met the temporal precision rule, but it did not improve
the frozen validation result. **H6B was not supported.**

### 8.4 Phase 6C: Decision Thresholds

Only native classifier operating thresholds changed. MulCas cascade confidence
thresholds and voting remained frozen.

| Model | Selected threshold | Eligible | Validation precision | Recall | F1 |
| --- | ---: | --- | ---: | ---: | ---: |
| Ridge | -0.9730 | No, fallback | 0.5000 | 0.9800 | 0.6622 |
| SVM | 1.6765 | Yes | 0.0000 | 0.0000 | 0.0000 |
| XGBoost | 0.9936 | Yes | 0.0000 | 0.0000 | 0.0000 |

The eligible SVM/XGBoost thresholds were too conservative on external
validation. The Ridge fallback raised recall but greatly reduced precision and
F1. **H6C was not supported.**

### 8.5 Phase 6D1: Structural Opcode Features

Seventy-four fit-free features described opcode size/diversity, canonical EVM
categories, selected operations, PUSH/stack structure, and sequence complexity.
No candidate met the CV precision floor. Ridge/SVM matched classification
counts but reduced PR-AUC; XGBoost retained the unchanged control.
**H6D1 was not supported.**

### 8.6 Phase 6D2: Normalized Operand Features

Operand extraction was feasible from `contractCode`. Thirty-seven normalized
features described operand volume, constant structure, widths, and size-pattern
indicators without exposing exact addresses, selectors, or raw constants.

The audit parsed all 6,498 contracts and recorded 3,425 truncated PUSH
instructions and 42,382 unsupported opcode bytes. Affected contracts were not
dropped.

| Selected diagnostic | Validation precision | Recall | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Ridge, operand only | 0.6667 | 0.0800 | 0.1429 | 0.7632 |
| SVM, baseline + width | 0.7931 | 0.9200 | 0.8519 | 0.8354 |
| XGBoost, baseline + width | 0.7931 | 0.9200 | 0.8519 | 0.8681 |

No candidate met the temporal eligibility rule. **H6D2 was not supported.**

### 8.7 Phase 6E: Alternative Classifiers

Logistic Regression and LightGBM 4.7.0 were tested on frozen feature views.
Neither produced an eligible temporal-CV candidate.

| Selected diagnostic | Validation precision | Recall | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression, term count | 0.7931 | 0.9200 | 0.8519 | 0.7635 |
| LightGBM, term count | 0.8036 | 0.9000 | 0.8491 | 0.8386 |

**H6E was not supported.**

### 8.8 Phase 6 Closure

None of the tested Phase 6 interventions demonstrated reliable improvement
under the predefined temporal protocol. Phase 6F was not executed because no
independently supported intervention existed to combine. This is a negative
experimental result, not evidence that improvement is impossible.

The frozen Phase 5.5 system remained the reference for Phase 7.

## 9. Robustness and Explanation Research: Phase 7

### 9.1 Period Definitions

| Period | Rows | Ponzi | Benign | Block range |
| --- | ---: | ---: | ---: | --- |
| P0 | 104 | 50 | 54 | 49,880-1,287,252 |
| P1 | 305 | 50 | 255 | 1,287,353-2,000,473 |
| P2 | 5,141 | 50 | 5,091 | 2,008,140-4,963,650 |
| P3 | 289 | 50 | 239 | 4,964,082-5,021,809 |
| P4 | 150 | 50 | 100 | 5,021,917-5,038,334 |
| P5 | 509 | 64 | 445 | 5,038,337-7,136,486 |

The relationship to Phase 5.5 is:

```text
Phase 5.5 train      = P0 + P1 + P2 + P3
Phase 5.5 validation = P4
Phase 5.5 test       = P5
```

### 9.2 Phase 7A: Temporal Robustness

| Predict | Training periods | Precision | Recall | F1 | PR-AUC | Balanced accuracy | TN | FP | FN | TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | P0 + P1 | 0.0492 | 0.2600 | 0.0828 | 0.0462 | 0.6053 | 4,840 | 251 | 37 | 13 |
| P3 | P0-P2 | 0.5385 | 0.1400 | 0.2222 | 0.4786 | 0.5574 | 233 | 6 | 43 | 7 |
| P4 | P0-P3 | 0.7931 | 0.9200 | 0.8519 | 0.8706 | 0.9000 | 88 | 12 | 4 | 46 |
| P5 | P0-P4 | 0.8710 | 0.4219 | 0.5684 | 0.6674 | 0.7064 | 441 | 4 | 37 | 27 |

Published temporal comparison:

| Period | Published precision / recall / F1 | Reconstruction precision / recall / F1 |
| --- | --- | --- |
| P2 | 0.8800 / 0.3800 / 0.5300 | 0.0492 / 0.2600 / 0.0828 |
| P3 | 0.9600 / 0.3200 / 0.4800 | 0.5385 / 0.1400 / 0.2222 |
| P4 | 0.8100 / 0.9400 / 0.8700 | 0.7931 / 0.9200 / 0.8519 |
| P5 | 0.9500 / 0.6700 / 0.7900 | 0.8710 / 0.4219 / 0.5684 |

Performance is strongly non-monotonic. P4 is close to the publication; P2,
P3, and P5 are materially weaker. This is temporal performance variation, not
proof of concept drift.

### 9.3 Phase 7B: Viewer Ablation

| Removed component | P2 F1 delta | P3 F1 delta | P4 F1 delta | P5 F1 delta |
| --- | ---: | ---: | ---: | ---: |
| First cascade viewer | -0.0073 | -0.0437 | 0.0000 | -0.1981 |
| Second cascade viewer | +0.0156 | -0.0768 | 0.0000 | -0.0740 |
| Voting stage | -0.0005 | 0.0000 | 0.0000 | -0.0152 |

The first viewer is particularly important in P5. The second viewer has mixed
effects, and voting contributes only marginally. Standalone viewers can fail
badly, including zero P3 recall.

### 9.4 Phase 7C: Creator-Feature Ablation

Delta means without creator minus with creator.

| Period | Flagged rows | Delta precision | Delta recall | Delta F1 | Delta PR-AUC | Delta FP | Delta FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | 1 | +0.0195 | -0.0400 | +0.0220 | +0.0261 | -102 | +2 |
| P3 | 2 | -0.0147 | +0.0800 | +0.0876 | +0.1453 | +4 | -4 |
| P4 | 6 | 0.0000 | 0.0000 | 0.0000 | -0.0087 | 0 | 0 |
| P5 | 11 | +0.0078 | +0.0312 | +0.0295 | +0.0165 | 0 | -2 |

Creator history is sparse and not uniformly beneficial. P4 classifications do
not depend on it, and removing it recovers two P5 positives without adding
false positives. These are predictive interactions, not causal findings.

### 9.5 Phase 7D: Error Analysis

| Period | FP | FN | Errors from unseen creators | Dominant route |
| --- | ---: | ---: | ---: | --- |
| P2 | 251 | 37 | 98.3% | Second-viewer false positives |
| P3 | 6 | 43 | 98.0% | First-viewer false negatives |
| P4 | 12 | 4 | 87.5% | First-viewer errors |
| P5 | 4 | 37 | 92.7% | First-viewer errors |

Among the 37 P5 false negatives:

- 36 involve unseen creators;
- all 37 are historically novel under opcode/runtime exact matching;
- 28 are confidently rejected by viewer 1;
- one is confidently rejected by viewer 2;
- eight reach voting and remain incorrect; and
- only three are near a predefined confidence boundary.

There is no single structural or operand profile that explains errors in every
period. Error mechanisms themselves are period-dependent.

### 9.6 Phase 7E: Historical Clone Sensitivity

| Period | Any exact clones | Cloned Ponzi | Recall cloned Ponzi | Recall novel Ponzi |
| --- | ---: | ---: | ---: | ---: |
| P2 | 29/5,141 (0.6%) | 0/50 | N/A | 26.0% |
| P3 | 1/289 (0.3%) | 0/50 | N/A | 14.0% |
| P4 | 15/150 (10.0%) | 14/50 | 100.0% | 88.9% |
| P5 | 8/509 (1.6%) | 1/64 | 100.0% | 41.3% |

P5 creator/novelty interaction:

| Group | Ponzi | TP | FN | Recall |
| --- | ---: | ---: | ---: | ---: |
| Seen creator + clone | 1 | 1 | 0 | 100.0% |
| Seen creator + novel code | 3 | 2 | 1 | 66.7% |
| Unseen creator + novel code | 60 | 24 | 36 | 40.0% |

Exact reuse is associated with easier detection where cloned positives occur,
but subgroup sizes are small and P4 still performs strongly on novel positives.
Code novelty therefore does not fully explain temporal variation and is not
shown to cause misclassification.

### 9.7 Phase 7 Closure and Research Questions

| Research question | Answer supported by current evidence |
| --- | --- |
| RQ4: Which components/features contribute most? | Viewer and creator contributions are complementary and period-dependent. Viewer 1 is important in P5, viewer 2 is mixed, voting is small, and creator history is sparse. |
| RQ5: Does effectiveness persist on later contracts? | It persists in some periods but is not stable. P5 retains high precision but only 0.4219 recall. The evidence is consistent with distribution variation, not proof of concept drift. |

Phase 7 is formally complete.

## 10. What the Project Has Demonstrated

The project has demonstrated that:

1. The released dataset can be audited and reproduced with explicit hashes.
2. Static opcode feature families can be implemented without fitting on future
   contracts.
3. The MulCas architecture can be independently reconstructed with inspectable
   routing and voting traces.
4. A paper-aligned protocol materially improves the credibility of the
   reconstruction compared with the initial percentage split.
5. The corrected reconstruction approaches published precision but remains
   below published recall and F1.
6. Several common imbalance and classifier interventions do not automatically
   improve performance under temporal testing.
7. Strong validation or single-period performance does not imply temporal
   robustness.
8. Different cascade components matter in different periods.
9. Creator history and exact code reuse are sparse signals.
10. Many later false negatives are confidently misclassified, so the issue is
    not merely that the system refuses to decide on difficult cases.

## 11. What the Project Has Not Demonstrated

The current evidence does not demonstrate that:

- the paper has been exactly replicated;
- a new model reliably outperforms the frozen baseline;
- concept drift has been statistically proven;
- novel code causes model errors;
- creator identity causes Ponzi behavior;
- exact clone prevalence fully explains period differences;
- a live deployment would generalize to present-day Ethereum contracts; or
- the current scores apply to contracts created after the dataset's 2019 end.

These boundaries are important parts of the research result, not weaknesses to
hide.

## 12. Current Repository Outputs

### Baseline and Protocol

- `results/baseline/dataset_audit.json`
- `results/baseline/temporal_split_metadata.json`
- `results/baseline/feature_view_summary.json`
- `results/baseline/individual_baseline_results.json`
- `results/baseline/mulcas_reproduction_results.json`
- `results/baseline/paper_aligned/split_metadata.json`
- `results/baseline/paper_aligned/feature_summary.json`
- `results/baseline/paper_aligned/individual_baseline_results.json`
- `results/baseline/paper_aligned/mulcas_results.json`

### Research Closures

- `docs/phase6_summary.md`
- `results/experiments/phase6_summary.json`
- `docs/phase7_summary.md`
- `results/experiments/phase7_summary.json`

### Phase 6 Experiments

- `results/experiments/phase6a/` through `phase6e/`

### Phase 7 Analyses

- `results/experiments/phase7a/` through `phase7e/`

### Notebooks

| Notebook | Status |
| --- | --- |
| `00_environment_check.ipynb` | Present but currently empty |
| `01_dataset_audit.ipynb` | Implemented |
| `02_dataset_eda.ipynb` | Implemented |
| `03_feature_reproduction.ipynb` | Implemented |
| `04_baseline_models.ipynb` | Implemented |
| `05_mulcas_reproduction.ipynb` | Implemented |
| `06_improvement_experiments.ipynb` | Implemented through Phase 6 closure |
| `07_ablation_study.ipynb` | Implemented through Phase 7E |
| `08_final_evaluation.ipynb` | Empty and pending |

## 13. How to Run and Verify the Current Project

From Windows PowerShell in the repository root:

```powershell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the environment and dataset:

```powershell
python -c "import sys; print(sys.executable); print(sys.version)"
Test-Path -LiteralPath data\raw\PonziDataset_20221114.csv
Get-FileHash -Algorithm SHA256 data\raw\PonziDataset_20221114.csv
```

Run all tests:

```powershell
python -m pytest -q
```

The current expected result is **118 passed**. A Windows permission warning for
`.pytest_cache` can appear without indicating a test failure.

Run Jupyter:

```powershell
python -m jupyter notebook
```

Core research logic belongs under `src/`; notebooks should remain presentation
layers.

Important: rerunning Phases 6 or 7 is not necessary to read the results. Their
machine-readable artifacts already exist. Reproduction runs should be performed
only when intentionally verifying the archived pipeline.

## 14. What Is Still Pending

### 14.1 Phase 8 Final Synthesis

The clearest required next step is to populate `08_final_evaluation.ipynb` as a
read-only synthesis of frozen outputs. It should:

- load Phase 5.5, Phase 6 closure, and Phase 7 closure artifacts;
- present the final baseline, published comparison, negative improvement
  results, and temporal robustness findings;
- include final confusion matrices and precision/recall/F1/PR-AUC tables;
- state RQ1-RQ5 answers and limitations;
- avoid retraining, tuning, or selecting a new model; and
- clearly label the frozen Phase 5.5 reconstruction as the retained reference.

### 14.2 Final-Test Status Must Be Described Honestly

The research plan originally proposed a final untouched test evaluation.
However, the Phase 5.5 test metrics were already inspected, and the same P5
period was subsequently analyzed in Phases 7A-7E. Therefore:

- P5 is a frozen evaluation partition, but it is no longer untouched;
- no new method should be selected or tuned using P5;
- notebook 08 should synthesize the existing P5 result, not treat it as a new
  first-time evaluation; and
- a genuinely untouched generalization result would require a new external or
  later-time labeled dataset.

This limitation should be stated explicitly in the final thesis/report.

### 14.3 Final Research Manuscript or Thesis Chapter

The final written research output remains to be assembled. It should include:

- problem statement and research questions;
- source-paper summary and reproducibility gaps;
- dataset description and audit;
- leakage-safe methodology;
- initial reconstruction and Phase 5.5 correction;
- Phase 6 hypotheses and negative results;
- Phase 7 temporal, ablation, error, and clone findings;
- threats to validity;
- answers to RQ1-RQ5;
- limitations and future work; and
- reproducibility instructions and artifact index.

### 14.4 Final Reproducibility Audit

Before submission or release, perform one clean-environment verification:

1. create a fresh `.venv`;
2. install dependencies;
3. verify the raw dataset hash;
4. run all tests;
5. execute implemented notebooks non-interactively;
6. verify frozen manifests and closure JSON files;
7. record Python and dependency versions; and
8. confirm no raw data, secrets, fitted artifacts, or temporary notebook files
   are staged for commit.

The current `requirements.txt` is mostly unpinned. Producing a fully pinned
environment snapshot or lock file remains advisable for archival reproduction.

### 14.5 Documentation Synchronization

Some earlier entry-point documents still describe the project only through
Phase 7C or list an older test count. They were intentionally not edited as part
of this request. A later documentation-only task should synchronize:

- `README.md`;
- `docs/RUN_AND_TEST_GUIDE.md`;
- the earlier `docs/PROJECT_PROGRESS_REPORT.md`; and
- notebook status descriptions.

This new report should be used as the current progress reference until that
synchronization is performed.

### 14.6 Optional External Validation

A stronger temporal-generalization claim would require a later labeled Ethereum
dataset that was not used anywhere in development or analysis. Such work would
need a new preregistered protocol covering label definitions, data provenance,
bytecode compatibility, creator-history availability, and one-time evaluation.

This is valuable future research, but it is not required to close the completed
Phase 7 work.

### 14.7 Optional Prototype

The original roadmap mentions an optional interface that accepts new contract
bytecode and creator information and produces a warning. This has not been
implemented and should remain secondary to final research reporting.

A prototype must not be presented as deployment-ready because:

- the data ends in 2019;
- P2-P5 performance is unstable;
- recall is low in several periods;
- current-day distribution compatibility is unverified; and
- the system has not been externally validated.

### 14.8 Optional Additional Research, Not Current Commitments

The following are possible future studies, not unfinished requirements of the
closed Phase 6/7 protocols:

- external later-time validation;
- repeated-seed robustness for genuinely stochastic future candidates;
- compiler/version-aware analysis;
- representation-learning methods designed for code novelty;
- calibrated uncertainty and abstention studied under a new protocol;
- inter-contract or transaction-based signals, which would change the static
  creation-time scope; and
- adversarial robustness after external validity is established.

These must be preregistered as new research rather than selected after reviewing
the existing P5 errors.

## 15. Recommended Next Actions

The recommended order is:

1. Freeze the current Phase 7 closure and this progress report.
2. Create Phase 8 as a read-only final synthesis, not a new experiment.
3. Prepare final thesis/report tables directly from machine-readable artifacts.
4. Complete a clean-environment reproducibility audit.
5. Synchronize entry-point documentation and current test counts.
6. Decide whether external validation is feasible.
7. Consider the optional prototype only after the research deliverables are
   complete and its limitations are clearly communicated.

## 16. Final Current Conclusion

The project has successfully progressed from repository setup and dataset audit
through independent MulCas reconstruction, paper-aligned correction,
controlled improvement experiments, temporal robustness analysis, component
ablation, creator-feature ablation, systematic error analysis, and historical
clone sensitivity.

The work produced a defensible frozen baseline and a substantial body of
negative and diagnostic evidence. It did not produce a reliably improved model,
but it did establish where the reconstruction succeeds, where it fails, how its
components behave over time, and why claims must remain conservative.

The research implementation through Phase 7 is complete. The main remaining
work is final synthesis, reproducibility packaging, documentation alignment,
and, optionally, external validation or a clearly caveated prototype.

