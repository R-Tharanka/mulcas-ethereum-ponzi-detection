# Complete Project Progress Report

## MulCas Ethereum Ponzi Smart-Contract Detection

| Status item | Current value |
| --- | --- |
| Report date | 2026-10-22 |
| Project status | Complete through Phase 9D under the frozen Phase 9B protocol |
| Retained reference | Frozen Phase 5.5 paper-aligned MulCas |
| External benchmark | Sealed STRICT Track B established; not yet evaluated |
| Next research phase | Phase 9E, not started and requiring explicit authorization |

## 1. Report Purpose

This document is a complete, current account of the project through Phase 9D,
including Track B construction, the Phase 9A protocol-gap audit, the frozen
Phase 9B protocol, and the Phase 9C/9D development experiments.
It explains what was implemented, why each phase was
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
11. Tested recency weighting, runtime CFG/GIN, score fusion, creation-code
    features, and self-supervised opcode-sequence representations.
12. Closed the Phase 8 static-model search without accepting a replacement for
    the frozen Phase 5.5 MulCas system.
13. Audited paper/reconstruction protocol gaps without changing the retained
    reference or treating non-deployable diagnostics as improvements.
14. Discovered, enriched, overlap-audited, and sealed a later-time external
    Track B benchmark containing 112 positive and 180 benign contracts.
15. Froze the prospective Phase 9B evaluation protocol before any Phase 9
    intervention was trained or Track B performance was observed.
16. Tested deployment-available creator and code-family state in Phase 9C;
    none of the fixed candidates met every prospective advancement condition.
17. Tested prior-period viewer calibration and fixed consensus routing in Phase
    9D; pooled results improved diagnostically, but temporal reliability gates
    rejected every candidate.

The frozen Phase 5.5 MulCas final-test result is:

| Precision | Recall | F1 | PR-AUC | Balanced accuracy | TN | FP | FN | TP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9063 | 0.4531 | 0.6042 | 0.6624 | 0.7232 | 442 | 3 | 35 | 29 |

This means the reconstructed system is usually correct when it predicts Ponzi,
but it misses more than half of the Ponzi contracts in the final Phase 5.5 test
partition.

The central result after the completed Phase 8 extension is:

> No tested Phase 8 intervention produced a reliable P2-P4 improvement under
> its predefined temporal criterion, so frozen Phase 5.5 MulCas remains the
> retained reference. The project nevertheless established a sealed,
> later-time, overlap-audited Track B benchmark for one future evaluation after
> the Phase 9 system is frozen.

The project does not claim that concept drift has been proven, that novel code
or unseen creators cause errors, or that the sealed Track B result is known.
No detector has yet been evaluated on Track B.

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
| 8R0 | Extension protocol and feasibility | Complete | RQ6, Track A/Track B governance, and staged static extension defined |
| 8R1A | Recency-aware MulCas | Complete | H8R1 not supported |
| 8R2A | Runtime CFG corpus | Complete and frozen | 6,498 deterministic graphs constructed |
| 8R2B | Graph aggregates and GIN | Complete | H8R2 not supported; diagnostic P3 benefit did not generalize |
| 8R3A | MulCas/GIN complementarity | Complete, analysis only | Complementarity justified one bounded fusion test |
| 8R3B | Lightweight score fusion | Complete | H8R3B not supported |
| 8R4A | Creation-code representation | Complete and frozen | Three deployment-time views constructed |
| 8R4B | Creation-code classification | Complete | H8R4 not supported |
| 8R5A | Opcode-sequence feasibility | Complete | Deterministic 1,024-token chunking selected |
| 8R5B0 | CUDA benchmark | Complete | Feasibility gate passed |
| 8R5B | Masked-opcode pretraining | Complete | Nine temporally valid checkpoints produced |
| 8R5C1 | Frozen sequence linear probe | Complete | H8R5 not supported by this probe; 8R5C2 not run |
| Phase 8 closure | Static extension closure | Complete | No accepted replacement; Phase 5.5 retained |
| 9A | Paper/reconstruction protocol-gap decomposition | Complete | Limited protocol explanation; no detector improvement claimed |
| 8R6A-8R6C | External-source audit and provider enrichment | Complete as staged predecessors | Candidate union and 787-row resumable evidence established |
| 8R6D | Complete enrichment and Track B sealing | Complete | 1,453 enriched; 292-contract STRICT Track B sealed |
| 9B | Prospective Phase 9 evaluation protocol | Frozen | Track A development and one-shot sealed Track B rules fixed |
| 9C | Creator/family-state intervention | Complete | H9C not supported; no candidate advanced |
| 9D | Cascade calibration and safer routing | Complete | H9D not supported; pooled D2 gain failed temporal gates |
| 9E | Selector and semantic static features | Not started | Requires explicit authorization under Phase 9B |
| Final synthesis | Notebook and manuscript packaging | Pending | `08_final_evaluation.ipynb` remains empty |
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
- **RQ6:** Can temporal adaptation or richer graph, sequence, and
  deployment-time representations improve historically novel and
  unseen-creator detection?
- **RQ7:** To what extent can deployment-available historical state,
  code-family similarity, and calibrated cascade routing improve forward
  temporal detection, and does a final frozen system generalize to sealed
  Track B?

The evidence supports independent reconstruction for RQ1/RQ2, but not exact
replication of the paper's scores. None of the Phase 6 changes reliably
improved the model for RQ3. For RQ4, contribution is period-dependent. For RQ5,
performance is useful in some periods but not temporally stable.
For RQ6, the tested Phase 8 methods did not reliably improve P2-P4 performance,
although GIN showed a period-specific P3 benefit. RQ7 is partially answered
internally: Phases 9C and 9D did not produce a candidate satisfying every
prospective advancement condition. External generalization remains unanswered
because Track B labels remain unavailable to development and have never been
scored.

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

The automated suite now covers imports, validation, leakage, deterministic
behavior, feature extraction, estimators, MulCas routing, frozen manifests,
Phase 6-9 governance, provider-resume behavior, overlap classification, and
sealed-label controls. The pre-seal Phase 8R6B/R6C assertions were migrated
after the authorized Phase 8R6D seal. The current 2026-10-22 branch run is
**345 passed and 27 failed**. The Phase 8R6D LFS object has now been restored
and its idempotence test passes. Every remaining failure is downstream of the
two frozen-artifact hash mismatches detailed in Section 16; the suite is not
misrepresented as passing.

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

## 10. Static Research Extension: Phase 8R0-8R5C1

### 10.1 Phase 8R0: Extension Protocol and Feasibility

**Work completed:** Phase 8R0 defined RQ6, separated exploratory Track A from
future independent Track B evidence, froze forward-only P2-P4 evaluation, and
specified bounded recency, graph, creation-code, and sequence branches. It also
audited whether deterministic runtime CFG construction and external-data
discovery were feasible.

**Outcome:** The extension was feasible, but no independent external benchmark
was ready. Phase 8R0 was a protocol phase and did not test an improvement
hypothesis.

### 10.2 Phase 8R1A: Recency-Aware MulCas

**Work completed:** Recent-window training and exponential-decay policies were
compared with the exact expanding-history MulCas control on P2-P4. P5 was
excluded.

| System | Macro precision | Macro recall | Macro F1 | Macro PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Frozen expanding control | 0.4603 | 0.4400 | 0.3856 | 0.4651 |
| Diagnostic decay alpha 0.75 | 0.4582 | 0.4533 | 0.4024 | 0.5044 |

**Outcome:** The diagnostic fallback improved some aggregate values but failed
the predefined precision requirement. **H8R1 was not supported.**

### 10.3 Phase 8R2A: Deterministic Runtime CFG Corpus

**Work completed:** All 6,498 runtime bytecodes were deterministically
disassembled into contract-local control-flow graphs with conservative static
jump recovery and explicit unresolved dynamic jumps.

| Graphs | Nodes | Edges | Node features | Static jump resolution |
| ---: | ---: | ---: | ---: | ---: |
| 6,498 | 1,514,545 | 1,464,910 | 55 | 74.48% |

**Outcome:** The frozen graph corpus was constructed successfully. No
classifier was trained in this construction phase.

### 10.4 Phase 8R2B: Graph Aggregates and GIN

**Work completed:** Fixed graph-aggregate Logistic Regression and a small,
predeclared GIN grid were evaluated under expanding P2-P4 prediction.

The diagnostic GIN fallback, with three layers, 128 hidden units, and dropout
0.5, produced macro P/R/F1 of 0.4652/0.5213/0.4721. Its period F1 values were
0.0074 on P2, 0.6680 on P3, and 0.7409 on P4.

**Outcome:** GIN substantially improved P3 relative to MulCas but failed P2
and the cross-period eligibility criterion. **H8R2 was not supported.**

### 10.5 Phase 8R3A: MulCas/GIN Complementarity

**Work completed:** Frozen MulCas and GIN predictions were compared without
training a new model. Across P2-P4, GIN recovered 28 positives missed by
MulCas, MulCas recovered 22 missed by GIN, and GIN correctly rejected 256
benign contracts misclassified by MulCas.

**Outcome:** Complementarity, concentrated mainly in P3, justified one bounded
fusion test. The oracle analysis was non-deployable and was not a hypothesis
winner.

### 10.6 Phase 8R3B: Lightweight Score Fusion

**Work completed:** Two fixed Logistic Regression meta-models combined frozen
MulCas and GIN scores, with one variant adding permitted historical context.
P3 was used for selection and P4 for confirmation.

**Outcome:** Both hybrids predicted no P3 positives. Diagnostic Hybrid A gave
P=0.6567, R=0.8800, and F1=0.7521 on P4, below MulCas F1=0.8519.
**H8R3B was not supported.**

### 10.7 Phase 8R4A: Creation-Code Representation

**Work completed:** All 6,498 `creationCode` values were parsed. The deployed
runtime appeared exactly once in 6,497 rows; the unmatched row received the
frozen zero-filled init representation. Full creation, deployment-init, and
combined views had 85, 90, and 183 dimensions.

**Outcome:** Deployment-time representations were frozen successfully. No
classifier was trained in this phase.

### 10.8 Phase 8R4B: Creation-Code Classification

**Work completed:** Fixed Logistic Regression and XGBoost candidates were
evaluated on the three creation-code views over P2-P4.

**Outcome:** The diagnostic Logistic Regression `creation_full` fallback had
macro P/R/F1/PR-AUC of 0.2703/0.3067/0.2818/0.5165 and recovered only one
positive missed by MulCas across all periods. **H8R4 was not supported.**

### 10.9 Phase 8R5A: Opcode-Sequence Feasibility

**Work completed:** Runtime opcode lengths, historical vocabularies, OOV rates,
truncation loss, and CPU cost were audited. Lengths ranged from 8 to 18,058
tokens; 80.92% of P0-P4 contracts would be truncated at 1,024 tokens.

**Outcome:** Deterministic 1,024-token chunking was frozen. The projected
three-seed CPU runtime was too high, so pretraining was not launched until a
separate CUDA gate passed.

### 10.10 Phase 8R5B0: CUDA Resource Gate

**Work completed:** CUDA-enabled PyTorch was validated on the local RTX 2050.
The bounded AMP benchmark estimated about 68.5 minutes for all three seeds.

**Outcome:** The resource gate was **GREEN**, authorizing only the predefined
masked-opcode pretraining work.

### 10.11 Phase 8R5B: Masked-Opcode Pretraining

**Work completed:** Nine historical-only fold/seed Transformer runs completed
with fixed masking, optimizer, architecture, and chunking. Future prediction
periods and P5 were excluded from each fold's vocabulary and pretraining data.

**Outcome:** Nine hash-recorded checkpoints were produced in 72.32 summed
minutes. This representation-learning phase did not itself test H8R5.

### 10.12 Phase 8R5C1: Frozen Sequence Linear Probe

**Work completed:** Frozen contextual and non-contextual 128-dimensional
contract embeddings were evaluated with a fixed linear probe over P2-P4.

| Representation | Macro precision | Macro recall | Macro F1 | Macro PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Contextual seed mean | 0.2841 | 0.3133 | 0.2921 | 0.4692 |
| Non-contextual control | 0.5297 | 0.4600 | 0.3870 | 0.5261 |
| Frozen MulCas | 0.4603 | 0.4400 | 0.3856 | 0.4651 |

**Outcome:** The contextual representation underperformed both controls. The
non-contextual control's small macro-F1 difference was ineligible and did not
establish improvement. **H8R5 was not supported by this frozen linear-probe
evaluation.** Phase 8R5C2 supervised fine-tuning was deliberately not executed
because its predefined gate failed; this does not disprove all possible
sequence fine-tuning methods.

### 10.13 Formal Phase 8 Closure

The closure reverified 199 frozen anchors, nine MLM checkpoints, 18 embedding
caches, and the raw dataset hash. H8R1, H8R2, H8R3B, H8R4, and the tested
H8R5 probe were not supported. No diagnostic fallback was promoted to a
research winner, and no Phase 8 intervention replaced frozen Phase 5.5 MulCas.

## 11. External Track B Construction: Phase 8R6A-8R6D

### 11.1 Phase 8R6A: External/Later Dataset Audit

**Work completed:** Eight candidate sources and 5,218 retrieved files were
audited for provenance, address overlap, temporal metadata, benign-cohort
availability, and compatibility with static creation-time features.

**Outcome:** No ready Track B existed. A promising union required provider
enrichment and exact overlap checks, so the decision was external-data
enrichment/manual verification first. No model was trained and P5 was not
evaluated.

### 11.2 Phase 8R6B: Initial Enrichment Gate

**Work completed:** The candidate union was frozen at 1,490 unique addresses:
563 positive and 927 benign records from `ponzi_ml` plus new class-A
Blockchain@Unica positives. Provider feasibility and runtime were tested.

**Outcome:** The run stopped at its performance gate before enrichment. Zero
counts were explicitly treated as unobserved, not evidence of absence. Track B
was not established.

### 11.3 Phase 8R6C: Authenticated Provider Resume

**Work completed:** Environment-only credentials, Ethereum mainnet identity,
historical RPC calls, Blockscout creation lookup, an end-to-end contract path,
a 20-address preflight, and a 100-address stability gate were validated.
Provider-backed enrichment then resumed from a deterministic ledger.

**Outcome:** The continuation stopped safely after 787 successful enrichments,
with row-level invalid, identity, and schema outcomes preserved for resumption.
Historical gate evidence and completed rows were retained; no success was
silently refetched.

### 11.4 Phase 8R6D: Complete Enrichment and Seal Track B

**Work completed:** Phase 8R6D froze the prior ledger, added safe row-level
handling for null/unsupported creation schemas while retaining strict identity
validation, resumed only unresolved rows, finalized chronology and exact
address/runtime/opcode/creation-code overlap, applied A/B provenance rules,
audited internal duplicates, and sealed eligible labels behind the default-deny
guard.

| Enrichment measure | Result |
| --- | ---: |
| Candidate addresses / attempted | 1,490 / 1,490 |
| Successful enrichment | 1,453 |
| Invalid contract / identity mismatch / not found | 34 / 2 / 1 |
| Provider failures / retries | 0 / 0 |
| MULCAS_READY | 1,453 |
| STATIC_RUNTIME_READY | 1,453 |
| CREATION_CODE_READY | 1,453 |
| Phase 8R6D runtime / cumulative provider runtime | 26.80 / 74.52 minutes |

The mandatory Track B funnel was:

| Stage | Total | Positive | Benign |
| --- | ---: | ---: | ---: |
| Starting / attempted | 1,490 | 563 | 927 |
| Successfully enriched and A/B eligible | 1,453 | 530 | 923 |
| Later deployment | 292 | 112 | 180 |
| Runtime-new and opcode-new | 292 | 112 | 180 |
| Creation-new, conflict-free, and MULCAS_READY | 292 | 112 | 180 |
| TRACK_B_STRICT | 292 | 112 | 180 |
| TRACK_B_PROVISIONAL | 0 | 0 | 0 |

**Outcome:** A **STRONG**, strict-only Track B benchmark was established and
sealed. It contains 292 unique-address contracts from blocks 7,143,846 through
10,311,201, dated 2019-01-29 through 2020-06-21. It has 287 unique runtime and
opcode identities, no unresolved A/B conflicts, and no critical manual-review
cases. Track A remains separate and byte-for-byte unchanged. No detector was
trained, scored, or evaluated on Track B.

## 12. Phase 9 Diagnostic and Prospective Protocol Work

### 12.1 Phase 9A: Paper/Reconstruction Protocol-Gap Decomposition

**Work completed:** Phase 9A audited training coverage, creator-state updates,
viewer selection, N-gram dimensions, period membership, cascade thresholds,
voting, and auxiliary-bytecode handling. It replayed frozen controls and used a
clearly labeled, non-deployable true-label online creator oracle. It introduced
no classifier, feature, threshold, or tuning change.

**Outcome:** On exploratory P5, the non-deployable creator oracle changed the
Phase 7A control from P/R/F1=0.8710/0.4219/0.5684 to
0.8824/0.4688/0.6122, adding three true positives without additional false
positives. This explained only about 19.87% of the F1 gap in isolation. The
test-selected-viewer replay was stopped by its runtime gate and was also
under-specified by the paper, so no partial ranking or combined oracle was
reported. The completed evidence was classified as **little protocol
explanation**, not as a model improvement.

### 12.2 Phase 9B: Prospective Evaluation Protocol Freeze

**Work completed:** After Track B status was known, the project froze RQ7 and
the Phase 9 selection/evaluation rules before Phase 9C began. Development is
limited to Track A expanding predictions: P0+P1 to P2, P0-P2 to P3, and P0-P3
to P4. P5 remains previously observed exploratory evidence only.

The primary measures are pooled prequential PR-AUC, pooled prequential F1, and
recall at the frozen MulCas pooled FPR ceiling of 0.04954. Advancement requires
strictly better pooled F1, no more than 0.01 absolute PR-AUC degradation, FPR
within that ceiling, nonnegative F1 change in at least two periods, no single
period degradation beyond 0.02, and paired contract-level bootstrap reporting
with 2,000 replicates and seed 42.

**Outcome:** The protocol is frozen. Phase 9C-9F may not access Track B labels.
Only after a Phase 9F system is frozen may one final Track B evaluation occur,
with no subsequent tuning.

### 12.3 Phase 9C: Deployment-Available Creator and Code-Family State

**Work completed:** Phase 9C evaluated exactly four candidates on Track A:
C0 frozen MulCas, C1 creator-state features, C2 historical code-family
features, and C3 their combination. Every feature used only strictly earlier
blocks; target-period labels did not update verified state. Model families,
cascade thresholds, viewer order, and voting remained frozen. P5 and Track B
were excluded.

| Candidate | Pooled precision | Pooled recall | Pooled F1 | Pooled PR-AUC | Pooled FPR | Advanced |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| C0 frozen control | 0.1970 | 0.4400 | 0.2722 | 0.3373 | 0.04954 | Control |
| C1 creator state | 0.2104 | 0.4600 | 0.2887 | 0.1837 | 0.04770 | No |
| C2 code family | 0.2951 | 0.4800 | 0.3655 | 0.2785 | 0.03168 | No |
| C3 combined state | 0.2969 | 0.5067 | 0.3744 | 0.2620 | 0.03315 | No |

**Outcome:** C3 improved operating-point F1, with P2/P3/P4 F1 of
0.1404/0.4000/0.8519 versus C0's 0.0828/0.2222/0.8519. Its PR-AUC fell by
more than the frozen 0.01 tolerance, however, so no candidate advanced and H9C
was not supported.

### 12.4 Phase 9D: Cascade Calibration and Safer Routing

**Work completed:** Phase 9D preregistered D0 frozen MulCas, D1 out-of-time
Platt calibration, D2 a fixed consensus safety guard, and D3 calibration plus
that guard. Calibration used the immediately preceding labeled period predicted
from still-earlier history. D2/D3 required two of four voting views to confirm
a positive and three to overturn a negative. No threshold search or underlying
viewer change occurred. P5 and Track B remained excluded.

| Candidate | Pooled precision | Pooled recall | Pooled F1 | Pooled PR-AUC | Pooled FPR | Advanced |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| D0 frozen control | 0.1970 | 0.4400 | 0.2722 | 0.3373 | 0.04954 | Control |
| D1 calibrated cascade | 0.2305 | 0.3733 | 0.2850 | 0.3439 | 0.03444 | No |
| D2 fixed consensus guard | 0.7869 | 0.3200 | 0.4550 | 0.4044 | 0.00239 | No |
| D3 calibrated guard | 0.7742 | 0.3200 | 0.4528 | 0.3872 | 0.00258 | No |

**Outcome:** D2 substantially improved pooled precision, F1, PR-AUC, and FPR,
and its paired bootstrap F1 and PR-AUC deltas were positive. It nevertheless
produced P2/P3/P4 F1 of 0.0755/0.0000/0.8519: P3 recall collapsed to zero and
P2 F1 declined slightly. D1 and D3 also lost more than 0.02 F1 in at least one
period. Every candidate therefore failed the frozen temporal-breadth and
maximum-period-degradation conditions. H9D was not supported, no replacement
was accepted, and Phase 9E was not started.

## 13. What the Project Has Demonstrated

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
11. Recency, graph, fusion, creation-code, and frozen sequence interventions
    did not reliably improve cross-period performance under their predefined
    Phase 8 rules.
12. Period-specific complementarity, especially GIN on P3, is not sufficient
    evidence for a replacement when P2/P4 behavior and eligibility fail.
13. The paper/reconstruction gap is only partly explained by completed,
    reproducible protocol diagnostics; under-specified viewer selection remains
    unresolved.
14. A later-time, exact-overlap-audited external benchmark can be constructed
    from the frozen sources while keeping Track A unchanged.
15. A prospective model-development and one-shot external-evaluation protocol
    can be frozen before any Track B performance is observed.
16. Deployment-available creator/code-family state can improve operating-point
    F1 while degrading ranking quality enough to fail a prospective rule.
17. Fixed consensus routing can sharply reduce false positives and improve
    pooled metrics while still failing temporal reliability because one period
    loses all positive detections.

## 14. What the Project Has Not Demonstrated

The current evidence does not demonstrate that:

- the paper has been exactly replicated;
- a new model reliably outperforms the frozen baseline;
- concept drift has been statistically proven;
- novel code causes model errors;
- creator identity causes Ponzi behavior;
- exact clone prevalence fully explains period differences;
- the non-deployable Phase 9A creator oracle is a usable improvement;
- Phase 9E or 9F will improve the retained baseline;
- the strong pooled Phase 9D D2 result is a reliable cross-period replacement;
- the frozen system generalizes to Track B, because sealed labels have not been
  loaded for evaluation;
- a live deployment would generalize to present-day Ethereum contracts; or
- the current results apply beyond Track B's June 2020 endpoint.

These boundaries are important parts of the research result, not weaknesses to
hide.

## 15. Current Repository Outputs

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

### Phase 8 Extension and Closure

- `results/extension/phase8r0/` through `phase8r5c1/`
- `results/extension/phase8_closure/`
- `docs/phase8_summary.md`
- `results/extension/phase8r6a/` through `phase8r6d/`
- `results/extension/phase8r6d/phase8r6d_results.json`
- `results/extension/phase8r6d/track_b_filter_funnel.csv`
- `results/extension/phase8r6d/phase8r6d_report.md`

### Sealed Track B

- `data/external/track_b/metadata.csv`
- `data/external/track_b/membership_manifest.json`
- `data/external/track_b_sealed/labels.csv` (protected and default-deny)

Track B is an external dataset, not an addition to the 6,498-row Track A file.
Its labels must remain inaccessible during Phase 9C-9F.

### Phase 9

- `results/phase9/phase9a/phase9a_results.json`
- `results/phase9/phase9a/phase9a_report.md`
- `configs/phase9/phase9b.yaml`
- `docs/phase9_evaluation_protocol.md`
- `results/phase9/phase9b/phase9b_protocol.json`
- `results/phase9/phase9b/phase9b_report.md`
- `results/phase9/phase9c/phase9c_results.json`
- `results/phase9/phase9c/phase9c_report.md`
- `configs/phase9/phase9d.yaml`
- `results/phase9/phase9d/phase9d_results.json`
- `results/phase9/phase9d/phase9d_report.md`

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

## 16. How to Run and Verify the Current Project

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

The current Phase 9D focused suite passes **10 tests**. After restoring the
tracked Phase 8R6D LFS object, the full 2026-10-22 branch run reported **345
passed, 27 failed, and 66 warnings in 298.72 seconds**. No Phase 9D test failed.
All 27 failures share one existing branch-integrity cause rather than a
candidate-performance failure:

- current `feature_summary.json` and `split_metadata.json` bytes do not match
  their frozen Phase 5.5/Phase 7 manifest hashes, so dependent frozen-chain
  tests fail as designed.

The earlier sealed-label lifecycle assertions were already migrated, and the
LFS failure is resolved. The 27 current failures must not be relabeled as
expected skips, and frozen manifests must not be silently rewritten to accept
changed historical artifacts. Exact reconciliation evidence is recorded in
`results/integrity/frozen_artifact_reconciliation.json`.

Run Jupyter:

```powershell
python -m jupyter notebook
```

Core research logic belongs under `src/`; notebooks should remain presentation
layers.

Important: rerunning completed training or provider-enrichment phases is not
necessary to read the results. Phase 8R2B graph training, Phase 8R5B sequence
pretraining, and Phase 8R6 provider enrichment are expensive and have frozen
outputs. Read and verify their reports/manifests unless an intentional
reproduction exercise has been approved. Never expose provider credentials or
unseal Track B labels during ordinary verification.

## 17. What Is Still Pending

### 17.1 Final Synthesis Notebook

`08_final_evaluation.ipynb` remains empty. It should become a read-only
synthesis of frozen outputs and must not train, tune, or select a model. It
should:

- load Phase 5.5, Phase 6-8 closures, Phase 9A, Track B construction metadata,
  the Phase 9B protocol, and completed Phase 9C/9D results;
- present the final baseline, published comparison, negative improvement
  results, and temporal robustness findings;
- include final confusion matrices and precision/recall/F1/PR-AUC tables;
- state the current RQ1-RQ6 answers and RQ7's prospective status;
- avoid retraining, tuning, or selecting a new model; and
- clearly label the frozen Phase 5.5 reconstruction as the retained reference.

### 17.2 P5 and Track B Must Be Described Honestly

The research plan originally proposed a final untouched test evaluation.
However, the Phase 5.5 test metrics were already inspected, and the same P5
period was subsequently analyzed in Phases 7A-7E. Therefore:

- P5 is a frozen evaluation partition, but it is no longer untouched;
- no new method should be selected or tuned using P5;
- notebook 08 should synthesize the existing P5 result, not treat it as a new
  first-time evaluation;
- P5 is Track A exploratory historical evidence, not an untouched final test;
- Track B now supplies the independent later-time benchmark, but it has not
  been evaluated;
- sealed Track B labels must remain inaccessible during Phase 9C-9F; and
- one final external evaluation is allowed only after the Phase 9F system is
  frozen, with no tuning afterward.

This limitation should be stated explicitly in the final thesis/report.

### 17.3 Remaining Phase 9 Research

Phases 9C and 9D are complete and neither supplied an advancing component.
Phase 9E is the next prospective research phase but has not started and
requires explicit authorization. The frozen sequence status is:

1. Phase 9C: deployment-available creator and code-family state - complete,
   H9C not supported.
2. Phase 9D: cascade calibration and safer routing - complete, H9D not
   supported.
3. Phase 9E: selector and semantic static features - not started.
4. Phase 9F: combine only independently supported components - conditional;
   no Phase 9C/9D component is currently eligible.

Arbitrary model search, reopening GIN/Transformer/creation-code models, and
unrestricted threshold search are prohibited by the Phase 9B protocol.

### 17.4 Final Research Manuscript or Thesis Chapter

The final written research output remains to be assembled. It should include:

- problem statement and research questions;
- source-paper summary and reproducibility gaps;
- dataset description and audit;
- leakage-safe methodology;
- initial reconstruction and Phase 5.5 correction;
- Phase 6 hypotheses and negative results;
- Phase 7 temporal, ablation, error, and clone findings;
- Phase 8 hypotheses, negative results, and closure;
- Phase 9A protocol-gap diagnostics;
- Phase 8R6 external-data provenance, enrichment, overlap, and sealing;
- Phase 9B prospective evaluation governance;
- Phase 9C creator/code-family state and its failed ranking-quality gate;
- Phase 9D calibration/routing results and their failed temporal gates;
- threats to validity;
- answers to RQ1-RQ6 and the status of RQ7;
- limitations and future work; and
- reproducibility instructions and artifact index.

### 17.5 Final Reproducibility Audit

Before submission or release, perform one clean-environment verification:

1. create a fresh `.venv`;
2. install dependencies;
3. verify the raw dataset hash;
4. restore the two metadata-only frozen artifacts after explicit approval;
   the required Phase 8R6D LFS object is now available locally;
5. execute implemented notebooks non-interactively;
6. verify frozen manifests and closure JSON files;
7. verify the sealed-label default-deny guard without loading labels into model
   code;
8. record Python and dependency versions; and
9. confirm no credentials, provider URLs containing secrets, large fitted
   artifacts, or temporary notebook files are staged for commit.

The current `requirements.txt` is mostly unpinned. Producing a fully pinned
environment snapshot or lock file remains advisable for archival reproduction.

### 17.6 Documentation Synchronization

Other entry-point documents may still list older phase or test status. A later
documentation-only task should reconcile:

- `README.md`;
- `docs/RUN_AND_TEST_GUIDE.md`;
- the earlier `docs/PROJECT_PROGRESS_REPORT.md`; and
- notebook status descriptions.

This report should be used as the broad current progress reference, while exact
JSON/CSV artifacts remain authoritative for metrics and membership.

### 17.7 Track B Final Evaluation

Track B construction is complete, but external validation is pending by design.
After Phase 9F is frozen, the one-shot evaluation should report precision,
recall, F1, PR-AUC, balanced accuracy, and the confusion matrix. No model
selection or post-evaluation tuning is allowed.

### 17.8 Optional Prototype

The original roadmap mentions an optional interface that accepts new contract
bytecode and creator information and produces a warning. This has not been
implemented and should remain secondary to final research reporting.

A prototype must not be presented as deployment-ready because:

- Track A ends in 2019 and Track B ends in June 2020;
- P2-P5 performance is unstable;
- recall is low in several periods;
- current-day distribution compatibility remains unverified; and
- Track B performance is still sealed and unknown.

### 17.9 Optional Additional Research, Not Current Commitments

The following are possible future studies, not substitutes for the frozen
Phase 9 program:

- a newer post-2020 external cohort after the sealed Track B evaluation;
- repeated-seed robustness for genuinely stochastic future candidates;
- compiler/version-aware analysis;
- representation-learning methods designed for code novelty;
- calibrated uncertainty and abstention studied under a new protocol;
- inter-contract or transaction-based signals, which would change the static
  creation-time scope; and
- adversarial robustness after external validity is established.

These must be preregistered as new research rather than selected after reviewing
the existing P5 errors.

## 18. Recommended Next Actions

The recommended order is:

1. Reconcile the branch's two frozen Phase 5.5/Phase 7 metadata artifacts after
   explicit approval; the Phase 8R6D LFS object is already restored.
2. Begin Phase 9E only after explicit authorization and under the frozen Phase
   9B protocol; do not access Track B labels or P5.
3. Run Phase 9F only if Phase 9E supplies an independently supported component;
   Phase 9C and 9D supplied none.
4. Freeze the final Phase 9 system before the single permitted Track B
   evaluation.
5. Populate notebook 08 as read-only synthesis and prepare thesis/manuscript
   tables from machine-readable artifacts.
6. Complete clean-environment and dependency-lock reproducibility audits.
7. Consider a prototype only after research closure and with explicit
   non-production limitations.

## 19. Final Current Conclusion

The project has successfully progressed from repository setup and dataset audit
through independent MulCas reconstruction, paper-aligned correction,
controlled improvement experiments, temporal robustness analysis, materially
different static representations, protocol-gap diagnostics, complete external
enrichment, and prospective evaluation governance.

The work produced a defensible frozen baseline and a substantial body of
negative and diagnostic evidence. No Phase 8 intervention reliably improved
the reference across P2-P4. Phase 9A explained only a limited portion of the
published gap using completed diagnostics and did not create a deployable
improvement.

Phase 8R6D established a sealed, strict-only STRONG Track B benchmark with 112
positive and 180 benign later-time contracts while preserving Track A. Phase
9B fixes how development and one-shot external evaluation must be conducted.
Phase 9C improved operating-point F1 but violated the ranking-quality gate.
Phase 9D improved pooled F1 and PR-AUC through conservative consensus routing,
but P3 recall collapsed to zero and the temporal gates rejected it. Neither
phase produced an accepted replacement.

Track B performance remains unknown and the retained reference remains frozen
Phase 5.5 paper-aligned MulCas. The immediate work is explicit authorization
and preregistration for Phase 9E, restoration of the branch's frozen-artifact
and LFS integrity, and final synthesis/reproducibility packaging. No claim of
external generalization is supported until a frozen Phase 9 system receives
the one permitted Track B evaluation.
