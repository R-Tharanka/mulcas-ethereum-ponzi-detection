# Phase 9A Paper/Reconstruction Protocol-Gap Decomposition

## 1. Purpose

Phase 9A is a protocol-decomposition study, not an improved detector evaluation.
It asks how reproducible protocol differences affect the paper/reconstruction
gap while leaving all Phase 0-8 conclusions closed.

## 2. Scope and non-deployable diagnostics

No classifier family, representation, hyperparameter, threshold, class weight,
resampling rule, or calibration was introduced. P5 remains previously observed
exploratory Track A data. Diagnostics that use target-period true labels for
creator updates or viewer selection are deliberately non-deployable and are
reported only to measure how permissive evaluation choices affect reproduction
of the published result.

## 3. Frozen artifact verification

- Verified Phase 0-8R5C1 anchors: 199
- Verified Phase 8 closure anchors: 11
- Raw dataset SHA-256: `48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf`
- Phase 8 conclusions preserved unchanged: yes

Any anchor mismatch raises before diagnostics run. H8R1, H8R2, H8R3B, H8R4,
and H8R5 retain their closed, not-supported statuses.

## 4. Paper-vs-reconstruction protocol evidence table

| protocol_item | paper_evidence | reconstruction_policy | difference | reproducible_in_9a | diagnostic_type | deployable |
| --- | --- | --- | --- | --- | --- | --- |
| training coverage | Table 5 training covers the first 250 chronological Ponzi contracts. (local paper PDF Section 5.2.2, Table 5) | Phase 7A P5 trains on P0-P4; Phase 5.5 stops at P3 and reserves P4. | Phase 7A matches the reported positive-history coverage more closely. | True | D1 descriptive frozen-control comparison | True |
| creator-state update | Creator vector is chronological and may expand with upcoming labeled Ponzi contracts. (local paper PDF pages 16-17, Algorithm 1) | Evaluation state is frozen from historical training labels. | Target-period true-label updates are excluded from controls. | True | D2 true-label online creator oracle | False |
| viewer selection | Best model for each feature is selected on the test set; five final models are claimed but only four are enumerated. (local paper PDF page 17, Section 4.3) | Viewer choice uses validation/forward data and a frozen four-viewer mapping. | Test-label selection is non-deployable; the full paper mapping/order is under-specified. | False | D3 candidate oracle runtime-gated; full replay unavailable | False |
| N-gram filtering | N-grams with document frequency less than 0.1 are discarded; Table 3 reports 77/767/2292 selected dimensions. (local paper PDF pages 13-14, Section 4.2.1/Table 3) | Whitespace opcode tokenization and integer min_df=2. | Formula semantics are under-specified and dimensions differ. | True | dimension-only audit | True |
| period row membership | Table 5 reports 5,990 train and 508 test rows; P2 benign membership is not reported. (local paper PDF Sections 5.2.1-5.2.2, Tables 4-5) | Whole-block positive-occurrence boundaries yield 5,989/509. | One-row Table-5 boundary difference; exact P2 benign comparison is not verifiable. | True | membership audit | True |
| cascade thresholds | Algorithm 2 describes theta and reports a commonly used value of 0.05; complete per-stage asymmetric thresholds are not specified. (local paper PDF pages 17-18, Section 4.4/Algorithm 2) | Frozen N=2 cascade uses 0.05/0.95 then 0.05/0.51. | Paper-exact per-stage threshold mapping is under-specified. | False | frozen mechanics only; no threshold search | True |
| voting rule | All viewers vote on residual rows and lambda controls the decision. (local paper PDF pages 17-18, Section 4.4/Algorithm 2) | Frozen voting threshold lambda=0. | No change is permitted in Phase 9A. | True | frozen mechanic | True |
| auxiliary-bytecode handling | Auxiliary bytecode fragment is discarded. (local paper PDF page 10, Section 3.2) | Paper-aligned baseline consumes released opcode; later diagnostic parsers retain complete contractCode. | Exact preprocessing equivalence cannot be established from released material. | False | evidence audit only | True |

Ambiguous publication details are marked under-specified and are not guessed.

## 5. Frozen P5 controls

| Control | Precision | Recall | F1 | TN/FP/FN/TP |
| --- | ---: | ---: | ---: | --- |
| Phase 5.5 P5 | 0.9062 | 0.4531 | 0.6042 | 442/3/35/29 |
| Phase 7A P5 | 0.8710 | 0.4219 | 0.5684 | 441/4/37/27 |

Both controls reproduce at 1e-12 metric tolerance and exact confusion counts.

## 6. Training-history comparison

D0B minus D0A changes TP by -2, FP by +1, FN by
+2, TN by -1, precision by -0.0353,
recall by -0.0312, F1 by -0.0357, PR-AUC by
+0.0049, and balanced accuracy by -0.0167.
This is a **training-history / protocol-coverage difference**, not a causal
estimate of the effect of adding P4, because the frozen protocols are not
otherwise identical.

## 7. True-label online creator oracle

The oracle predicts each row before revealing its label and commits true-positive
creator updates only after the whole creation block. Same-block rows cannot
affect one another. Frozen replays matched archived Phase 7A predictions for
P2-P5: True.

| Period | Condition | Creator flags | Flag changes | Prediction changes | Precision | Recall | F1 | TN/FP/FN/TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P2 | frozen | 1 | 0 | 0 | 0.0492 | 0.2600 | 0.0828 | 4840/251/37/13 |
| P2 | oracle | 69 | 68 | 15 | 0.0538 | 0.3000 | 0.0912 | 4827/264/35/15 |
| P3 | frozen | 2 | 0 | 0 | 0.5385 | 0.1400 | 0.2222 | 233/6/43/7 |
| P3 | oracle | 20 | 18 | 0 | 0.5385 | 0.1400 | 0.2222 | 233/6/43/7 |
| P4 | frozen | 6 | 0 | 0 | 0.7931 | 0.9200 | 0.8519 | 88/12/4/46 |
| P4 | oracle | 36 | 30 | 0 | 0.7931 | 0.9200 | 0.8519 | 88/12/4/46 |
| P5 | frozen | 11 | 0 | 0 | 0.8710 | 0.4219 | 0.5684 | 441/4/37/27 |
| P5 | oracle | 34 | 23 | 3 | 0.8824 | 0.4688 | 0.6122 | 441/4/34/30 |

The TRUE-LABEL ONLINE CREATOR ORACLE is NON-DEPLOYABLE / EXPLORATORY. On P5 it
adds three true positives without adding false positives; on P2 it also raises
false positives from 251 to 264.

## 8. Test-selected-viewer oracle

The fixed 18-candidate replay was stopped by the resource gate after the
representative XGBoost estimate proved unrepresentative: the complete pool was
still unfinished after at least 37 minutes due to a slow SVM/high-dimensional-view combination. The run was
stopped and no partial label-selected ranking is reported. Independently, a full
TEST-SELECTED VIEWER ORACLE is **not reproducible from the available
specification** because the paper claims five final viewers but enumerates only
four and gives no unique fifth viewer or complete ordering.

## 9. Combined paper-permissive oracle

D4 is unavailable. Combining D2 with D3 would require inventing the missing
viewer mapping/order. No combined result is fabricated.

## 10. Published P5 gap decomposition

D0B is the primary control. D2 changes precision by +0.0114, recall by +0.0469,
and F1 by +0.0438, closing 18.59% of the recall gap and 19.87% of the F1 gap
in isolation. D3 and D4 are unavailable, so D4 gap-closure fractions and a
D2+D3 interaction estimate are N/A. Based only on completed reproducible
diagnostics, the descriptive classification is **C. LITTLE PROTOCOL
EXPLANATION**; it is explicitly provisional because D3/D4 could not be run.

## 11. N-gram dimension reproduction audit

| Corpus | N-grams | Raw observed | min_df=2 | Paper selected | Nearest integer DF | Nearest dimension |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P0-P3 | 1-1 | 142 | 142 | 77 | 869 | 77 |
| P0-P3 | 1-2 | 13623 | 8717 | 767 | 628 | 766 |
| P0-P3 | 1-3 | 82301 | 44671 | 2292 | 602 | 2293 |
| P0-P4 | 1-1 | 142 | 142 | 77 | 913 | 77 |
| P0-P4 | 1-2 | 13751 | 8817 | 767 | 644 | 767 |
| P0-P4 | 1-3 | 83277 | 45103 | 2292 | 612 | 2291 |

No classifier was trained. With the repository's existing whitespace opcode
tokenization, a cutoff exactly reaches 77 unigrams on both corpora and 767
unigram+bigrams on P0-P4. P0-P3 unigram+bigrams and both 1-3-gram corpora miss
their selected targets by one at the nearest cutoff. The raw 1-2 and 1-3
dimensions remain mismatched, and the paper's "document frequency less than
0.1" wording is under-specified.

## 12. P2 period-membership audit

| Check | Reconstructed value | Paper value | Classification |
| --- | --- | --- | --- |
| positive_count | 50 | 50.0 | MATCH |
| row_count | 5141 | nan | NOT VERIFIABLE FROM PAPER |
| benign_count | 5091 | nan | NOT VERIFIABLE FROM PAPER |
| first_block | 2008140 | nan | NOT VERIFIABLE FROM PAPER |
| last_block | 4963650 | nan | NOT VERIFIABLE FROM PAPER |
| first_positive | 0x143e8bd26d95b230680c25300f93ef56f86eccac@2129975#row_521 | nan | NOT VERIFIABLE FROM PAPER |
| last_positive | 0xe3f64dc522a66405c51d96aae234217a03502bb4@4963650#row_5549 | nan | NOT VERIFIABLE FROM PAPER |
| same_block_boundary_behavior | all rows at each positive boundary block remain in the same period | nan | NOT VERIFIABLE FROM PAPER |

Only the 50-positive definition is a MATCH. The publication does not expose
enough information to verify P2 benign membership or block boundaries.

## 13. P2 false-positive routing analysis

All 251 archived P2 false positives reconcile with
Phase 7D. Accepted-stage counts are `{"viewer_2": 250, "voting": 1}`;
route counts are `{"vote": 1, "xgboost_ngram_count_df_filtered": 250}` and route
percentages are `{"vote": 0.398406374501992, "xgboost_ngram_count_df_filtered": 99.60159362549801}`. There are
5 FPs from historically seen creators and
92 rows belonging to creators repeated within
the P2 FP set. Final-score quantiles are min=0.5224, q25=0.7276,
median=0.8540, q75=0.9241, max=0.9799.

## 14. P2 duplicate/family concentration

The 251 FPs contain 240 unique normalized opcode
hashes and 243 unique runtime hashes. Repeated
opcode/runtime clusters account for 17
(6.8%) and
11 (4.4%)
rows; largest cluster sizes are 5 and
5. This quantifies concentration without learned
or label-tuned clustering.

## 15. What the paper gap can and cannot be explained by

The creator permission explains a modest part of the P5 recall/F1 gap in
isolation. Viewer-selection and combined explanations remain unresolved because
the runtime gate stopped D3 and the full cascade mapping is under-specified.
Closer agreement under a paper-permissive replay would not prove that the
publication's result was caused by any single protocol difference.

Failure to reproduce the published result even under the permissive replay
would indicate that additional differences remain, such as feature construction,
row membership, implementation details, or under-specified methodology.

## 16. Limitations

P5-label-based diagnostics are non-deployable. D1 is descriptive rather than
causal. D3 has no completed candidate ranking, D4 is unavailable, paper-exact
thresholds/order are under-specified, and approximate P2 paper confusion counts
are not presented as facts.

## 17. Reproducibility/tests

The configuration freezes the candidate pool and ranking before P5 scoring,
same-block creator updates, allowed metrics/grouping, paper benchmarks, and
no-search constraints. Machine-readable tables reconcile frozen controls,
creator changes, P2 totals, dimensions, and unavailable diagnostics. The test
suite verifies frozen hashes, leakage ordering, deterministic grouping, output
existence, and governance constraints.

## 18. Decision for the next phase

No scientific-model, data-membership, or leakage defect was discovered. The
archived Phase 8 checkpoint/embedding registries do contain checkout-specific
absolute paths; Phase 9A verifies relocated copies by frozen hash without
modifying those prior registries. This is a portability defect, not a metric or
membership defect, and does not require reopening Phase 8.
Recommendation A: proceed to Phase 8R6A external/later-data audit, then Phase 9B
prospective evaluation design. Phase 8R6A is not started by Phase 9A.
