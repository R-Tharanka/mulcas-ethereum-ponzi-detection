# Phase 9A Paper/Reconstruction Protocol-Gap Decomposition

## Purpose and Scope

Phase 9A is a diagnostic reproduction analysis. It introduces no classifier,
feature family, hyperparameter, threshold, resampling method, calibration,
GNN, or Transformer. P5 is reused only as previously observed exploratory
Track A evidence. The creator and viewer-selection analyses are explicitly
non-deployable oracles.

## Frozen Inputs

- Verified Phase 0-8R5C1 anchors: 199
- Verified Phase 8 closure anchors: 11
- Raw dataset SHA-256: `48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf`
- Phase 8 conclusions preserved unchanged: yes

## D0 Frozen Controls

| Control | Precision | Recall | F1 | TN/FP/FN/TP |
| --- | ---: | ---: | ---: | --- |
| Phase 5.5 P5 | 0.9062 | 0.4531 | 0.6042 | 442/3/35/29 |
| Phase 7A P5 | 0.8710 | 0.4219 | 0.5684 | 441/4/37/27 |

The difference is described as a training-history/protocol-coverage difference,
not as a causal P4 effect.

## D2 Creator Oracle

The oracle predicts each row before revealing its label and commits true-positive
creator updates only after the whole creation block. Same-block rows cannot
affect one another. Frozen replays matched archived Phase 7A predictions for
P2-P5: True.

| Period | Condition | Precision | Recall | F1 | TN/FP/FN/TP |
| --- | --- | ---: | ---: | ---: | --- |
| P2 | frozen | 0.0492 | 0.2600 | 0.0828 | 4840/251/37/13 |
| P2 | oracle | 0.0538 | 0.3000 | 0.0912 | 4827/264/35/15 |
| P3 | frozen | 0.5385 | 0.1400 | 0.2222 | 233/6/43/7 |
| P3 | oracle | 0.5385 | 0.1400 | 0.2222 | 233/6/43/7 |
| P4 | frozen | 0.7931 | 0.9200 | 0.8519 | 88/12/4/46 |
| P4 | oracle | 0.7931 | 0.9200 | 0.8519 | 88/12/4/46 |
| P5 | frozen | 0.8710 | 0.4219 | 0.5684 | 441/4/37/27 |
| P5 | oracle | 0.8824 | 0.4688 | 0.6122 | 441/4/34/30 |

The oracle is not deployable because it reveals target-period true labels after
prediction. It improves P5 by three true positives without changing false
positives, but it also increases P2 false positives from 251 to 264.

## D3 Viewer-Selection Oracle

The fixed 18-candidate replay was stopped by the resource gate after the
representative XGBoost estimate proved unrepresentative: the complete pool was
still unfinished after 37 minutes due to a slow SVM/high-dimensional-view
combination. No partial ranking is presented. Independently, a full paper-exact
MulCas replay is **not reproducible from the available specification** because
the paper claims five final viewers but enumerates only four and gives no unique
fifth viewer or complete ordering.

## D4 Combined Oracle and Gap Closure

D4 is unavailable. Combining D2 with D3 would require inventing the missing
viewer mapping/order. Recall/F1 gap-closure fractions are therefore N/A.

## N-gram Dimensionality Audit

The audit measures raw dimensions, min_df=2 dimensions, document-frequency
histograms, and the nearest integer cutoffs to the paper's 77/767/2,292 targets.
No classifier is trained in this audit. The paper's phrase "document frequency
less than 0.1" does not uniquely specify the integer filtering operation used.

## P2 False-Positive Audit

All 251 archived P2 false positives reconcile with
Phase 7D. Routing counts are `{"vote": 1, "xgboost_ngram_count_df_filtered": 250}`.
Exact normalized opcode/runtime hashes, creators, routes, scores, block numbers,
clone indicators, and structural summaries are retained in the row/group CSVs.

## Research Conclusion

Phase 9A decomposes protocol differences without changing the retained Phase 5.5
MulCas reference or any Phase 8 conclusion. Test-label creator updates and test-set
viewer selection can only be interpreted as non-deployable upper-bound diagnostics.
Because the publication does not uniquely specify the fifth viewer and full order,
the combined oracle is not fabricated. The recommended next research step remains
external/later dataset discovery, provenance, and overlap audit before further
model development.
