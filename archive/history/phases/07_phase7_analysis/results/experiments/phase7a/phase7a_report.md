# Phase 7A: Temporal Robustness and Model Aging

Phase 7A evaluates the unchanged frozen Phase 5.5 MulCas reconstruction. No model, feature, threshold, routing, or ensemble setting was optimized from P2-P5 results.

## Period Definitions

| Period | Blocks | Rows | Positive | Negative | Neg:Pos | Boundary delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | 49880-1287252 | 104 | 50 | 54 | 1.080 | +0 |
| P1 | 1287353-2000473 | 305 | 50 | 255 | 5.100 | +0 |
| P2 | 2008140-4963650 | 5141 | 50 | 5091 | 101.820 | +0 |
| P3 | 4964082-5021809 | 289 | 50 | 239 | 4.780 | +0 |
| P4 | 5021917-5038334 | 150 | 50 | 100 | 2.000 | +0 |
| P5 | 5038337-7136486 | 509 | 64 | 445 | 6.953 | +0 |

Whole creation blocks remain intact. `Boundary delta` records any positive-count difference caused by contracts sharing a boundary block. The paper reports 5,990 P0-P4 rows and 508 P5 rows; this reconstruction measures 5,989 and 509 under the explicit whole-block policy. The publication does not expose row membership, so the exact source of this one-row difference cannot be resolved.

## Current Reconstruction

| Predict | Train periods | Accuracy | Balanced acc. | Precision | Recall | F1 | PR-AUC | ROC-AUC | TN | FP | FN | TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | P0 + P1 | 0.9440 | 0.6053 | 0.0492 | 0.2600 | 0.0828 | 0.0462 | 0.6388 | 4840 | 251 | 37 | 13 |
| P3 | P0 + P1 + P2 | 0.8304 | 0.5574 | 0.5385 | 0.1400 | 0.2222 | 0.4786 | 0.8511 | 233 | 6 | 43 | 7 |
| P4 | P0 + P1 + P2 + P3 | 0.8933 | 0.9000 | 0.7931 | 0.9200 | 0.8519 | 0.8706 | 0.9435 | 88 | 12 | 4 | 46 |
| P5 | P0 + P1 + P2 + P3 + P4 | 0.9194 | 0.7064 | 0.8710 | 0.4219 | 0.5684 | 0.6674 | 0.9123 | 441 | 4 | 37 | 27 |

## Routing

| Predict | Routing distribution |
| --- | --- |
| P2 | xgboost_term_count: 4859, xgboost_ngram_count_df_filtered: 263, vote: 19 |
| P3 | xgboost_term_count: 271, xgboost_ngram_count_df_filtered: 13, vote: 5 |
| P4 | xgboost_term_count: 134, xgboost_ngram_count_df_filtered: 15, vote: 1 |
| P5 | xgboost_term_count: 454, xgboost_ngram_count_df_filtered: 29, vote: 26 |

## Temporal Interpretation

From P2 to P5, precision changed by +0.8217, recall by +0.1619, and F1 by +0.4856. These are descriptive temporal variations and do not by themselves prove concept drift.

Performance is not monotonic: recall is 0.2600, 0.1400, 0.9200, and 0.4219 across P2-P5. P2 is also unusually imbalanced at 101.82 negatives per positive, making its high accuracy misleading. P5 retains high precision but misses 37 of 64 positives.

The expanding training history changes together with the prediction period, so these scenarios describe operational forward performance; they do not isolate time from training-set-size effects.

## Published Paper Comparison

**PUBLISHED PAPER:** Zheng et al., Securing the Ethereum from Smart Ponzi Schemes, DOI 10.1145/3571847. Table 4 reports precision, recall, and F1 for P2-P5. It does not report accuracy, balanced accuracy, PR-AUC, ROC-AUC, or confusion counts. Values absent from the publication are not inferred.

| Period | Metric | Published paper | Current reconstruction | Difference |
| --- | --- | ---: | ---: | ---: |
| P2 | precision | 0.8800 | 0.0492 | -0.8308 |
| P2 | recall | 0.3800 | 0.2600 | -0.1200 |
| P2 | f1 | 0.5300 | 0.0828 | -0.4472 |
| P3 | precision | 0.9600 | 0.5385 | -0.4215 |
| P3 | recall | 0.3200 | 0.1400 | -0.1800 |
| P3 | f1 | 0.4800 | 0.2222 | -0.2578 |
| P4 | precision | 0.8100 | 0.7931 | -0.0169 |
| P4 | recall | 0.9400 | 0.9200 | -0.0200 |
| P4 | f1 | 0.8700 | 0.8519 | -0.0181 |
| P5 | precision | 0.9500 | 0.8710 | -0.0790 |
| P5 | recall | 0.6700 | 0.4219 | -0.2481 |
| P5 | f1 | 0.7900 | 0.5684 | -0.2216 |

**CURRENT RECONSTRUCTION:** the exact measured values are reported above. Direct exact replication is not claimed because the N-gram cutoff, full model settings, and boundary handling are under-specified.
