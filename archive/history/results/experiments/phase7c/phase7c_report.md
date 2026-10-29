# Phase 7C: Chronological Creator Feature Ablation

The only independent variable is whether the final leakage-safe creator feature column is present. The no-creator matrices contain no replacement or constant placeholder column.

## Full-System Metrics

| Period | Condition | Precision | Recall | F1 | PR-AUC | Bal. acc. | TN | FP | FN | TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | with_creator | 0.0492 | 0.2600 | 0.0828 | 0.0462 | 0.6053 | 4840 | 251 | 37 | 13 |
| P2 | without_creator | 0.0688 | 0.2200 | 0.1048 | 0.0723 | 0.5954 | 4942 | 149 | 39 | 11 |
| P3 | with_creator | 0.5385 | 0.1400 | 0.2222 | 0.4786 | 0.5574 | 233 | 6 | 43 | 7 |
| P3 | without_creator | 0.5238 | 0.2200 | 0.3099 | 0.6238 | 0.5891 | 229 | 10 | 39 | 11 |
| P4 | with_creator | 0.7931 | 0.9200 | 0.8519 | 0.8706 | 0.9000 | 88 | 12 | 4 | 46 |
| P4 | without_creator | 0.7931 | 0.9200 | 0.8519 | 0.8620 | 0.9000 | 88 | 12 | 4 | 46 |
| P5 | with_creator | 0.8710 | 0.4219 | 0.5684 | 0.6674 | 0.7064 | 441 | 4 | 37 | 27 |
| P5 | without_creator | 0.8788 | 0.4531 | 0.5979 | 0.6839 | 0.7221 | 441 | 4 | 35 | 29 |

## Without Minus With Creator

| Period | Delta precision | Delta recall | Delta F1 | Delta PR-AUC | Delta bal. acc. | Delta FP | Delta FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | +0.0195 | -0.0400 | +0.0220 | +0.0261 | -0.0100 | -102 | +2 |
| P3 | -0.0147 | +0.0800 | +0.0876 | +0.1453 | +0.0316 | +4 | -4 |
| P4 | +0.0000 | +0.0000 | +0.0000 | -0.0087 | +0.0000 | +0 | +0 |
| P5 | +0.0078 | +0.0312 | +0.0295 | +0.0165 | +0.0156 | +0 | -2 |

## Creator History Diagnostics

| Period | Previously observed creator | Creator feature = 1 | Ponzi with feature = 1 | Benign with feature = 1 | With TP/FN (feature 1) | Without TP/FN (feature 1) | With TP/FN (feature 0) | Without TP/FN (feature 0) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| P2 | 148 | 1 | 0 | 1 | 0/0 | 0/0 | 13/37 | 11/39 |
| P3 | 49 | 2 | 1 | 1 | 0/1 | 0/1 | 7/42 | 11/38 |
| P4 | 20 | 6 | 4 | 2 | 4/0 | 4/0 | 42/4 | 42/4 |
| P5 | 85 | 11 | 4 | 7 | 3/1 | 3/1 | 24/36 | 26/34 |

## Viewer-Level Diagnostics

| Period | Viewer | Delta precision | Delta recall | Delta F1 | Delta PR-AUC | Delta FP | Delta FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | xgboost_term_count | -0.1061 | -0.0200 | -0.0338 | -0.0002 | +2 | +1 |
| P2 | xgboost_ngram_count_df_filtered | +0.0000 | +0.0000 | +0.0000 | -0.0000 | +0 | +0 |
| P2 | xgboost_ngram_tfidf_df_filtered | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0 | +0 |
| P2 | ridge_tfidf | +0.2000 | +0.1000 | +0.1333 | -0.0056 | +15 | -5 |
| P3 | xgboost_term_count | +0.0000 | +0.0000 | +0.0000 | +0.1120 | +1 | +0 |
| P3 | xgboost_ngram_count_df_filtered | +0.0000 | +0.0000 | +0.0000 | +0.0508 | +1 | +0 |
| P3 | xgboost_ngram_tfidf_df_filtered | -0.2083 | +0.0000 | -0.0062 | -0.0113 | +2 | +0 |
| P3 | ridge_tfidf | +0.0000 | +0.0000 | +0.0000 | -0.0035 | +0 | +0 |
| P4 | xgboost_term_count | -0.0217 | +0.0400 | +0.0057 | -0.0016 | +2 | -2 |
| P4 | xgboost_ngram_count_df_filtered | +0.0000 | +0.0000 | +0.0000 | +0.0131 | +0 | +0 |
| P4 | xgboost_ngram_tfidf_df_filtered | +0.0000 | +0.0000 | +0.0000 | +0.0058 | +0 | +0 |
| P4 | ridge_tfidf | +0.0934 | +0.7400 | +0.6481 | +0.0499 | +8 | -37 |
| P5 | xgboost_term_count | +0.0089 | +0.0312 | +0.0331 | -0.0155 | +0 | -2 |
| P5 | xgboost_ngram_count_df_filtered | +0.0000 | +0.0000 | +0.0000 | +0.0080 | +0 | +0 |
| P5 | xgboost_ngram_tfidf_df_filtered | -0.0074 | -0.0156 | -0.0204 | -0.0159 | +0 | +1 |
| P5 | ridge_tfidf | +0.0074 | +0.0156 | +0.0204 | +0.0684 | +0 | -1 |

## Interpretation

- Removing creator history changes recall by -0.0400 in P2, +0.0800 in P3, +0.0000 in P4, and +0.0312 in P5. The creator feature therefore does not improve recall consistently.
- In P2, creator history trades two additional true positives for 102 additional false positives. There are no P2 Ponzi contracts with creator feature 1, so this is a learned model interaction rather than direct recognition of a known Ponzi creator.
- P4 has four Ponzi contracts associated with historical Ponzi creators, and all four are detected under both conditions. Final classifications are identical, although PR-AUC decreases by 0.0087 without creator history. The strong P4 classification result is not dependent on the creator column in this ablation.
- In P5, removing creator history recovers two positives without changing false positives. Both gains occur among the 60 Ponzi contracts with creator feature 0; the four feature-1 Ponzi contracts retain the same 3 TP / 1 FN result.
- Later false negatives are concentrated among creators without prior positive history: 36 of 37 P5 false negatives under the reference condition have creator feature 0. This is descriptive historical signal, not evidence that creator identity causes Ponzi behavior.

Neither condition is used for model selection, and the viewer-level results are diagnostics only.
