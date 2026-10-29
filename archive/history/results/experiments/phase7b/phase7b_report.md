# Phase 7B: Frozen MulCas Component/View Ablation

This is contribution analysis only. No feature, model, class weight, sampling rule, decision threshold, cascade threshold, voting threshold, or period boundary was tuned.

## Actual Frozen Viewers

| Order | Viewer | Feature | Classifier | Cascade stage | Voting |
| ---: | --- | --- | --- | ---: | --- |
| 1 | xgboost_term_count | opcode_term_count | xgboost | 1 | yes |
| 2 | xgboost_ngram_count_df_filtered | opcode_ngram_count_1_3_df_filtered | xgboost | 2 | yes |
| 3 | xgboost_ngram_tfidf_df_filtered | opcode_ngram_tfidf_1_3_df_filtered | xgboost | - | yes |
| 4 | ridge_tfidf | opcode_tfidf | ridge | - | yes |

The no-voting condition uses the last cascade viewer's native prediction for unresolved samples. Standalone conditions use each viewer's full-historical-training voting copy and native model prediction. Removed viewers are excluded from both cascade routing and voting.

## Metrics

| Ablation | Period | Precision | Recall | F1 | PR-AUC | Bal. acc. | TN | FP | FN | TP | Changed predictions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_frozen_mulcas | P2 | 0.0492 | 0.2600 | 0.0828 | 0.0462 | 0.6053 | 4840 | 251 | 37 | 13 | 0 |
| remove_first_cascade_viewer | P2 | 0.6667 | 0.0400 | 0.0755 | 0.0788 | 0.5199 | 5090 | 1 | 48 | 2 | 261 |
| remove_second_cascade_viewer | P2 | 0.2727 | 0.0600 | 0.0984 | 0.1417 | 0.5292 | 5083 | 8 | 47 | 3 | 253 |
| remove_voting_stage | P2 | 0.0489 | 0.2600 | 0.0823 | 0.0463 | 0.6052 | 4838 | 253 | 37 | 13 | 4 |
| first_viewer_only | P2 | 0.2727 | 0.0600 | 0.0984 | 0.1018 | 0.5292 | 5083 | 8 | 47 | 3 | 253 |
| second_viewer_only | P2 | 0.6667 | 0.0400 | 0.0755 | 0.0673 | 0.5199 | 5090 | 1 | 48 | 2 | 261 |
| full_frozen_mulcas | P3 | 0.5385 | 0.1400 | 0.2222 | 0.4786 | 0.5574 | 233 | 6 | 43 | 7 | 0 |
| remove_first_cascade_viewer | P3 | 0.8333 | 0.1000 | 0.1786 | 0.3732 | 0.5479 | 238 | 1 | 45 | 5 | 9 |
| remove_second_cascade_viewer | P3 | 0.8000 | 0.0800 | 0.1455 | 0.4895 | 0.5379 | 238 | 1 | 46 | 4 | 8 |
| remove_voting_stage | P3 | 0.5385 | 0.1400 | 0.2222 | 0.4773 | 0.5574 | 233 | 6 | 43 | 7 | 0 |
| first_viewer_only | P3 | 0.0000 | 0.0000 | 0.0000 | 0.4571 | 0.5000 | 239 | 0 | 50 | 0 | 13 |
| second_viewer_only | P3 | 0.0000 | 0.0000 | 0.0000 | 0.3673 | 0.5000 | 239 | 0 | 50 | 0 | 13 |
| full_frozen_mulcas | P4 | 0.7931 | 0.9200 | 0.8519 | 0.8706 | 0.9000 | 88 | 12 | 4 | 46 | 0 |
| remove_first_cascade_viewer | P4 | 0.7931 | 0.9200 | 0.8519 | 0.8315 | 0.9000 | 88 | 12 | 4 | 46 | 0 |
| remove_second_cascade_viewer | P4 | 0.7931 | 0.9200 | 0.8519 | 0.8396 | 0.9000 | 88 | 12 | 4 | 46 | 0 |
| remove_voting_stage | P4 | 0.7931 | 0.9200 | 0.8519 | 0.8706 | 0.9000 | 88 | 12 | 4 | 46 | 0 |
| first_viewer_only | P4 | 0.8148 | 0.8800 | 0.8462 | 0.8708 | 0.8900 | 90 | 10 | 6 | 44 | 4 |
| second_viewer_only | P4 | 0.8000 | 0.8800 | 0.8381 | 0.8328 | 0.8850 | 89 | 11 | 6 | 44 | 3 |
| full_frozen_mulcas | P5 | 0.8710 | 0.4219 | 0.5684 | 0.6674 | 0.7064 | 441 | 4 | 37 | 27 | 0 |
| remove_first_cascade_viewer | P5 | 0.8824 | 0.2344 | 0.3704 | 0.6436 | 0.6149 | 443 | 2 | 49 | 15 | 14 |
| remove_second_cascade_viewer | P5 | 0.8800 | 0.3438 | 0.4944 | 0.6793 | 0.6685 | 442 | 3 | 42 | 22 | 6 |
| remove_voting_stage | P5 | 0.8667 | 0.4062 | 0.5532 | 0.6641 | 0.6986 | 441 | 4 | 38 | 26 | 1 |
| first_viewer_only | P5 | 0.8800 | 0.3438 | 0.4944 | 0.6715 | 0.6685 | 442 | 3 | 42 | 22 | 6 |
| second_viewer_only | P5 | 0.8824 | 0.2344 | 0.3704 | 0.6335 | 0.6149 | 443 | 2 | 49 | 15 | 14 |

## F1 Changes Relative to Full MulCas

| Ablation | P2 F1 delta | P3 F1 delta | P4 F1 delta | P5 F1 delta | Descriptive effect |
| --- | ---: | ---: | ---: | ---: | --- |
| remove_first_cascade_viewer | -0.0073 | -0.0437 | +0.0000 | -0.1981 | full_component_nonnegative_in_every_period |
| remove_second_cascade_viewer | +0.0156 | -0.0768 | +0.0000 | -0.0740 | period_specific_mixed_effect |
| remove_voting_stage | -0.0005 | +0.0000 | +0.0000 | -0.0152 | full_component_nonnegative_in_every_period |
| first_viewer_only | +0.0156 | -0.2222 | -0.0057 | -0.0740 | period_specific_mixed_effect |
| second_viewer_only | -0.0073 | -0.2222 | -0.0138 | -0.1981 | full_component_nonnegative_in_every_period |

## Contribution Interpretation

- **P2:** removing the second cascade viewer, or using the first viewer alone, raises F1 by only 0.0156. The full first stage supplies more true positives but also most of P2's false positives; removing it changes 261 predictions and reduces recall from 0.2600 to 0.0400.
- **P5:** the first viewer has the strongest positive contribution. Removing it lowers recall from 0.4219 to 0.2344 and F1 by 0.1981. Removing the second viewer lowers F1 by 0.0740.
- **Consistency:** removing the first viewer never improves F1, while removing the second has a mixed effect: slightly positive in P2, neutral in P4, and negative in P3/P5.
- **Voting:** removing voting changes 4 P2 predictions, no P3/P4 predictions, and one P5 prediction. Its F1 contribution is correspondingly small but nonnegative in every period.
- **Routing and time:** the first viewer handles most contracts in every period, yet its outcomes range from very weak P2 performance to strong P4 performance. Routing volume alone does not explain the temporal variation.

Positive deltas do not identify replacement systems; they only describe component effects for a specific period. No ablation is promoted as a candidate model. Routing details are available in `phase7b_routing.csv`.
