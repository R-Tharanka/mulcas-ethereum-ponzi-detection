# Phase 8R1A: Recency-Aware Temporal Backtesting

The expanding-history control reproduced frozen Phase 7A P2-P4 exactly. P5 was neither loaded nor evaluated.

## Full MulCas Period Metrics

| Strategy | Period | Precision | Recall | F1 | PR-AUC | Balanced acc. | TN | FP | FN | TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| expanding_history | P2 | 0.0492 | 0.2600 | 0.0828 | 0.0462 | 0.6053 | 4840 | 251 | 37 | 13 |
| expanding_history | P3 | 0.5385 | 0.1400 | 0.2222 | 0.4786 | 0.5574 | 233 | 6 | 43 | 7 |
| expanding_history | P4 | 0.7931 | 0.9200 | 0.8519 | 0.8706 | 0.9000 | 88 | 12 | 4 | 46 |
| recent_1_period | P2 | 0.0680 | 0.2800 | 0.1094 | 0.0663 | 0.6211 | 4899 | 192 | 36 | 14 |
| recent_1_period | P3 | 0.3750 | 0.1200 | 0.1818 | 0.3569 | 0.5391 | 229 | 10 | 44 | 6 |
| recent_1_period | P4 | 0.7931 | 0.9200 | 0.8519 | 0.8685 | 0.9000 | 88 | 12 | 4 | 46 |
| recent_2_periods | P2 | 0.0492 | 0.2600 | 0.0828 | 0.0462 | 0.6053 | 4840 | 251 | 37 | 13 |
| recent_2_periods | P3 | 0.5000 | 0.1400 | 0.2188 | 0.3521 | 0.5554 | 232 | 7 | 43 | 7 |
| recent_2_periods | P4 | 0.7931 | 0.9200 | 0.8519 | 0.8441 | 0.9000 | 88 | 12 | 4 | 46 |
| recent_3_periods | P2 | 0.0492 | 0.2600 | 0.0828 | 0.0462 | 0.6053 | 4840 | 251 | 37 | 13 |
| recent_3_periods | P3 | 0.5385 | 0.1400 | 0.2222 | 0.4786 | 0.5574 | 233 | 6 | 43 | 7 |
| recent_3_periods | P4 | 0.7797 | 0.9200 | 0.8440 | 0.8618 | 0.8950 | 87 | 13 | 4 | 46 |
| exponential_decay_0.25 | P2 | 0.0630 | 0.3000 | 0.1042 | 0.0342 | 0.6281 | 4868 | 223 | 35 | 15 |
| exponential_decay_0.25 | P3 | 0.3846 | 0.1000 | 0.1587 | 0.4570 | 0.5333 | 231 | 8 | 45 | 5 |
| exponential_decay_0.25 | P4 | 0.7931 | 0.9200 | 0.8519 | 0.8566 | 0.9000 | 88 | 12 | 4 | 46 |
| exponential_decay_0.50 | P2 | 0.0673 | 0.3000 | 0.1099 | 0.0374 | 0.6296 | 4883 | 208 | 35 | 15 |
| exponential_decay_0.50 | P3 | 0.5000 | 0.1400 | 0.2188 | 0.4826 | 0.5554 | 232 | 7 | 43 | 7 |
| exponential_decay_0.50 | P4 | 0.7931 | 0.9200 | 0.8519 | 0.8333 | 0.9000 | 88 | 12 | 4 | 46 |
| exponential_decay_0.75 | P2 | 0.0520 | 0.2600 | 0.0867 | 0.0841 | 0.6067 | 4854 | 237 | 37 | 13 |
| exponential_decay_0.75 | P3 | 0.5294 | 0.1800 | 0.2687 | 0.5579 | 0.5733 | 231 | 8 | 41 | 9 |
| exponential_decay_0.75 | P4 | 0.7931 | 0.9200 | 0.8519 | 0.8711 | 0.9000 | 88 | 12 | 4 | 46 |

## Macro P2-P4 Strategy Summary

| Strategy | Mean precision | Mean recall | Mean F1 | F1 std | Mean PR-AUC | Mean balanced acc. | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| expanding_history | 0.4603 | 0.4400 | 0.3856 | 0.3345 | 0.4651 | 0.6876 | False |
| recent_1_period | 0.4120 | 0.4400 | 0.3810 | 0.3342 | 0.4305 | 0.6867 | False |
| recent_2_periods | 0.4474 | 0.4400 | 0.3845 | 0.3351 | 0.4141 | 0.6869 | False |
| recent_3_periods | 0.4558 | 0.4400 | 0.3830 | 0.3309 | 0.4622 | 0.6859 | False |
| exponential_decay_0.25 | 0.4136 | 0.4400 | 0.3716 | 0.3403 | 0.4493 | 0.6871 | False |
| exponential_decay_0.50 | 0.4535 | 0.4533 | 0.3935 | 0.3271 | 0.4511 | 0.6950 | False |
| exponential_decay_0.75 | 0.4582 | 0.4533 | 0.4024 | 0.3264 | 0.5044 | 0.6933 | False |

Selection status: `no_eligible_strategy`. Selected strategy: `None`. Diagnostic fallback: `exponential_decay_0.75`.

## Period-Specific Effects vs Control

| Strategy | Period | Delta recall | Delta F1 | Delta FP | P4 preserved |
| --- | --- | ---: | ---: | ---: | --- |
| expanding_history | P2 | +0.0000 | +0.0000 | +0 | N/A |
| expanding_history | P3 | +0.0000 | +0.0000 | +0 | N/A |
| expanding_history | P4 | +0.0000 | +0.0000 | +0 | True |
| recent_1_period | P2 | +0.0200 | +0.0266 | -59 | N/A |
| recent_1_period | P3 | -0.0200 | -0.0404 | +4 | N/A |
| recent_1_period | P4 | +0.0000 | +0.0000 | +0 | True |
| recent_2_periods | P2 | +0.0000 | +0.0000 | +0 | N/A |
| recent_2_periods | P3 | +0.0000 | -0.0035 | +1 | N/A |
| recent_2_periods | P4 | +0.0000 | +0.0000 | +0 | True |
| recent_3_periods | P2 | +0.0000 | +0.0000 | +0 | N/A |
| recent_3_periods | P3 | +0.0000 | +0.0000 | +0 | N/A |
| recent_3_periods | P4 | +0.0000 | -0.0078 | +1 | False |
| exponential_decay_0.25 | P2 | +0.0400 | +0.0214 | -28 | N/A |
| exponential_decay_0.25 | P3 | -0.0400 | -0.0635 | +2 | N/A |
| exponential_decay_0.25 | P4 | +0.0000 | +0.0000 | +0 | True |
| exponential_decay_0.50 | P2 | +0.0400 | +0.0271 | -43 | N/A |
| exponential_decay_0.50 | P3 | +0.0000 | -0.0035 | +1 | N/A |
| exponential_decay_0.50 | P4 | +0.0000 | +0.0000 | +0 | True |
| exponential_decay_0.75 | P2 | +0.0000 | +0.0039 | -14 | N/A |
| exponential_decay_0.75 | P3 | +0.0400 | +0.0464 | +2 | N/A |
| exponential_decay_0.75 | P4 | +0.0000 | +0.0000 | +0 | True |

## Novelty and Creator Diagnostics

| Strategy | Period | Subgroup | Positives | Recall |
| --- | --- | --- | ---: | ---: |
| expanding_history | P2 | historically_novel_code | 50 | 0.2600 |
| expanding_history | P2 | exact_historical_code_clone | 0 | N/A |
| expanding_history | P2 | unseen_creator | 50 | 0.2600 |
| expanding_history | P2 | seen_creator | 0 | N/A |
| expanding_history | P3 | historically_novel_code | 50 | 0.1400 |
| expanding_history | P3 | exact_historical_code_clone | 0 | N/A |
| expanding_history | P3 | unseen_creator | 49 | 0.1429 |
| expanding_history | P3 | seen_creator | 1 | 0.0000 |
| expanding_history | P4 | historically_novel_code | 36 | 0.8889 |
| expanding_history | P4 | exact_historical_code_clone | 14 | 1.0000 |
| expanding_history | P4 | unseen_creator | 46 | 0.9130 |
| expanding_history | P4 | seen_creator | 4 | 1.0000 |
| exponential_decay_0.75 | P2 | historically_novel_code | 50 | 0.2600 |
| exponential_decay_0.75 | P2 | exact_historical_code_clone | 0 | N/A |
| exponential_decay_0.75 | P2 | unseen_creator | 50 | 0.2600 |
| exponential_decay_0.75 | P2 | seen_creator | 0 | N/A |
| exponential_decay_0.75 | P3 | historically_novel_code | 50 | 0.1800 |
| exponential_decay_0.75 | P3 | exact_historical_code_clone | 0 | N/A |
| exponential_decay_0.75 | P3 | unseen_creator | 49 | 0.1837 |
| exponential_decay_0.75 | P3 | seen_creator | 1 | 0.0000 |
| exponential_decay_0.75 | P4 | historically_novel_code | 36 | 0.8889 |
| exponential_decay_0.75 | P4 | exact_historical_code_clone | 14 | 1.0000 |
| exponential_decay_0.75 | P4 | unseen_creator | 46 | 0.9130 |
| exponential_decay_0.75 | P4 | seen_creator | 4 | 1.0000 |

## Conclusion

H8R1 is **not supported** under the predefined macro-precision eligibility rule.
