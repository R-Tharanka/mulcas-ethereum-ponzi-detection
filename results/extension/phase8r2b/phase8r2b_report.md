# Phase 8R2B Temporal Graph-Model Backtesting

Phase 8R2B used only P2-P4 for forward evaluation. P5 was not loaded or evaluated.

## Environment

- PyTorch: `2.14.1+cpu`
- PyTorch Geometric: `2.8.0.post1`
- Device: `cpu`
- Total GIN runtime: 22650.8 seconds

## Aggregate Baseline

| Period | Precision | Recall | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| P2 | 0.0323 | 0.1600 | 0.0537 | 0.0142 |
| P3 | 0.0000 | 0.0000 | 0.0000 | 0.1630 |
| P4 | 0.7895 | 0.9000 | 0.8411 | 0.7489 |

## GIN Architecture Summary

| Architecture | Macro precision | Macro recall | Macro F1 | Seed F1 SD | Eligible |
| --- | ---: | ---: | ---: | ---: | --- |
| gin_l2_h128_d02 | 0.5025 | 0.4827 | 0.4712 | 0.0771 | False |
| gin_l2_h128_d05 | 0.4143 | 0.4373 | 0.3788 | 0.0999 | False |
| gin_l2_h64_d02 | 0.4190 | 0.3987 | 0.3849 | 0.1159 | False |
| gin_l2_h64_d05 | 0.4155 | 0.4173 | 0.3980 | 0.0797 | False |
| gin_l3_h128_d02 | 0.4617 | 0.4827 | 0.4422 | 0.0797 | False |
| gin_l3_h128_d05 | 0.4652 | 0.5213 | 0.4721 | 0.0250 | False |
| gin_l3_h64_d02 | 0.4516 | 0.4813 | 0.4398 | 0.1044 | False |
| gin_l3_h64_d05 | 0.3879 | 0.3800 | 0.3517 | 0.1025 | False |

Selected status: **diagnostic_fallback_no_eligible_gin** (`gin_l3_h128_d05`).

## Selected GIN vs Frozen MulCas

| Period | GIN P | GIN R | GIN F1 | MulCas P | MulCas R | MulCas F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | 0.0500 | 0.0040 | 0.0074 | 0.0492 | 0.2600 | 0.0828 |
| P3 | 0.7113 | 0.6600 | 0.6680 | 0.5385 | 0.1400 | 0.2222 |
| P4 | 0.6343 | 0.9000 | 0.7409 | 0.7931 | 0.9200 | 0.8519 |

## Diagnostic Positive Subgroups

These diagnostics did not participate in architecture selection.

| Period | Subgroup | Positive support | Mean recall | Seed SD |
| --- | --- | ---: | ---: | ---: |
| P2 | cloned_ponzi | 0 | N/A | N/A |
| P2 | historically_novel_ponzi | 50 | 0.0040 | 0.0080 |
| P2 | seen_creator_ponzi | 0 | N/A | N/A |
| P2 | unseen_creator_ponzi | 50 | 0.0040 | 0.0080 |
| P3 | cloned_ponzi | 0 | N/A | N/A |
| P3 | historically_novel_ponzi | 50 | 0.6600 | 0.1187 |
| P3 | seen_creator_ponzi | 1 | 0.4000 | 0.4899 |
| P3 | unseen_creator_ponzi | 49 | 0.6653 | 0.1115 |
| P4 | cloned_ponzi | 14 | 1.0000 | 0.0000 |
| P4 | historically_novel_ponzi | 36 | 0.8611 | 0.0556 |
| P4 | seen_creator_ponzi | 4 | 0.8000 | 0.1000 |
| P4 | unseen_creator_ponzi | 46 | 0.9087 | 0.0374 |

## H8R2

H8R2 is **not supported** under the predefined rule.
No architecture, threshold, loss, or subgroup policy was changed after observing results.
