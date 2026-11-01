# Phase 8R5C1 Frozen Sequence-Embedding Linear Probe

Status: **complete**.

The Transformer checkpoints remained frozen. P5 was not loaded, thresholds and probe hyperparameters were not tuned, and no hybrid or fine-tuning experiment was run.

## Preflight And Resources

- Estimated embedding extraction: 9.73 minutes.
- Actual embedding extraction: 2.25 minutes.
- Total phase runtime: 2.82 minutes.
- Cached artifacts: 18 files, 46224050 bytes.

## Contextual Seed Results

| Period | Seed | Precision | Recall | F1 | PR-AUC | Balanced accuracy | TN/FP/FN/TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P2 | 42 | 0.0889 | 0.0800 | 0.0842 | 0.0340 | 0.5360 | 5050/41/46/4 |
| P2 | 123 | 0.0161 | 0.0400 | 0.0230 | 0.0180 | 0.5080 | 4969/122/48/2 |
| P2 | 2026 | 0.0478 | 0.2000 | 0.0772 | 0.0249 | 0.5805 | 4892/199/40/10 |
| P3 | 42 | 0.0000 | 0.0000 | 0.0000 | 0.6868 | 0.5000 | 239/0/50/0 |
| P3 | 123 | 0.0000 | 0.0000 | 0.0000 | 0.5301 | 0.5000 | 239/0/50/0 |
| P3 | 2026 | 0.0000 | 0.0000 | 0.0000 | 0.4047 | 0.5000 | 239/0/50/0 |
| P4 | 42 | 0.8000 | 0.7200 | 0.7579 | 0.8270 | 0.8150 | 91/9/14/36 |
| P4 | 123 | 0.7931 | 0.9200 | 0.8519 | 0.8452 | 0.9000 | 88/12/4/46 |
| P4 | 2026 | 0.8113 | 0.8600 | 0.8350 | 0.8518 | 0.8800 | 90/10/7/43 |

## Contextual Seed Variability

| Period | Precision mean +/- SD | Recall mean +/- SD | F1 mean +/- SD | PR-AUC mean +/- SD |
| --- | ---: | ---: | ---: | ---: |
| P2 | 0.0510 +/- 0.0298 | 0.1067 +/- 0.0680 | 0.0615 +/- 0.0274 | 0.0256 +/- 0.0065 |
| P3 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.5405 +/- 0.1154 |
| P4 | 0.8015 +/- 0.0075 | 0.8333 +/- 0.0838 | 0.8149 +/- 0.0409 | 0.8413 +/- 0.0105 |

## Non-Contextual Control

| Period | Seed | Precision | Recall | F1 | PR-AUC | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | 42 | 0.0348 | 0.3200 | 0.0627 | 0.0245 | 0.6164 |
| P2 | 123 | 0.0312 | 0.2800 | 0.0562 | 0.0220 | 0.5974 |
| P2 | 2026 | 0.0222 | 0.3000 | 0.0413 | 0.0201 | 0.5850 |
| P3 | 42 | 0.8000 | 0.2400 | 0.3692 | 0.7725 | 0.6137 |
| P3 | 123 | 0.6667 | 0.0400 | 0.0755 | 0.7154 | 0.5179 |
| P3 | 2026 | 0.8333 | 0.2000 | 0.3226 | 0.8167 | 0.5958 |
| P4 | 42 | 0.7931 | 0.9200 | 0.8519 | 0.7617 | 0.9000 |
| P4 | 123 | 0.7931 | 0.9200 | 0.8519 | 0.8072 | 0.9000 |
| P4 | 2026 | 0.7931 | 0.9200 | 0.8519 | 0.7947 | 0.9000 |

## Three-Seed Ensembles

| Representation | Period | Precision | Recall | F1 | PR-AUC | Balanced accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| contextual | P2 | 0.0588 | 0.0800 | 0.0678 | 0.0341 | 0.5337 |
| non_contextual | P2 | 0.0343 | 0.3400 | 0.0623 | 0.0247 | 0.6230 |
| contextual | P3 | 0.0000 | 0.0000 | 0.0000 | 0.6137 | 0.5000 |
| non_contextual | P3 | 0.6667 | 0.1200 | 0.2034 | 0.7653 | 0.5537 |
| contextual | P4 | 0.8036 | 0.9000 | 0.8491 | 0.8393 | 0.8950 |
| non_contextual | P4 | 0.7931 | 0.9200 | 0.8519 | 0.8071 | 0.9000 |

## Macro P2-P4 Evidence

| System | Precision | Recall | F1 | PR-AUC | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Contextual seed mean | 0.2841 | 0.3133 | 0.2921 | 0.4692 | 0.6355 |
| Non-contextual seed mean | 0.5297 | 0.4600 | 0.3870 | 0.5261 | 0.6918 |
| Frozen MulCas | 0.4603 | 0.4400 | 0.3856 | 0.4651 | 0.6876 |

## Period Comparisons

| Period | Contextual ensemble F1 | Non-contextual ensemble F1 | Delta context-control | Frozen MulCas F1 | Delta context-MulCas |
| --- | ---: | ---: | ---: | ---: | ---: |
| P2 | 0.0678 | 0.0623 | +0.0055 | 0.0828 | -0.0150 |
| P3 | 0.0000 | 0.2034 | -0.2034 | 0.2222 | -0.2222 |
| P4 | 0.8491 | 0.8519 | -0.0028 | 0.8519 | -0.0028 |

## Contextual Ensemble Subgroups

| Period | Subgroup | Ponzi count | TP | FN | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| P2 | historically_novel_ponzi | 50 | 4 | 46 | 0.0800 |
| P2 | exact_clone_ponzi | 0 | 0 | 0 | N/A |
| P2 | unseen_creator_ponzi | 50 | 4 | 46 | 0.0800 |
| P2 | seen_creator_ponzi | 0 | 0 | 0 | N/A |
| P3 | historically_novel_ponzi | 50 | 0 | 50 | 0.0000 |
| P3 | exact_clone_ponzi | 0 | 0 | 0 | N/A |
| P3 | unseen_creator_ponzi | 49 | 0 | 49 | 0.0000 |
| P3 | seen_creator_ponzi | 1 | 0 | 1 | 0.0000 |
| P4 | historically_novel_ponzi | 36 | 31 | 5 | 0.8611 |
| P4 | exact_clone_ponzi | 14 | 14 | 0 | 1.0000 |
| P4 | unseen_creator_ponzi | 46 | 42 | 4 | 0.9130 |
| P4 | seen_creator_ponzi | 4 | 3 | 1 | 0.7500 |

## Complementarity With Frozen MulCas

| Scope | Both correct | Both wrong | MulCas only correct | Sequence only correct | MulCas-only TP | Sequence-only TP | Union recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | 4796 | 53 | 57 | 235 | 10 | 1 | 0.2800 |
| P3 | 233 | 43 | 7 | 6 | 7 | 0 | 0.1400 |
| P4 | 133 | 15 | 1 | 1 | 1 | 0 | 0.9200 |
| P2-P4 | 5162 | 111 | 65 | 242 | 18 | 1 | 0.4467 |

## Conclusion

- Contextual precision eligibility (>= 0.80): **False**.
- H8R5 supported by Phase 8R5C1: **False**.
- Evidence justifies supervised fine-tuning in Phase 8R5C2: **False**.
- This conclusion applies only to the frozen linear-probe protocol. It does not claim that sequence learning or future fine-tuning cannot improve performance.
