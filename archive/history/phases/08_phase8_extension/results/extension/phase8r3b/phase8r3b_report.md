# Phase 8R3B: Forward MulCas-GIN Score Fusion

MulCas and GIN were not retrained. The meta-classifier used P2 for feature-set selection on P3, then the selected schema was frozen and refitted on P2+P3 for one P4 confirmation. P5 was not loaded.

## Score Semantics

`mulcas_score` is the frozen accepted viewer's positive-class score for cascade decisions. For voting rows it is the maximum frozen voting-viewer score. It is a deployment-time confidence proxy, not a route-independent calibrated probability. Hybrid B therefore includes route one-hot indicators.

## Runtime Preflight

CPU execution was used (`torch 2.14.1+cpu`, CUDA=False). The representative fit took 0.0890s; estimated full runtime was 5.00s and actual experiment runtime was 2.3877s.

## P3 Selection

| System | Precision | Recall | F1 | PR-AUC | Balanced accuracy | TN/FP/FN/TP |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| frozen_mulcas | 0.5385 | 0.1400 | 0.2222 | 0.4786 | 0.5574 | 233/6/43/7 |
| frozen_gin_ensemble | 0.8235 | 0.5600 | 0.6667 | 0.7074 | 0.7674 | 233/6/22/28 |
| hybrid_a | 0.0000 | 0.0000 | 0.0000 | 0.5760 | 0.5000 | 239/0/50/0 |
| hybrid_b | 0.0000 | 0.0000 | 0.0000 | 0.6097 | 0.4979 | 238/1/50/0 |

Selected: **hybrid_a**. Eligible: **False**. No candidate met the P3 precision floor; selected as a diagnostic fallback.

## P4 Forward Confirmation

| System | Precision | Recall | F1 | PR-AUC | Balanced accuracy | TN/FP/FN/TP |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| frozen_mulcas | 0.7931 | 0.9200 | 0.8519 | 0.8706 | 0.9000 | 88/12/4/46 |
| frozen_gin_ensemble | 0.6984 | 0.8800 | 0.7788 | 0.8369 | 0.8450 | 81/19/6/44 |
| hybrid_a | 0.6567 | 0.8800 | 0.7521 | 0.8026 | 0.8250 | 77/23/6/44 |

## Novelty And Creator Subgroups

| System | Period | Subgroup | Positive support | MulCas recall | GIN recall | Hybrid recall |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| hybrid_a | P3 | historically_novel_ponzi | 50 | 0.1400 | 0.5600 | 0.0000 |
| hybrid_a | P3 | historical_clone_ponzi | 0 | N/A | N/A | N/A |
| hybrid_a | P3 | unseen_creator_ponzi | 49 | 0.1429 | 0.5714 | 0.0000 |
| hybrid_a | P3 | seen_creator_ponzi | 1 | 0.0000 | 0.0000 | 0.0000 |
| hybrid_a | P4 | historically_novel_ponzi | 36 | 0.8889 | 0.8333 | 0.8333 |
| hybrid_a | P4 | historical_clone_ponzi | 14 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_a | P4 | unseen_creator_ponzi | 46 | 0.9130 | 0.8913 | 0.8913 |
| hybrid_a | P4 | seen_creator_ponzi | 4 | 1.0000 | 0.7500 | 0.7500 |
| hybrid_b | P3 | historically_novel_ponzi | 50 | 0.1400 | 0.5600 | 0.0000 |
| hybrid_b | P3 | historical_clone_ponzi | 0 | N/A | N/A | N/A |
| hybrid_b | P3 | unseen_creator_ponzi | 49 | 0.1429 | 0.5714 | 0.0000 |
| hybrid_b | P3 | seen_creator_ponzi | 1 | 0.0000 | 0.0000 | 0.0000 |

## Shared-Error Corrections

- hybrid_a on P3: 2
- hybrid_a on P4: 0
- hybrid_b on P3: 2

Counts are reported separately because the two P3 candidates evaluate the same rows; they must not be summed as distinct contracts.

## Conclusion

H8R3B is **not supported** under the predeclared P3 eligibility and P4 confirmation rule.
