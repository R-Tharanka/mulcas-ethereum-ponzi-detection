# Phase 7E: Historical Clone and Novelty Sensitivity

This analysis reuses exact Phase 7A predictions. It performs no retraining, tuning, fuzzy matching, or model changes.

## Exact Historical Clone Prevalence

| Period | Any clone / all | Ponzi clone / Ponzi | Benign clone / benign |
| --- | ---: | ---: | ---: |
| P2 | 29/5141 (0.6%) | 0/50 (0.0%) | 29/5091 (0.6%) |
| P3 | 1/289 (0.3%) | 0/50 (0.0%) | 1/239 (0.4%) |
| P4 | 15/150 (10.0%) | 14/50 (28.0%) | 1/100 (1.0%) |
| P5 | 8/509 (1.6%) | 1/64 (1.6%) | 7/445 (1.6%) |

## Ponzi Recall by Historical Novelty

| Period | Cloned TP/FN | Recall cloned | Novel TP/FN | Recall novel |
| --- | --- | ---: | --- | ---: |
| P2 | 0/0 | N/A | 13/37 | 26.0% |
| P3 | 0/0 | N/A | 7/43 | 14.0% |
| P4 | 14/0 | 100.0% | 32/4 | 88.9% |
| P5 | 1/0 | 100.0% | 26/37 | 41.3% |

## Benign False-Positive Behavior

| Period | Cloned FP/TN | Clone FP rate | Novel FP/TN | Novel FP rate |
| --- | --- | ---: | --- | ---: |
| P2 | 0/29 | 0.0% | 251/4811 | 5.0% |
| P3 | 0/1 | 0.0% | 6/232 | 2.5% |
| P4 | 0/1 | 0.0% | 12/87 | 12.1% |
| P5 | 1/6 | 14.3% | 3/435 | 0.7% |

## P5 Creator and Novelty Interaction

| Creator status | Clone status | Ponzi | TP | FN | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| seen_creator | historical_code_clone | 1 | 1 | 0 | 100.0% |
| seen_creator | novel_code | 3 | 2 | 1 | 66.7% |
| unseen_creator | novel_code | 60 | 24 | 36 | 40.0% |

## Routing and Interpretation

Among 63 novel P5 Ponzi contracts, 28 were accepted at viewer 1, 29 were confidently misclassified before voting, and 8 were misclassified after reaching voting.

P4 contains 14 cloned positives (28.0%) and achieves 88.9% recall even on novel positives. P5 contains only 1 cloned positive (1.6%); its novel-positive recall is 41.3%.

Historical code reuse is associated with easier detection where cloned Ponzi examples exist, but clone prevalence does not fully explain temporal variation. P2 and P3 have no cloned positives, P4 still performs strongly on novel positives, and subgroup sizes for clones are often very small. These are descriptive associations, not causal evidence.
