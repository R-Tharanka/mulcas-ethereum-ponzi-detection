# Phase 8R3A: Frozen MulCas-GIN Complementarity

This is an analysis of saved P2-P4 predictions only. No model was trained, no threshold was tuned, and P5 was not loaded.

## Ensemble Disagreement

| Period | Both correct | Both wrong | MulCas only correct | GIN only correct |
| --- | ---: | ---: | ---: | ---: |
| P2 | 4838 | 38 | 15 | 250 |
| P3 | 231 | 19 | 9 | 30 |
| P4 | 121 | 12 | 13 | 4 |
| P2-P4 | 5190 | 69 | 37 | 284 |

## Positive Complementarity

| Period | TP both | TP MulCas only | TP GIN only | FN both | MulCas recall | Oracle union recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | 0 | 13 | 0 | 37 | 0.2600 | 0.2600 |
| P3 | 2 | 5 | 26 | 17 | 0.1400 | 0.6600 |
| P4 | 42 | 4 | 2 | 2 | 0.9200 | 0.9600 |
| P2-P4 | 44 | 22 | 28 | 56 | 0.4400 | 0.6267 |

## Negative Complementarity

| Period | TN both | TN MulCas only | TN GIN only | FP both |
| --- | ---: | ---: | ---: | ---: |
| P2 | 4838 | 2 | 250 | 1 |
| P3 | 229 | 4 | 4 | 2 |
| P4 | 79 | 9 | 2 | 10 |
| P2-P4 | 5146 | 15 | 256 | 13 |

## Non-Deployable Oracle Ceiling

| Period | Precision | Recall | F1 | Balanced accuracy | TN/FP/FN/TP |
| --- | ---: | ---: | ---: | ---: | --- |
| P2 | 0.9286 | 0.2600 | 0.4062 | 0.6299 | 5090/1/37/13 |
| P3 | 0.9429 | 0.6600 | 0.7765 | 0.8258 | 237/2/17/33 |
| P4 | 0.8276 | 0.9600 | 0.8889 | 0.9300 | 90/10/2/48 |
| P2-P4 | 0.8785 | 0.6267 | 0.7315 | 0.8121 | 5417/13/56/94 |

These oracle values assume a perfect per-row selector and are not achievable model results.

## Routing

GIN rescued 28 MulCas false negatives overall; 28 were confidently rejected by MulCas.

## Hybrid Feasibility

- meaningful_period_recoveries: True
- material_overall_union_recall_gain: True
- errors_not_overwhelmingly_identical: True
- novel_or_unseen_creator_complementarity: True

Decision: **justified**.
Proceed to a small temporally trained Phase 8R3B hybrid/novelty-routing experiment.

The project-wide performance preflight rule is recorded in `docs/research_extension_performance_preflight.md`; the original Phase 8R0 protocol remains hash-frozen.
