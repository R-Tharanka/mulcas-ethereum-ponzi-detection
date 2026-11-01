# Phase 7D: Frozen MulCas Error Analysis

This analysis reads the exact Phase 7A predictions. It performs no model fitting, optimization, threshold changes, or feature changes.

## Error Inventory

| Period | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| P2 | 4840 | 251 | 37 | 13 |
| P3 | 233 | 6 | 43 | 7 |
| P4 | 88 | 12 | 4 | 46 |
| P5 | 441 | 4 | 37 | 27 |

## Routing of Errors

| Period | Route | Count | Percent of errors | FP | FN | Near boundary |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2 | confident_wrong_first_viewer | 33 | 11.5% | 0 | 33 | 0 |
| P2 | confident_wrong_second_viewer | 250 | 86.8% | 250 | 0 | 0 |
| P2 | unresolved_until_voting_wrong | 5 | 1.7% | 1 | 4 | 1 |
| P3 | confident_wrong_first_viewer | 41 | 83.7% | 0 | 41 | 0 |
| P3 | confident_wrong_second_viewer | 6 | 12.2% | 6 | 0 | 0 |
| P3 | unresolved_until_voting_wrong | 2 | 4.1% | 0 | 2 | 0 |
| P4 | confident_wrong_first_viewer | 11 | 68.8% | 8 | 3 | 0 |
| P4 | confident_wrong_second_viewer | 4 | 25.0% | 4 | 0 | 0 |
| P4 | unresolved_until_voting_wrong | 1 | 6.2% | 0 | 1 | 0 |
| P5 | confident_wrong_first_viewer | 29 | 70.7% | 1 | 28 | 2 |
| P5 | confident_wrong_second_viewer | 4 | 9.8% | 3 | 1 | 1 |
| P5 | unresolved_until_voting_wrong | 8 | 19.5% | 0 | 8 | 0 |

Errors are usually accepted confidently before voting. P2 is dominated by second-viewer false positives, while P3 and P5 are dominated by first-viewer false negatives. Only 1.7%-19.5% of period errors reach voting.

## False Negatives vs True Positives

| Period | Mean opcode count FN / TP | Mean unique opcodes FN / TP | Mean entropy FN / TP | Mean operands FN / TP | Mean confidence FN / TP |
| --- | --- | --- | --- | --- | --- |
| P2 | 2892.3 / 764.6 | 60.2 / 52.7 | 4.745 / 4.750 | 723.4 / 194.4 | 0.942 / 0.857 |
| P3 | 2187.6 / 2304.4 | 72.7 / 64.3 | 4.915 / 4.772 | 565.1 / 540.3 | 0.971 / 0.874 |
| P4 | 2137.8 / 2435.1 | 62.5 / 77.2 | 4.790 / 4.939 | 528.5 / 630.7 | 0.917 / 0.970 |
| P5 | 3574.2 / 4019.4 | 68.8 / 75.7 | 4.838 / 4.882 | 773.1 / 1003.6 | 0.905 / 0.763 |

There is no single size or operand pattern across periods. P2 false negatives are much longer than P2 true positives, whereas P3-P5 contrasts are smaller or reverse in direction. This period dependence is consistent with changing distributions, not proof of concept drift.

## False Positives vs True Negatives

| Period | Mean opcode count FP / TN | Mean entropy FP / TN | Mean operands FP / TN | Mean confidence FP / TN |
| --- | --- | --- | --- | --- |
| P2 | 1507.9 / 2725.9 | 4.758 / 4.719 | 384.1 / 685.9 | 0.819 / 0.991 |
| P3 | 1859.5 / 2812.8 | 4.745 / 4.729 | 448.7 / 734.7 | 0.826 / 0.990 |
| P4 | 2512.3 / 3100.0 | 4.931 / 4.730 | 657.5 / 785.7 | 0.965 / 0.994 |
| P5 | 2989.5 / 3236.1 | 4.785 / 4.730 | 792.8 / 824.0 | 0.816 / 0.983 |

False positives generally have lower mean prediction confidence than true negatives. P2 false positives are also substantially shorter on average and are concentrated at the second cascade viewer.

## Historical Context and Clones

| Period | Errors from unseen creators | Errors with Ponzi-creator flag | Error opcode clones | Error runtime clones |
| --- | ---: | ---: | ---: | ---: |
| P2 | 98.3% | 0.0% | 0.0% | 0.0% |
| P3 | 98.0% | 2.0% | 0.0% | 0.0% |
| P4 | 87.5% | 12.5% | 0.0% | 0.0% |
| P5 | 92.7% | 7.3% | 2.4% | 2.4% |

## P5 False Negatives

P5 contains 37 false negatives and 27 true positives.

- Unseen-creator prevalence: FN 97.3%; TP 88.9%.
- Historical Ponzi-creator flag: FN 2.7%; TP 11.1%.
- Exact historical opcode clone: FN 0.0%; TP 3.7%.
- Exact historical runtime clone: FN 0.0%; TP 3.7%.
- Mean opcode count: FN 3574.2; TP 4019.4.
- Mean parseable operands: FN 773.1; TP 1003.6.
- Reached voting: FN 21.6%; TP 3.7%.
- Routing of the 37 misses: 28 first-viewer confident errors, 1 second-viewer confident error, and 8 voting-stage errors. Only 3 are near a configured boundary under the predeclared 0.01 margin.

The missed P5 Ponzi contracts are mainly unseen-creator, non-clone, confidently misclassified cases: 36/37 have unseen creators, 36/37 lack the historical Ponzi-creator flag, all 37 lack exact historical opcode/runtime clones, and 29/37 are accepted incorrectly by a cascade viewer. Eight uncertain misses reach voting. These findings are descriptive and do not establish scam motive or concept drift.

## Interpretation Limits

Observed period differences are consistent with temporal distribution shift, but this descriptive analysis does not prove concept drift. Diagnostic structural and operand features were not supplied to the frozen model, and no result was used to tune it.
