# Phase 6A: Controlled Cost-Sensitive Learning

Final-test rows and metrics were not used or generated. Parameters were selected
with forward-only temporal CV, followed by one validation evaluation per family.

## Selected Variants

| Model | Cost | CV precision | CV recall | CV F1 | CV PR-AUC | Validation precision | Validation recall | Validation F1 | Validation PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ridge | `{'0': 1.0, '1': 10.0}` | 0.307542 | 0.653333 | 0.227214 | 0.314530 | 0.793103 | 0.920000 | 0.851852 | 0.716222 |
| svm | `{'0': 1.0, '1': 28.195}` | 0.476871 | 0.353333 | 0.293931 | 0.484114 | 0.793103 | 0.920000 | 0.851852 | 0.788327 |
| xgboost | `1.0` | 0.686869 | 0.293333 | 0.297325 | 0.467277 | 0.793103 | 0.920000 | 0.851852 | 0.866902 |

## Outcome

- Best candidate: `phase6a_xgboost_scale_1`
- H6A supported: `false`
- Selection criterion: maximum validation F1 subject to precision >= 0.80; ties recall, PR-AUC, precision
- CV precision-constraint fallback by family: `{"ridge": true, "svm": true, "xgboost": true}`
- Validation precision-constraint fallback: `true`

H6A is assessed against the frozen Phase 5.5 **validation** metrics because
Phase 6 candidates may not be evaluated against the final-test partition.
No candidate satisfying the configured precision floor was available when a
fallback is reported; the named candidate is then a diagnostic ranking, not a
claim that the precision-constrained objective was achieved.
