# Phase 6D1: Structural Static Features

Creation-time, opcode-name-only features were evaluated with frozen models, native thresholds, no sampling, and no class weighting. Final-test rows were not loaded.

| Model | Selected representation | CV precision | CV recall | CV F1 | Validation precision | Validation recall | Validation F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ridge | baseline_plus_group_c | 0.147592 | 0.493333 | 0.193679 | 0.793103 | 0.920000 | 0.851852 |
| svm | baseline_plus_group_e | 0.305051 | 0.253333 | 0.261998 | 0.793103 | 0.920000 | 0.851852 |
| xgboost | frozen_baseline | 0.686869 | 0.293333 | 0.297325 | 0.793103 | 0.920000 | 0.851852 |

- Added structural features: 74
- H6D1 supported: `false`
