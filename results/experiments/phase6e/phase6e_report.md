# Phase 6E: Selected Alternative Classifiers

Only Logistic Regression and LightGBM were tested against frozen Phase 5.5 views. Final-test rows were not loaded.

| Classifier | Feature | CV precision | CV recall | CV F1 | Validation precision | Validation recall | Validation F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | opcode_term_count | 0.521771 | 0.586667 | 0.490360 | 0.793103 | 0.920000 | 0.851852 |
| lightgbm | opcode_term_count | 0.568277 | 0.286667 | 0.346353 | 0.803571 | 0.900000 | 0.849057 |

- LightGBM version: `4.7.0`
- H6E supported: `false`
