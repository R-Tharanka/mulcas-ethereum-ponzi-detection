# Phase 6C: Controlled Decision-Threshold Optimization

Only native classifier operating thresholds were changed. MulCas cascade theta, lambda, viewer order, routing, and voting were not changed. Final-test rows and labels were not loaded.

## Results

| Model | Default | Selected | Eligible | CV precision | CV recall | CV F1 | Default val F1 | Selected val F1 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| ridge | 0 | -0.973011 | False | 0.035118 | 0.953333 | 0.067740 | 0.851852 | 0.662162 |
| svm | 0 | 1.67652 | True | 1.000000 | 0.006667 | 0.013245 | 0.851852 | 0.000000 |
| xgboost | 0.5 | 0.993604 | True | 1.000000 | 0.086667 | 0.159509 | 0.851852 | 0.000000 |

## Outcome

- H6C supported: `false`
- Eligibility: pooled temporal-CV precision >= 0.80.
- PR-AUC is computed from continuous scores and is invariant across thresholds for each fitted score vector.
