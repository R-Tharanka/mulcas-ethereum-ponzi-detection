# Phase 6B: Controlled Training-Data Sampling

Final-test rows and metrics were not used. Ratios were selected using forward-only temporal CV; each selected family variant was evaluated once on validation.

## Selected variants

| Model | Sampling | CV precision | CV recall | CV F1 | CV PR-AUC | Val precision | Val recall | Val F1 | Val PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ridge | random_under 1:10 | 0.445027 | 0.740000 | 0.427778 | 0.380306 | 0.793103 | 0.920000 | 0.851852 | 0.717527 |
| svm | random_over 1:10 | 0.865591 | 0.266667 | 0.258951 | 0.544419 | 0.793103 | 0.920000 | 0.851852 | 0.807356 |
| xgboost | random_under 1:1 | 0.356803 | 0.313333 | 0.303273 | 0.407016 | 0.730159 | 0.920000 | 0.814159 | 0.879579 |

## Outcome

- Best candidate: `phase6b_svm_over_1_to_10`
- H6B supported: `false`
- Frozen validation reference: precision 0.793103, recall 0.920000, F1 0.851852, PR-AUC 0.870640.
