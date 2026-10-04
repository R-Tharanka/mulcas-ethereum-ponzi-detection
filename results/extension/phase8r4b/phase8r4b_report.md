# Phase 8R4B: Forward Creation-Code Classification

The frozen Phase 8R4A matrices were evaluated with six predeclared candidates on P2-P4 only. P5 was never loaded, no threshold was tuned, and MulCas was not retrained.

## Runtime Preflight

CPU execution was used (`torch 2.14.1+cpu`, CUDA=False). Representative runtimes were {'logistic_regression': 0.2963875999994343, 'xgboost': 2.3058765000023413}; projected full runtime was 23.42s and actual runtime was 7.02s.

## Candidate Summary

| Candidate | Macro precision | Macro recall | Macro F1 | Macro PR-AUC | Macro BA | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| logistic_regression__creation_full | 0.2703 | 0.3067 | 0.2818 | 0.5165 | 0.6280 | False |
| logistic_regression__deployment_init | 0.3379 | 0.0867 | 0.0217 | 0.3224 | 0.5144 | False |
| logistic_regression__combined | 0.2664 | 0.4933 | 0.2750 | 0.4588 | 0.6289 | False |
| xgboost__creation_full | 0.5243 | 0.2067 | 0.2609 | 0.5292 | 0.5983 | False |
| xgboost__deployment_init | 0.3319 | 0.1667 | 0.2214 | 0.3760 | 0.5778 | False |
| xgboost__combined | 0.4655 | 0.1867 | 0.2441 | 0.4815 | 0.5883 | False |

Selected: **logistic_regression__creation_full**. Eligible: **False**. No candidate met macro precision >= 0.80; selected as diagnostic fallback.

## Selected Candidate By Period

| Period | Precision | Recall | F1 | PR-AUC | BA | TN/FP/FN/TP | Delta F1 vs MulCas |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| P2 | 0.0183 | 0.0800 | 0.0299 | 0.0135 | 0.5190 | 4877/214/46/4 | -0.0530 |
| P3 | 0.0000 | 0.0000 | 0.0000 | 0.6889 | 0.5000 | 239/0/50/0 | -0.2222 |
| P4 | 0.7925 | 0.8400 | 0.8155 | 0.8471 | 0.8650 | 89/11/8/42 | -0.0363 |

## Positive Complementarity

| Period | TP both | MulCas-only TP | Creation-only TP | FN both | Oracle union recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| P2 | 3 | 10 | 1 | 36 | 0.2800 |
| P3 | 0 | 7 | 0 | 43 | 0.1400 |
| P4 | 42 | 4 | 0 | 4 | 0.9200 |
| P2-P4 | 45 | 21 | 1 | 83 | 0.4467 |

Oracle union recall is a non-deployable diagnostic; no hybrid was trained.

## Conclusion

H8R4 is **not supported** under the predefined macro-precision eligibility rule.
