# Clone-Balanced Group-Safe Ensemble Result

## Purpose

This controlled successor tested whether reducing repeated clone-family
influence and replacing equal ensemble weights with validation-AP weights could
improve the exact duplicate-group-safe random result. Candidate definitions,
model hyperparameters, selection order, and the one-confirmation test rule were
frozen before execution.

## Method

- The parent fit/validation/test membership was unchanged.
- Connected address/exact-opcode groups remained isolated.
- Clone-capped fitting retained one deterministic row per group-label pair.
- XGBoost and Ridge hyperparameters were inherited from the parent.
- Four candidates were compared on validation only.
- Only the validation winner was evaluated on the held-out grouped test.
- P5 and Track B were not loaded.

Clone capping reduced fit rows from 4,159 to 3,829, removing 330 repeated rows.

## Validation Selection

| Candidate | Validation F1 | Validation AP |
| --- | ---: | ---: |
| Full, equal weights | 0.8261 | 0.8613 |
| Clone-capped, equal weights | 0.8411 | 0.8809 |
| Full, AP weights | 0.8261 | 0.8631 |
| Clone-capped, AP weights | 0.8411 | 0.8821 |

Validation selected the clone-capped AP-weighted candidate.

## Held-Out Result

| System | Precision | Recall | F1 | AP | TN/FP/FN/TP |
| --- | ---: | ---: | ---: | ---: | --- |
| Parent grouped ensemble | 0.7778 | 0.7778 | 0.7778 | 0.7510 | 1223/14/14/49 |
| Selected clone-balanced candidate | 0.6923 | 0.8571 | 0.7660 | 0.7435 | 1213/24/9/54 |

Recall increased by 0.0794, but precision fell by 0.0855. F1 decreased by
0.0118 and AP decreased by 0.0075. The intervention is therefore **not
supported as an overall improvement**.

## Conclusion

Clone capping reduced a known training-dominance issue and improved validation
metrics, but that gain did not generalize to the held-out grouped test. The
result must not replace the parent grouped reference or be described as a win.
Runtime was 1,213.03 seconds and all frozen input hashes were preserved.

