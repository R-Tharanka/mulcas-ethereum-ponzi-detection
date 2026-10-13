# Phase 9C Deployment-Available Creator and Code-Family State

Phase 9C evaluates deployment-available historical state on Track A only. Track B remains sealed and is not used for development.

## Results

| Candidate | Pooled P | Pooled R | Pooled F1 | PR-AUC | FPR | FPR-constrained R | Advanced |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C0 | 0.1970 | 0.4400 | 0.2722 | 0.3373 | 0.0495 | 0.4400 | control |
| C1 | 0.2104 | 0.4600 | 0.2887 | 0.1837 | 0.0477 | 0.4600 | False |
| C2 | 0.2951 | 0.4800 | 0.3655 | 0.2785 | 0.0317 | 0.5400 | False |
| C3 | 0.2969 | 0.5067 | 0.3744 | 0.2620 | 0.0331 | 0.5467 | False |

## Period F1

| Candidate | P2 F1 | P3 F1 | P4 F1 |
| --- | ---: | ---: | ---: |
| C0 | 0.0828 | 0.2222 | 0.8519 |
| C1 | 0.1042 | 0.2222 | 0.8519 |
| C2 | 0.1290 | 0.3478 | 0.8519 |
| C3 | 0.1404 | 0.4000 | 0.8519 |

## Advancement Checks

| Candidate | F1 | PR-AUC | FPR | Breadth | Period floor | Uncertainty | Advanced |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | True | False | True | True | True | True | False |
| C2 | True | False | True | True | True | True | False |
| C3 | True | False | True | True | True | True | False |

## Paired Bootstrap

| Candidate | Delta F1 95% CI | Delta PR-AUC 95% CI | Delta recall 95% CI |
| --- | --- | --- | --- |
| C1 | [-0.0018, 0.0357] | [-0.1892, -0.1069] | [-0.0067, 0.0494] |
| C2 | [0.0553, 0.1334] | [-0.1067, -0.0056] | [-0.0135, 0.0958] |
| C3 | [0.0660, 0.1408] | [-0.1262, -0.0214] | [0.0138, 0.1233] |

## Error Recovery: C3

| Period | FN recovered | New FN | FP corrected | New FP |
| --- | ---: | ---: | ---: | ---: |
| P2 | 7 | 4 | 154 | 65 |
| P3 | 7 | 0 | 2 | 2 |
| P4 | 0 | 0 | 0 | 0 |

## Decision

The tested creator/code-family state features did not satisfy the prospective advancement criteria. Track B remains sealed and the negative result does not invalidate the future Phase 9D routing experiment.

Selected candidate: `None`. H9C supported: **False**.

## Reproducibility and Tests

Focused Phase 9C tests: **16 passed**. Full repository suite: **349 passed, 65 warnings in 151.34 seconds**.

No P5 or Track B evaluation occurred. Cascade thresholds, voting, viewers, and model families remained frozen. Phase 9D was not started.
