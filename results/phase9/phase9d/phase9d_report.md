# Phase 9D Cascade Calibration and Safer Routing

## Scope

Phase 9D evaluates the frozen MulCas viewer scores on Track A P2-P4 only. Track B remains sealed, P5 is not evaluated, and no underlying viewer or static representation is changed.

## Candidates

- **D0:** exact frozen MulCas control.
- **D1:** prior-period Platt calibration with frozen cascade thresholds and voting rule.
- **D2:** fixed two-vote positive confirmation and three-vote negative recovery on raw scores.
- **D3:** D1 calibration followed by the same fixed consensus guard.

## Pooled Results

| Candidate | Precision | Recall | F1 | PR-AUC | FPR | Recall at FPR ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 | 0.1970 | 0.4400 | 0.2722 | 0.3373 | 0.0495 | 0.4400 |
| D1 | 0.2305 | 0.3733 | 0.2850 | 0.3439 | 0.0344 | 0.4267 |
| D2 | 0.7869 | 0.3200 | 0.4550 | 0.4044 | 0.0024 | 0.5067 |
| D3 | 0.7742 | 0.3200 | 0.4528 | 0.3872 | 0.0026 | 0.5133 |

## Period F1

| Candidate | P2 | P3 | P4 |
| --- | ---: | ---: | ---: |
| D0 | 0.0828 | 0.2222 | 0.8519 |
| D1 | 0.0789 | 0.0000 | 0.8174 |
| D2 | 0.0755 | 0.0000 | 0.8519 |
| D3 | 0.0755 | 0.0000 | 0.8440 |

## Prospective Decision

- **D1:** NOT ADVANCED.
- **D2:** NOT ADVANCED.
- **D3:** NOT ADVANCED.

H9D is not supported. The tested calibration and fixed routing guard did not satisfy every prospective Phase 9B advancement condition.

## Integrity

Track B labels were not opened, no Track B row was fitted or scored, and no Track B metric was generated. P5 was not evaluated. Phase 9E was not started.

## Bootstrap

Paired contract-level bootstrap used 2000 replicates and seed 42.
