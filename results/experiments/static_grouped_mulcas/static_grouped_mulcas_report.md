# Duplicate-Group-Safe Random-Split Successor

## Purpose

This experiment addresses the exact duplicate leakage in commit `5ed8509`
without overwriting its historical result. Rows connected by either normalized
contract address or exact normalized opcode sequence are assigned to one
connected duplicate group. Those groups remain intact across fit, validation,
test, and all three internal cross-validation folds.

This is still a random in-distribution experiment. It is not chronological,
does not reproduce the published MulCas cascade, and does not establish
later-time generalization.

## Preserved Evidence

- Track A SHA-256 remained
  `48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf`.
- The historical result remained byte-identical at SHA-256
  `4f8bd7d3c028317495fa7b8a3fa0efed1247f2b8f5c12287cf93192372204916`.
- The historical script and result were not used as successor output targets.

## Grouping And Split

| Item | Result |
| --- | ---: |
| Dataset rows | 6,498 |
| Connected duplicate groups | 5,981 |
| Multirow groups | 195 |
| Rows in multirow groups | 712 |
| Largest group | 34 rows |
| Mixed-label groups | 11 |
| Fit rows / positives | 4,159 / 201 |
| Validation rows / positives | 1,039 / 50 |
| Test rows / positives | 1,300 / 63 |

Fit/validation and development/test overlap counts are zero for connected
group ID, normalized address, and exact normalized opcode hash.

## Results

| Protocol | Precision | Recall | F1 | Average precision | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical row-random `5ed8509` | 0.8833 | 0.8413 | 0.8618 | 0.8833 | 1230 | 7 | 10 | 53 |
| Duplicate-group-safe successor | 0.7778 | 0.7778 | 0.7778 | 0.7510 | 1223 | 14 | 14 | 49 |

The group-safe result is lower by 0.1056 precision, 0.0635 recall, 0.0840 F1,
and 0.1323 average precision. The old score remains valid as the outcome of its
recorded row-random protocol, but it should not be used as duplicate-isolated
evidence.

## Interpretation

Exact duplicate isolation materially reduces the measured score. This is
consistent with optimistic bias in the row-random estimate, but the experiment
does not identify how much of the change is caused by each duplicated family,
split composition, or model-selection variation. The corrected result remains
close to the published Table 5 F1 numerically (0.7778 versus 0.7890), but the
protocols and systems differ, so this is not an exact replication or a valid
superiority claim.

The grouping policy resolves only exact address and exact normalized-opcode
links. Near-duplicate, compiler-derived, template-derived, or semantically
equivalent contract families may still cross the split.

## Artifacts

- `static_grouped_mulcas_results.json`: configuration, hashes, metrics, and environment.
- `split_membership.csv`: auditable fit/validation/test membership and group IDs.
- `test_predictions.csv`: test identities, scores, and predictions.
- `split_audit.json`: pre-training split and overlap audit.

Full fitting completed in 1,829.71 seconds. No Track A row was modified.
