# Clone-Capped Group-Safe Sparse SVM Result

## Purpose And Method

This single fixed candidate tested a paper-motivated sparse linear margin model
without a classifier grid. It used clone-capped fit data, training-only opcode
1-3-gram TF-IDF, `LinearSVC(C=1, class_weight="balanced")`, validation-only
threshold selection, and the unchanged exact group-safe test partition. P5 and
Track B were not loaded.

## Result

| System | Precision | Recall | F1 | AP | TN/FP/FN/TP |
| --- | ---: | ---: | ---: | ---: | --- |
| Parent grouped ensemble | 0.7778 | 0.7778 | 0.7778 | 0.7510 | 1223/14/14/49 |
| Clone-capped sparse SVM | 0.7302 | 0.7302 | 0.7302 | 0.6935 | 1220/17/17/46 |

F1 decreased by 0.0476 and AP decreased by 0.0575. The sparse SVM is **not
supported** as an improvement. Runtime was 63.89 seconds and frozen inputs were
preserved.

## Governance Decision

The grouped test has now been observed by multiple explicitly recorded
diagnostics. Further classifier or threshold shopping on this same partition is
not justified. Subsequent model development should use the frozen Phase 9
forward P2-P4 protocol; sealed Track B must remain unavailable until the final
system is frozen.

