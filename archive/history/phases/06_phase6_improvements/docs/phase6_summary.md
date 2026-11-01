# Phase 6 Research Closure

## Conclusion

None of the tested Phase 6 interventions demonstrated a reliable improvement
under the predefined temporal evaluation protocol.

This result does not show that improvement is impossible. It shows that the
specific, predeclared interventions evaluated in Phases 6A-6E did not provide
reliable evidence of improvement under the forward-only temporal protocol and
precision eligibility rule used by this project.

Phase 6F was **not executed because no independently supported intervention
existed to combine.** Combining diagnostic fallbacks after observing their
results would be an arbitrary post-hoc experiment and would violate the Phase 6
protocol.

## Evaluation Protocol

All Phase 6 experiments reused the same three expanding, forward-only temporal
folds inside the Phase 5.5 training partition. Feature preprocessing was fitted
on each fold's historical training rows, and its later interval was transform-only.
The external Phase 5.5 validation partition was evaluated once after temporal-CV
selection. The final-test partition was not evaluated during Phase 6.

The frozen Phase 5.5 paper-aligned MulCas baseline remains the model carried
into Phase 7:

| Precision | Recall | F1 | PR-AUC | Balanced accuracy | TN | FP | FN | TP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7931 | 0.9200 | 0.8519 | 0.8706 | 0.9000 | 88 | 12 | 4 | 46 |

## Phase Outcomes

| Phase | Hypothesis and intervention | Temporal-CV conclusion | One-time validation conclusion | Status |
| --- | --- | --- | --- | --- |
| 6A | H6A: cost-sensitive Ridge/SVM/XGBoost could improve recall and F1 while preserving precision. | No selected family variant met the predefined mean-CV precision floor; reported selections were diagnostic fallbacks. | Selected variants did not improve frozen precision/recall/F1; ranking quality was generally lower. | Not supported |
| 6B | H6B: controlled random under/over-sampling could improve recall and F1 while preserving precision. | Only the selected SVM sampling candidate met the CV precision rule; the remaining selections were fallbacks, and CV evidence did not establish a useful general improvement. | Ridge and SVM matched frozen precision/recall/F1 but reduced PR-AUC; XGBoost reduced precision and F1. | Not supported |
| 6C | H6C: pooled temporal-CV operating-threshold selection could improve F1 and/or recall with useful precision. | Ridge had no eligible threshold. SVM and XGBoost selected very conservative eligible thresholds with negligible CV recall. | SVM and XGBoost predicted no validation positives; the Ridge fallback increased recall but sharply reduced precision and F1. | Not supported |
| 6D1 | H6D1: compact opcode-sequence structural features could improve the frozen static views. | No representation met the mean-CV precision floor; all selected representations were diagnostic fallbacks. | Ridge and SVM matched frozen classification metrics but reduced PR-AUC; XGBoost retained the frozen control. | Not supported |
| 6D2 | H6D2: normalized deployed-bytecode operand features could recover useful discarded information. | No operand representation met the mean-CV precision floor; all selections were diagnostic fallbacks. | SVM/XGBoost width features matched frozen classification metrics but reduced PR-AUC; operand-only Ridge degraded substantially. | Not supported |
| 6E | H6E: Logistic Regression or LightGBM could exploit the frozen Phase 5.5 views more effectively. | Neither classifier family produced an eligible candidate; both selections were diagnostic fallbacks. | Logistic Regression matched frozen classification metrics but reduced PR-AUC. LightGBM slightly raised precision while lowering recall, F1, PR-AUC, and balanced accuracy. | Not supported |
| 6F | Combine independently supported Phase 6 interventions. | No independently supported intervention was available. | No combination or validation evaluation was performed. | Not executed |

## Archived Outputs

- Phase 6A: [`results/experiments/phase6a/phase6a_results.json`](../results/experiments/phase6a/phase6a_results.json), [`phase6a_candidates.csv`](../results/experiments/phase6a/phase6a_candidates.csv), [`phase6a_validation.csv`](../results/experiments/phase6a/phase6a_validation.csv)
- Phase 6B: [`results/experiments/phase6b/phase6b_results.json`](../results/experiments/phase6b/phase6b_results.json), [`phase6b_candidates.csv`](../results/experiments/phase6b/phase6b_candidates.csv), [`phase6b_validation.csv`](../results/experiments/phase6b/phase6b_validation.csv)
- Phase 6C: [`results/experiments/phase6c/phase6c_results.json`](../results/experiments/phase6c/phase6c_results.json), [`phase6c_threshold_search.csv`](../results/experiments/phase6c/phase6c_threshold_search.csv), [`phase6c_validation.csv`](../results/experiments/phase6c/phase6c_validation.csv)
- Phase 6D1: [`results/experiments/phase6d1/phase6d1_results.json`](../results/experiments/phase6d1/phase6d1_results.json), [`phase6d1_candidates.csv`](../results/experiments/phase6d1/phase6d1_candidates.csv), [`phase6d1_validation.csv`](../results/experiments/phase6d1/phase6d1_validation.csv)
- Phase 6D2: [`results/experiments/phase6d2/phase6d2_results.json`](../results/experiments/phase6d2/phase6d2_results.json), [`phase6d2_candidates.csv`](../results/experiments/phase6d2/phase6d2_candidates.csv), [`phase6d2_validation.csv`](../results/experiments/phase6d2/phase6d2_validation.csv)
- Phase 6E: [`results/experiments/phase6e/phase6e_results.json`](../results/experiments/phase6e/phase6e_results.json), [`phase6e_candidates.csv`](../results/experiments/phase6e/phase6e_candidates.csv), [`phase6e_validation.csv`](../results/experiments/phase6e/phase6e_validation.csv)

The machine-readable closure record is
[`results/experiments/phase6_summary.json`](../results/experiments/phase6_summary.json).
