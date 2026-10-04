# Experiment Log

## Phase 6 Closure

**Status:** Closed without a supported improvement.

Phases 6A, 6B, 6C, 6D1, 6D2, and 6E were completed under the same
forward-only temporal selection protocol. Their respective hypotheses were not
supported. None of the tested Phase 6 interventions demonstrated a reliable
improvement under the predefined temporal evaluation protocol.

This conclusion does not claim that improvement is impossible. It records that
the tested cost weighting, sampling, operating thresholds, structural features,
normalized operand features, Logistic Regression, and LightGBM interventions
did not satisfy the project's reliability criteria.

Phase 6F was not executed because no independently supported intervention
existed to combine. No post-hoc combination was attempted, and the final-test
partition was not evaluated.

The frozen Phase 5.5 paper-aligned MulCas baseline is carried forward into
Phase 7. See [Phase 6 Research Closure](phase6_summary.md) and the
machine-readable record at
[`results/experiments/phase6_summary.json`](../results/experiments/phase6_summary.json).

## Phase 7 Closure

**Status:** Complete as a frozen-system analysis phase.

Phases 7A-7E evaluated the unchanged Phase 5.5 MulCas reconstruction through
forward temporal testing, viewer ablation, creator-feature ablation, error
analysis, and exact historical-clone sensitivity. No model was optimized,
replaced, or selected from these analyses.

The reconstructed system exhibits substantial temporal performance variation.
Component and creator-feature contributions are period-dependent. Later errors
are concentrated among contracts with unseen creators and historically novel
code, and many are confidently misclassified. Exact historical code reuse is
associated with easier detection where cloned Ponzi contracts occur, but clone
prevalence does not fully explain the temporal results.

This closure does not claim that concept drift was proven, that novel code
causes misclassification, or that creator identity causes Ponzi behavior. See
[Phase 7 Research Closure](phase7_summary.md) and the machine-readable record
at [`results/experiments/phase7_summary.json`](../results/experiments/phase7_summary.json).
