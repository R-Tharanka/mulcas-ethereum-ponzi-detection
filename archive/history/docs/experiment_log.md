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

## Phase 8 Extension Closure

**Status:** Complete without an accepted replacement for Phase 5.5 MulCas.

Phase 8R0 through Phase 8R5C1 tested materially different static extension
directions while keeping P5 outside extension model selection and evaluation:
recency-aware training, runtime CFG/GIN representations, MulCas-GIN
complementarity and score fusion, creation-code features, and temporally valid
self-supervised opcode-sequence representations.

H8R1, H8R2, H8R3B, and H8R4 were not supported. H8R5 was not supported by the
frozen Phase 8R5C1 linear-probe evaluation; this does not disprove every
possible sequence fine-tuning method. Phase 8R5C2 was not executed because its
predefined gate failed. Diagnostic fallbacks were retained as evidence, not
promoted as winners.

The frozen Phase 5.5 paper-aligned MulCas remains the reference system. Its P5
result is P=0.9063, R=0.4531, F1=0.6042, and PR-AUC=0.6624. P5 is now
previously observed Track A evidence, not an untouched independent test.

The current static-model search is closed. Phase 8R6A, an external/later
dataset provenance and overlap audit, is the next recommended research phase
and has **not started**. See [Phase 8 Research Extension Closure](phase8_summary.md)
and [`phase8_closure_results.json`](../results/extension/phase8_closure/phase8_closure_results.json).
