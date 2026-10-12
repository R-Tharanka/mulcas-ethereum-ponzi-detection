# Phase 8 Static Research Extension Closure

## 1. Purpose

This document is the formal closure record for Phase 8R0 through Phase 8R5C1. It consolidates frozen evidence; it does not add an experiment or reinterpret an unsuccessful intervention as a success.

## 2. Scope

No model was trained, no threshold was changed, no P5 evaluation was run, and Phase 8R6A was not started. All metrics below were read from completed artifacts.

## 3. Frozen Inputs Verified

- Frozen artifact anchors verified: **199**
- Phase 8R5B checkpoints reverified: **9**
- Phase 8R5C1 embedding caches reverified: **18**
- Raw dataset SHA-256: `48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf`
- Verification status: **PASS**

## 4. Phase 8 Hypothesis Register

| Hypothesis | Tested claim | Status | Evidence |
| --- | --- | --- | --- |
| H8R1 | recency-aware training | NOT SUPPORTED | 8R1A |
| H8R2 | runtime CFG and GIN under the predefined eligibility criterion | NOT SUPPORTED | 8R2B |
| H8R3B | lightweight learned MulCas-GIN score fusion | NOT SUPPORTED | 8R3B |
| H8R4 | creation/deployment-bytecode representation | NOT SUPPORTED | 8R4B |
| H8R5 | temporally pretrained opcode-sequence representation; does not disprove all fine-tuning methods | NOT SUPPORTED BY FROZEN LINEAR PROBE | 8R5C1 |

Phase 8R0, 8R2A, 8R3A, 8R4A, 8R5A, 8R5B0, and 8R5B were protocol, construction, analysis, feasibility, systems, or representation-learning stages. They did not independently establish a supervised hypothesis result. Phase 8R5C2 was not executed because the predefined Phase 8R5C1 gate failed. This does not disprove every possible sequence fine-tuning method.

## 5. Results by Phase

| Phase | System | Macro P | Macro R | Macro F1 | Macro PR-AUC | Role | Eligible |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| control | Frozen MulCas | 0.4603 | 0.4400 | 0.3856 | 0.4651 | retained_reference | False |
| 8R1A | Decay alpha 0.75 | 0.4582 | 0.4533 | 0.4024 | 0.5044 | diagnostic_fallback | False |
| 8R2B | GIN 3x128 dropout 0.5 | 0.4652 | 0.5213 | 0.4721 | 0.4744 | diagnostic_fallback | False |
| 8R3B | Hybrid A | N/A | N/A | N/A | N/A | diagnostic_fallback | False |
| 8R4B | Logistic Regression + creation_full | 0.2703 | 0.3067 | 0.2818 | 0.5165 | diagnostic_fallback | False |
| 8R5C1 | Contextual seed mean | 0.2841 | 0.3133 | 0.2921 | 0.4692 | hypothesis_candidate | False |
| 8R5C1 | Non-contextual seed mean | 0.5297 | 0.4600 | 0.3870 | 0.5261 | scientific_control | False |

Diagnostic fallbacks are preserved for interpretation only. None is a research winner.

## 6. P2-P4 Temporal Comparison

| System | Period | Precision | Recall | F1 | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen MulCas | P2 | 0.0492 | 0.2600 | 0.0828 | 0.0462 |
| Frozen MulCas | P3 | 0.5385 | 0.1400 | 0.2222 | 0.4786 |
| Frozen MulCas | P4 | 0.7931 | 0.9200 | 0.8519 | 0.8706 |
| GIN 3x128 dropout 0.5 | P2 | 0.0500 | 0.0040 | 0.0074 | 0.0204 |
| GIN 3x128 dropout 0.5 | P3 | 0.7113 | 0.6600 | 0.6680 | 0.6254 |
| GIN 3x128 dropout 0.5 | P4 | 0.6343 | 0.9000 | 0.7409 | 0.7773 |
| LR + creation_full | P2 | 0.0183 | 0.0800 | 0.0299 | 0.0135 |
| LR + creation_full | P3 | 0.0000 | 0.0000 | 0.0000 | 0.6889 |
| LR + creation_full | P4 | 0.7925 | 0.8400 | 0.8155 | 0.8471 |
| Contextual sequence ensemble | P2 | 0.0588 | 0.0800 | 0.0678 | 0.0341 |
| Non-contextual sequence control | P2 | 0.0343 | 0.3400 | 0.0623 | 0.0247 |
| Contextual sequence ensemble | P3 | 0.0000 | 0.0000 | 0.0000 | 0.6137 |
| Non-contextual sequence control | P3 | 0.6667 | 0.1200 | 0.2034 | 0.7653 |
| Contextual sequence ensemble | P4 | 0.8036 | 0.9000 | 0.8491 | 0.8393 |
| Non-contextual sequence control | P4 | 0.7931 | 0.9200 | 0.8519 | 0.8071 |

P4 was consistently easier for several systems. P2 and P3 were much less stable, and some methods improved one period while degrading another. No Phase 8 method produced a reliable cross-period improvement under its predefined criterion. This is temporal instability and evidence consistent with distribution variation; it is not proof of concept drift.

## 7. Failure-Mode Synthesis

Phase 7 found 37 P5 false negatives: 36 involved unseen creators, all 37 lacked an exact historical opcode/runtime clone, and 28 were confidently rejected by MulCas viewer 1. These are descriptive associations, not causes.

Phase 8R3A showed genuine period-specific complementarity: the GIN recovered 28 P2-P4 Ponzi contracts missed by MulCas, all with historically novel code and unseen creators, with most of the benefit in P3. The learned Phase 8R3B fusion nevertheless predicted no P3 positives and degraded P4. Creation-code models showed weak positive-class complementarity. The contextual sequence representation underperformed frozen MulCas and its non-contextual control in macro F1; the non-contextual control's tiny macro-F1 increase remained ineligible because macro precision was only 0.5297.

## 8. Comparison With Frozen MulCas

No Phase 8 intervention replaces the frozen Phase 5.5 paper-aligned MulCas. Its P5 result remains P=0.9063, R=0.4531, F1=0.6042, PR-AUC=0.6624, with TN/FP/FN/TP=442/3/35/29.

P5 is no longer an untouched independent test after Phase 5.5 and Phase 7. Any future use must be labeled exploratory Track A evidence.

## 9. Published Benchmark Gap

| Metric | Published Table 5 | Frozen Phase 5.5 | Difference |
| --- | ---: | ---: | ---: |
| Precision | 0.9510 | 0.9063 | -0.0447 |
| Recall | 0.6740 | 0.4531 | -0.2209 |
| F1 | 0.7890 | 0.6042 | -0.1848 |

This is not an exact replication comparison. The paper reports 5,990/508 rows while whole-block reconstruction gives 5,989/509; Table 5 trains through the 250th Ponzi while Phase 5.5 fits through the 200th and reserves P4 for validation; the paper's viewer selection appears test-based while this project uses validation/temporal protocols; and the N-gram dimensionality does not match exactly.

## 10. Why No Phase 8 Replacement Was Accepted

Recency weighting, runtime CFG aggregates and GINs, learned score fusion, creation-code views, and self-supervised opcode-sequence embeddings all failed their predefined temporal acceptance rules. Period-specific improvements were insufficient evidence of reliable P2-P4 generalization.

## 11. What Was Deliberately Not Executed

Phase 8R5C2 was not run because its gate failed. GAT, GraphSAGE, larger GINs, extra decay values, further threshold searches, additional fusion models, creation-code fusion, and arbitrary model shopping were not launched. Any future attempt requires a new preregistered rationale.

## 12. Track A / Track B Status

- **Track A:** the existing 6,498-contract corpus is heavily inspected and remains useful for exploratory analysis and controlled temporal experiments, but not untouched final-generalization claims.
- **Track B:** no genuinely external, later, independently labeled and overlap-audited benchmark has been established.

## 13. Updated RQ6 Answer

The tested Phase 8 approaches did not produce a reliable improvement under the predefined temporal criteria. Some methods showed period-specific or diagnostic benefits, especially GIN on P3, but no candidate generalized strongly enough across P2-P4 to replace frozen MulCas.

## 14. Limitations

P5 has been observed; the corpus ends in January 2019; no Track B benchmark exists; period imbalance is extreme; P2 offers sparse positive history for representation learning; CFG jump resolution is conservative; the sequence conclusion is linear-probe-specific; CUDA retraining is not guaranteed bit-identical; present-day generalization is unknown; and the paper's protocol and feature details remain under-specified.

## 15. Research-Governance Decision

The current Phase 8 static-model search is closed. Continuing to add classifiers, thresholds, graph variants, or post-hoc combinations on the same corpus is not justified by the current evidence. This does not mean improvement is impossible.

## 16. Recommended Next Phase

The next recommended phase is **Phase 8R6A: external/later dataset discovery, provenance, and overlap audit**. It is **not started**. New independent temporal coverage is more informative now than another unmotivated classifier search.

## 17. Reproducibility and Tests

The closure generator verifies the full frozen-manifest chain, raw data, nine MLM checkpoints, and 18 embedding caches. Closure outputs are deterministic and contain no training code. The complete repository test command is `python -m pytest -q`.

## 18. Final Phase 8 Closure Statement

Phase 8 is closed without an accepted replacement for the frozen Phase 5.5 MulCas reference. Recency-aware learning, runtime CFG/GIN, learned score fusion, creation-code features, and frozen self-supervised sequence embeddings did not produce a reliable improvement under their predefined temporal criteria. These negative results do not establish that improvement is impossible. They show that the tested static interventions did not resolve the observed temporal-generalization weakness on the current dataset. The next recommended research step is an external/later dataset provenance and overlap audit before further model development.
