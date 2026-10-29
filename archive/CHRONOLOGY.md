# Research Chronology

This index orders the archived research from the beginning of the project to
the final prospective phases. The physical archive preserves original paths;
`archive/history/TRACKED_FILE_MANIFEST.csv` assigns each file to one of these
chronological groups.

## 00-05: Foundation and Baseline Reproduction

- Repository/environment setup and deterministic utilities.
- Dataset integrity audit of 6,498 contracts.
- Initial chronological split and static feature reproduction.
- Ridge, SVM, and XGBoost individual baselines.
- Independent MulCas cascade and voting reconstruction.

Archive group: `00_05_foundation_and_baseline`

## 05.5: Paper-Aligned Correction

- Positive-occurrence chronological split.
- Filtered N-grams, IDF-weighted Word2Vec, and creator feature alignment.
- Frozen Phase 5.5 MulCas reference and paper comparison.

Archive group: `05_5_paper_aligned`

The active repository retains only
`results/baseline/paper_aligned/mulcas_results.json` because the random-split
script reads it to print a descriptive comparison.

## 06: Controlled Improvements

- Cost weighting, sampling, threshold, structural, operand, and classifier
  interventions.
- No intervention satisfied the predefined reliability criteria.

Archive group: `06_phase6_improvements`

## 07: Temporal and Failure Analysis

- Expanding temporal evaluation over P2-P5.
- Viewer and creator ablations, error analysis, and code-novelty analysis.
- Strong temporal variation documented without claiming causal concept drift.

Archive group: `07_phase7_analysis`

## 08: Static Research Extensions

- Recency-aware learning.
- Runtime CFG construction, graph aggregates, GIN, and score fusion.
- Creation-code representations.
- Self-supervised opcode-sequence pretraining and frozen linear probes.
- External dataset discovery, provider-backed enrichment, and Track B sealing.
- Formal Phase 8 closure.

Archive group: `08_phase8_extension`

## 09: Prospective Protocol Work

- Paper/reconstruction gap decomposition.
- Prospective Phase 9 evaluation protocol.
- Deployment-available creator/code-family state and routing experiments.
- Track B remained governed by its sealed-label boundary during development.

Archive group: `09_phase9_prospective`

## Cross-Phase Synthesis and Support

- Complete progress reports.
- Run guides, environment support, licenses, and shared project metadata that
  did not belong exclusively to one experimental phase.

Archive group: `10_cross_phase_synthesis_and_support`

## Current Active Flow

The active flow follows commit `5ed8509` and is intentionally separate from
the chronological research archive:

```text
raw opcode CSV
-> random stratified fit/validation/test split
-> count XGBoost + TF-IDF XGBoost + calibrated Ridge/Word2Vec
-> equal probability average
-> validation-selected F1 threshold
-> random test metrics
```

Its stronger F1 is a random-holdout result and must retain the duplicate and
temporal-mixing limitations documented in
`docs/RANDOM_SPLIT_THREE_VIEWER_FLOW.md`.

