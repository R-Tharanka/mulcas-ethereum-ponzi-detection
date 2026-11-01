# Phase 7: Temporal and Failure Analysis

This package explains the retained MulCas system rather than proposing a new
replacement:

- Phase 7A temporal robustness over P2-P5;
- Phase 7B viewer/component ablation;
- Phase 7C creator-feature ablation;
- Phase 7D systematic error analysis; and
- Phase 7E clone and historical-novelty sensitivity.

Important locations:

- `configs/phase7/`: Phase 7 configurations and chained manifests;
- `src/experiments/phase7*.py`: analyses;
- `tests/test_phase7*.py`: temporal and integrity tests;
- `results/experiments/phase7*/`: metrics and traces;
- `notebooks/07_ablation_study.ipynb`: presentation notebook; and
- `docs/phase7_summary.md`: closure summary.

The Phase 8 file named `frozen_phase7_manifest.yaml` is intentionally stored in
the Phase 8 package because it freezes Phase 7 as Phase 8's predecessor input.
