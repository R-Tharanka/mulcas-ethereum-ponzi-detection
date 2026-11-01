# Phase 5.5: Paper-Aligned Correction

This package contains the corrected chronological protocol that became the
retained reference system.

It includes:

- positive-occurrence chronological splitting;
- filtered opcode N-grams;
- IDF-weighted Word2Vec;
- creator-history alignment across views;
- paper-aligned individual baselines; and
- the frozen paper-aligned MulCas result.

Important locations:

- `configs/paper_aligned_*.yaml`: the four principal configurations;
- `src/data/paper_aligned_split.py`: corrected split construction;
- `src/features/paper_aligned.py`: corrected feature construction;
- `src/models/run_baselines.py` and `src/models/run_mulcas.py`: runners;
- `results/baseline/paper_aligned/`: metrics, traces, and manifests; and
- `docs/paper_aligned_baseline_report.md`: human-readable report.
