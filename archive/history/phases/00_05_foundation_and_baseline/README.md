# Phases 0-5: Foundation and Baseline

This package contains the chronological beginning of the project:

- Phase 0 repository, environment, path, and seed setup;
- Phase 1 dataset integrity audit;
- Phase 2 initial chronological split;
- Phase 3 static feature reproduction;
- Phase 4 Ridge, SVM, and XGBoost baselines; and
- Phase 5 independent MulCas cascade reconstruction.

Important locations:

- `configs/`: initial split, feature, baseline, and MulCas settings;
- `src/data/`: loading, validation, audit, and split logic;
- `src/features/`: opcode, Word2Vec, and creator features;
- `src/models/`: baseline and MulCas implementations;
- `notebooks/00_environment_check.ipynb` through
  `notebooks/05_mulcas_reproduction.ipynb`: presentation notebooks;
- `results/baseline/`: frozen outputs and traces; and
- `docs/`: dataset, methodology, and initial reproduction reports.

Phase 5 was the initial reconstruction. The corrected retained reference is in
the next package, `05_5_paper_aligned`.
