# Phase 6: Controlled Improvements

This package contains forward-only controlled tests of possible improvements to
the frozen Phase 5.5 reference:

- Phase 6A cost-sensitive learning;
- Phase 6B training-data sampling;
- Phase 6C decision-threshold optimization;
- Phase 6D1 structural opcode features;
- Phase 6D2 normalized operand features; and
- Phase 6E alternative classifiers.

Phase 6F was not run because no independent intervention passed its predefined
criterion. No Phase 6 candidate replaced Phase 5.5.

Important locations:

- `configs/experiments/phase6*/`: phase-specific configs and input manifests;
- `src/experiments/phase6*.py`: experiment runners;
- `src/experiments/temporal_cv.py`: shared forward-only evaluation logic;
- `tests/test_phase6*.py`: experiment and leakage tests;
- `results/experiments/phase6*/`: machine-readable evidence; and
- `docs/phase6_summary.md`: closure summary.
