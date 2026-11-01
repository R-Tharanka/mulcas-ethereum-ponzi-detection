# Phase-Ordered Historical Archive

The historical research is organized chronologically under `phases/`. Each
phase package contains the configuration, source, tests, results, notebooks,
and documentation owned by that phase.

```text
phases/
|-- 00_05_foundation_and_baseline/
|-- 05_5_paper_aligned/
|-- 06_phase6_improvements/
|-- 07_phase7_analysis/
|-- 08_phase8_extension/
|-- 09_phase9_prospective/
`-- 10_cross_phase_synthesis_and_support/
```

Inside every package, paths remain relative to the original repository root.
For example, the archived Phase 7 implementation remains under
`07_phase7_analysis/src/experiments/`, while its configuration remains under
`07_phase7_analysis/configs/phase7/`.

Use `TRACKED_FILE_MANIFEST.csv` as the authoritative lookup table. It records:

- chronological owner;
- original repository path;
- current archive path; and
- original Git blob identifier.

`LFS_MATERIALIZATION_AUDIT.json` separately verifies the large Phase 8R6D
enrichment table that was materialized from Git LFS.

The phase packages are historical evidence, not directly executable projects.
To run a phase, restore its original paths into a separate worktree or copy of
the repository and include the required predecessor packages.
