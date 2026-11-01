# Archive Directory Guide

## 1. Purpose

The `archive/` directory preserves the research program that existed before
the repository root was reduced to the active random-split three-viewer flow.
The archive keeps the chronological experiments, configurations, source code,
tests, reports, notebooks, and tracked result evidence available without
mixing them into the active execution path.

Archiving did not mean that the earlier research was invalid or deleted. It
means that the repository now has two clearly separated responsibilities:

| Area | Responsibility |
| --- | --- |
| Repository root | Current executable random-split workflow |
| `archive/history/` | Git-tracked historical research evidence |
| `archive/local-only/` | Ignored local datasets, generated artifacts, caches, references, and private material |

The historical archive should be treated as preserved evidence. Most archived
Python modules cannot be executed directly from inside `archive/history/`
because their imports and configuration paths were written for their original
locations such as `src/`, `configs/`, and `results/`.

## 2. Directory Map

```text
archive/
|-- README.md
|-- CHRONOLOGY.md
|-- ARCHIVE_MOVE_SUMMARY.json
|-- history/
|   |-- README.md
|   |-- TRACKED_FILE_MANIFEST.csv
|   |-- LFS_MATERIALIZATION_AUDIT.json
|   |-- configs/
|   |-- data/
|   |-- docs/
|   |-- notebooks/
|   |-- results/
|   |-- src/
|   `-- tests/
`-- local-only/                 # ignored by Git
    |-- cache/
    |-- data/
    |-- docs/
    |-- generated/
    `-- private/
```

The `history/` tree mirrors original repository-relative paths. For example:

```text
Original: configs/paper_aligned_mulcas.yaml
Archive:  archive/history/configs/paper_aligned_mulcas.yaml
```

This path-preserving layout makes it possible to identify where a file
originally belonged and to restore a historical workflow deliberately.

## 3. Tracked Historical Archive

`archive/history/` contains the files that were tracked by Git before the
cleanup. The archive operation moved 567 tracked files. Additional archive
control files document and verify that migration.

| Historical area | Preserved content |
| --- | --- |
| `configs/` | Baseline, Phase 6-9, extension, and frozen-manifest YAML files |
| `docs/` | Progress reports, protocols, phase summaries, and the earlier run guide |
| `notebooks/` | Presentation notebooks for Phases 0-8 |
| `results/` | Compact metrics, traces, reports, manifests, and audit tables |
| `src/` | Historical data, feature, model, experiment, extension, and Phase 9 code |
| `tests/` | Tests that protected the historical pipelines and frozen artifacts |
| `data/README.md` | Historical explanation of local data storage |

The archived source and tests are retained together because experiment results
are not understandable without the implementation and integrity checks that
produced them.

## 4. Local-Only Archive

`archive/local-only/` is intentionally excluded by `.gitignore`. It can contain:

- downloaded or external datasets;
- processed graph, sequence, and creation-code data;
- fitted models and reusable generated artifacts;
- temporary experiment outputs and caches;
- local reference documents; and
- private provider configuration.

This directory is not a portable or reviewable part of the Git repository. Its
contents can differ between machines. Never use `git add -f` on
`archive/local-only/`, and never copy private configuration into tracked
Markdown, YAML, JSON, CSV, logs, or tests.

The active raw dataset and active virtual environment were not moved into this
area because the current root workflow still uses them.

## 5. Archive Control Files

### `archive/CHRONOLOGY.md`

This is the human-readable timeline. It explains the purpose and outcome of
each chronological group:

| Group | Research stage |
| --- | --- |
| `00_05_foundation_and_baseline` | Environment, audit, initial split, features, baselines, and MulCas reconstruction |
| `05_5_paper_aligned` | Corrected chronological split and frozen paper-aligned MulCas |
| `06_phase6_improvements` | Controlled weighting, sampling, threshold, feature, and classifier experiments |
| `07_phase7_analysis` | Temporal robustness, ablations, errors, and clone/novelty analysis |
| `08_phase8_extension` | Recency, graph, creation-code, sequence, external-data, and closure work |
| `09_phase9_prospective` | Protocol-gap analysis and prospective Phase 9 development |
| `10_cross_phase_synthesis_and_support` | Cross-phase reports, guides, and shared support files |

These groups are logical labels in the manifest. They are not separate physical
directories because the physical tree preserves each file's original path.

### `archive/history/TRACKED_FILE_MANIFEST.csv`

This is the authoritative file-level index. Every row records:

- `chronology_group`;
- `original_path`;
- `archived_path`; and
- `git_blob`.

Find every file assigned to Phase 5.5:

```powershell
Import-Csv archive\history\TRACKED_FILE_MANIFEST.csv |
    Where-Object chronology_group -eq "05_5_paper_aligned" |
    Select-Object original_path, archived_path
```

Find a file by part of its original name:

```powershell
Import-Csv archive\history\TRACKED_FILE_MANIFEST.csv |
    Where-Object original_path -like "*mulcas*" |
    Select-Object chronology_group, original_path, archived_path
```

### `archive/ARCHIVE_MOVE_SUMMARY.json`

This machine-readable summary records the archive date, source commit, number
of moved tracked files, integrity exceptions, active-flow hashes, and the test
status observed during cleanup. It is the quickest way to confirm what the
archive operation claimed and what it deliberately did not do.

### `archive/history/LFS_MATERIALIZATION_AUDIT.json`

One archived result is a 107,441,279-byte Git LFS object:

```text
archive/history/results/extension/phase8r6d/enriched_contracts_final.csv
```

The archive stores the materialized CSV rather than the small LFS pointer text.
The audit records the original pointer blob, expected LFS SHA-256, expected
size, materialized SHA-256, and verified status. `.gitattributes` continues to
mark this archived path for Git LFS.

## 6. How to Find a Historical Phase

Use this order:

1. Read `archive/CHRONOLOGY.md` to identify the phase and chronology group.
2. Filter `TRACKED_FILE_MANIFEST.csv` by that group.
3. Start with the archived report or result JSON under `history/docs/` or
   `history/results/`.
4. Read the corresponding YAML under `history/configs/`.
5. Trace implementation code under `history/src/`.
6. Use the matching files under `history/tests/` to understand integrity and
   leakage controls.

For example, the Phase 5.5 paper-aligned flow is represented by:

```text
archive/history/configs/paper_aligned_split.yaml
archive/history/configs/paper_aligned_features.yaml
archive/history/configs/paper_aligned_baseline_models.yaml
archive/history/configs/paper_aligned_mulcas.yaml
archive/history/src/data/paper_aligned_split.py
archive/history/src/features/paper_aligned.py
archive/history/src/models/run_baselines.py
archive/history/src/models/run_mulcas.py
archive/history/results/baseline/paper_aligned/
archive/history/docs/paper_aligned_baseline_report.md
archive/history/tests/test_paper_aligned_protocol.py
```

## 7. Running Historical Work

Do not run commands such as:

```powershell
python -m archive.history.src.models.run_mulcas
```

That is not a supported package layout. Historical imports expect modules such
as `src.models`, and historical configurations expect root-relative paths.

To reproduce an archived phase safely:

1. Create a separate branch and preferably a separate Git worktree.
2. Use the manifest to identify all files in the required phase and its earlier
   dependencies.
3. Restore those files to their original repository-relative paths in the
   isolated worktree.
4. Provide required ignored datasets and generated dependencies separately.
5. install the archived requirements appropriate to that phase;
6. run focused historical tests before running an experiment; and
7. compare regenerated evidence with the archived result files without
   overwriting the archive.

A branch in the same working directory does not isolate ignored datasets or
generated artifacts. A separate worktree or repository copy is safer for full
historical reproduction.

## 8. Integrity Checks

Confirm that tracked historical files are known to Git:

```powershell
git ls-files archive/history
```

Confirm that local-only material is ignored:

```powershell
git check-ignore -v archive/local-only/private/Alchemy.txt
```

Inspect archive changes without scanning private material:

```powershell
git status --short -- archive/history archive/README.md archive/CHRONOLOGY.md
git diff --check -- archive/history archive/README.md archive/CHRONOLOGY.md
```

Inspect the migration summary and LFS audit:

```powershell
Get-Content archive\ARCHIVE_MOVE_SUMMARY.json
Get-Content archive\history\LFS_MATERIALIZATION_AUDIT.json
git lfs status
```

Do not use broad commands that print files under `archive/local-only/private/`.

## 9. What the Archive Is Not

The archive is not:

- the active model-training path;
- a second Python package intended to run in place;
- proof that every ignored local artifact is available in a fresh clone;
- permission to reopen frozen experiments without a new research protocol;
- a combined Track A and Track B dataset; or
- a safe location for tracked credentials.

Archived results retain their original evaluation boundaries. Diagnostic
fallbacks, validation metrics, P5 exploratory evidence, and random-split
results must still be described according to their original protocols.

## 10. Maintenance Rules

- Keep `archive/history/` read-only except for a documented archive correction.
- Do not rewrite historical conclusions to match later work.
- Do not move active-flow dependencies into the archive without updating the
  active documentation and tests.
- Add new archive entries to the manifest with their original paths and Git
  blob IDs.
- Preserve Git LFS attributes and audits for large archived objects.
- Keep `archive/local-only/` ignored and free from tracked secrets.
- Record any restoration or archive correction in a separate auditable change.

## 11. Related Documents

- `docs/ARCHIVED_CODE_REFERENCE.md`: module-by-module catalog of every public
  class, function, and class method in the preserved historical Python source.
- `archive/README.md`: concise archive landing page.
- `archive/CHRONOLOGY.md`: chronological research index.
- `archive/history/README.md`: historical repository README preserved as
  evidence of the earlier root layout.
- `docs/RUN_AND_TEST_GUIDE.md`: instructions for the current active workflow.
- `docs/RANDOM_SPLIT_THREE_VIEWER_FLOW.md`: current workflow architecture and
  its relationship to archived evidence.

