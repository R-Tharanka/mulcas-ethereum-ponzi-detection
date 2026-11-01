# Historical Research Archive

For a detailed explanation of trust boundaries, control files, phase lookup,
integrity checks, and safe restoration, see
[`docs/ARCHIVE_DIRECTORY_GUIDE.md`](../docs/ARCHIVE_DIRECTORY_GUIDE.md).
For a module-by-module index of the preserved Python implementation, see
[`docs/ARCHIVED_CODE_REFERENCE.md`](../docs/ARCHIVED_CODE_REFERENCE.md).

The active repository root now contains only the random-split three-viewer
workflow and the files required to understand, run, and verify it.

Nothing from the prior research program was deleted.

## Archive Layout

```text
archive/
|-- CHRONOLOGY.md
|-- history/
|   |-- TRACKED_FILE_MANIFEST.csv
|   |-- LFS_MATERIALIZATION_AUDIT.json
|   `-- phases/
|       |-- 00_05_foundation_and_baseline/
|       |-- 05_5_paper_aligned/
|       |-- 06_phase6_improvements/
|       |-- 07_phase7_analysis/
|       |-- 08_phase8_extension/
|       |-- 09_phase9_prospective/
|       `-- 10_cross_phase_synthesis_and_support/
`-- local-only/                 # ignored by Git
    |-- data/
    |-- generated/
    |-- private/
    `-- cache/
```

`archive/history/phases/` orders tracked historical files by research phase.
Inside each phase package, a file retains its original repository-relative
path:

```text
archive/history/phases/<chronology group>/<original repository path>
```

For example:

```text
configs/data_split.yaml
-> archive/history/phases/00_05_foundation_and_baseline/configs/data_split.yaml

configs/phase7/temporal_robustness.yaml
-> archive/history/phases/07_phase7_analysis/configs/phase7/temporal_robustness.yaml

configs/extension/frozen_phase7_manifest.yaml
-> archive/history/phases/08_phase8_extension/configs/extension/frozen_phase7_manifest.yaml
```

The last example is intentionally in Phase 8: it freezes the completed Phase 7
state as an input to the Phase 8 extension. A filename that mentions an earlier
phase does not necessarily belong to that earlier phase.

The CSV manifest records:

- chronological group;
- original path;
- archived path; and
- original Git blob ID.

One archived file,
`phases/08_phase8_extension/results/extension/phase8r6d/enriched_contracts_final.csv`,
is stored as its fully materialized Git LFS object rather than its pointer text.
`archive/history/LFS_MATERIALIZATION_AUDIT.json` proves that its byte size and
SHA-256 match the OID and size declared by the original Git blob.

## Local-Only Material

`archive/local-only/` is ignored because it can contain downloaded external
datasets, generated model artifacts, caches, nested repositories, and private
provider configuration. Do not force-add that directory to Git.

The active raw dataset and `.venv` remain at their original locations because
the current workflow uses them.

## Restoring a Historical File

Use `archive/history/TRACKED_FILE_MANIFEST.csv` to locate the phase package and
original path. Restore the contents beneath a phase package to the repository
root while preserving their internal paths. Historical manifests and imports
assume the original layout and are not expected to execute from inside the
archive.

