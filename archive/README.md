# Historical Research Archive

The active repository root now contains only the random-split three-viewer
workflow and the files required to understand, run, and verify it.

Nothing from the prior research program was deleted.

## Archive Layout

```text
archive/
|-- CHRONOLOGY.md
|-- history/
|   |-- TRACKED_FILE_MANIFEST.csv
|   |-- configs/
|   |-- docs/
|   |-- notebooks/
|   |-- results/
|   |-- src/
|   `-- tests/
`-- local-only/                 # ignored by Git
    |-- data/
    |-- generated/
    |-- private/
    `-- cache/
```

`archive/history/` preserves each moved tracked file at:

```text
archive/history/<original repository path>
```

For example:

```text
configs/data_split.yaml
-> archive/history/configs/data_split.yaml
```

The CSV manifest records:

- chronological group;
- original path;
- archived path; and
- original Git blob ID.

One archived file, `results/extension/phase8r6d/enriched_contracts_final.csv`,
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

Use `archive/history/TRACKED_FILE_MANIFEST.csv` to locate the original path.
Copy or move the archived file back to that exact path only when intentionally
restoring the historical pipeline. Historical manifests and imports assume the
original layout and are not expected to execute from inside the archive.

