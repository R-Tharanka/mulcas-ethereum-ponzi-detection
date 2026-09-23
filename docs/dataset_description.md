# Dataset Description

This document records the Phase 1 integrity audit for the released dataset:

`data/raw/PonziDataset_20221114.csv`

The raw CSV must not be modified. Machine-readable audit output is saved at
`results/baseline/dataset_audit.json`.

## Source File Identity

| Field | Value |
| --- | --- |
| File size | 307,176,676 bytes |
| SHA-256 | `48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf` |
| Audit generated | See `generated_at_utc` in `results/baseline/dataset_audit.json` |

## Paper Comparison

The published research reports 6,498 contracts, including 314 Ponzi contracts.
The released CSV matches these headline counts.

| Measure | Published | Measured | Difference |
| --- | ---: | ---: | ---: |
| Total contracts | 6,498 | 6,498 | 0 |
| Ponzi contracts | 314 | 314 | 0 |

Measured class counts:

| Label | Count |
| --- | ---: |
| 0 | 6,184 |
| 1 | 314 |

Observed label values are `0` and `1`; no unexpected label values were found.

## Columns And Types

| Column | Audit dtype | Null count |
| --- | --- | ---: |
| `address` | `str` | 0 |
| `opcode` | `str` | 0 |
| `code` | `str` | 0 |
| `contractCode` | `str` | 0 |
| `createdBlockNumber` | `float64` | 0 |
| `createdTransactionHash` | `str` | 0 |
| `creationCode` | `str` | 0 |
| `creator` | `str` | 0 |
| `label` | `int64` | 0 |
| `timestamp` | `str` | 0 |

The initial audit avoids loading all large text/code columns unless needed.
Metadata columns are loaded first; `opcode` is loaded separately for opcode
integrity checks.

## Integrity Checks

| Check | Result |
| --- | ---: |
| Duplicate address rows | 30 |
| Duplicate transaction-hash rows | 197 |
| Duplicate `createdBlockNumber` + `address` rows | 30 |
| Block-number minimum | 49,880 |
| Block-number maximum | 7,136,486 |
| Timestamp minimum | 2015-08-07 20:48:27 |
| Timestamp maximum | 2019-01-27 23:52:51 |
| Chronological by block number | True |
| Chronological by timestamp | True |
| Unique creators | 3,360 |
| Creators with multiple contracts | 1,093 |
| Creators with multiple positive contracts | 55 |
| Missing or empty opcode values | 0 |
| Rows in exact duplicate opcode groups | 712 |
| Exact duplicate opcode groups | 195 |

## Approximate Memory Usage

| Measure | Bytes |
| --- | ---: |
| Metadata-only DataFrame, deep memory | 2,475,870 |
| Opcode Series, deep memory | 100,264,775 |
| Metadata plus opcode, deep memory | 102,740,645 |
| Raw CSV file size | 307,176,676 |

These values are approximate runtime memory indicators. They are intended for
planning local audit and preprocessing workflows, not as model inputs.

## Phase 1 Scope Notes

This audit does not perform feature extraction, model training, balancing,
splitting, or experiment execution. Duplicate rows and repeated creators are
reported for later research decisions, but no rows are removed in Phase 1.
