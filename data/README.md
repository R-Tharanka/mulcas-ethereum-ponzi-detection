# Active Dataset

The active random-split experiment reads:

```text
data/raw/PonziDataset_20221114.csv
```

The CSV is intentionally ignored by Git. Do not edit, sort, clean, deduplicate,
or resave it.

Expected properties:

| Property | Value |
| --- | ---: |
| Rows | 6,498 |
| Ponzi labels | 314 |
| Benign labels | 6,184 |
| SHA-256 | `48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf` |

Only `opcode`, `label`, and `address` are loaded by the active script. The
address is used to report split overlap and is not a classifier feature.

Historical external datasets, processed matrices, and interim files were moved
to ignored local storage under `archive/local-only/data/`.
