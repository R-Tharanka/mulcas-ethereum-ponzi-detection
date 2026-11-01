# Phase 8R4A: Creation-Code Feasibility And Features

This phase constructs and audits deployment-time features only. No classifier was trained, no model metric was calculated, and P5 was used only in descriptive period summaries.

## Runtime Boundary Policy

The normalized deployed runtime must occur exactly once as a contiguous byte substring of `creationCode`. Only then is the prefix treated as the deployment-init region. Multiple or missing matches never receive an inferred init boundary. Bytes after the last exact runtime match are retained as trailing-data diagnostics and are not interpreted as constructor arguments.

## Preflight

The deterministic 100-contract CPU sample took 0.836s (8.361 ms/contract), projecting 54.3s for the corpus. Actual processing took 60.5s.

## Corpus Audit

- Parseable creationCode: 6,498/6,498
- Exact runtime substring found: 6,497/6,498
- Unique / ambiguous / absent runtime matches: 6,497 / 0 / 1
- Init region available: 6,497/6,498
- Nonempty trailing region: 2,604/6,498

## Frozen Views

- `creation_full`: 85 features
- `deployment_init`: 90 features
- `creation_full_plus_init_and_boundary`: 183 features
- Matrix SHA-256: `ecc80b2ab83c5954eb8790e7f693619441116506d64f8c6e3aa8aec87bcdf897`
- Compressed size: 2,750,110 bytes

## Period Description

| Period | Rows | Runtime match | Init available | Median creation/runtime ratio | Median init bytes | Trailing nonempty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | 104 | 1.0000 | 1.0000 | 1.0434 | 46.0 | 0.0577 |
| P1 | 305 | 1.0000 | 1.0000 | 1.0728 | 89.0 | 0.2098 |
| P2 | 5141 | 0.9998 | 0.9998 | 1.1172 | 326.0 | 0.4299 |
| P3 | 289 | 1.0000 | 1.0000 | 1.1074 | 265.0 | 0.3426 |
| P4 | 150 | 1.0000 | 1.0000 | 1.0547 | 129.0 | 0.3467 |
| P5 | 509 | 1.0000 | 1.0000 | 1.1128 | 476.0 | 0.3399 |

These period and class summaries are descriptive only. They did not alter the frozen feature schema. H8R4 remains unevaluated until a later modeling phase.
