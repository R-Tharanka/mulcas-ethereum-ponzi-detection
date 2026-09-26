# Phase 6D2: Normalized Operand Features

Authoritative source: `contractCode` deployed runtime bytecode. Historical-EVM disassembly matches all released opcode sequences. Full runtime bytes are preserved; no new metadata stripping policy was introduced.

| Model | Selection | CV precision | CV recall | CV F1 | Validation precision | Validation recall | Validation F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ridge | operand_only | 0.234045 | 0.260000 | 0.224936 | 0.666667 | 0.080000 | 0.142857 |
| svm | baseline_plus_group_c | 0.255191 | 0.273333 | 0.255282 | 0.793103 | 0.920000 | 0.851852 |
| xgboost | baseline_plus_group_c | 0.692708 | 0.293333 | 0.301055 | 0.793103 | 0.920000 | 0.851852 |

- Contracts parsed: 6498
- Truncated PUSH instructions: 3425
- Unsupported opcode bytes: 42382
- H6D2 supported: `false`
