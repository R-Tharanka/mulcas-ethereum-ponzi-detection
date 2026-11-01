# Phase 8R5A Opcode-Sequence Feasibility

## Scope

This phase audited deployed-runtime opcode sequences and compute feasibility only. No supervised Ponzi classifier was trained, no full masked-opcode pretraining was launched, and P5 was used only in global descriptive statistics.

## Frozen Research

All configured Phase 0-8R4B anchors verified (129 unique files). The raw dataset SHA-256 remained `48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf`.

## Sequence Audit

Across 6,498 contracts, opcode-token length ranged from 8 to 18058, with median 2206.0, mean 2664.7, P90 5306.0, P95 6756.8, and P99 9298.8.

The policy decision uses P0-P4 only; P5 and all-contract values are descriptive.

| P0-P4 limit | Contracts exceeding | Percent | Tokens retained by truncation |
| ---: | ---: | ---: | ---: |
| 512 | 5486 | 91.60% | 0.1884 |
| 1024 | 4846 | 80.92% | 0.3574 |
| 2048 | 3198 | 53.40% | 0.6201 |
| 4096 | 1077 | 17.98% | 0.8726 |

| Period | Median tokens | Mean tokens | P95 | Max |
| --- | ---: | ---: | ---: | ---: |
| P0 | 712.0 | 1288.8 | 4447.5 | 7011 |
| P1 | 1187.0 | 2007.7 | 5907.2 | 9150 |
| P2 | 2182.0 | 2662.6 | 6550.0 | 16266 |
| P3 | 2206.0 | 2687.7 | 6982.6 | 14983 |
| P4 | 2380.0 | 2823.5 | 6818.1 | 10942 |
| P5 (descriptive only) | 2622.0 | 3300.3 | 8186.0 | 18058 |

## Temporal Vocabulary

| Evaluation | Historical periods | Vocabulary | OOV tokens | OOV rate |
| --- | --- | ---: | ---: | ---: |
| P2 | P0+P1 | 141 | 279 | 0.000020 |
| P3 | P0+P1+P2 | 142 | 0 | 0.000000 |
| P4 | P0+P1+P2+P3 | 142 | 0 | 0.000000 |

## Representation Recommendation

Use deterministic 1,024-token chunking with future chunk-level pooling. It preserves all tokens, avoids the information loss of truncation, has lower quadratic attention cost than a single 2,048/4,096-token sequence, and remains label-independent. Prefix and prefix+suffix remain documented alternatives, not performance-selected policies.

## Compute Preflight

PyTorch `2.14.1+cpu` reports CUDA unavailable. The OS sees `NVIDIA GeForce RTX 2050` with 4096 MiB, so the installed Torch build is CPU-only despite a GPU being present.

| Length | Seconds/batch | Examples/s | Tokens/s | RSS delta MiB |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1.5067 | 0.6637 | 339.8 | 25.3 |
| 1024 | 0.1443 | 6.9290 | 7095.3 | 139.7 |
| 2048 | 0.5511 | 1.8146 | 3716.2 | 370.0 |

Estimated P2-P4 pretraining time for one seed and 10 epochs/fold: 862.7 minutes. Three seeds: 43.1 hours.

Recommendation: Before Phase 8R5B, install a compatible CUDA PyTorch build in a separately verified environment, benchmark again, and seek explicit approval.

## Frozen Future Design

The proposed encoder uses 128-dimensional opcode embeddings, two Transformer encoder layers, four attention heads, a 256-dimensional feed-forward layer, dropout 0.1, and masked-opcode prediction. Each P2-P4 fold must build its vocabulary and pretraining corpus from completed historical periods only.

H8R5 is not evaluated in Phase 8R5A.

## Blockers Before Phase 8R5B

- The installed PyTorch build cannot use the available GPU.
- The CPU estimate exceeds the 30-minute automatic-launch limit by a wide margin.
- A CUDA-compatible environment needs a fresh representative benchmark before choosing batch size or launching pretraining.
- Phase 8R5B requires explicit approval after that benchmark and must keep separate historical pretraining corpora for P2, P3, and P4.
