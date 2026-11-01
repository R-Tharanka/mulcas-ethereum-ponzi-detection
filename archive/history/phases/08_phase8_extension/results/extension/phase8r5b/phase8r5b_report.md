# Phase 8R5B Masked-Opcode Pretraining

Status: **complete** (9/9 runs complete).

This phase trains only fold-local masked-opcode encoders. It does not load Ponzi labels, evaluate P5, or train a classifier. H8R5 remains unevaluated until Phase 8R5C.

## Frozen Policy

- Folds: P0+P1 -> P2, P0-P2 -> P3, P0-P3 -> P4; future periods are excluded from each vocabulary and corpus.
- Seeds: 42, 123, 2026; every fold/seed starts from a fresh model and optimizer.
- Transformer: embedding 128, 2 layers, 4 heads, feed-forward 256, dropout 0.1.
- Input: deterministic 1,024-token chunks; 15% masking with the fixed 80/10/10 replacement policy.
- Optimizer: AdamW, learning rate 1e-4, weight decay 1e-2, exactly 10 epochs.
- Execution: CUDA AMP, physical batch size 8, no CPU fallback.

## Corpus And Vocabulary

| Fold | Historical periods | Contracts | Chunks | Non-padding tokens | Vocabulary | Vocabulary SHA-256 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| P2 | P0+P1 | 409 | 957 | 746365 | 144 | `1627ebcc9d70e07c4ab5c9f530ac7b4f454f7f581eaf4c3e126eec6c50c8c1ce` |
| P3 | P0+P1+P2 | 5550 | 16999 | 14434964 | 145 | `23f52bcfe1050bbe25a008431aa647fca433ba5eef11f7d2cb598d7393f0c3ec` |
| P4 | P0+P1+P2+P3 | 5839 | 17908 | 15211698 | 145 | `23f52bcfe1050bbe25a008431aa647fca433ba5eef11f7d2cb598d7393f0c3ec` |

## Completed Runs

| Fold | Seed | Final loss | Final top-1 accuracy | Runtime (min) | Peak VRAM MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| P2 | 42 | 3.067678 | 0.247371 | 0.66 | 132.0 |
| P2 | 123 | 3.078008 | 0.246176 | 0.65 | 132.0 |
| P2 | 2026 | 3.075212 | 0.248084 | 0.65 | 132.0 |
| P3 | 42 | 1.741869 | 0.523564 | 11.34 | 132.0 |
| P3 | 123 | 1.653006 | 0.549481 | 11.49 | 132.0 |
| P3 | 2026 | 1.918116 | 0.474557 | 11.63 | 132.0 |
| P4 | 42 | 1.616402 | 0.558130 | 11.98 | 140.0 |
| P4 | 123 | 1.636456 | 0.553231 | 12.02 | 140.0 |
| P4 | 2026 | 1.813668 | 0.499201 | 11.92 | 140.0 |

## Seed Variability And Learning Trend

| Fold | Final loss mean +/- SD | Final accuracy mean +/- SD | Epoch 1 -> 10 loss | Epoch 1 -> 10 accuracy |
| --- | ---: | ---: | ---: | ---: |
| P2 | 3.073633 +/- 0.004362 | 0.247210 +/- 0.000787 | 3.897618 -> 3.073633 | 0.156792 -> 0.247210 |
| P3 | 1.770997 +/- 0.110173 | 0.515867 +/- 0.031068 | 3.145798 -> 1.770997 | 0.225294 -> 0.515867 |
| P4 | 1.688842 +/- 0.088644 | 0.536854 +/- 0.026700 | 3.138375 -> 1.688842 | 0.226324 -> 0.536854 |

## Resource Accounting

- Summed completed-run time: 72.32 minutes.
- Phase 8R5B0 estimate for all nine runs: 68.53 minutes.
- Actual/estimated ratio so far: 1.055.
- CUDA determinism controls are enabled, but bit-identical GPU retraining is not claimed.

## Embedding Readiness

The frozen encoder exposes final-layer per-token hidden states. Phase 8R5C is intended to use padding-aware mean pooling for a chunk representation. No supervised embeddings or classifier were produced here.
