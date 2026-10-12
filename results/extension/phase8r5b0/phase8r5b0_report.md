# Phase 8R5B0 CUDA Validation and GPU Benchmark

## Outcome

CUDA validation succeeded and the bounded benchmark completed. Feasibility rating: **GREEN**. Full Phase 8R5B was not launched and remains explicitly approval-gated.

## Environment

- OS: Windows-11-10.0.26300-SP0
- Python: 3.13.14
- Torch: 2.14.1+cu130
- CUDA runtime: 13.0
- GPU: NVIDIA GeForce RTX 2050
- VRAM: 4096 MiB
- NVIDIA driver: 610.78
- torch-geometric: 2.8.0.post1

## GPU Benchmark

| Precision | Batch | Status | Peak allocated MiB | Peak reserved MiB | Headroom | Examples/s | Tokens/s | Forward s | Backward s | Finite loss |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fp32 | 1 | success | 36.4 | 60.0 | 98.5% | 82.05 | 84017 | 0.0039 | 0.0083 | True |
| fp32 | 2 | success | 56.6 | 82.0 | 98.0% | 120.86 | 123764 | 0.0052 | 0.0113 | True |
| fp32 | 4 | success | 89.0 | 110.0 | 97.3% | 122.04 | 124964 | 0.0104 | 0.0223 | True |
| fp32 | 8 | success | 157.1 | 188.0 | 95.4% | 137.47 | 140774 | 0.0177 | 0.0404 | True |
| amp | 1 | success | 31.1 | 56.0 | 98.6% | 157.54 | 161318 | 0.0020 | 0.0043 | True |
| amp | 2 | success | 42.5 | 66.0 | 98.4% | 195.02 | 199702 | 0.0030 | 0.0072 | True |
| amp | 4 | success | 67.5 | 80.0 | 98.0% | 237.91 | 243622 | 0.0042 | 0.0126 | True |
| amp | 8 | success | 108.2 | 132.0 | 96.8% | 261.66 | 267940 | 0.0076 | 0.0230 | True |

## Execution Policy

Recommended physical batch: **8** using **amp**, with 96.8% measured reserved-memory headroom.
Gradient accumulation is not required by this systems benchmark, though Phase 8R5B may predeclare it for a larger effective batch.

## Runtime Estimates

- FP32 batch 8: one seed at 10 epochs/fold = 0.725 hours; three seeds = 2.174 hours.
- AMP batch 8: one seed at 10 epochs/fold = 0.381 hours; three seeds = 1.142 hours.

| Fold | Chunks/epoch | 1 epoch min | 5 epochs min | 10 epochs min |
| --- | ---: | ---: | ---: | ---: |
| P2 | 957 | 0.06 | 0.30 | 0.61 |
| P3 | 16999 | 1.08 | 5.41 | 10.83 |
| P4 | 17908 | 1.14 | 5.70 | 11.41 |

The selected one-seed estimate is **0.381 hours**; three seeds are **1.142 hours**.

## Safeguards

Only the first 5,989 frozen P0-P4 rows were loaded, without labels. P5 was not loaded. No full masked-opcode pretraining or supervised classifier training occurred.
