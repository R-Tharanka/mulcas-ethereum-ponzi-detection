# Phase 8R2A: Deterministic Static EVM CFG Dataset

This phase constructs and validates graphs only. No classifier or GNN was trained, and no P5 model evaluation was performed.

## Corpus

- Contracts/graphs: 6,498
- Total nodes: 1,514,545
- Total edges: 1,464,910
- Feature dimension: 55
- Artifact chunks: 102
- Compressed artifact size: 68,757,791 bytes
- Corpus hash: `ece393232abad38fb329f4ccbe12e617d9c3ec90ba5f265d867ccf5670fec680`
- Zero-node graphs: 0
- Zero-edge graphs: 1
- Non-finite features: 0

## Graph-Size Audit

| Measure | Min | Median | Mean | P90 | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nodes | 2 | 184.00 | 233.08 | 462.00 | 582.00 | 883.45 | 1302 |
| Edges | 0 | 175.00 | 225.44 | 449.00 | 558.00 | 854.00 | 1801 |
| Unresolved jumps | 0 | 21.00 | 25.85 | 52.00 | 65.00 | 85.03 | 146 |
| Truncated PUSH | 0 | 1.00 | 0.53 | 1.00 | 1.00 | 1.00 | 1 |

## Jump Resolution

- Total JUMP/JUMPI: 963,899
- Statically resolved: 717,928
- Unresolved: 167,989
- Invalid direct targets: 77,982
- Static resolution rate: 0.7448

Unresolved jumps receive no speculative destination edges. JUMPI retains only its sequential fall-through edge when its jump target is unresolved.

## Period-Level Structural Audit

| Period | Graphs | Median nodes | Mean nodes | Median edges | Mean edges | Resolution rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | 104 | 49.00 | 78.38 | 51.00 | 88.75 | 0.6619 |
| P1 | 305 | 70.00 | 115.30 | 80.00 | 130.91 | 0.6296 |
| P2 | 5141 | 187.00 | 239.08 | 179.00 | 229.99 | 0.7380 |
| P3 | 289 | 197.00 | 232.36 | 194.00 | 224.86 | 0.8013 |
| P4 | 150 | 232.00 | 248.53 | 233.00 | 235.67 | 0.7833 |
| P5 | 509 | 203.00 | 270.47 | 193.00 | 261.39 | 0.8067 |

These are descriptive graph statistics, not model-performance measurements. P5 labels were not loaded or evaluated.

## Frozen Representation

The `phase8r2a-cfg-v1` block policy, 55-feature node order, three edge types, direct-PUSH jump rule, and deterministic chunk schema are frozen for Phase 8R2B. Exact PUSH operands are retained only in provenance arrays and are excluded from `x`.
