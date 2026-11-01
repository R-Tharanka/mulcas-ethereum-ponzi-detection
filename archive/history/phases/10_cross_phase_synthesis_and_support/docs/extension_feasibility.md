# Phase 8R0 Extension Feasibility Audit

## Outcome

The targeted extension is technically feasible as a staged research program,
but Phase 8R0 trained no model and provides no performance claim. Runtime CFG
extraction is feasible for every released contract under a deterministic,
conservative policy. External validation remains the main research-design
blocker because the immediately available public candidates substantially
overlap the current corpus or are incomplete binary benchmarks.

## Frozen-Artifact Verification

The extension manifest verifies 18 anchors spanning the raw dataset, Phase 5.5
configuration/results, Phase 6 closure, and Phase 7A-7E results. All matched.
The raw dataset remains:

`48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf`

No Phase 0-7 file was modified by Phase 8R0.

## Reference Audit

The request's metric triplet and confusion counts refer to two distinct runs:

- Phase 5.5 final test: P=0.9063, R=0.4531, F1=0.6042, TN=442, FP=3,
  FN=35, TP=29.
- Phase 7A incremental P5: P=0.8710, R=0.4219, F1=0.5684, TN=441, FP=4,
  FN=37, TP=27.

This audit keeps them separate. The published reference remains P=0.951,
R=0.674, and F1=0.789. On 64 positives, TP=44 and FP=2 imply P=0.9565,
R=0.6875, and F1=0.8000, but those values are not a P5 tuning objective.

## Temporal Adaptation

`RidgeClassifier.fit`, `SVC.fit`, and `XGBClassifier.fit` all accept per-row
`sample_weight`. Therefore, equal-weight expanding history, recent 1/2/3-period
windows, and fixed half-life exponential weighting can be implemented without
changing estimator families. Strategy selection is predefined over P2-P4 only.

No fit call was made in this audit.

## Runtime CFG Audit

Authoritative source: `contractCode`.

| Runtime statistic | Result |
| --- | ---: |
| Contracts audited/disassemblable | 6,498 / 6,498 |
| Invalid or empty bytecode | 0 |
| Graphs with basic blocks | 6,498 |
| Median / mean / max instructions | 2,206 / 2,665.20 / 18,059 |
| Median / mean / max basic blocks | 190 / 239.24 / 1,306 |
| Median / mean / max identified edges | 175 / 225.42 / 1,801 |
| Fall-through edges | 746,822 |
| Conservative static unconditional jump edges | 226,005 |
| Conservative static conditional jump edges | 491,923 |
| Unresolved jump targets | 245,971 |
| Contracts with unresolved targets | 6,468 |
| Truncated PUSH instructions | 3,425 |
| Unsupported opcode bytes | 42,382 |

Basic blocks and fall-through edges are directly identifiable. Many jump
targets remain unresolved because EVM destinations are stack values. The large
unresolved count is expected under the intentionally narrow static rule and is
not treated as a parser failure.

All 6,498 runtime values are valid non-empty hex. A truncated final `PUSH` is
recorded rather than repaired. Unsupported bytes are deterministic `INVALID`
instructions. Full deployed bytes are retained to match the frozen metadata
policy.

## Creation-Code Audit

Authoritative secondary source: `creationCode`.

| Creation-code statistic | Result |
| --- | ---: |
| Contracts audited/disassemblable | 6,498 / 6,498 |
| Invalid or empty bytecode | 0 |
| Graphs with basic blocks | 6,498 |
| Median / mean / max instructions | 2,518 / 3,070.22 / 18,450 |
| Median / mean / max basic blocks | 238 / 307.00 / 2,402 |
| Median / mean / max identified edges | 106 / 139.44 / 1,164 |
| Unresolved jump targets | 990,115 |
| Truncated PUSH instructions | 3,052 |
| Unsupported opcode bytes | 67,339 |

Creation code is available at deployment and fully parseable, but its bytes can
mix constructor execution, embedded runtime code, metadata, and constructor
arguments. The very high unresolved-target count reflects that semantic mixing
as well as dynamic jumps. It is feasible as a separately declared deployment
view, but runtime `contractCode` is the safer primary CFG representation.

## Proposed Graph Model

The design is a small deterministic PyTorch Geometric graph classifier:

- primary model: 2-3 layer GIN;
- declared alternative: GAT;
- hidden size: 64 or 128;
- global add or mean pooling;
- small binary MLP head;
- dropout 0.2;
- node features from basic-block opcode categories, instruction count, calls,
  storage, stack, control-flow type, and PUSH summaries.

Neither `torch` nor `torch_geometric` is installed in the project `.venv`.
Their installation and exact version pinning are deferred to a training phase.

## Novelty Routing

The design permits kNN embedding distance, regularized Mahalanobis distance, or
IsolationForest. Every statistic and model must fit historical training graph
embeddings only. P2-P4 forward backtests must select the method and cutoff. No
novelty score or threshold was fitted in Phase 8R0.

## External Dataset Candidates

| Candidate | Availability | Main limitation | Track B readiness |
| --- | --- | --- | --- |
| PonziGuard ground truth | Public GitHub archives | Same reported 6,498/314 XBlock/Zheng corpus | Not independent |
| XBlock smart Ponzi labels | Public listing | Known release is the current corpus; no verified newer compatible snapshot found | Not independent yet |
| PonziGuard 2022-July 2023 result list | Public address-list directory | Positive-only; provenance, benign cohort, bytecode, and overlap need audit | Not ready |

The [PonziGuard repository](https://github.com/PonziDetection/PonziGuard) lists
separate Ponzi and non-Ponzi archives. The accompanying
[PonziGuard paper](https://arxiv.org/abs/2406.00921) states that its ground truth
contains 6,498 contracts and 314 Ponzi contracts from the XBlock/Zheng dataset.
[XBlock](https://www.xblock.pro/) remains a possible discovery source, but this
audit did not identify a clearly newer, separately labeled, bytecode-compatible
Ponzi benchmark.

No external dataset was downloaded.

## Blockers and Dependencies

1. Install and pin CPU/GPU-compatible `torch` and `torch-geometric` versions
   before graph training.
2. Decide whether full-bytecode metadata preservation remains the final graph
   policy; any change must be predefined and evaluated only through P2-P4.
3. Implement stronger static jump recovery only if it is deterministic and
   fixed before performance inspection; unresolved edges must remain explicit.
4. Acquire a defensible external binary dataset with labels, bytecode, block
   dates, provenance, reuse permission, and enough overlap-free samples.
5. Keep the dynamic PonziGuard CRBG and proposed static CFG conceptually
   separate: the former uses generated/runtime behavior and instrumented EVM
   execution, while this project preserves creation-time static availability.

## Recommended Next Extension Phase

Proceed first with **Phase 8R1A: controlled recency-aware temporal
backtesting**. It requires no new dependency and directly tests H8R1 using the
already frozen P2-P4 protocol. Only after that protocol is complete should a
separate Phase 8R1B implement graph extraction/model training. P5 must remain
outside selection in both stages.

Machine-readable records are in `results/extension/phase8r0/`.

