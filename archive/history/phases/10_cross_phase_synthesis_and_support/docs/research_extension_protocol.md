# Research Extension Protocol

## Phase 8R0 Scope

Phase 8R0 defines a targeted post-Phase-7 research extension. It is a protocol
and feasibility audit only. It does not train a model, fit a novelty detector,
select a threshold, or evaluate a new system.

Phases 0-7 remain frozen. Their configurations, results, reports, model
definitions, and notebooks are not extension work and must not be overwritten.
All new artifacts live under `configs/extension/`, `src/extension/`, and
`results/extension/`.

## Motivation

The frozen reconstruction under-detects later Ponzi contracts. Phase 7 found
that the P5 errors are concentrated among historically novel code and mostly
unseen creators: all 37 Phase 7A P5 false negatives lack an exact historical
opcode/runtime clone, 36 have unseen creators, and 28 are confidently rejected
by viewer 1. Only three are close to its configured confidence boundary.

Phase 6 also showed that generic class weighting, sampling, operating-threshold
search, compact structural/operand additions, and alternative classifiers did
not produce reliable temporal-CV improvements. The extension therefore targets
temporal adaptation and program structure rather than reopening those searches.

## Reference Reconciliation

Two valid but different P5-like evaluations exist:

| Evaluation | Precision | Recall | F1 | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen Phase 5.5 final test | 0.9063 | 0.4531 | 0.6042 | 442 | 3 | 35 | 29 |
| Phase 7A incremental P5 | 0.8710 | 0.4219 | 0.5684 | 441 | 4 | 37 | 27 |

The Phase 8R0 request supplied the Phase 5.5 metric triplet together with the
Phase 7A confusion counts. They cannot describe one confusion matrix. The two
conditions are retained separately because their training histories differ.

The published Table 5 reference is precision 0.951, recall 0.674, and F1 0.789.
For a 64-positive partition, `TP >= 44` and `FP <= 2` would imply at least
precision 0.9565, recall 0.6875, and F1 0.8000. This is an interpretation of the
published benchmark, never a rule for selecting parameters from P5 labels.

## RQ6 and Hypotheses

**RQ6:** Can a temporal-adaptive, graph-augmented detector improve detection of
historically novel and unseen-creator Ponzi contracts relative to the frozen
MulCas reconstruction while retaining high precision?

- **H8R1:** Recency-aware training improves later-period generalization relative
  to equal-weight expanding-history training.
- **H8R2:** A static EVM control-flow graph representation captures
  discriminative information not represented adequately by flat opcode
  counts/N-grams.
- **H8R3:** A novelty-aware routing mechanism reduces confident false-negative
  decisions on historically novel contracts.
- **H8R4:** A temporally trained hybrid of MulCas and the graph view improves F1
  and recall relative to frozen MulCas.

## Evaluation Tracks

### Track A: Paper-Comparable Exploratory Benchmark

Track A uses the released 6,498-contract dataset. P5 has already been inspected
in Phases 5.5 and 7, so it is not an untouched test set. P5 labels cannot select
the temporal strategy, graph design, novelty method, novelty cutoff, hybrid
rule, model parameters, or operating threshold. A later P5 comparison may be
reported only as exploratory and paper-comparable, not as unbiased external
generalization evidence.

### Track B: External Validation

Track B requires a separately labeled external dataset. No external label may
influence feature, model, hyperparameter, or threshold design. Before scoring,
the candidate dataset must be audited against all 6,498 released contracts for:

- normalized address equality;
- exact normalized opcode equality;
- exact normalized runtime-bytecode equality;
- exact normalized creation-code equality.

All overlaps must be reported. An overlap-free evaluation must be produced when
enough data remains. Dataset provenance, labeling procedure, time coverage,
access conditions, and undefined labels must also be documented.

## Temporal Adaptation Design

The following candidates are fixed before training:

| Strategy | Historical training data |
| --- | --- |
| Expanding history | All complete prior periods, equal row weights |
| Recent 1 | Most recent complete prior period |
| Recent 2 | Most recent two complete prior periods |
| Recent 3 | Most recent three complete prior periods |
| Exponential decay | All prior periods, weight `0.5 ** period_age` |

Only these forward backtests may select a strategy:

1. Train from available P0-P1 history and predict P2.
2. Train from available P0-P2 history and predict P3.
3. Train from available P0-P3 history and predict P4.

Selection eligibility is mean temporal-backtest precision at least 0.80. Among
eligible candidates, rank mean F1, recall, and balanced accuracy in that order;
prefer the simpler equal-weight history if still tied. P5 and external labels
are prohibited during selection.

The current Ridge, SVM, and XGBoost estimator APIs all expose `sample_weight` in
`fit`. The existing factories do not need modification to pass time-decay
weights in a future phase. This is API feasibility only; Phase 8R0 called no
estimator `fit` method.

## Static CFG Design

`contractCode` is the authoritative runtime-bytecode source. Phase 6D2 already
showed deterministic disassembly aligns with the released opcode sequence for
all 6,498 rows. `creationCode` is audited separately as a deployment-time view.
No Solidity source or transaction history is required.

The feasibility parser defines basic-block leaders as the first instruction,
every `JUMPDEST`, and the instruction after a branch or terminator. It identifies
fall-through edges directly. It resolves a jump edge only when a valid,
non-truncated `PUSH` immediately precedes `JUMP`/`JUMPI` and names an existing
`JUMPDEST`; all other jump targets remain explicitly unresolved. This is a
conservative static approximation, not full EVM stack analysis.

Malformed hex produces a recorded non-disassemblable row. A truncated `PUSH`
is retained and counted. Unsupported bytes are represented deterministically as
`INVALID`. Contracts are never silently dropped.

The frozen reproduction preserves full deployed bytecode, including any
compiler metadata tail. Phase 8R0 uses that same policy. A later change to
metadata stripping would be a separately declared representation change.

## Graph Classifier Plan

The future graph unit is one EVM basic block per node. Candidate node features
are contract-local and label-independent:

- opcode-category histogram;
- instruction count;
- CALL-family indicators;
- storage-operation counts;
- stack-operation summaries;
- control-flow type;
- PUSH summaries.

The predefined small model is a 2-3 layer GIN, with GAT as the single declared
alternative, hidden size 64 or 128, global add or mean pooling, a small binary
MLP head, dropout 0.2, and deterministic seed 42. The future selection process
must remain within P2-P4 forward backtesting. A large transformer is out of
scope.

## Novelty-Aware Routing Plan

Novelty must be derived from graph embeddings fitted on historical training
graphs only. Predeclared candidate estimators are k-nearest-neighbor embedding
distance, regularized Mahalanobis distance, and IsolationForest. Evaluation or
future embeddings may be scored but may not fit centroids, covariance,
neighbors, isolation trees, or normalization statistics.

The intended routing is conceptual at this stage:

```text
historically familiar -> frozen MulCas or predeclared normal hybrid route
high novelty          -> graph-informed route or predeclared hybrid decision
```

No novelty threshold or hybrid decision rule has been selected. Both must be
selected using P2-P4 only before any exploratory P5 scoring.

## External Validation Gate

The feasibility audit found public candidates, but no immediately usable,
clearly independent Track B benchmark:

- [PonziGuard repository](https://github.com/PonziDetection/PonziGuard) exposes
  Ponzi/non-Ponzi archives, but its paper says the ground-truth experiment uses
  the same 6,498-contract, 314-Ponzi XBlock/Zheng corpus. It is therefore a
  reproduction/overlap source, not independent evidence.
- [XBlock/Kaggle Smart Ponzi labels](https://www.kaggle.com/xblock/datasets)
  expose the known label resource. No verified newer compatible labeled
  snapshot was found in this audit; using the known release would repeat the
  current corpus.
- PonziGuard's public result directory contains a named 2022-July 2023 address
  list. It may offer later positives, but it is not a complete binary benchmark:
  independent confirmation, bytecode retrieval, overlap checking, and a
  defensible benign cohort are still required.

The PonziGuard extension paper reports use of the same 6,498/314 ground-truth
dataset and describes a separate recent-contract deployment study. Its dynamic
runtime behavior graph also requires transaction generation and instrumented
execution, unlike this proposed static creation-time CFG. These are related but
not equivalent representations.

## Non-Negotiable Guards

- No Phase 0-7 artifact may be changed.
- No model or novelty detector is trained in Phase 8R0.
- No P5 or external label is used for selection.
- No source-code, transaction-history, or future-contract feature is introduced.
- Track A must remain labeled exploratory.
- Track B results require a complete provenance and overlap audit.

## Next Phase Gate

The recommended next phase is **Phase 8R1A: temporal-adaptation backtesting**.
It should execute only the five fixed temporal strategies on P2-P4 with the
frozen feature/model behavior and produce a selected strategy, or a documented
negative result, before graph training begins. In parallel, dependency and
external-dataset acquisition can be prepared without inspecting external labels.

