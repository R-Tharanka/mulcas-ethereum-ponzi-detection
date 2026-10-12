# Full Current Research Progress Report

## MulCas Ethereum Ponzi Smart-Contract Detection

**Report date:** 2026-10-12  
**Project status:** Phase 8 extension complete and formally closed
**Current retained reference system:** Frozen Phase 5.5 paper-aligned MulCas  
**Latest completed experiment:** Phase 8R5C1 frozen sequence-embedding linear probe  
**Raw dataset status:** Unmodified and hash-verified

## 1. Purpose of This Report

This document gives one ordered, understandable account of the project from
repository setup through the latest sequence-representation experiment. It
describes the project, execution flow, environment, dataset, preprocessing,
splits, features, thresholds, models, each research phase, results,
conclusions, tests, limitations, and remaining work.

This is a new report. It does not replace or modify earlier reports. Exact JSON
and CSV artifacts remain the source of truth when a rounded value in this
document differs in displayed precision.

## 2. Executive Summary

The project independently reproduces and investigates MulCas, a static
multi-view cascade for detecting Ethereum Ponzi smart contracts. The detector
uses contract bytecode/opcode information and a leakage-safe historical creator
indicator. It does not rely on transaction behavior for the frozen baseline.

The work completed four broad stages:

1. **Reproduction foundation, Phases 0-5:** environment setup, dataset audit,
   chronological splitting, feature reproduction, individual classifiers, and
   an independent MulCas cascade.
2. **Paper alignment, Phase 5.5:** corrected the development protocol and
   feature representations without overwriting the initial reconstruction.
3. **Controlled improvement and analysis, Phases 6-7:** tested weighting,
   sampling, thresholds, features, and alternative classifiers; then analyzed
   temporal robustness, component contribution, creator history, errors, and
   code novelty.
4. **Targeted research extension, Phase 8R0-8R5C1:** tested recency-aware
   learning, runtime control-flow graphs and GIN, score fusion, creation-code
   features, and temporally valid self-supervised opcode-sequence learning.

The current frozen Phase 5.5 final-test result is:

| Accuracy | Balanced accuracy | Precision | Recall | F1 | PR-AUC | TN | FP | FN | TP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9253 | 0.7232 | 0.9063 | 0.4531 | 0.6042 | 0.6624 | 442 | 3 | 35 | 29 |

Compared with published MulCas Table 5, current precision is reasonably close,
but recall and F1 remain substantially lower:

| Metric | Published | Current Phase 5.5 | Current - published |
| --- | ---: | ---: | ---: |
| Precision | 0.9510 | 0.9063 | -0.0448 |
| Recall | 0.6740 | 0.4531 | -0.2209 |
| F1 | 0.7890 | 0.6042 | -0.1848 |

On the 64-positive test partition, approximately **TP >= 44 and FP <= 2**
would exceed all three published precision/recall/F1 references. The current
result is TP=29 and FP=3, so it is about 15 additional correctly detected Ponzi
contracts and one fewer false positive away from that benchmark interpretation.
This is not a tuning rule.

No Phase 6 or Phase 8 intervention met its predefined reliability criterion.
The main unresolved weakness is temporal generalization: performance is strong
in P4 but poor or unstable in P2, P3, and P5. The project has nevertheless
produced a reproducible baseline, extensive negative results, and clear evidence
about where and how the system fails.

### 2.1 Current Status Matrix

| Phase | Purpose | Status | Main outcome |
| --- | --- | --- | --- |
| 0 | Repository/environment foundation | Complete | Local `.venv`, packages, paths, seeds, tests |
| 1 | Dataset integrity | Complete | 6,498 rows and 314 Ponzi verified |
| 2 | Initial split | Complete, historical | Chronological 70/15/15 split exposed class-allocation weakness |
| 3 | Published feature families | Complete | Count, TF-IDF, N-gram, Word2Vec, creator |
| 4 | Individual baselines | Complete | Ridge, SVM, XGBoost on initial split |
| 5 | Independent MulCas | Complete, historical | Cascade/voting reproduced; weak initial test recall |
| 5.5 | Paper-aligned correction | Complete and frozen | Current retained MulCas reference |
| 6A-6E | Controlled improvements | Complete | All hypotheses not supported |
| 6F | Combine supported improvements | Not executed by design | No independently supported input existed |
| 6 closure | Improvement-study closure | Complete | Frozen Phase 5.5 retained |
| 7A-7E | Robustness, ablation, errors, novelty | Complete | Strong temporal variation characterized |
| 7 closure | Analysis closure | Complete | RQ4/RQ5 evidence consolidated |
| 8R0 | Extension protocol/feasibility | Complete | RQ6 and two-track extension defined |
| 8R1A | Recency-aware MulCas | Complete | H8R1 not supported |
| 8R2A | Runtime CFG corpus | Complete and frozen | 6,498 deterministic graphs |
| 8R2B | Graph aggregate/GIN | Complete | H8R2 not supported |
| 8R3A | MulCas-GIN complementarity | Complete, analysis only | Complementarity justified one small hybrid test |
| 8R3B | Score fusion | Complete | H8R3B not supported |
| 8R4A | Creation-code representation | Complete and frozen | Three deployment-time feature views |
| 8R4B | Creation-code classification | Complete | H8R4 not supported |
| 8R5A | Sequence feasibility | Complete | 1,024-token chunking frozen |
| 8R5B0 | CUDA benchmark | Complete | GREEN feasibility gate |
| 8R5B | Masked-opcode pretraining | Complete | Nine temporal checkpoints produced |
| 8R5C1 | Frozen embedding linear probe | Complete | H8R5 not supported; fine-tuning gate failed |
| Phase 8 closure | Consolidated extension closure | Complete | No accepted Phase 8 replacement; Phase 5.5 retained |
| Phase 8R6A | External/later data provenance and overlap audit | Not started | Next recommended research phase |
| Final synthesis | Notebook/report packaging | Pending | Notebook 08 remains empty |

### 2.2 Source Paper and Reproduction Target

The primary publication is:

> Zibin Zheng, Weili Chen, Zhijie Zhong, Zhiguang Chen, and Yutong Lu.
> *Securing the Ethereum from Smart Ponzi Schemes: Identification Using Static
> Features.* ACM Transactions on Software Engineering and Methodology, 32(5),
> Article 130, July 2023. DOI: `10.1145/3571847`.

The archived local reference is `docs/reference/P09 - Ponzi-SmartContract.pdf`.
The planning references are
`docs/reference/research_continuing_and_improving_plan.md` and
`docs/reference/phase 6 - realign.md`; they are guidance documents, while
configs, source code, and generated result artifacts record what was actually
executed.

The source paper's central claim is creation-time detection using deployed
bytecode and developer information, without waiting for post-deployment
transaction behavior. It reports four principal motivations: prior datasets
were small, transaction-dependent detection can react too late, random splitting
can leak future contract evolution into the past, and simple single-view models
have limited accuracy under severe imbalance.

The paper built labels from contracts with verified source code. Seven
researchers participated in a cross-check process: contracts were independently
read by two researchers, disagreements were escalated, unresolved cases were
discarded, and another researcher challenged a random subset. This strengthens
label provenance, but creates an important sampling limitation: evaluation is
on source-available contracts, whose distribution may differ from all deployed
Ethereum contracts.

The source paper's research questions differ from this continuation project's
RQ1-RQ6 and must not be conflated:

| Source-paper question | Topic |
| --- | --- |
| Paper RQ1 | MulCas sustainability/model aging across P2-P5 |
| Paper RQ2 | Whether MulCas outperforms baseline approaches |
| Paper RQ3 | Effectiveness of developer features |
| Paper RQ4 | Robustness under different class-imbalance ratios |

The continuation project broadens these into independent reproduction,
controlled improvement, component contribution, temporal robustness, and
new-representation research.

The paper uses more than one evaluation protocol. Its setup describes a default
chronological 80/20 split and additional train/test-ratio checks. Its
sustainability analysis defines P0-P5 by groups of 50 chronological Ponzi
contracts and retrains incrementally. Its Table 5 benchmark trains on contracts
through the 250th Ponzi and tests on the remaining 64 Ponzi contracts, reporting
5,990 training rows and 508 test rows. These protocols must be distinguished
when comparing a reported metric.

### 2.3 Published Feature and Model Results

The paper discards PUSH operands, treats opcode sequences as documents, uses
term count, TF-IDF, unigram/bigram/trigram count and TF-IDF, a 100-dimensional
Word2Vec IDF-weighted contract vector, and a creator vector. It reports the
following N-gram dimensions after document-frequency selection:

| Published N-gram family | Before selection | After selection |
| --- | ---: | ---: |
| Unigram | 141 | 77 |
| Unigram + bigram | 7,211 | 767 |
| Unigram + bigram + trigram | 52,532 | 2,292 |

Published Table 5 reports:

| Model | Feature | Recall | Precision | F1 |
| --- | --- | ---: | ---: | ---: |
| Ridge | N-gram count | 0.453 | 0.829 | 0.586 |
| Ridge | Term count | 0.406 | 0.839 | 0.547 |
| Ridge | Word2Vec | 0.000 | 0.000 | 0.000 |
| Ridge | N-gram TF-IDF | 0.328 | 0.913 | 0.483 |
| Ridge | TF-IDF | 0.250 | 0.889 | 0.390 |
| SVM | N-gram count | 0.375 | 0.923 | 0.533 |
| SVM | Term count | 0.328 | 0.913 | 0.483 |
| SVM | Word2Vec | 0.266 | 0.895 | 0.410 |
| SVM | N-gram TF-IDF | 0.281 | 0.900 | 0.429 |
| SVM | TF-IDF | 0.281 | 0.900 | 0.429 |
| XGBoost | N-gram count | 0.219 | 0.875 | 0.350 |
| XGBoost | Term count | 0.219 | 0.875 | 0.350 |
| XGBoost | Word2Vec | 0.219 | 0.875 | 0.350 |
| XGBoost | N-gram TF-IDF | 0.234 | 0.882 | 0.370 |
| XGBoost | TF-IDF | 0.234 | 0.882 | 0.370 |
| MulCas | Hybrid | 0.674 | 0.951 | 0.789 |

The paper also compares against the rule/symbolic-execution system SadPonzi.
SadPonzi successfully analyzed 4,881 of 6,498 contracts and reported overall
P=0.51, R=0.59, F1=0.55 on successful cases. Its P2-P5 F1 values were
0.50, 0.53, 0.21, and 0.20, compared with published MulCas values 0.53, 0.48,
0.87, and 0.79. SadPonzi was not reconstructed in this repository; these are
reference-only paper results.

Published Table 6 reports a much stronger creator-feature effect than this
project later observes under strict forward reconstruction:

| Published viewer | Without creator R/P/F1 | With creator R/P/F1 |
| --- | --- | --- |
| TF-IDF XGBoost | 0.234 / 0.882 / 0.370 | 0.547 / 0.946 / 0.693 |
| Term-count XGBoost | 0.219 / 0.875 / 0.350 | 0.563 / 0.947 / 0.706 |
| Word2Vec SVM | 0.266 / 0.895 / 0.410 | 0.266 / 0.895 / 0.410 |
| N-gram-TFIDF Ridge | 0.328 / 0.913 / 0.483 | 0.531 / 0.971 / 0.596 |

The paper states that 61% of its Ponzi contracts come from experienced
creators. It also attributes part of P5 behavior to 34 contracts derived from a
new Ponzi-token demo, reporting 15 tainted creators in one discussion and 14 in
another. The released CSV has no explicit scam-family label, so this project
cannot directly reproduce that semantic subtype analysis and does not infer it
from predictions.

The nearby prose says MulCas F1 is 0.743, conflicting with Table 5's 0.789.
This project uses the internally consistent Table 5 triplet for comparison and
preserves the inconsistency as a publication limitation.

## 3. Project Description

### 3.1 Research Goal

The project aims to:

- reproduce the published MulCas approach independently;
- measure any disagreement between the released dataset, paper, and
  reconstruction;
- prevent temporal and label leakage;
- test controlled improvements without tuning on the final test;
- explain temporal failures and component contributions; and
- investigate materially different static program representations.

### 3.2 Research Questions

The completed work addresses:

- **RQ1:** Can the published static feature views and individual classifiers be
  independently reproduced?
- **RQ2:** Can the MulCas cascade and voting algorithm be reconstructed?
- **RQ3:** Do controlled interventions improve the frozen baseline under
  forward-only temporal evaluation?
- **RQ4:** Which feature groups and MulCas components contribute to predictions?
- **RQ5:** Does reconstructed MulCas retain effectiveness on later contracts?
- **RQ6:** Can temporal adaptation or richer graph/sequence/deployment-time
  representations improve historically novel and unseen-creator detection?

### 3.3 Supported High-Level Answers

- **RQ1/RQ2:** Yes for independent reconstruction, but exact paper replication
  is not claimed.
- **RQ3:** No tested Phase 6 intervention reliably improved the baseline.
- **RQ4:** Viewer and creator contributions are complementary and
  period-dependent.
- **RQ5:** Effectiveness is temporally unstable; this does not by itself prove
  concept drift.
- **RQ6:** The completed recency, graph, fusion, creation-code, and frozen
  sequence-probe experiments did not meet the predefined improvement criteria.

## 4. How the System Works

### 4.1 End-to-End Research Flow

```text
Immutable raw CSV
    -> integrity audit and SHA-256 verification
    -> chronological split or P0-P5 period assignment
    -> historical-only preprocessing
       - fit vocabulary/IDF/Word2Vec/scalers on training rows only
       - build creator state from strictly earlier labels only
    -> static feature views
    -> individual classifiers
    -> MulCas cascade and voting
    -> validation/future-period prediction
    -> metrics, traces, ablations, and error analysis
```

Extension experiments add separate representations, but do not overwrite the
frozen MulCas pipeline:

```text
contractCode -> deterministic disassembly -> CFG -> graph aggregates / GIN
creationCode -> full creation and deployment-init features
runtime opcode sequence -> historical MLM pretraining -> frozen embeddings
```

### 4.2 Frozen MulCas Prediction Flow

Every static view includes the one-dimensional chronological creator feature.

1. **Viewer 1:** XGBoost on opcode term-count features.
   - score <= 0.05: accept benign;
   - score >= 0.95: accept Ponzi;
   - otherwise: pass the contract onward.
2. **Viewer 2:** XGBoost on filtered 1-3-gram count features.
   - score <= 0.05: accept benign;
   - score >= 0.51: accept Ponzi;
   - otherwise: send to voting.
3. **Voting stage:** four frozen viewers vote:
   - term-count XGBoost;
   - filtered N-gram-count XGBoost;
   - filtered N-gram-TFIDF XGBoost; and
   - opcode-TFIDF Ridge.
4. **Voting rule:** `lambda = 0`; at least one positive vote produces a positive
   final prediction.
5. **Traceability:** row order, address, scores, accepted viewer, stage, and
   voting route are retained in trace files.

During fitting, cascade viewer 1 is trained on the full historical training
partition. Confidently handled training rows are removed, and cascade viewer 2
is fitted on the residual training rows. The voting viewers are separately
fitted on the complete historical training partition because the residual set
is too small to train every voting model reliably. Prediction applies the same
stage order and sends only unresolved rows to voting.

MulCas is the final frozen baseline system, not a single classifier. It tries to
let confident early viewers decide cheaply while escalating uncertain contracts
to additional views.

### 4.3 Codebase Architecture

The implementation keeps reusable logic out of notebooks:

| Area | Main modules | Responsibility |
| --- | --- | --- |
| Data | `src/data/load.py`, `validate.py`, `audit.py` | Typed loading, schema checks, integrity audit |
| Splits | `src/data/split.py`, `paper_aligned_split.py` | Initial and Ponzi-occurrence chronological membership |
| Core features | `src/features/opcode.py`, `word2vec.py`, `developer.py` | Opcode vectorizers, semantic vectors, creator history |
| Paper-aligned features | `src/features/paper_aligned.py` | Frozen Phase 5.5 feature generation/artifacts |
| Additional static features | `structural.py`, `operands.py`, `opcode_categories.py` | Phase 6D1/6D2 and diagnostic feature extractors |
| Baseline models | `src/models/baselines.py`, `run_baselines.py` | Ridge, SVM, XGBoost construction and evaluation |
| MulCas | `src/models/mulcas.py`, `run_mulcas.py` | Cascade routing, confidence acceptance, voting, traces |
| Metrics | `src/evaluation/metrics.py`, `temporal.py` | Binary metrics and temporal period utilities |
| Controlled experiments | `src/experiments/phase6*.py`, `phase7*.py` | Phase 6 selection and Phase 7 analysis |
| Shared temporal CV | `src/experiments/temporal_cv.py` | Expanding forward-only folds and assertions |
| CFG/graph extension | `src/extension/cfg.py`, `graph_features.py`, `graph_dataset.py`, `graph_models.py` | Disassembly, graphs, frozen loading, GIN |
| Creation-code extension | `creation_code.py`, `creation_features.py` | Runtime-boundary audit and deployment-time features |
| Sequence extension | `sequence_data.py`, `sequence_model.py`, `sequence_pretraining.py`, `sequence_embeddings.py` | Tokenization, Transformer MLM, checkpoints, embeddings |
| Extension runners | `src/extension/phase8*.py` | Protocol enforcement and result generation |
| Utilities | `src/utils/paths.py`, `seed.py` | Repository-relative paths and deterministic seeds |

Configuration is stored in YAML under `configs/`. Compact metrics, manifests,
reports, and traces are tracked under `results/`. Large generated matrices,
graphs, fitted estimators, checkpoints, and embedding caches are stored in
ignored artifact directories.

### 4.4 Experiment Governance

Each controlled phase follows the same evidence pattern:

```text
predeclared YAML configuration
    -> frozen-input hash verification
    -> deterministic/repeated execution
    -> machine-readable JSON and CSV
    -> human-readable report
    -> tests for leakage, scope, and artifact immutability
```

Frozen manifest files create a hash chain across phases. They do not make files
physically immutable, but tests fail when an anchored predecessor changes. Git
commit IDs are recorded when available; a clean working tree is still required
for strong archival provenance.

## 5. Current Environment

The following environment was verified live on 2026-10-12:

| Item | Current value |
| --- | --- |
| OS | Windows 11, build `10.0.26300` |
| Python | 3.13.14 |
| Virtual environment | `.venv` |
| CPU | Intel64 Family 6 Model 154, 10 physical / 16 logical cores |
| RAM | approximately 15.71 GiB |
| GPU | NVIDIA GeForce RTX 2050 |
| VRAM | 4,096 MiB |
| NVIDIA driver | 610.78 |
| PyTorch | 2.14.1+cu130 |
| CUDA runtime | 13.0 |
| CUDA available | Yes |
| PyTorch Geometric | 2.8.0.post1 |
| NumPy | 2.5.3 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| XGBoost | 3.4.1 |
| LightGBM | 4.7.0 in the Phase 6E environment record |
| joblib | 1.6.0 |
| PyYAML | 6.0.3 |

The project uses a CUDA overlay in `requirements-cuda.txt` so historical CPU
requirements are not silently rewritten. Most versions in `requirements.txt`
remain unpinned, which is a reproducibility limitation discussed later.

## 6. Dataset

### 6.1 Source and Integrity

The immutable source file is:

`data/raw/PonziDataset_20221114.csv`

| Property | Measured value |
| --- | ---: |
| File size | 307,176,676 bytes |
| SHA-256 | `48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf` |
| Rows/contracts | 6,498 |
| Ponzi | 314 |
| Benign | 6,184 |
| Benign:Ponzi ratio | 19.69:1 |
| Earliest block | 49,880 |
| Latest block | 7,136,486 |
| Timestamp range | 2015-08-07 to 2019-01-27 |
| Unique creators | 3,360 |

Columns are `address`, `opcode`, `code`, `contractCode`,
`createdBlockNumber`, `createdTransactionHash`, `creationCode`, `creator`,
`label`, and `timestamp`.

### 6.2 Audit Findings

- Headline row and Ponzi counts match the publication.
- No null values were measured in the released CSV.
- No missing or empty opcode values were measured.
- 30 duplicate-address rows exist.
- 197 duplicate transaction-hash rows exist.
- 30 duplicate block/address combinations exist.
- 712 rows belong to 195 exact duplicate-opcode groups.
- 1,093 creators have multiple contracts.
- 55 creators have multiple positive contracts.
- The file is already ordered chronologically by block and timestamp.

Duplicates were reported, not silently removed. The raw CSV was never changed.

## 7. Dataset Splitting and Temporal Protocols

### 7.1 Initial Phase 2 Split

The first protocol used chronological 70/15/15 row partitions without
shuffling:

| Split | Rows | Ponzi | Benign | Benign:Ponzi | Block range |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 4,548 | 140 | 4,408 | 31.49:1 | 49,880-4,572,153 |
| Validation | 974 | 8 | 966 | 120.75:1 | 4,572,291-4,955,839 |
| Test | 976 | 166 | 810 | 4.88:1 | 4,957,244-7,136,486 |

This was leakage-safe but poorly balanced across time: validation contained
only eight Ponzi contracts, making model selection unstable.

### 7.2 Phase 5.5 Paper-Aligned Split

The corrected protocol assigns whole blocks by chronological Ponzi occurrence:

| Split | Ponzi boundary | Rows | Ponzi | Benign | Benign:Ponzi | Block range |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Train | 1st-200th Ponzi | 5,839 | 200 | 5,639 | 28.195:1 | 49,880-5,021,809 |
| Validation | 201st-250th | 150 | 50 | 100 | 2.000:1 | 5,021,917-5,038,334 |
| Final test | 251st-314th | 509 | 64 | 445 | 6.953:1 | 5,038,337-7,136,486 |

The paper reports 5,990 rows through P4 and 508 in P5. The released file plus
the whole-block policy gives 5,989 and 509. This one-row boundary discrepancy is
documented rather than manually changed.

### 7.3 P0-P5 Periods

Phase 7 and all Phase 8 temporal work use these fixed periods:

| Period | Chronological Ponzi range | Rows | Ponzi | Benign | Block range |
| --- | --- | ---: | ---: | ---: | --- |
| P0 | 1-50 | 104 | 50 | 54 | 49,880-1,287,252 |
| P1 | 51-100 | 305 | 50 | 255 | 1,287,353-2,000,473 |
| P2 | 101-150 | 5,141 | 50 | 5,091 | 2,008,140-4,963,650 |
| P3 | 151-200 | 289 | 50 | 239 | 4,964,082-5,021,809 |
| P4 | 201-250 | 150 | 50 | 100 | 5,021,917-5,038,334 |
| P5 | 251-314 | 509 | 64 | 445 | 5,038,337-7,136,486 |

Relationship to Phase 5.5:

```text
Phase 5.5 train      = P0 + P1 + P2 + P3
Phase 5.5 validation = P4
Phase 5.5 final test = P5
```

Phase 7A instead uses expanding forward prediction:

```text
P0+P1       -> P2
P0+P1+P2    -> P3
P0-P3       -> P4
P0-P4       -> P5
```

This explains why Phase 5.5 P5 and Phase 7A P5 metrics differ: Phase 7A adds P4
to the P5 training history, while Phase 5.5 does not.

### 7.4 Phase 6 Temporal CV

Phase 6 uses three forward-only folds within the Phase 5.5 training partition:

| Fold | Earlier training rows/positives | Later CV rows/positives |
| --- | --- | --- |
| 1 | 104 / 50 | 305 / 50 |
| 2 | 409 / 100 | 5,141 / 50 |
| 3 | 5,550 / 150 | 289 / 50 |

Preprocessing is fitted inside each fold. The external P4 validation partition
is used once after temporal-CV selection, and P5 is excluded.

## 8. Preprocessing and Leakage Controls

### 8.1 Common Static Pipeline

- Sort and partition by `createdBlockNumber`; never shuffle before temporal
  assignment.
- Normalize opcode sequences into canonical opcode names.
- Fit CountVectorizer, TF-IDF vocabulary/IDF, N-gram vocabulary, and Word2Vec on
  historical training rows only.
- Transform validation/future rows without updating learned state.
- Fit scalers on historical training features only.
- Keep high-dimensional N-gram features sparse.
- Preserve row/index/address identity through every cascade stage.

### 8.2 Creator Feature

The feature is 1 only when the contract creator was associated with a Ponzi
contract strictly earlier than the current contract under the allowed history.
Same-block and future labels cannot affect it. For validation/test, creator
state is frozen from historical training data. Removing the creator feature
removes the column completely; it is not replaced with a constant.

### 8.3 Runtime Bytecode and Operands

`contractCode` is the authoritative deployed runtime source. Deterministic EVM
disassembly records malformed/truncated PUSH instructions and unsupported bytes
instead of repairing or dropping rows. Exact raw constants, addresses, and
selectors are not emitted as model features.

### 8.4 CFG Processing

- Basic blocks begin at entry, `JUMPDEST`, and after terminators/branches.
- Edge types are `FALL_THROUGH`, `STATIC_JUMP`, and `CONDITIONAL_JUMP`.
- A jump target is resolved only when a directly preceding PUSH safely points
  to a valid `JUMPDEST`.
- Unresolved dynamic jumps never receive invented all-to-all edges.
- Every graph is contract-local and label-independent.

### 8.5 Creation-Code Processing

`creationCode` is the deployment-bytecode source. The deployed runtime is used
only to find a conservative boundary. An init region is accepted only when the
runtime appears exactly once as a contiguous substring. Missing or ambiguous
matches do not receive an invented boundary. Trailing bytes are described but
not interpreted as constructor arguments.

### 8.6 Sequence Processing

- Use deployed-runtime canonical opcode tokens without raw operands.
- Fit a separate vocabulary on each fold's historical corpus only.
- Use deterministic 1,024-token chunking; do not truncate away later chunks.
- Pad final chunks deterministically and exclude PAD from pooling/loss.
- MLM masks 15% of eligible tokens using fixed 80% MASK, 10% historical random
  opcode, and 10% unchanged behavior.
- Future-period contracts never participate in vocabulary or pretraining.

## 9. Feature Representations

### 9.1 Frozen Phase 5.5 Views

Dimensions include the creator column:

| View | Dimension | Description |
| --- | ---: | --- |
| Opcode term count | 143 | 142 opcode counts + creator |
| Opcode TF-IDF | 143 | training-IDF weighted opcode counts + creator |
| Filtered 1-3-gram count | 44,672 | 44,671 DF>=2 N-grams + creator |
| Filtered 1-3-gram TF-IDF | 44,672 | N-gram TF-IDF + creator |
| Mean Word2Vec | 101 | 100 embedding values + creator |
| IDF-weighted Word2Vec | 101 | 100 IDF-weighted values + creator |

The initial unfiltered N-gram implementation had 70,867 dimensions and remains
preserved as a historical alternative.

### 9.2 Later Diagnostic/Experimental Features

| Feature family | Phase | Dimension/status |
| --- | --- | --- |
| Structural opcode features | 6D1 | 74 compact static features |
| Normalized operand features | 6D2 | 37 features |
| Runtime CFG node features | 8R2A | 55 features per basic block |
| Creation full | 8R4A | 85 contract features |
| Deployment init | 8R4A | 90 contract features |
| Combined creation/init/boundary | 8R4A | 183 features |
| Contextual sequence embedding | 8R5C1 | 128 per contract |
| Non-contextual token embedding control | 8R5C1 | 128 per contract |

Structural and operand features were reused for analysis in Phase 7D but were
not added to the frozen MulCas model.

### 9.3 Important Paper-to-Reconstruction Feature Differences

| Topic | Source paper | Current reconstruction | Consequence |
| --- | --- | --- | --- |
| Selected 1-3-gram dimension | 2,292 | 44,671 before creator, using `min_df=2` | The main sparse representation is much larger and not feature-identical |
| Initial unfiltered 1-3-grams | 52,532 reported before selection | 70,867 in the initial implementation | Tokenization/corpus details differ |
| Word2Vec | 100-d IDF-weighted sum; limited training details | 100-d, window 5, min count 1, 10 epochs, seed 42, one worker | Reproducible assumption, not exact replication |
| Operand policy | Discard operands | Baseline opcode views discard them; Phase 6D2 separately studies normalized operands | Baseline principle preserved; operand phase is an extension |
| Auxiliary bytecode | Paper says non-executable auxiliary fragment is discarded | Released `opcode` is used for the baseline; later runtime disassembly preserves full released `contractCode` bytes | CFG/operand extensions may include metadata-like tails excluded by the paper |
| Creator state | Previously tainted creator | Strictly earlier block labels; later partitions use frozen historical state | Stronger explicit anti-leakage interpretation |
| Viewer collection | Text says five final models but explicitly enumerates four | Four viewers: three XGBoost plus one Ridge | Publication wording is internally incomplete |

The N-gram difference is one of the largest remaining reproduction gaps. The
paper describes a document-frequency rule but does not provide enough detail to
reliably recreate the reported 2,292-column matrix from the released CSV. The
project therefore freezes `min_df=2` as an explicit assumption instead of
tuning a cutoff against P5.

## 10. Models, Algorithms, and Thresholds

### 10.1 Frozen Baseline Classifiers

- **RidgeClassifier:** `alpha=1.0`, `class_weight=None`, native decision
  threshold 0.
- **Linear SVM:** `C=1.0`, linear kernel, `class_weight=None`, native decision
  threshold 0.
- **XGBoost:** 100 trees, learning rate 0.1, max depth 3,
  `scale_pos_weight=1`, native probability threshold 0.5.

### 10.2 Other Controlled Models

- Logistic Regression with L2, `C=1.0`, no class weighting, threshold 0.5.
- LightGBM with small predefined fixed grids in Phase 6E.
- GIN with 2/3 layers, 64/128 hidden units, dropout 0.2/0.5, fixed threshold
  0.5, and training-only positive loss weight.
- Logistic score-fusion meta-classifier with fixed L2/C=1.0/threshold 0.5.
- Small opcode Transformer: embedding 128, 2 encoder layers, 4 heads, FFN 256,
  dropout 0.1, 1,024-token chunks, MLM head.
- Frozen sequence linear probe: StandardScaler + L2 Logistic Regression,
  `C=1.0`, no class weighting, threshold 0.5.

### 10.3 Research Selection Thresholds

Most controlled improvement phases require **macro or mean temporal precision
>= 0.80** before a candidate can be called eligible. F1, recall, balanced
accuracy, and simplicity are then used in the predeclared order. Diagnostic
fallbacks are reported when nothing is eligible, but they are not research
winners.

Classifier operating thresholds, MulCas cascade confidence thresholds, voting
lambda, and research eligibility thresholds are distinct concepts. They are not
interchanged.

### 10.4 Metric Meanings

- **Precision = TP / (TP + FP):** among contracts predicted Ponzi, the fraction
  that are truly Ponzi.
- **Recall = TP / (TP + FN):** among true Ponzi contracts, the fraction found.
- **F1:** harmonic balance of precision and recall.
- **PR-AUC:** ranking quality across precision-recall operating points; useful
  for rare positives.
- **Balanced accuracy:** average of positive recall and negative-class recall.
- **ROC-AUC:** ranking across true/false-positive rates; retained as a secondary
  metric because severe imbalance can make it look optimistic.
- **Accuracy:** total fraction correct; never treated as sufficient by itself.

For this project, recall and F1 are central because missed Ponzi contracts are
the dominant baseline error, while precision remains a guard against excessive
false warnings.

## 11. Phase-by-Phase Progress and Results

### 11.1 Phase 0 - Repository and Environment

**Work:** created `.venv` workflow, Python packages, path and deterministic seed
helpers, import tests, `.gitignore`, and PowerShell instructions.  
**Result:** local reproducible development foundation.  
**ML performed:** none.

### 11.2 Phase 1 - Dataset Integrity Audit

**Work:** reusable loading/validation, metadata-first audit, hashes, duplicate
and chronology analysis, creator reuse, opcode audit, and notebook.  
**Result:** 6,498 rows and 314 Ponzi match the paper; raw SHA-256 frozen.  
**Conclusion:** released data is usable, but contains documented duplicate
identities/code and extreme imbalance.

### 11.3 Phase 2 - Initial Temporal Split

**Work:** deterministic 70/15/15 chronological split and cross-split duplicate
checks.  
**Result:** validation had only 8 positives while test had 166.  
**Conclusion:** leakage-safe, but not suitable as the final paper-aligned
development protocol.

### 11.4 Phase 3 - Feature Reproduction

**Work:** opcode counts, TF-IDF, unfiltered 1-3-grams, N-gram TF-IDF, mean
Word2Vec, and chronological creator feature.  
**Result:** dimensions 142, 142, 70,867, 70,867, 100, and 1 before
concatenation.  
**Conclusion:** published feature families were independently implemented with
training-only fitting.

### 11.5 Phase 4 - Individual Baselines on Initial Split

| Model | Validation-selected view | Validation F1 | Test P | Test R | Test F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Ridge | N-gram TF-IDF | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SVM | N-gram count | 0.3333 | 0.5897 | 0.1386 | 0.2244 |
| XGBoost | Term count | 0.5455 | 0.0000 | 0.0000 | 0.0000 |

**Conclusion:** the percentage split produced unstable selection and weak test
recall.

### 11.6 Phase 5 - Initial MulCas Reconstruction

| Split | Precision | Recall | F1 | PR-AUC | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.5714 | 0.5000 | 0.5333 | 0.4160 | 963 | 3 | 4 | 4 |
| Test | 0.6000 | 0.1084 | 0.1837 | 0.5222 | 798 | 12 | 148 | 18 |

**Conclusion:** architecture and traces were reproduced, but performance was far
below the paper.

### 11.7 Phase 5.5 - Paper-Aligned Correction and Frozen Baseline

**Corrections:** 200/50/64 positive-occurrence split, DF-filtered N-grams,
IDF-weighted Word2Vec, and creator concatenation to every static view. Historical
70/15/15 artifacts were preserved.

Selected individual final-test results:

| Model | Selected view | Precision | Recall | F1 | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Ridge | Term count | 0.9310 | 0.4219 | 0.5806 | 0.6365 |
| SVM | N-gram TF-IDF | 0.8824 | 0.2344 | 0.3704 | 0.6717 |
| XGBoost | N-gram TF-IDF | 0.8333 | 0.2344 | 0.3659 | 0.5943 |

Frozen MulCas:

| Split | Accuracy | Balanced acc. | Precision | Recall | F1 | PR-AUC | TN/FP/FN/TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Validation/P4 | 0.8933 | 0.9000 | 0.7931 | 0.9200 | 0.8519 | 0.8706 | 88/12/4/46 |
| Final test/P5 | 0.9253 | 0.7232 | 0.9063 | 0.4531 | 0.6042 | 0.6624 | 442/3/35/29 |

**Conclusion:** much closer to the paper than Phase 5, particularly in
precision, but recall and F1 remain below the published result. Phase 5.5 is the
retained reference system.

The label **paper-aligned** means closer alignment, not an exact protocol copy.
The publication's Table 5 model is trained on the first 250 Ponzi contracts and
corresponding benign rows, then tested on P5. Phase 5.5 instead fits/selects the
frozen model on the first 200 Ponzi contracts, reserves the next 50 as P4
validation, and reports P5 without refitting on train+validation. This provides
a defensible validation boundary but gives the final model less labeled history
than the published Table 5 model. Phase 7A's P5 run does train on P0-P4 and is
closer in training-period coverage, but it is a separate incremental analysis
and produces P=0.8710, R=0.4219, F1=0.5684.

The publication also states that candidate viewers were selected by their
performance on the test set. This project deliberately uses validation or
forward temporal CV for selection instead. That methodological safeguard is a
further reason exact numerical replication is neither expected nor claimed.

### 11.8 Phase 6 - Controlled Improvement Experiments

All candidates used forward-only CV and excluded P5. The frozen P4 validation
reference was P=0.7931, R=0.9200, F1=0.8519, PR-AUC=0.8706.

| Phase | Intervention | Key selected validation result | Conclusion |
| --- | --- | --- | --- |
| 6A | Cost-sensitive Ridge/SVM/XGBoost | All selected variants P=0.7931, R=0.9200, F1=0.8519; lower PR-AUC | H6A not supported |
| 6B | Random under/over-sampling | Ridge/SVM matched F1 0.8519; XGBoost F1 0.8142 | H6B not supported |
| 6C | Native threshold optimization | Ridge fallback F1 0.6622; eligible SVM/XGB predicted no positives | H6C not supported |
| 6D1 | 74 structural opcode features | Ridge/SVM matched F1 0.8519; no eligible CV representation | H6D1 not supported |
| 6D2 | 37 normalized operand features | Ridge F1 0.1429; SVM/XGB matched F1 0.8519 | H6D2 not supported |
| 6E | Logistic Regression and LightGBM | LR F1 0.8519; LightGBM F1 0.8491 | H6E not supported |
| 6F | Combine independently supported changes | Not run because none existed | Not executed by design |

**Phase 6 conclusion:** none of the tested interventions demonstrated a
reliable improvement under the predefined temporal protocol. This does not mean
improvement is impossible.

### 11.9 Phase 7A - Temporal Robustness

| Predict | Training | Precision | Recall | F1 | PR-AUC | Balanced acc. | TN/FP/FN/TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| P2 | P0+P1 | 0.0492 | 0.2600 | 0.0828 | 0.0462 | 0.6053 | 4840/251/37/13 |
| P3 | P0-P2 | 0.5385 | 0.1400 | 0.2222 | 0.4786 | 0.5574 | 233/6/43/7 |
| P4 | P0-P3 | 0.7931 | 0.9200 | 0.8519 | 0.8706 | 0.9000 | 88/12/4/46 |
| P5 | P0-P4 | 0.8710 | 0.4219 | 0.5684 | 0.6674 | 0.7064 | 441/4/37/27 |

Published vs reconstructed temporal F1: P2 0.5300 vs 0.0828, P3 0.4800 vs
0.2222, P4 0.8700 vs 0.8519, and P5 0.7900 vs 0.5684.

**Conclusion:** substantial non-monotonic temporal variation. This is consistent
with distribution variation but does not prove concept drift.

### 11.10 Phase 7B - Viewer Ablation

F1 delta after removing each component:

| Removed component | P2 | P3 | P4 | P5 |
| --- | ---: | ---: | ---: | ---: |
| First cascade viewer | -0.0073 | -0.0437 | 0.0000 | -0.1981 |
| Second cascade viewer | +0.0156 | -0.0768 | 0.0000 | -0.0740 |
| Voting stage | -0.0005 | 0.0000 | 0.0000 | -0.0152 |

**Conclusion:** viewer 1 is particularly important in P5, viewer 2 has mixed
effects, and voting contributes only slightly.

### 11.11 Phase 7C - Creator Feature Ablation

Delta is without creator minus with creator:

| Period | Creator flag prevalence | Delta precision | Delta recall | Delta F1 | Delta FP/FN |
| --- | ---: | ---: | ---: | ---: | --- |
| P2 | 1/5,141 | +0.0195 | -0.0400 | +0.0220 | -102/+2 |
| P3 | 2/289 | -0.0147 | +0.0800 | +0.0876 | +4/-4 |
| P4 | 6/150 | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| P5 | 11/509 | +0.0078 | +0.0312 | +0.0295 | 0/-2 |

**Conclusion:** creator history is sparse and not uniformly helpful. It is a
historical predictive signal, not evidence that creator identity causes Ponzi
behavior.

### 11.12 Phase 7D - Error Analysis

| Period | FP | FN | Errors from unseen creators | Dominant route |
| --- | ---: | ---: | ---: | --- |
| P2 | 251 | 37 | 98.3% | second-viewer false positives |
| P3 | 6 | 43 | 98.0% | first-viewer false negatives |
| P4 | 12 | 4 | 87.5% | first-viewer errors |
| P5 | 4 | 37 | 92.7% | first-viewer errors |

For the 37 P5 false negatives: 36 have unseen creators, all 37 have no exact
historical opcode/runtime clone, 28 are confidently rejected by viewer 1, one
by viewer 2, eight fail after voting, and only three are near the configured
confidence boundary.

### 11.13 Phase 7E - Clone/Novelty Sensitivity

| Period | Any clone prevalence | Cloned Ponzi | Recall cloned | Recall novel |
| --- | ---: | ---: | ---: | ---: |
| P2 | 29/5,141 | 0 | N/A | 0.2600 |
| P3 | 1/289 | 0 | N/A | 0.1400 |
| P4 | 15/150 | 14 | 1.0000 | 0.8889 |
| P5 | 8/509 | 1 | 1.0000 | 0.4127 |

**Phase 7 conclusion:** temporal performance varies substantially. Later errors
are concentrated among unseen creators and historically novel code, but exact
clone prevalence does not fully explain the variation and novelty is not shown
to cause errors.

### 11.14 Phase 8R0 - Extension Protocol and Feasibility

**Work:** defined RQ6, exploratory Track A and external-validation Track B,
recency policies, static CFG/GIN design, novelty-routing plan, and external data
audit.  
**Result:** runtime CFG construction was feasible for all 6,498 contracts;
public external candidates were overlapping or incomplete.  
**Conclusion:** staged extension feasible; independent external validation
remains a blocker.

### 11.15 Phase 8R1A - Recency-Aware MulCas

| Strategy | Macro P | Macro R | Macro F1 | Macro PR-AUC | Eligible |
| --- | ---: | ---: | ---: | ---: | --- |
| Expanding history control | 0.4603 | 0.4400 | 0.3856 | 0.4651 | No |
| Recent 1 period | 0.4120 | 0.4400 | 0.3810 | 0.4305 | No |
| Recent 2 periods | 0.4474 | 0.4400 | 0.3845 | 0.4141 | No |
| Recent 3 periods | 0.4558 | 0.4400 | 0.3830 | 0.4622 | No |
| Decay alpha 0.25 | 0.4136 | 0.4400 | 0.3716 | 0.4493 | No |
| Decay alpha 0.50 | 0.4535 | 0.4533 | 0.3935 | 0.4511 | No |
| Decay alpha 0.75 | 0.4582 | 0.4533 | 0.4024 | 0.5044 | No |

The control reproduced Phase 7A P2-P4 exactly. Alpha 0.75 was the diagnostic
fallback but failed the 0.80 precision floor. **H8R1 not supported.**

### 11.16 Phase 8R2A - Frozen Runtime CFG Corpus

| Item | Result |
| --- | ---: |
| Graphs | 6,498 |
| Nodes | 1,514,545 |
| Edges | 1,464,910 |
| Node feature dimension | 55 |
| Static jump resolution | 717,928 / 963,899 = 0.7448 |
| Zero-node / zero-edge graphs | 0 / 1 |
| Compressed artifact size | 68,757,791 bytes |
| Corpus SHA-256 | `ece393232abad38fb329f4ccbe12e617d9c3ec90ba5f265d867ccf5670fec680` |

**Conclusion:** deterministic graph construction succeeded. No classifier was
trained in this phase.

### 11.17 Phase 8R2B - Graph Aggregate and GIN Backtesting

Graph-aggregate Logistic Regression F1 was 0.0537 on P2, 0 on P3, and 0.8411
on P4.

| GIN architecture | Macro P | Macro R | Macro F1 | Eligible |
| --- | ---: | ---: | ---: | --- |
| 2 layers, 128, dropout .2 | 0.5025 | 0.4827 | 0.4712 | No |
| 2 layers, 128, dropout .5 | 0.4143 | 0.4373 | 0.3788 | No |
| 2 layers, 64, dropout .2 | 0.4190 | 0.3987 | 0.3849 | No |
| 2 layers, 64, dropout .5 | 0.4155 | 0.4173 | 0.3980 | No |
| 3 layers, 128, dropout .2 | 0.4617 | 0.4827 | 0.4422 | No |
| 3 layers, 128, dropout .5 | 0.4652 | 0.5213 | 0.4721 | No |
| 3 layers, 64, dropout .2 | 0.4516 | 0.4813 | 0.4398 | No |
| 3 layers, 64, dropout .5 | 0.3879 | 0.3800 | 0.3517 | No |

The diagnostic fallback (3 layers, 128, dropout .5) produced mean F1 0.0074,
0.6680, and 0.7409 on P2/P3/P4. It strongly improved P3 over MulCas but failed
P2 and did not satisfy macro precision. **H8R2 not supported.**

### 11.18 Phase 8R3A - MulCas/GIN Complementarity

No model was trained. Across P2-P4, GIN recovered 28 Ponzi contracts missed by
MulCas, while MulCas recovered 22 that GIN missed. GIN correctly rejected 256
benign contracts that MulCas misclassified.

| Diagnostic | P2 | P3 | P4 | P2-P4 |
| --- | ---: | ---: | ---: | ---: |
| MulCas recall | 0.2600 | 0.1400 | 0.9200 | 0.4400 |
| Oracle union recall | 0.2600 | 0.6600 | 0.9600 | 0.6267 |

The non-deployable perfect-selector oracle gave P=0.8785, R=0.6267, and
F1=0.7315 overall. All 28 GIN-only Ponzi recoveries had novel code, unseen
creators, and confident MulCas viewer-1 rejection. **Conclusion:** a small
hybrid experiment was justified diagnostically, not proven effective.

### 11.19 Phase 8R3B - Lightweight Score Fusion

P3 selection:

| System | Precision | Recall | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| MulCas | 0.5385 | 0.1400 | 0.2222 | 0.4786 |
| GIN ensemble | 0.8235 | 0.5600 | 0.6667 | 0.7074 |
| Hybrid A scores only | 0 | 0 | 0 | 0.5760 |
| Hybrid B scores + context | 0 | 0 | 0 | 0.6097 |

Hybrid A was an ineligible diagnostic fallback. On P4 it produced P=0.6567,
R=0.8800, F1=0.7521, below MulCas F1=0.8519. **H8R3B not supported.**

### 11.20 Phase 8R4A - Creation-Code Features

All 6,498 creation-code values were parseable. Runtime was found exactly once
inside creationCode for 6,497 contracts; the remaining row received the frozen
zero-filled init representation. Nonempty trailing data appeared in 2,604
contracts. Frozen views were 85, 90, and 183 dimensions. Matrix SHA-256:
`ecc80b2ab83c5954eb8790e7f693619441116506d64f8c6e3aa8aec87bcdf897`.

**Conclusion:** representation construction was feasible; no model was trained.

### 11.21 Phase 8R4B - Creation-Code Classification

| Candidate | Macro P | Macro R | Macro F1 | Macro PR-AUC | Eligible |
| --- | ---: | ---: | ---: | ---: | --- |
| LR creation full | 0.2703 | 0.3067 | 0.2818 | 0.5165 | No |
| LR deployment init | 0.3379 | 0.0867 | 0.0217 | 0.3224 | No |
| LR combined | 0.2664 | 0.4933 | 0.2750 | 0.4588 | No |
| XGB creation full | 0.5243 | 0.2067 | 0.2609 | 0.5292 | No |
| XGB deployment init | 0.3319 | 0.1667 | 0.2214 | 0.3760 | No |
| XGB combined | 0.4655 | 0.1867 | 0.2441 | 0.4815 | No |

The diagnostic fallback, LR creation-full, had F1 0.0299/0/0.8155 on
P2/P3/P4. It recovered only one Ponzi missed by MulCas over all three periods.
**H8R4 not supported.**

### 11.22 Phase 8R5A - Opcode-Sequence Feasibility

Across all contracts, sequence length ranged from 8 to 18,058 tokens, median
2,206, mean 2,664.7, P95 6,756.8. On P0-P4, simple 1,024-token truncation would
truncate 80.92% of contracts and retain only 35.74% of tokens. Therefore
deterministic 1,024-token chunking was frozen.

Historical vocabulary/OOV:

| Predict | Vocabulary | OOV tokens | OOV rate |
| --- | ---: | ---: | ---: |
| P2 | 141 ordinary opcodes | 279 | 0.000020 |
| P3 | 142 | 0 | 0 |
| P4 | 142 | 0 | 0 |

The CPU estimate was 14.4 hours for one seed and 43.1 hours for three, so full
training was not launched.

### 11.23 Phase 8R5B0 - CUDA Validation

CUDA-enabled PyTorch was validated on the RTX 2050. AMP batch 8 achieved about
261.66 examples/s and 267,940 tokens/s with 132 MiB peak reserved VRAM in the
bounded benchmark. Estimated all-fold 10-epoch time was 22.8 minutes for one
seed and 68.5 minutes for three. Feasibility was **GREEN**.

### 11.24 Phase 8R5B - Masked-Opcode Pretraining

Nine independent fold/seed runs completed using AdamW, learning rate 1e-4,
weight decay 1e-2, 10 epochs, CUDA AMP, and batch 8. No labels or P5 data were
loaded.

| Fold | Historical contracts | Chunks | Vocabulary | Final loss mean +/- SD | Accuracy mean +/- SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| P2 | 409 | 957 | 144 incl. specials | 3.0736 +/- 0.0044 | 0.2472 +/- 0.0008 |
| P3 | 5,550 | 16,999 | 145 | 1.7710 +/- 0.1102 | 0.5159 +/- 0.0311 |
| P4 | 5,839 | 17,908 | 145 | 1.6888 +/- 0.0886 | 0.5369 +/- 0.0267 |

Total summed run time was 72.32 minutes versus the 68.53-minute estimate.
All nine final checkpoints were hash-recorded. This phase established
representation-learning artifacts only; it did not test H8R5.

### 11.25 Phase 8R5C1 - Frozen Sequence Embeddings and Linear Probe

The encoder was frozen. Contextual and non-contextual 128-dimensional contract
embeddings were extracted for all three seeds, then evaluated with a fixed
linear probe. P5 was excluded.

Contextual seed-mean results:

| Period | Precision mean | Recall mean | F1 mean | PR-AUC mean |
| --- | ---: | ---: | ---: | ---: |
| P2 | 0.0510 | 0.1067 | 0.0615 | 0.0256 |
| P3 | 0.0000 | 0.0000 | 0.0000 | 0.5405 |
| P4 | 0.8015 | 0.8333 | 0.8149 | 0.8413 |

Three-seed ensemble:

| Representation | Period | Precision | Recall | F1 | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Contextual | P2 | 0.0588 | 0.0800 | 0.0678 | 0.0341 |
| Contextual | P3 | 0.0000 | 0.0000 | 0.0000 | 0.6137 |
| Contextual | P4 | 0.8036 | 0.9000 | 0.8491 | 0.8393 |
| Non-contextual | P2 | 0.0343 | 0.3400 | 0.0623 | 0.0247 |
| Non-contextual | P3 | 0.6667 | 0.1200 | 0.2034 | 0.7653 |
| Non-contextual | P4 | 0.7931 | 0.9200 | 0.8519 | 0.8071 |

Macro evidence:

| System | Macro precision | Macro recall | Macro F1 | Macro PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Contextual seed mean | 0.2841 | 0.3133 | 0.2921 | 0.4692 |
| Non-contextual control | 0.5297 | 0.4600 | 0.3870 | 0.5261 |
| Frozen MulCas | 0.4603 | 0.4400 | 0.3856 | 0.4651 |

Although the non-contextual control has a tiny macro-F1 difference of +0.0014
over MulCas, it is a control, fails the 0.80 precision requirement, and does not
establish a reliable improvement. The contextual representation underperforms
the control and MulCas. **H8R5 is not supported by Phase 8R5C1, and supervised
fine-tuning in Phase 8R5C2 is not justified under the predefined gate.**

## 12. Consolidated Current Results

### 12.1 Original Paper vs Reconstructions

| System/protocol | Precision | Recall | F1 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Published MulCas Table 5 | 0.9510 | 0.6740 | 0.7890 | Publication reference |
| Initial 70/15/15 MulCas | 0.6000 | 0.1084 | 0.1837 | Initial independent reconstruction |
| Frozen Phase 5.5 MulCas | 0.9063 | 0.4531 | 0.6042 | Current paper-aligned reference |
| Phase 7A incremental P5 | 0.8710 | 0.4219 | 0.5684 | Different training history; analysis result |

The paper also states F1=0.743 in nearby prose while Table 5 reports 0.789.
Both are retained as an unresolved source inconsistency; Table 5 is used for
the primary comparison.

These rows are useful benchmarks but not perfectly controlled head-to-head
experiments. The initial reconstruction uses the 70/15/15 split, Phase 5.5 uses
200 positives for fitting plus a held-out 50-positive validation set, the paper
uses a 250-positive two-way training set, and Phase 7A incremental P5 trains on
P0-P4. The report therefore emphasizes direction and gap, not a claim that every
metric difference is caused only by model implementation.

### 12.2 What Came Closest After the Baseline

- P4 frozen MulCas is close to the published temporal P4 result: F1 0.8519 vs
  0.8700.
- Phase 8R2B GIN substantially improved P3 F1 to 0.6680, but failed P2 and the
  macro precision criterion.
- Phase 8R3A showed real complementarity, but the trainable Phase 8R3B hybrid
  failed selection and confirmation.
- Phase 8R5C1 non-contextual control produced macro F1 0.3870 vs MulCas 0.3856,
  but precision was only 0.5297 and the representation was a scientific control,
  not an eligible winner.
- No completed intervention has beaten the frozen baseline reliably across
  P2-P4, and no extension candidate was evaluated on P5.

### 12.3 Current Research Status

The retained system remains Phase 5.5 MulCas. All Phase 6 hypotheses, H8R1,
H8R2, H8R3B, H8R4, and the Phase 8R5C1 evaluation of H8R5 are not supported
under their predefined criteria. Negative results are preserved rather than
hidden or post-hoc combined.

## 13. Tests and Validation

### 13.1 Current Automated Test Result

Executed on 2026-10-12:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result after Phase 8 closure: **256 passed, 65 warnings in 74.01 seconds**.

Warnings were non-failing deprecation/future warnings from PyTorch,
PyTorch Geometric, and scikit-learn, plus a Windows permission warning when
pytest attempted to write `.pytest_cache`.

### 13.2 What the Tests Cover

- repository paths, imports, and random-seed determinism;
- data schema, chronology, labels, missing values, and duplicates;
- chronological split disjointness and leakage assertions;
- training-only vectorizer, IDF, Word2Vec, scaler, and creator state;
- baseline metrics and classifier score behavior;
- MulCas routing, order preservation, and voting;
- frozen artifact and raw dataset hashes;
- sampling/class-weight/threshold isolation;
- structural and operand edge cases;
- P0-P5 period definitions and historical-only clone matching;
- graph disassembly, blocks, edges, features, normalization, and GIN forward
  behavior;
- creation-code boundary and feature determinism;
- sequence vocabulary, chunking, masking, pooling, CUDA gates, checkpoint
  hashes, embedding caches, and fixed linear-probe behavior;
- explicit exclusion of P5 from extension selection/evaluation.

### 13.3 Validation Boundaries

Passing tests proves the implemented contracts and archived outputs are
internally consistent. It does not prove present-day Ethereum generalization,
paper-identical software behavior, causal concept drift, or production safety.

## 14. How to Run the Project

### 14.1 CPU/Base Environment

From Windows PowerShell in the repository root:

```powershell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA sequence work, apply the pinned overlay after base requirements:

```powershell
python -m pip install -r requirements-cuda.txt
```

Verify Python, dataset, and CUDA:

```powershell
python --version
Get-FileHash -Algorithm SHA256 data\raw\PonziDataset_20221114.csv
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 14.2 Tests

```powershell
python -m pytest -q
```

### 14.3 Read Existing Results

Retraining is not required to inspect the project. Start with:

- `docs/paper_aligned_baseline_report.md`
- `docs/phase6_summary.md`
- `docs/phase7_summary.md`
- `results/extension/phase8r5c1/phase8r5c1_report.md`
- this report

### 14.4 Notebooks

```powershell
python -m jupyter notebook
```

Implemented notebooks cover Phases 1-7. `00_environment_check.ipynb` and
`08_final_evaluation.ipynb` are currently empty. Core research logic belongs in
`src/`; notebooks should remain thin presentation layers.

### 14.5 Main Executable Entry Points

The baseline can be regenerated in dependency order with:

```powershell
python -m src.data.audit
python -m src.data.paper_aligned_split --config configs/paper_aligned_split.yaml
python -m src.features.paper_aligned --config configs/paper_aligned_features.yaml
python -m src.models.run_baselines --config configs/paper_aligned_baseline_models.yaml
python -m src.models.run_mulcas --config configs/paper_aligned_mulcas.yaml
```

Controlled phases expose module entry points such as:

```powershell
python -m src.experiments.phase6a
python -m src.experiments.phase7a
python -m src.extension.phase8r2a
python -m src.extension.phase8r5c1
```

Do not rerun all phases merely to read existing results. In particular,
Phase 8R2B graph training and Phase 8R5B nine-run GPU pretraining are expensive
and already have frozen outputs. Read their reports/manifests first and rerun
only as an intentional reproducibility exercise.

### 14.6 Important Do/Do-Not Rules

Do:

- preserve raw data and frozen manifests;
- use repository-relative path helpers;
- fit learned preprocessing on historical training only;
- keep P5 outside new model selection;
- label diagnostic fallbacks honestly;
- record hashes, seeds, versions, and exact split memberships.

Do not:

- edit the raw CSV;
- overwrite prior phase results;
- tune using P5 or external labels;
- describe a diagnostic fallback as a supported improvement;
- treat P5 as untouched after Phases 5.5 and 7;
- commit large generated graphs, matrices, checkpoints, or secrets;
- claim concept drift or causality from descriptive subgroup differences.

## 15. Current Limitations

1. **Paper under-specification:** exact N-gram cutoff, several model details,
   software versions, and exact row membership are not fully reported.
2. **Paper inconsistency:** Table 5 F1=0.789 conflicts with nearby prose
   F1=0.743; the methodology also says five final viewers while explicitly
   naming only four.
3. **N-gram representation mismatch:** the paper reports 2,292 selected
   unigram/bigram/trigram columns, while the reconstruction has 44,671 before
   creator concatenation.
4. **Different Table 5 fitting protocol:** the paper trains through the 250th
   Ponzi; frozen Phase 5.5 fits through the 200th and reserves the next 50 for
   validation instead of refitting before P5.
5. **Viewer-selection difference:** the paper describes choosing viewers by
   test performance; this project uses validation/temporal CV to avoid that
   leakage-prone practice.
6. **P5 is no longer untouched:** it was inspected in Phase 5.5 and analyzed in
   Phase 7, so future P5 experiments would be exploratory, not unbiased final
   evidence.
7. **Temporal instability:** P2/P3/P5 results are substantially weaker than P4.
8. **Severe imbalance:** period class ratios vary dramatically; accuracy can be
   misleading.
9. **Source-available sampling bias:** labels were created from contracts with
   verified source code, so the benchmark may not represent all deployed
   bytecode-only contracts.
10. **Old time coverage:** the dataset ends in January 2019 and cannot establish
   present-day deployment performance.
11. **No independent Track B benchmark:** known public candidates overlap this
   dataset or lack a complete labeled benign cohort.
12. **Exact-clone subgroups are small:** P2/P3 contain no cloned Ponzi contracts,
   limiting clone conclusions.
13. **Creator history is sparse:** its observed contribution is unstable and is
   not a causal signal.
14. **Bytecode policy difference:** the paper discards auxiliary bytecode;
   later extension parsers preserve the complete released runtime value,
   including metadata-like tails.
15. **CFG jump recovery is conservative:** unresolved dynamic targets are
    explicit; full symbolic execution is intentionally absent.
16. **Static scope:** transaction behavior and live economic context are not
    part of the frozen model.
17. **Sequence result is probe-specific:** Phase 8R5C1 rejects the frozen linear
    probe, not every possible sequence-learning method.
18. **Adversarial robustness is unevaluated:** the source paper discusses dead
   code/opcode injection, but this project has not tested evasion attacks.
19. **Dependency reproducibility:** base requirements are mostly unpinned; a
    final lock/snapshot is still advisable.
20. **CUDA reproducibility:** deterministic controls are enabled, but
    bit-identical retraining across GPU/software environments is not claimed.
21. **No production validation:** the detector is a research artifact, not a
    deployment-ready security decision system.

## 16. Pending Work

### 16.1 Required Research Packaging

1. Populate `notebooks/08_final_evaluation.ipynb` as a read-only synthesis of
   frozen results. It should not train or select a model.
2. Build final thesis/manuscript tables directly from machine-readable
   artifacts.
3. Run a clean-environment reproducibility audit and archive a pinned dependency
   snapshot.

### 16.2 External Validation

A genuinely new claim of temporal generalization requires a later, independently
labeled Ethereum dataset. Before evaluation it must be audited for address,
opcode, runtime-bytecode, and creation-code overlap with all 6,498 current
contracts. Labels must not influence design or tuning.

### 16.3 Phase 8R5C2 Decision

Supervised Transformer fine-tuning is **not currently justified** by the
predefined Phase 8R5C1 gate. It should not be treated as routine pending work.
Any future fine-tuning study requires a new preregistered rationale and must
remain P5-free during design and selection.

### 16.4 Optional Future Work

- acquire an overlap-free external binary benchmark;
- investigate newer deployment periods under a fresh protocol;
- consider compiler/version-aware or richer semantic representations;
- study calibrated abstention only as a newly preregistered experiment;
- build a clearly caveated demonstration interface after research closure.

## 17. Artifact Index

Key source-of-truth artifacts:

- Dataset: `results/baseline/dataset_audit.json`
- Paper-aligned split: `results/baseline/paper_aligned/split_metadata.json`
- Feature summary: `results/baseline/paper_aligned/feature_summary.json`
- Individual baseline results:
  `results/baseline/paper_aligned/individual_baseline_results.json`
- Frozen MulCas: `results/baseline/paper_aligned/mulcas_results.json`
- Phase 6 closure: `results/experiments/phase6_summary.json`
- Phase 7 closure: `results/experiments/phase7_summary.json`
- Phase 8 closure: `results/extension/phase8_closure/phase8_closure_results.json`
- Extension outputs: `results/extension/phase8r0/` through
  `results/extension/phase8r5c1/`
- Sequence checkpoints: manifest at
  `results/extension/phase8r5b/checkpoint_manifest.json`; large files are in
  ignored artifact storage.
- Latest extension result:
  `results/extension/phase8r5c1/phase8r5c1_results.json`

### 17.1 Generated Data and Trained-Artifact Locations

These directories are intentionally ignored by Git and may be absent in a
fresh clone until the relevant phase is run:

| Location | Current contents | Current local size |
| --- | --- | ---: |
| `data/processed/phase8r2a_graphs/schema_v1/` | 102 deterministic CFG NPZ chunks/manifests | 68,757,791 bytes |
| `data/processed/phase8r4a_creation/schema_v1/` | Frozen creation-code feature matrix | 2,750,110 bytes |
| `results/artifacts/paper_aligned/features/` | Fitted vectorizers, Word2Vec, creator artifacts | included below |
| `results/artifacts/paper_aligned/baseline_models/` | Ridge/SVM/XGBoost feature-view models | included below |
| `results/artifacts/paper_aligned/mulcas/` | Two cascade viewers and four full-train voting viewers | included below |
| `results/artifacts/paper_aligned/` total | 31 paper-aligned fitted artifacts | 4,298,367 bytes |
| `artifacts/phase8r5b/schema_v1/` | Nine final temporal MLM checkpoints | 15,834,579 bytes |
| `artifacts/phase8r5c1/schema_v1/` | 18 contextual/non-contextual embedding caches | 46,224,050 bytes |

This explains the role of `data/processed`: baseline text features are generally
generated and stored as fitted artifacts rather than copied full CSVs, while
large reusable CFG and creation-code matrices now populate `data/processed`.
Tracked `results/` files contain compact metrics, summaries, manifests, hashes,
and prediction traces rather than the 307 MB source data or large matrices.

### 17.2 Documentation State

- `docs/paper_aligned_baseline_report.md`, `docs/phase6_summary.md`, and
  `docs/phase7_summary.md` are authoritative frozen-stage summaries.
- `docs/research_extension_protocol.md` defines RQ6 and the exploratory versus
  external-validation distinction.
- `docs/phase8_summary.md` is the formal Phase 8 closure and records the
  retained reference, hypothesis outcomes, and research-governance decision.
- `README.md`, `docs/RUN_AND_TEST_GUIDE.md`, and `docs/experiment_log.md` are
  synchronized with the Phase 8 closure.
- `docs/COMPLETE_PROJECT_PROGRESS_REPORT.md` stops at Phase 7 and is historical.
- This file is the current cross-phase progress report through the Phase 8
  closure.

## 18. Final Current Conclusion

The project has progressed from a basic independent reproduction to a
paper-aligned frozen baseline, controlled improvement studies, systematic
temporal/component/error analysis, and several materially different program
representations.

The strongest honest final statement is:

> The frozen reconstruction approaches the published precision but remains
> substantially below the published recall and F1. Performance varies sharply
> across time. Cost weighting, sampling, threshold changes, compact structural
> and operand features, alternative classifiers, recency weighting, runtime
> CFG/GIN, score fusion, creation-code features, and the frozen contextual
> sequence linear probe did not produce a reliable improvement under their
> predefined temporal criteria.

This is not evidence that improvement is impossible. It is evidence that the
tested interventions did not solve the temporal-generalization problem under
the project's leakage-safe protocol. The most important next research need is
an independently labeled, overlap-audited later-time dataset, while the most
important immediate project task is final synthesis and reproducibility
packaging.
