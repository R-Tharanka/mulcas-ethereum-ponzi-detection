# Archived Code Reference

## 1. Purpose

This document is a non-invasive documentation layer for the Python source
preserved under `archive/history/src/`. It explains the historical modules
and indexes every public top-level class and function, plus every public
class method, without editing the archived evidence or changing its Git blob
identity.

The catalog was generated from Python syntax trees on 2026-10-31. It did not
open `archive/local-only/`, external datasets, provider credentials, result
payloads, or fitted artifacts.

## 2. Integrity Boundary

- Files under `archive/history/src/` remain historical evidence.
- The catalog does not make archived modules directly executable.
- Underscore-prefixed private helpers are omitted from the symbol tables.
- A missing callable description means the preserved callable lacked a
  dedicated docstring; the guide supplies only a neutral navigation note.
- Source line numbers refer to the preserved archive snapshot and should
  remain stable unless an explicit archive correction is made.

## 3. Package Overview

| Package | Modules | Public symbols | Responsibility |
| --- | ---: | ---: | --- |
| `(root)` | 1 | 0 | Package identity and shared top-level declarations. |
| `data` | 6 | 31 | Raw-data loading, validation, integrity auditing, and chronological split construction. |
| `evaluation` | 3 | 5 | Shared binary metrics and chronological period definitions. |
| `experiments` | 13 | 97 | Controlled Phase 6 interventions and Phase 7 temporal/diagnostic analyses. |
| `extension` | 39 | 303 | Phase 8 representations, graph and sequence work, external enrichment, Track B governance, and closure. |
| `features` | 9 | 42 | Static opcode, creator, Word2Vec, structural, and operand feature construction. |
| `models` | 6 | 22 | Baseline estimators, MulCas cascade/voting, and model execution entry points. |
| `phase9` | 11 | 28 | Prospective Phase 9 diagnostics, state features, metrics, and protocol-gap analysis. |
| `utils` | 3 | 2 | Repository-relative paths and deterministic random seeding. |

Catalog total: **91 modules** and **530 public symbols**.

## 4. How to Use This Reference

1. Start with `archive/CHRONOLOGY.md` to identify the research phase.
2. Use `archive/history/TRACKED_FILE_MANIFEST.csv` to find all files in that phase.
3. Find the implementation module in the package sections below.
4. Read the linked configuration, result report, and tests before reusing code.
5. Restore code to its original root-relative path in an isolated worktree before execution.

Do not import modules as `archive.history.src...`; historical imports expect
the original `src...` package layout.

## 5. Package: `(root)`

Package identity and shared top-level declarations.

### `__init__.py`

**Source:** `archive/history/src/__init__.py`

**Purpose:** MulCas Ethereum Ponzi detection research package.

No public top-level classes or functions are declared in this module.

## 6. Package: `data`

Raw-data loading, validation, integrity auditing, and chronological split construction.

### `data/__init__.py`

**Source:** `archive/history/src/data/__init__.py`

**Purpose:** Data access and validation helpers.

No public top-level classes or functions are declared in this module.

### `data/audit.py`

**Source:** `archive/history/src/data/audit.py`

**Purpose:** Dataset integrity audit for the released raw Ponzi dataset.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `compute_sha256` | 49 | Compute SHA-256 without modifying the input file. |
| Function | `build_dataset_audit` | 117 | Build the Phase 1 integrity audit as a machine-readable dictionary. |
| Function | `save_dataset_audit` | 211 | Save the audit JSON under the baseline results directory. |
| Function | `main` | 222 | Command-line entry point for this historical module. |

### `data/load.py`

**Source:** `archive/history/src/data/load.py`

**Purpose:** Dataset loading helpers for the released MulCas/Ponzi CSV.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `read_dataset_header` | 23 | Return CSV column names without loading data rows. |
| Function | `read_dataset_columns` | 28 | Read selected columns from the raw dataset. |
| Function | `read_dataset_metadata` | 36 | Read non-text columns for integrity checks that do not need code fields. |

### `data/paper_aligned_split.py`

**Source:** `archive/history/src/data/paper_aligned_split.py`

**Purpose:** Ponzi-occurrence-based chronological split for the paper-aligned protocol.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `PonziOccurrenceSplitConfig` | 29 | Public historical class; inspect its source and matching tests before reuse. |
| Function | `load_config` | 46 | Load historical inputs or configuration for load config. |
| Function | `assign_ponzi_occurrence_splits` | 61 | Assign whole creation blocks using the configured positive occurrences. |
| Function | `build_paper_aligned_split` | 107 | Build the declared historical representation or result for build paper aligned split. |
| Function | `save_outputs` | 142 | Persist historical outputs for save outputs. |
| Function | `main` | 157 | Command-line entry point for this historical module. |

### `data/split.py`

**Source:** `archive/history/src/data/split.py`

**Purpose:** Leakage-safe chronological train/validation/test splitting.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `SplitConfig` | 25 | Configuration for deterministic chronological splitting. |
| Function | `load_split_config` | 46 | Load split configuration from YAML. |
| Function | `validate_split_config` | 64 | Validate split fraction settings. |
| Function | `prepare_split_frame` | 77 | Validate and chronologically sort split metadata without shuffling. |
| Function | `assign_chronological_splits` | 112 | Assign train/validation/test labels by chronological row position. |
| Function | `assert_temporal_split_boundaries` | 135 | Assert chronological boundaries between train, validation, and test. |
| Function | `summarize_split_counts` | 149 | Return class counts and imbalance ratios for each split. |
| Function | `detect_cross_split_duplicates` | 172 | Detect duplicate values that appear in more than one split. |
| Function | `build_temporal_split` | 209 | Build split membership and metadata for the configured raw dataset. |
| Function | `save_temporal_split` | 250 | Save compact split membership and metadata. |
| Function | `main` | 266 | Command-line entry point for this historical module. |

### `data/validate.py`

**Source:** `archive/history/src/data/validate.py`

**Purpose:** Reusable validation primitives for dataset integrity audits.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `missing_columns` | 11 | Return required columns not present in ''columns''. |
| Function | `require_columns` | 17 | Raise ''ValueError'' if required columns are missing. |
| Function | `unexpected_values` | 24 | Return non-null values outside an allowed set. |
| Function | `missing_or_empty_count` | 30 | Count nulls and strings that become empty after stripping whitespace. |
| Function | `duplicate_value_count` | 39 | Count rows whose value duplicates another row. |
| Function | `duplicate_group_count` | 49 | Count duplicated rows by a subset of columns. |
| Function | `is_chronologically_ordered` | 64 | Return whether non-null values are monotonically non-decreasing. |

## 7. Package: `evaluation`

Shared binary metrics and chronological period definitions.

### `evaluation/__init__.py`

**Source:** `archive/history/src/evaluation/__init__.py`

**Purpose:** Standardized evaluation helpers.

No public top-level classes or functions are declared in this module.

### `evaluation/metrics.py`

**Source:** `archive/history/src/evaluation/metrics.py`

**Purpose:** Standardized binary-classification metrics.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `binary_classification_metrics` | 20 | Calculate the standardized Phase 4 binary metrics. |

### `evaluation/temporal.py`

**Source:** `archive/history/src/evaluation/temporal.py`

**Purpose:** Chronological period construction for temporal robustness evaluation.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `TemporalScenario` | 14 | One expanding-history train period and its immediately later target. |
| Function | `build_ponzi_occurrence_periods` | 22 | Assign all rows to whole-block periods delimited by positive occurrences. |
| Function | `incremental_scenarios` | 113 | Return the four predeclared expanding-history Phase 7A scenarios. |
| Function | `validate_scenario_chronology` | 125 | Assert that every training block strictly precedes the target period. |

## 8. Package: `experiments`

Controlled Phase 6 interventions and Phase 7 temporal/diagnostic analyses.

### `experiments/__init__.py`

**Source:** `archive/history/src/experiments/__init__.py`

**Purpose:** Controlled post-baseline research experiments.

No public top-level classes or functions are declared in this module.

### `experiments/phase6a.py`

**Source:** `archive/history/src/experiments/phase6a.py`

**Purpose:** Controlled cost-sensitive learning experiments for Phase 6A.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `sha256_file` | 43 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `load_config` | 51 | Load historical inputs or configuration for load config. |
| Function | `verify_frozen_manifest` | 69 | Fail before experimentation if any frozen Phase 5.5 file changed. |
| Function | `load_development_data` | 84 | Load train and validation only; final-test rows are excluded immediately. |
| Function | `fold_metadata` | 109 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `build_feature_pair` | 131 | Fit static and creator transformers on the supplied past-only rows. |
| Function | `candidate_parameters` | 168 | Apply only the configured class-cost parameter to frozen model settings. |
| Function | `evaluate_once` | 191 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `aggregate_cv` | 209 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `select_candidate` | 218 | Select by CV F1 with predeclared precision constraint and tie breakers. |
| Function | `run_phase6a` | 251 | Run CV selection and one validation evaluation per model family. |
| Function | `save_results` | 464 | Persist historical outputs for save results. |
| Function | `main` | 538 | Command-line entry point for this historical module. |

### `experiments/phase6b.py`

**Source:** `archive/history/src/experiments/phase6b.py`

**Purpose:** Controlled training-data sampling experiments for Phase 6B.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 44 | Load historical inputs or configuration for load config. |
| Function | `baseline_parameters` | 56 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `apply_training_sampler` | 68 | Resample a training matrix only and return a complete class-count audit. |
| Function | `run_phase6b` | 118 | Execute the declared historical workflow for run phase6b. |
| Function | `save_results` | 232 | Persist historical outputs for save results. |
| Function | `main` | 255 | Command-line entry point for this historical module. |

### `experiments/phase6c.py`

**Source:** `archive/history/src/experiments/phase6c.py`

**Purpose:** Controlled decision-threshold optimization for Phase 6C.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 42 | Load historical inputs or configuration for load config. |
| Function | `candidate_thresholds` | 57 | Return a bounded deterministic quantile grid including the native default. |
| Function | `threshold_metrics` | 68 | Evaluate one operating threshold while retaining continuous-score PR-AUC. |
| Function | `select_threshold` | 76 | Select an eligible threshold, plus an always-available diagnostic fallback. |
| Function | `run_phase6c` | 96 | Execute the declared historical workflow for run phase6c. |
| Function | `save_results` | 252 | Persist historical outputs for save results. |
| Function | `main` | 279 | Command-line entry point for this historical module. |

### `experiments/phase6d1.py`

**Source:** `archive/history/src/experiments/phase6d1.py`

**Purpose:** Controlled structural static-feature experiments for Phase 6D1.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 38 | Load historical inputs or configuration for load config. |
| Function | `build_representation_pair` | 53 | Build one sparse pair, fitting any scaler on historical training rows only. |
| Function | `run_phase6d1` | 113 | Execute the declared historical workflow for run phase6d1. |
| Function | `save_results` | 180 | Persist historical outputs for save results. |
| Function | `main` | 204 | Command-line entry point for this historical module. |

### `experiments/phase6d2.py`

**Source:** `archive/history/src/experiments/phase6d2.py`

**Purpose:** Controlled normalized operand-derived static-feature experiments.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 33 | Load historical inputs or configuration for load config. |
| Function | `load_development_with_operands` | 40 | Load historical inputs or configuration for load development with operands. |
| Function | `build_operand_audit` | 45 | Build the declared historical representation or result for build operand audit. |
| Function | `build_representation_pair` | 58 | Build the declared historical representation or result for build representation pair. |
| Function | `run_phase6d2` | 82 | Execute the declared historical workflow for run phase6d2. |
| Function | `save_results` | 114 | Persist historical outputs for save results. |
| Function | `main` | 128 | Command-line entry point for this historical module. |

### `experiments/phase6e.py`

**Source:** `archive/history/src/experiments/phase6e.py`

**Purpose:** Controlled Logistic Regression and LightGBM experiments for Phase 6E.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 30 | Load historical inputs or configuration for load config. |
| Function | `environment_versions` | 37 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `build_feature_pair` | 40 | Build the declared historical representation or result for build feature pair. |
| Function | `candidate_parameters` | 51 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `evaluate` | 56 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `run_phase6e` | 70 | Execute the declared historical workflow for run phase6e. |
| Function | `save_results` | 103 | Persist historical outputs for save results. |
| Function | `main` | 111 | Command-line entry point for this historical module. |

### `experiments/phase7a.py`

**Source:** `archive/history/src/experiments/phase7a.py`

**Purpose:** Phase 7A temporal robustness analysis for the frozen MulCas reconstruction.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 47 | Load Phase 7A and reject settings that could alter frozen behavior. |
| Function | `load_full_dataset` | 69 | Load only fields needed by the frozen static temporal analysis. |
| Function | `build_historical_feature_matrices` | 98 | Fit every configured view on historical rows and transform the target. |
| Function | `fit_predict_frozen_mulcas` | 167 | Fit the exact frozen residual cascade and predict one later period. |
| Function | `evaluate_scenario` | 255 | Run one predeclared forward-only temporal evaluation scenario. |
| Function | `run_phase7a` | 361 | Execute all four fixed temporal robustness scenarios. |
| Function | `save_results` | 515 | Write all Phase 7A machine-readable and narrative artifacts. |
| Function | `main` | 555 | Command-line entry point for this historical module. |

### `experiments/phase7b.py`

**Source:** `archive/history/src/experiments/phase7b.py`

**Purpose:** Phase 7B component/view ablation of the frozen MulCas reconstruction.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 40 | Load and validate the predeclared Phase 7B ablation protocol. |
| Function | `describe_frozen_viewers` | 77 | Describe the actual reconstruction instead of assuming paper composition. |
| Function | `build_ablation_predictions` | 183 | Apply only the six predeclared component interventions. |
| Function | `run_phase7b` | 270 | Run fixed viewer/component ablations over the Phase 7A periods. |
| Function | `save_results` | 475 | Save the Phase 7B analysis and all requested machine-readable tables. |
| Function | `main` | 512 | Command-line entry point for this historical module. |

### `experiments/phase7c.py`

**Source:** `archive/history/src/experiments/phase7c.py`

**Purpose:** Phase 7C chronological creator-feature ablation for frozen MulCas.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 41 | Load Phase 7C and reject any deviation from the fixed ablation. |
| Function | `build_creator_ablation_matrices` | 70 | Fit static views once and expose matrices with/without creator column. |
| Function | `creator_diagnostics` | 128 | Describe historical creator prevalence and errors without updating state. |
| Function | `run_phase7c` | 195 | Evaluate frozen MulCas with and without its creator column. |
| Function | `save_results` | 461 | Write all requested Phase 7C outputs. |
| Function | `main` | 486 | Command-line entry point for this historical module. |

### `experiments/phase7d.py`

**Source:** `archive/history/src/experiments/phase7d.py`

**Purpose:** Systematic error analysis over exact frozen Phase 7A MulCas predictions.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 39 | Load Phase 7D and reject settings that could alter frozen behavior. |
| Function | `verify_prior_phase_chain` | 67 | Verify Phase 5.5 through Phase 7C without rewriting prior artifacts. |
| Function | `classify_outcomes` | 85 | Return one mutually exclusive confusion category for every row. |
| Function | `parse_boolean_series` | 105 | Parse CSV booleans without treating the string ''False'' as true. |
| Function | `load_analysis_dataset` | 116 | Load only identity and existing diagnostic source fields. |
| Function | `add_historical_context` | 140 | Add creator and exact-clone state from strictly earlier periods only. |
| Function | `add_contract_diagnostics` | 202 | Add fit-free, contract-local diagnostics from existing extractors. |
| Function | `add_routing_diagnostics` | 242 | Classify wrong routes and flag scores close to an applicable boundary. |
| Function | `validate_reference_counts` | 338 | Require exact row predictions and confusion counts from Phase 7A. |
| Function | `build_period_summary` | 361 | Build the declared historical representation or result for build period summary. |
| Function | `build_routing_summary` | 385 | Build the declared historical representation or result for build routing summary. |
| Function | `build_clone_summary` | 406 | Build the declared historical representation or result for build clone summary. |
| Function | `render_report` | 429 | Render the historical report for render report. |
| Function | `run` | 549 | Create deterministic Phase 7D diagnostics without model training. |
| Function | `main` | 711 | Command-line entry point for this historical module. |

### `experiments/phase7e.py`

**Source:** `archive/history/src/experiments/phase7e.py`

**Purpose:** Historical exact-clone sensitivity analysis for frozen Phase 7A predictions.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 51 | Load Phase 7E and reject settings that could change frozen behavior. |
| Function | `verify_prior_phase_chain` | 85 | Verify Phase 5.5 through Phase 7D artifacts before analysis. |
| Function | `normalize_opcode_sequence` | 104 | Normalize opcode names without fuzzy matching or operand interpretation. |
| Function | `normalize_bytecode` | 111 | Normalize a valid hexadecimal byte sequence for exact comparison. |
| Function | `build_clone_membership` | 149 | Assign exactly one novelty group using strictly earlier periods only. |
| Function | `safe_group_metrics` | 221 | Compute subgroup metrics and use None for mathematically undefined values. |
| Function | `build_group_metrics` | 273 | Build the declared historical representation or result for build group metrics. |
| Function | `build_class_analysis` | 294 | Summarize positive recall and benign false-positive behavior. |
| Function | `build_creator_interaction` | 343 | Build the declared historical representation or result for build creator interaction. |
| Function | `build_routing_interaction` | 370 | Build the declared historical representation or result for build routing interaction. |
| Function | `validate_partition_counts` | 411 | Require clone/novel confusion counts to recover every full period. |
| Function | `render_report` | 433 | Render the historical report for render report. |
| Function | `run` | 499 | Generate deterministic clone sensitivity artifacts without model fitting. |
| Function | `main` | 601 | Command-line entry point for this historical module. |

### `experiments/temporal_cv.py`

**Source:** `archive/history/src/experiments/temporal_cv.py`

**Purpose:** Forward-only temporal cross-validation utilities.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `TemporalFold` | 12 | Public historical class; inspect its source and matching tests before reuse. |
| Function | `make_positive_occurrence_folds` | 20 | Create expanding-window folds with whole-block positive boundaries. |

## 9. Package: `extension`

Phase 8 representations, graph and sequence work, external enrichment, Track B governance, and closure.

### `extension/__init__.py`

**Source:** `archive/history/src/extension/__init__.py`

**Purpose:** Planning and feasibility utilities for post-Phase-7 research extensions.

No public top-level classes or functions are declared in this module.

### `extension/artifact_resolver.py`

**Source:** `archive/history/src/extension/artifact_resolver.py`

**Purpose:** Hash-preserving resolver for historical artifacts relocated with a repository.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `ArtifactResolution` | 12 | Public historical class; inspect its source and matching tests before reuse. |
| Function | `resolve_artifact` | 20 | Public historical function; inspect its implementation and matching tests before reuse. |

### `extension/cfg.py`

**Source:** `archive/history/src/extension/cfg.py`

**Purpose:** Deterministic, contract-local EVM control-flow graph construction.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `Instruction` | 23 | Public historical class; inspect its source and matching tests before reuse. |
| Class | `Disassembly` | 35 | Public historical class; inspect its source and matching tests before reuse. |
| Class | `BasicBlock` | 44 | Public historical class; inspect its source and matching tests before reuse. |
| Class | `Edge` | 55 | Public historical class; inspect its source and matching tests before reuse. |
| Class | `JumpStatistics` | 62 | Public historical class; inspect its source and matching tests before reuse. |
| Method | `JumpStatistics.resolution_rate` | 69 | Public class method retained by the historical implementation. |
| Class | `ContractCFG` | 74 | Public historical class; inspect its source and matching tests before reuse. |
| Function | `disassemble` | 85 | Disassemble runtime bytecode without inferring missing operand bytes. |
| Function | `build_basic_blocks` | 127 | Partition every instruction exactly once using the frozen leader policy. |
| Function | `build_edges` | 181 | Build the declared historical representation or result for build edges. |
| Function | `build_cfg` | 211 | Construct one deterministic CFG using only the supplied contract bytecode. |

### `extension/creation_code.py`

**Source:** `archive/history/src/extension/creation_code.py`

**Purpose:** Deterministic creation/runtime bytecode boundary analysis.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `RuntimeBoundary` | 11 | Public historical class; inspect its source and matching tests before reuse. |
| Function | `exact_occurrence_offsets` | 29 | Return all exact offsets, including overlapping occurrences. |
| Function | `analyze_runtime_boundary` | 44 | Locate runtime bytes conservatively without inventing an init boundary. |

### `extension/creation_features.py`

**Source:** `archive/history/src/extension/creation_features.py`

**Purpose:** Frozen label-independent features for creation and deployment-init bytecode.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `extract_region_features` | 105 | Extract a finite fixed-order region vector without exposing constants. |
| Function | `extract_creation_full_features` | 165 | Extract the declared features or records for extract creation full features. |
| Function | `extract_init_features` | 174 | Extract the declared features or records for extract init features. |
| Function | `boundary_feature_vector` | 199 | Public historical function; inspect its implementation and matching tests before reuse. |

### `extension/external_enrichment.py`

**Source:** `archive/history/src/extension/external_enrichment.py`

**Purpose:** Candidate construction, resumable ledger, and deterministic preflight.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `candidate_rows` | 12 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `deterministic_sample` | 30 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `enrich` | 34 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `environment_record` | 58 | Public historical function; inspect its implementation and matching tests before reuse. |

### `extension/external_normalization.py`

**Source:** `archive/history/src/extension/external_normalization.py`

**Purpose:** Pure normalization helpers for the Phase 8R6A external-data audit.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `normalize_address` | 11 | Normalize values for normalize address. |
| Function | `normalize_hex` | 20 | Normalize values for normalize hex. |
| Function | `normalize_opcode` | 27 | Normalize values for normalize opcode. |
| Function | `digest` | 34 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `normalize_label` | 38 | Normalize values for normalize label. |

### `extension/external_overlap.py`

**Source:** `archive/history/src/extension/external_overlap.py`

**Purpose:** Exact identity overlap calculations; no fuzzy similarity or model use.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `add_hashes` | 10 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `identity_sets` | 19 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `address_summary` | 23 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `pairwise_overlap` | 39 | Public historical function; inspect its implementation and matching tests before reuse. |

### `extension/external_provenance.py`

**Source:** `archive/history/src/extension/external_provenance.py`

**Purpose:** Governance rules for external label provenance and Track B feasibility.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `track_b_rating` | 5 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `complete_benchmark` | 18 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `ground_truth_eligible` | 22 | Public historical function; inspect its implementation and matching tests before reuse. |

### `extension/external_rpc.py`

**Source:** `archive/history/src/extension/external_rpc.py`

**Purpose:** Credential-free Ethereum deployment/static metadata clients.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `blockscout_creation` | 21 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `rpc_deployment_bundle` | 32 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `retrieve_with_retry` | 57 | Public historical function; inspect its implementation and matching tests before reuse. |

### `extension/external_sources.py`

**Source:** `archive/history/src/extension/external_sources.py`

**Purpose:** Source-specific readers that preserve raw labels and missing values.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_unica` | 25 | Load historical inputs or configuration for load unica. |
| Function | `load_ctrf` | 32 | Load historical inputs or configuration for load ctrf. |
| Function | `load_ponzi_ml` | 46 | Load historical inputs or configuration for load ponzi ml. |
| Function | `load_positive_list` | 57 | Load historical inputs or configuration for load positive list. |

### `extension/feasibility.py`

**Source:** `archive/history/src/extension/feasibility.py`

**Purpose:** Deterministic Phase 8R0 feasibility audits without model training.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `Instruction` | 30 | One decoded EVM instruction and its byte offset. |
| Class | `Disassembly` | 41 | Deterministic bytecode disassembly result. |
| Class | `CFGSummary` | 52 | Compact control-flow graph feasibility summary for one contract. |
| Function | `sha256_file` | 68 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `disassemble` | 76 | Decode bytecode while retaining malformed PUSH instructions explicitly. |
| Function | `summarize_cfg` | 117 | Build conservative basic blocks and count statically identifiable edges. |
| Function | `audit_bytecode_values` | 195 | Aggregate CFG feasibility without requiring labels or other contracts. |
| Function | `audit_dataset_bytecode` | 227 | Audit runtime and creation bytecode fields without loading labels. |
| Function | `sample_weight_support` | 244 | Inspect fit signatures only; no estimator is fitted. |
| Function | `verify_manifest` | 260 | Verify frozen inputs or integrity conditions for verify manifest. |
| Function | `run` | 277 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `main` | 334 | Command-line entry point for this historical module. |

### `extension/graph_dataset.py`

**Source:** `archive/history/src/extension/graph_dataset.py`

**Purpose:** Verified readers and fixed transforms for the frozen Phase 8R2A corpus.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `sha256_file` | 23 | Public historical function; inspect its implementation and matching tests before reuse. |
| Class | `NodeNormalizer` | 32 | Public historical class; inspect its source and matching tests before reuse. |
| Method | `NodeNormalizer.transform` | 37 | Public class method retained by the historical implementation. |
| Class | `FrozenGraphStore` | 41 | Read frozen compressed chunks and return independent PyG graphs. |
| Method | `FrozenGraphStore.load_rows` | 64 | Public class method retained by the historical implementation. |
| Function | `fit_node_normalizer` | 93 | Fit population moments using nodes from declared training graphs only. |
| Function | `normalized_graphs` | 114 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `aggregate_feature_names` | 123 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `graph_aggregate_vector` | 135 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `chronological_inner_split` | 154 | Split historical rows at a whole-block boundary without future input. |

### `extension/graph_features.py`

**Source:** `archive/history/src/extension/graph_features.py`

**Purpose:** Frozen block-local feature schema for Phase 8R2A control-flow graphs.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `extract_block_features` | 58 | Return fixed-order finite features without exposing exact operand values. |
| Function | `graph_arrays` | 101 | Convert a CFG into framework-neutral node and edge arrays. |

### `extension/graph_models.py`

**Source:** `archive/history/src/extension/graph_models.py`

**Purpose:** Frozen Phase 8R2B GIN architecture and deterministic training helpers.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `GINClassifier` | 16 | Public historical class; inspect its source and matching tests before reuse. |
| Method | `GINClassifier.forward` | 33 | Public class method retained by the historical implementation. |
| Function | `set_torch_seed` | 41 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `graph_loader` | 50 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `epoch_loss` | 55 | Public historical function; inspect its implementation and matching tests before reuse. |
| Class | `TrainingResult` | 77 | Public historical class; inspect its source and matching tests before reuse. |
| Function | `train_with_early_stopping` | 85 | Train the declared historical component for train with early stopping. |
| Function | `predict_probabilities` | 123 | Generate predictions for predict probabilities. |

### `extension/phase8_closure.py`

**Source:** `archive/history/src/extension/phase8_closure.py`

**Purpose:** Deterministic evidence consolidation for the formal Phase 8 closure.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 27 | Load historical inputs or configuration for load config. |
| Function | `verify_frozen_inputs` | 57 | Verify frozen inputs or integrity conditions for verify frozen inputs. |
| Function | `build_closure` | 113 | Build the declared historical representation or result for build closure. |
| Function | `build_temporal_comparison` | 245 | Build the declared historical representation or result for build temporal comparison. |
| Function | `write_outputs` | 285 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `render_report` | 309 | Render the historical report for render report. |
| Function | `main` | 431 | Command-line entry point for this historical module. |

### `extension/phase8r1a.py`

**Source:** `archive/history/src/extension/phase8r1a.py`

**Purpose:** Controlled recency-aware temporal backtesting for Phase 8R1A.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 42 | Load historical inputs or configuration for load config. |
| Function | `verify_frozen_chain` | 80 | Verify frozen inputs or integrity conditions for verify frozen chain. |
| Function | `load_p0_p4_membership` | 90 | Read exactly the frozen P0-P4 prefix; no P5 row is loaded. |
| Function | `strategy_training_periods` | 139 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `period_sample_weights` | 150 | Return estimator-only temporal weights; controls/windows remain unweighted. |
| Function | `fit_estimator` | 164 | Apply temporal sample weights to estimator fitting only. |
| Function | `fit_predict_mulcas` | 176 | Run frozen MulCas, changing only optional estimator row weights. |
| Function | `evaluate_strategy_period` | 244 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `verify_control_reproduction` | 348 | Verify frozen inputs or integrity conditions for verify control reproduction. |
| Function | `summarize_strategies` | 406 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `select_strategy` | 435 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `load_subgroup_membership` | 459 | Read exactly the P2-P4 prefix from the frozen Phase 7E membership export. |
| Function | `subgroup_metrics` | 483 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `routing_summary` | 521 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `period_effects` | 542 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `run_phase8r1a` | 579 | Execute the declared historical workflow for run phase8r1a. |
| Function | `build_report` | 675 | Build the declared historical representation or result for build report. |
| Function | `save_outputs` | 731 | Persist historical outputs for save outputs. |
| Function | `main` | 766 | Command-line entry point for this historical module. |

### `extension/phase8r2a.py`

**Source:** `archive/history/src/extension/phase8r2a.py`

**Purpose:** Build and validate the frozen Phase 8R2A static EVM CFG corpus.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 40 | Load historical inputs or configuration for load config. |
| Function | `verify_frozen_chain` | 60 | Verify frozen inputs or integrity conditions for verify frozen chain. |
| Function | `write_deterministic_npz` | 80 | Write byte-stable NPZ data using fixed ZIP metadata and sorted keys. |
| Function | `graph_content_hash` | 91 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `chunk_arrays` | 106 | Pack multiple graphs with pointer arrays for later framework conversion. |
| Function | `graph_structure_statistics` | 182 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `graph_schema` | 254 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `build_graph_corpus` | 353 | Stream all contracts into deterministic ignored graph chunks and audits. |
| Function | `build_report` | 491 | Build the declared historical representation or result for build report. |
| Function | `save_outputs` | 554 | Persist historical outputs for save outputs. |
| Function | `main` | 583 | Command-line entry point for this historical module. |

### `extension/phase8r2b.py`

**Source:** `archive/history/src/extension/phase8r2b.py`

**Purpose:** Run Phase 8R2B graph aggregate and temporal GIN backtests.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 53 | Load historical inputs or configuration for load config. |
| Function | `verify_frozen_inputs` | 75 | Verify frozen inputs or integrity conditions for verify frozen inputs. |
| Function | `load_temporal_frame` | 90 | Load historical inputs or configuration for load temporal frame. |
| Function | `architecture_grid` | 109 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `run_aggregate_baseline` | 123 | Execute the declared historical workflow for run aggregate baseline. |
| Function | `run_gin_experiments` | 165 | Execute the declared historical workflow for run gin experiments. |
| Function | `summarize_gin` | 255 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `subgroup_metrics` | 303 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `frozen_mulcas_metrics` | 331 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `build_report` | 340 | Build the declared historical representation or result for build report. |
| Function | `run` | 391 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `main` | 446 | Command-line entry point for this historical module. |

### `extension/phase8r3a.py`

**Source:** `archive/history/src/extension/phase8r3a.py`

**Purpose:** Frozen MulCas-GIN complementarity and hybrid-headroom analysis.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 31 | Load historical inputs or configuration for load config. |
| Function | `verify_frozen_inputs` | 51 | Verify frozen inputs or integrity conditions for verify frozen inputs. |
| Function | `align_predictions` | 65 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `prediction_policies` | 148 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `disagreement_summary` | 154 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `positive_complementarity` | 174 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `negative_complementarity` | 195 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `oracle_headroom` | 212 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `score_disagreement` | 228 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `novelty_creator_interaction` | 251 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `routing_interaction` | 271 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `hybrid_decision` | 291 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `build_report` | 329 | Build the declared historical representation or result for build report. |
| Function | `run` | 383 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `main` | 425 | Command-line entry point for this historical module. |

### `extension/phase8r3b.py`

**Source:** `archive/history/src/extension/phase8r3b.py`

**Purpose:** Forward-only score fusion of frozen MulCas and GIN predictions.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 38 | Load historical inputs or configuration for load config. |
| Function | `verify_frozen_inputs` | 70 | Verify frozen inputs or integrity conditions for verify frozen inputs. |
| Function | `load_frozen_predictions` | 77 | Load historical inputs or configuration for load frozen predictions. |
| Function | `feature_schemas` | 109 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `runtime_preflight` | 129 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `select_candidate` | 214 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `error_transitions` | 258 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `subgroup_metrics` | 277 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `run` | 351 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `main` | 457 | Command-line entry point for this historical module. |

### `extension/phase8r4a.py`

**Source:** `archive/history/src/extension/phase8r4a.py`

**Purpose:** Creation-code feasibility audit and frozen deployment-time feature corpus.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 50 | Load historical inputs or configuration for load config. |
| Function | `verify_frozen_inputs` | 69 | Verify frozen inputs or integrity conditions for verify frozen inputs. |
| Function | `load_dataset` | 83 | Load historical inputs or configuration for load dataset. |
| Function | `performance_preflight` | 151 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `construct_features` | 184 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `creation_audit` | 225 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `period_summary` | 252 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `label_summary` | 276 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `boundary_statistics` | 294 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `feature_schema` | 310 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `build_report` | 329 | Build the declared historical representation or result for build report. |
| Function | `run` | 363 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `main` | 434 | Command-line entry point for this historical module. |

### `extension/phase8r4b.py`

**Source:** `archive/history/src/extension/phase8r4b.py`

**Purpose:** Forward temporal classification with frozen Phase 8R4A creation features.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 43 | Load historical inputs or configuration for load config. |
| Function | `verify_frozen_inputs` | 72 | Verify frozen inputs or integrity conditions for verify frozen inputs. |
| Function | `load_p0_p4_metadata` | 82 | Read the exact frozen 5,989-row P0-P4 prefix, never P5. |
| Function | `load_frozen_views` | 111 | Load historical inputs or configuration for load frozen views. |
| Function | `load_diagnostics` | 135 | Load historical inputs or configuration for load diagnostics. |
| Function | `fit_predict` | 181 | Fit the declared historical component for fit predict. |
| Function | `performance_preflight` | 218 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `frozen_mulcas_metrics` | 255 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `summarize_candidates` | 266 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `select_candidate` | 284 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `subgroup_diagnostics` | 304 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `complementarity` | 344 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `build_report` | 375 | Build the declared historical representation or result for build report. |
| Function | `run` | 416 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `main` | 518 | Command-line entry point for this historical module. |

### `extension/phase8r5a.py`

**Source:** `archive/history/src/extension/phase8r5a.py`

**Purpose:** Phase 8R5A opcode-sequence feasibility and compute audit.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 46 | Load historical inputs or configuration for load config. |
| Function | `verify_frozen_inputs` | 68 | Verify frozen inputs or integrity conditions for verify frozen inputs. |
| Function | `load_sequence_corpus` | 82 | Load label-free runtime sequences and frozen descriptive period membership. |
| Function | `sequence_statistics` | 120 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `truncation_coverage` | 131 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `temporal_vocabulary_audit` | 159 | Public historical function; inspect its implementation and matching tests before reuse. |
| Class | `TinyOpcodeEncoder` | 188 | Benchmark-only encoder matching the proposed Phase 8R5B dimensions. |
| Method | `TinyOpcodeEncoder.forward` | 205 | Public class method retained by the historical implementation. |
| Function | `runtime_environment` | 219 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `run_tiny_benchmark` | 250 | Execute the declared historical workflow for run tiny benchmark. |
| Function | `runtime_estimates` | 297 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `build_report` | 336 | Build the declared historical representation or result for build report. |
| Function | `run` | 408 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `main` | 466 | Command-line entry point for this historical module. |

### `extension/phase8r5b.py`

**Source:** `archive/history/src/extension/phase8r5b.py`

**Purpose:** Phase 8R5B temporally valid masked-opcode pretraining runner.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 42 | Load historical inputs or configuration for load config. |
| Function | `verify_frozen_inputs` | 83 | Verify frozen inputs or integrity conditions for verify frozen inputs. |
| Function | `verify_cuda` | 90 | Verify frozen inputs or integrity conditions for verify cuda. |
| Function | `run_phase8r5b` | 246 | Execute the declared historical workflow for run phase8r5b. |
| Function | `main` | 330 | Command-line entry point for this historical module. |

### `extension/phase8r5b0.py`

**Source:** `archive/history/src/extension/phase8r5b0.py`

**Purpose:** CUDA validation and bounded GPU benchmark for the frozen opcode Transformer.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 47 | Load historical inputs or configuration for load config. |
| Function | `verify_frozen_inputs` | 73 | Verify frozen inputs or integrity conditions for verify frozen inputs. |
| Function | `collect_environment` | 106 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `cuda_ready` | 142 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `load_historical_sequences` | 150 | Load only the exact frozen P0-P4 prefix, with no labels or P5 rows. |
| Function | `deterministic_benchmark_chunks` | 168 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `is_cuda_oom` | 182 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `run_gpu_benchmarks` | 264 | Execute the declared historical workflow for run gpu benchmarks. |
| Function | `select_execution_policy` | 290 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `build_runtime_estimates` | 304 | Build the declared historical representation or result for build runtime estimates. |
| Function | `feasibility_rating` | 330 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `summarize_mode_runtimes` | 338 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `run` | 495 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `main` | 581 | Command-line entry point for this historical module. |

### `extension/phase8r5c1.py`

**Source:** `archive/history/src/extension/phase8r5c1.py`

**Purpose:** Phase 8R5C1 forward temporal linear probes over frozen opcode encoders.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 49 | Load historical inputs or configuration for load config. |
| Function | `verify_frozen_inputs` | 82 | Verify frozen inputs or integrity conditions for verify frozen inputs. |
| Function | `verify_cuda` | 90 | Verify frozen inputs or integrity conditions for verify cuda. |
| Function | `load_temporal_data` | 115 | Load historical inputs or configuration for load temporal data. |
| Function | `load_frozen_diagnostics` | 133 | Load historical inputs or configuration for load frozen diagnostics. |
| Function | `performance_preflight` | 164 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `mean_seed_probabilities` | 229 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `run_phase8r5c1` | 401 | Execute the declared historical workflow for run phase8r5c1. |
| Function | `main` | 639 | Command-line entry point for this historical module. |

### `extension/phase8r6a.py`

**Source:** `archive/history/src/extension/phase8r6a.py`

**Purpose:** Phase 8R6A deterministic external dataset audit (strictly no model scoring).

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `run` | 70 | Public historical function; inspect its implementation and matching tests before reuse. |

### `extension/phase8r6b.py`

**Source:** `archive/history/src/extension/phase8r6b.py`

**Purpose:** Phase 8R6B external enrichment and Track B construction; no detector access.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `verify_prior` | 31 | Verify frozen inputs or integrity conditions for verify prior. |
| Function | `portability_audit` | 41 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `conflict_audit` | 49 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `build_outputs` | 80 | Build the declared historical representation or result for build outputs. |
| Function | `write_report` | 156 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `main` | 184 | Command-line entry point for this historical module. |

### `extension/phase8r6c.py`

**Source:** `archive/history/src/extension/phase8r6c.py`

**Purpose:** Phase 8R6C provider-gated external enrichment finalization (no model access).

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `provider_state` | 29 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `verify_prior` | 35 | Verify frozen inputs or integrity conditions for verify prior. |
| Function | `normalized_status` | 45 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `run` | 60 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `write_report` | 99 | Public historical function; inspect its implementation and matching tests before reuse. |

### `extension/phase8r6c_provider_resume.py`

**Source:** `archive/history/src/extension/phase8r6c_provider_resume.py`

**Purpose:** Audited provider validation and bounded Phase 8R6C continuation.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `GateFailure` | 34 | Public historical class; inspect its source and matching tests before reuse. |
| Class | `Metrics` | 41 | Public historical class; inspect its source and matching tests before reuse. |
| Method | `Metrics.record` | 47 | Public class method retained by the historical implementation. |
| Function | `provider_configuration` | 68 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `rpc_call` | 96 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `creation_schema` | 103 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `creation_lookup` | 112 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `rpc_health` | 158 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `enrich_one` | 168 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `provider_stability_failure` | 219 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `successful_ledger_addresses` | 525 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `successful_ledger_rows` | 534 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `run` | 592 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `run_approved_full` | 694 | Execute the declared historical workflow for run approved full. |

### `extension/phase8r6d.py`

**Source:** `archive/history/src/extension/phase8r6d.py`

**Purpose:** Phase 8R6D external enrichment, Track B sealing, and Phase 9B protocol freeze.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `RowFailure` | 37 | Public historical class; inspect its source and matching tests before reuse. |
| Class | `ProviderMetrics` | 45 | Public historical class; inspect its source and matching tests before reuse. |
| Method | `ProviderMetrics.record` | 51 | Public class method retained by the historical implementation. |
| Function | `sha256_file` | 57 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `verify_frozen_inputs` | 75 | Verify frozen inputs or integrity conditions for verify frozen inputs. |
| Function | `schema_fingerprint` | 124 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `creation_lookup` | 156 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `rpc_call` | 197 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `enrich_one` | 204 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `failure_category` | 253 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `should_stop_systemic` | 266 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `finalize_track_b` | 344 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `freeze_phase9b_protocol` | 574 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `pending_candidates` | 728 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `run` | 734 | Public historical function; inspect its implementation and matching tests before reuse. |

### `extension/sealed_track_b.py`

**Source:** `archive/history/src/extension/sealed_track_b.py`

**Purpose:** Default-deny access to any future sealed Track B labels.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_sealed_labels` | 8 | Load historical inputs or configuration for load sealed labels. |

### `extension/sequence_data.py`

**Source:** `archive/history/src/extension/sequence_data.py`

**Purpose:** Deterministic opcode-sequence preparation for temporal experiments.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `OpcodeVocabulary` | 15 | Public historical class; inspect its source and matching tests before reuse. |
| Method | `OpcodeVocabulary.pad_id` | 19 | Public class method retained by the historical implementation. |
| Method | `OpcodeVocabulary.mask_id` | 23 | Public class method retained by the historical implementation. |
| Method | `OpcodeVocabulary.unk_id` | 27 | Public class method retained by the historical implementation. |
| Method | `OpcodeVocabulary.encode` | 30 | Public class method retained by the historical implementation. |
| Function | `canonical_opcode_tokens` | 34 | Return canonical names while discarding hexadecimal operand literals. |
| Function | `build_vocabulary` | 39 | Fit a deterministic vocabulary from the supplied historical corpus only. |
| Function | `prefix_truncate` | 46 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `prefix_suffix_truncate` | 52 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `fixed_size_chunks` | 62 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `padded_ids` | 70 | Public historical function; inspect its implementation and matching tests before reuse. |

### `extension/sequence_embeddings.py`

**Source:** `archive/history/src/extension/sequence_embeddings.py`

**Purpose:** Frozen opcode-encoder inference and deterministic embedding caches.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `LoadedEncoder` | 24 | Public historical class; inspect its source and matching tests before reuse. |
| Function | `load_frozen_encoder` | 31 | Load historical inputs or configuration for load frozen encoder. |
| Function | `prepare_contract_chunks` | 67 | Return padded chunks, owner positions, contract rows, and chunk counts. |
| Function | `extract_contract_embeddings` | 97 | Extract contextual and embedding-table controls in one frozen pass. |
| Function | `write_deterministic_embedding_cache` | 169 | Write a deterministic compressed NPZ with fixed ZIP metadata. |
| Function | `load_embedding_cache` | 186 | Load historical inputs or configuration for load embedding cache. |
| Function | `checkpoint_entries` | 195 | Public historical function; inspect its implementation and matching tests before reuse. |

### `extension/sequence_model.py`

**Source:** `archive/history/src/extension/sequence_model.py`

**Purpose:** Frozen small opcode Transformer used by the Phase 8R5 benchmark.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `SinusoidalPositionalEncoding` | 11 | Public historical class; inspect its source and matching tests before reuse. |
| Method | `SinusoidalPositionalEncoding.forward` | 24 | Public class method retained by the historical implementation. |
| Class | `OpcodeMaskedLanguageModel` | 28 | Exact frozen Phase 8R5 opcode encoder with no classification head. |
| Method | `OpcodeMaskedLanguageModel.forward` | 57 | Public class method retained by the historical implementation. |
| Function | `masked_opcode_loss` | 67 | Public historical function; inspect its implementation and matching tests before reuse. |

### `extension/sequence_pretraining.py`

**Source:** `archive/history/src/extension/sequence_pretraining.py`

**Purpose:** Reusable, temporally safe masked-opcode pretraining utilities.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `PreparedCorpus` | 35 | Public historical class; inspect its source and matching tests before reuse. |
| Function | `vocabulary_sha256` | 49 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `load_p0_p4_sequences` | 56 | Load only label-free P0-P4 rows and attach frozen period membership. |
| Function | `prepare_fold_corpus` | 87 | Fit a vocabulary and construct chunks from historical rows only. |
| Function | `seed_everything` | 127 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `apply_mlm_mask` | 149 | Apply BERT-style 80/10/10 masking using one explicit generator. |
| Function | `encoder_hidden_states` | 193 | Expose final-layer token states without adding a classifier. |
| Function | `masked_mean_pool` | 203 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `model_state_sha256` | 209 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `gpu_utilization_percent` | 217 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `train_pretraining_run` | 231 | Train one fresh fold/seed model and save its epoch-10 checkpoint. |

### `extension/successor_migration.py`

**Source:** `archive/history/src/extension/successor_migration.py`

**Purpose:** Verify frozen manifests while honoring explicit successor test migrations.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_test_migrations` | 21 | Load historical inputs or configuration for load test migrations. |
| Function | `verify_manifest_with_successor_migrations` | 36 | Verify exact hashes, allowing only recorded old-to-new test migrations. |

### `extension/track_b_builder.py`

**Source:** `archive/history/src/extension/track_b_builder.py`

**Purpose:** Strict, model-free Track B construction from verified enrichment records.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `enrich_static_identities` | 13 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `internal_reference` | 31 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `classify` | 38 | Classify records according to the module policy for classify. |
| Function | `subset_counts` | 56 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `primary_membership` | 66 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `seal` | 81 | Public historical function; inspect its implementation and matching tests before reuse. |

## 10. Package: `features`

Static opcode, creator, Word2Vec, structural, and operand feature construction.

### `features/__init__.py`

**Source:** `archive/history/src/features/__init__.py`

**Purpose:** Static feature views used to reproduce the published MulCas inputs.

No public top-level classes or functions are declared in this module.

### `features/developer.py`

**Source:** `archive/history/src/features/developer.py`

**Purpose:** Chronological creator feature without future-label leakage.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `ChronologicalCreatorTransformer` | 14 | Flag creators known from strictly earlier positive training contracts. |
| Method | `ChronologicalCreatorTransformer.fit` | 35 | Learn positive creators from the training partition only. |
| Method | `ChronologicalCreatorTransformer.fit_transform` | 46 | Create prequential training features and then retain final train state. |
| Method | `ChronologicalCreatorTransformer.transform` | 73 | Transform later data from frozen training state without reading labels. |

### `features/opcode.py`

**Source:** `archive/history/src/features/opcode.py`

**Purpose:** Training-only opcode count and TF-IDF feature views.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `tokenize_opcodes` | 12 | Tokenize a whitespace-separated opcode document. |
| Function | `make_opcode_term_count_vectorizer` | 27 | Create the published unigram opcode term-count view. |
| Function | `make_opcode_tfidf_vectorizer` | 32 | Create the published unigram opcode TF-IDF view. |
| Function | `make_opcode_ngram_count_vectorizer` | 37 | Create the opcode unigram/bigram/trigram count view. |
| Function | `make_opcode_ngram_tfidf_vectorizer` | 42 | Create the opcode unigram/bigram/trigram TF-IDF view. |
| Function | `make_paper_aligned_ngram_count_vectorizer` | 47 | Create a 1-3 gram count view with explicit document-frequency filtering. |
| Function | `make_paper_aligned_ngram_tfidf_vectorizer` | 58 | Create a 1-3 gram TF-IDF view with explicit DF filtering. |
| Function | `fit_training_transformer` | 69 | Fit a vectorizer on training documents only. |
| Function | `transform_documents` | 78 | Transform documents without changing fitted vocabulary or IDF state. |

### `features/opcode_categories.py`

**Source:** `archive/history/src/features/opcode_categories.py`

**Purpose:** Canonical opcode categories for deterministic structural features.

No public top-level classes or functions are declared in this module.

### `features/operands.py`

**Source:** `archive/history/src/features/operands.py`

**Purpose:** Deterministic normalized features from deployed EVM PUSH operands.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `OperandParseResult` | 43 | Public historical class; inspect its source and matching tests before reuse. |
| Function | `decode_bytecode` | 52 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `disassemble_operands` | 62 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `disassemble_opcode_names` | 78 | Disassemble names with the same PUSH-byte consumption policy. |
| Function | `extract_operand_features` | 92 | Extract the declared features or records for extract operand features. |
| Function | `feature_names_for_groups` | 122 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `transform_operand_features` | 126 | Transform inputs for transform operand features. |

### `features/paper_aligned.py`

**Source:** `archive/history/src/features/paper_aligned.py`

**Purpose:** Paper-aligned feature variants for the frozen Phase 5.5 protocol.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 44 | Load historical inputs or configuration for load config. |
| Function | `reproduce_paper_aligned_features` | 66 | Fit all paper-aligned transformers on training data only. |
| Function | `main` | 157 | Command-line entry point for this historical module. |

### `features/reproduce.py`

**Source:** `archive/history/src/features/reproduce.py`

**Purpose:** Generate reproducible summaries for the published static feature views.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_feature_config` | 39 | Load and resolve Phase 3 feature configuration. |
| Function | `load_feature_source` | 47 | Load only fields required to reproduce feature views and attach splits. |
| Function | `reproduce_feature_views` | 118 | Fit on train only and summarize transforms for all chronological splits. |
| Function | `save_feature_summary` | 206 | Save compact machine-readable feature-view measurements. |
| Function | `main` | 213 | Command-line entry point for this historical module. |

### `features/structural.py`

**Source:** `archive/history/src/features/structural.py`

**Purpose:** Fit-free, opcode-name-only structural contract features.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `tokenize_opcode_names` | 40 | Return uppercase opcode-name tokens without interpreting operand content. |
| Function | `extract_structural_features` | 67 | Extract the declared features or records for extract structural features. |
| Function | `transform_structural_features` | 117 | Transform inputs for transform structural features. |
| Function | `feature_names_for_groups` | 123 | Public historical function; inspect its implementation and matching tests before reuse. |

### `features/word2vec.py`

**Source:** `archive/history/src/features/word2vec.py`

**Purpose:** Deterministic training-only Word2Vec contract representations.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `OpcodeWord2VecTransformer` | 16 | Represent a contract by the mean of its in-vocabulary opcode vectors. |
| Method | `OpcodeWord2VecTransformer.fit` | 27 | Fit Word2Vec using training documents only. |
| Method | `OpcodeWord2VecTransformer.transform` | 42 | Transform documents without updating the fitted embedding model. |
| Method | `OpcodeWord2VecTransformer.fit_transform` | 55 | Fit on training documents and return their representations. |
| Method | `OpcodeWord2VecTransformer.vocabulary_size_` | 60 | Return the fitted vocabulary size. |
| Class | `IDFWeightedOpcodeWord2VecTransformer` | 68 | Represent contracts by the IDF-weighted sum of opcode embeddings. |
| Method | `IDFWeightedOpcodeWord2VecTransformer.fit` | 80 | Fit embeddings and IDF statistics using training documents only. |
| Method | `IDFWeightedOpcodeWord2VecTransformer.transform` | 106 | Transform documents without updating embeddings or IDF statistics. |
| Method | `IDFWeightedOpcodeWord2VecTransformer.fit_transform` | 124 | Public class method retained by the historical implementation. |
| Method | `IDFWeightedOpcodeWord2VecTransformer.vocabulary_size_` | 128 | Public class method retained by the historical implementation. |

## 11. Package: `models`

Baseline estimators, MulCas cascade/voting, and model execution entry points.

### `models/__init__.py`

**Source:** `archive/history/src/models/__init__.py`

**Purpose:** Model definitions and training helpers.

No public top-level classes or functions are declared in this module.

### `models/alternative.py`

**Source:** `archive/history/src/models/alternative.py`

**Purpose:** Alternative classifier constructors for the controlled Phase 6E study.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `build_alternative_model` | 11 | Build the declared historical representation or result for build alternative model. |

### `models/baselines.py`

**Source:** `archive/history/src/models/baselines.py`

**Purpose:** Fixed-parameter baseline viewer families from the published study.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `build_baseline_model` | 14 | Construct one deterministic baseline classifier without tuning. |
| Function | `positive_class_scores` | 29 | Return continuous positive-class scores for ranking metrics. |

### `models/mulcas.py`

**Source:** `archive/history/src/models/mulcas.py`

**Purpose:** Reusable confidence cascade and final voting for MulCas reconstruction.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `ViewerSpec` | 13 | One configured viewer in cascade order. |
| Class | `ViewerPrediction` | 22 | Index-aligned predictions and positive-class probabilities. |
| Class | `CascadeResult` | 30 | Final predictions plus sample-level and viewer-level traces. |
| Class | `MulCasCascade` | 37 | Route confident samples through N viewers, then vote on residuals. |
| Method | `MulCasCascade.predict` | 90 | Predict while preserving the exact supplied sample index order. |

### `models/run_baselines.py`

**Source:** `archive/history/src/models/run_baselines.py`

**Purpose:** Run fixed individual baseline models across reproduced feature views.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_baseline_config` | 42 | Load fixed Phase 4 model and output configuration. |
| Function | `validate_evaluation_protocol` | 50 | Reject configurations that authorize final-test-driven selection. |
| Function | `build_feature_matrices` | 65 | Transform all splits using a training-fitted Phase 3 transformer. |
| Function | `evaluate_model_splits` | 111 | Evaluate an already-fitted model without changing model state. |
| Function | `run_individual_baselines` | 134 | Fit fixed baselines on train and evaluate unchanged on later splits. |
| Function | `save_baseline_results` | 209 | Save complete JSON and flat CSV baseline result records. |
| Function | `main` | 228 | Command-line entry point for this historical module. |

### `models/run_mulcas.py`

**Source:** `archive/history/src/models/run_mulcas.py`

**Purpose:** Train, run, and report the frozen independent MulCas reconstruction.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_mulcas_config` | 41 | Load and resolve the exact frozen MulCas reproduction configuration. |
| Function | `positive_probability` | 64 | Return probabilities or deterministic sigmoid-mapped decision margins. |
| Function | `build_mulcas_feature_matrices` | 87 | Build configured views with the leakage-safe developer feature appended. |
| Function | `train_mulcas_viewers` | 143 | Train residual cascade viewers and full-training voting viewers. |
| Function | `run_mulcas_reproduction` | 247 | Train and evaluate the paper-aligned frozen reconstruction. |
| Function | `save_mulcas_outputs` | 461 | Save machine-readable result, detailed traces, and Markdown report. |
| Function | `main` | 548 | Command-line entry point for this historical module. |

## 12. Package: `phase9`

Prospective Phase 9 diagnostics, state features, metrics, and protocol-gap analysis.

### `phase9/__init__.py`

**Source:** `archive/history/src/phase9/__init__.py`

**Purpose:** Phase 9 diagnostic reproduction analyses.

No public top-level classes or functions are declared in this module.

### `phase9/creator_oracle.py`

**Source:** `archive/history/src/phase9/creator_oracle.py`

**Purpose:** Non-deployable true-label online creator-state oracle for Phase 9A.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `online_true_label_creator_flags` | 19 | Flag known-positive creators, revealing target labels only after each block. |
| Function | `replace_creator_column` | 32 | Replace the final frozen creator column without touching static features. |
| Function | `run_creator_oracle` | 60 | Replay frozen MulCas with only the target creator state changed by the oracle. |

### `phase9/family_similarity.py`

**Source:** `archive/history/src/phase9/family_similarity.py`

**Purpose:** Strictly historical opcode-family similarity features for Phase 9C.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `FamilyStateResult` | 24 | Public historical class; inspect its source and matching tests before reuse. |
| Function | `transform_family_features` | 32 | Transform inputs for transform family features. |
| Function | `build_family_state` | 72 | Build labeled-training and label-free prequential family state. |

### `phase9/ngram_reproduction_audit.py`

**Source:** `archive/history/src/phase9/ngram_reproduction_audit.py`

**Purpose:** Label-independent N-gram dimensionality audit for Phase 9A.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `audit_ngram_dimensions` | 20 | Measure observed dimensions and integer document-frequency cutoffs. |

### `phase9/p2_anomaly_audit.py`

**Source:** `archive/history/src/phase9/p2_anomaly_audit.py`

**Purpose:** Deterministic descriptive audit of the 251 frozen P2 false positives.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `audit_p2_false_positives` | 24 | Attach exact hashes and summarize deterministic repeated FP groups. |

### `phase9/phase9_metrics.py`

**Source:** `archive/history/src/phase9/phase9_metrics.py`

**Purpose:** Frozen Phase 9B pooled metrics and paired uncertainty.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `prevalence_standardized_precision` | 12 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `recall_at_fpr_ceiling` | 17 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `pooled_metrics` | 32 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `paired_bootstrap` | 49 | Public historical function; inspect its implementation and matching tests before reuse. |

### `phase9/phase9a.py`

**Source:** `archive/history/src/phase9/phase9a.py`

**Purpose:** Command-line entry point for Phase 9A.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `main` | 11 | Command-line entry point for this historical module. |

### `phase9/phase9c.py`

**Source:** `archive/history/src/phase9/phase9c.py`

**Purpose:** Phase 9C deployment-available creator and code-family state experiment.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `sha256_file` | 39 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `load_config` | 48 | Load historical inputs or configuration for load config. |
| Function | `run_preflight` | 177 | Execute the declared historical workflow for run preflight. |
| Function | `run` | 368 | Public historical function; inspect its implementation and matching tests before reuse. |
| Function | `finalize_existing_outputs` | 486 | Regenerate derived diagnostics/report without fitting or scoring models. |
| Function | `main` | 518 | Command-line entry point for this historical module. |

### `phase9/protocol_gap.py`

**Source:** `archive/history/src/phase9/protocol_gap.py`

**Purpose:** Orchestration and evidence consolidation for Phase 9A.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `load_config` | 32 | Load historical inputs or configuration for load config. |
| Function | `verify_inputs` | 44 | Verify frozen inputs or integrity conditions for verify inputs. |
| Function | `build_protocol_evidence` | 99 | Build the declared historical representation or result for build protocol evidence. |
| Function | `run_phase9a` | 332 | Execute the declared historical workflow for run phase9a. |

### `phase9/state_features.py`

**Source:** `archive/history/src/phase9/state_features.py`

**Purpose:** Leakage-safe deployment-available creator state for Phase 9C.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Class | `CreatorStateResult` | 22 | Public historical class; inspect its source and matching tests before reuse. |
| Function | `transform_creator_features` | 54 | Transform inputs for transform creator features. |
| Function | `build_creator_state` | 66 | Build train-prequential and target-forward creator state. |

### `phase9/viewer_selection_oracle.py`

**Source:** `archive/history/src/phase9/viewer_selection_oracle.py`

**Purpose:** Fixed-candidate, non-deployable P5 viewer-selection oracle.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `build_candidate_matrices` | 35 | Fit only the six frozen Phase 5.5 views on P0-P4. |
| Function | `run_viewer_selection_oracle` | 74 | Evaluate the fixed 18-candidate pool on observed P5 labels. |

## 13. Package: `utils`

Repository-relative paths and deterministic random seeding.

### `utils/__init__.py`

**Source:** `archive/history/src/utils/__init__.py`

**Purpose:** Shared utility helpers.

No public top-level classes or functions are declared in this module.

### `utils/paths.py`

**Source:** `archive/history/src/utils/paths.py`

**Purpose:** Repository-relative filesystem paths.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `project_path` | 19 | Return a path resolved relative to the repository root. |

### `utils/seed.py`

**Source:** `archive/history/src/utils/seed.py`

**Purpose:** Deterministic random seed helpers for reproducible experiments.

| Type | Symbol | Line | Responsibility |
| --- | --- | ---: | --- |
| Function | `set_random_seed` | 13 | Seed supported random number generators and return the seed used. |

## 14. Tests and Supporting Evidence

The historical test suite is preserved separately under
`archive/history/tests/`. Tests are intentionally not cataloged function by
function here because they are verification evidence rather than reusable
source APIs. Match tests to modules by phase or feature name, for example:

| Source area | Representative tests |
| --- | --- |
| Data and chronology | `test_data_validation.py`, `test_temporal_split.py` |
| Features and leakage | `test_feature_leakage.py`, `test_structural_features.py`, `test_operand_features.py` |
| Baselines and MulCas | `test_baseline_models.py`, `test_mulcas.py`, `test_paper_aligned_protocol.py` |
| Phase 6 | `test_phase6a.py` through `test_phase6e.py` |
| Phase 7 | `test_phase7a.py` through `test_phase7e.py` |
| Phase 8 | `test_phase8*.py`, graph, creation, and sequence tests |
| Phase 9 | `test_phase9a.py`, `test_phase9c.py` |

## 15. Related Archive Documents

- `docs/ARCHIVE_DIRECTORY_GUIDE.md`: archive structure, trust boundaries,
  integrity verification, and restoration procedure.
- `archive/README.md`: concise archive landing page.
- `archive/CHRONOLOGY.md`: research phases in chronological order.
- `archive/history/TRACKED_FILE_MANIFEST.csv`: file-level phase and Git-blob index.
- `archive/history/README.md`: preserved README from the historical repository layout.
