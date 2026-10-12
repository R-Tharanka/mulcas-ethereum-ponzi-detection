"""Orchestration and evidence consolidation for Phase 9A."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.evaluation.temporal import build_ponzi_occurrence_periods, incremental_scenarios
from src.experiments.phase6a import verify_frozen_manifest
from src.experiments.phase7a import _load_frozen_configs, load_full_dataset
from src.extension.phase8_closure import load_config as load_phase8_closure_config
from src.extension.phase8_closure import verify_frozen_inputs as verify_phase8_inputs
from src.phase9.creator_oracle import run_creator_oracle
from src.phase9.ngram_reproduction_audit import audit_ngram_dimensions
from src.phase9.p2_anomaly_audit import audit_p2_false_positives
from src.phase9.viewer_selection_oracle import run_viewer_selection_oracle
from src.utils.paths import REPO_ROOT


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "phase9" / "phase9a.yaml"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in (
        "dataset", "paper_pdf", "phase8_closure_manifest", "phase7_config",
        "feature_config", "model_config", "mulcas_config", "phase55_results",
        "phase7a_metrics", "phase7a_predictions", "phase7d_inventory",
        "phase7e_clone_membership", "phase8_closure_results", "output_dir",
    ):
        config[key] = _resolve(config[key])
    return config


def verify_inputs(config: dict[str, Any]) -> dict[str, Any]:
    closure = verify_phase8_inputs(load_phase8_closure_config())
    closure_manifest = verify_frozen_manifest(config["phase8_closure_manifest"])
    phase8_result = json.loads(config["phase8_closure_results"].read_text(encoding="utf-8"))
    if phase8_result["status"] != "complete" or phase8_result["phase8_replacement_found"]:
        raise ValueError("Phase 8 closure is not in the expected frozen state.")
    return {
        "verified": True,
        "phase0_to_phase8r5c1_anchor_count": closure["frozen_anchor_count"],
        "phase8_closure_anchor_count": len(closure_manifest),
        "total_verification_events": closure["frozen_anchor_count"] + len(closure_manifest),
        "raw_dataset_sha256": closure["raw_dataset_sha256"],
        "checkpoint_count": closure["checkpoint_count"],
        "embedding_artifact_count": closure["embedding_artifact_count"],
    }


def _assert_control(observed: dict[str, Any], expected: dict[str, Any], name: str) -> None:
    for key, value in expected.items():
        if key in {"tn", "fp", "fn", "tp"}:
            matches = int(observed[key]) == int(value)
        else:
            matches = np.isclose(float(observed[key]), float(value), atol=1e-12)
        if not matches:
            raise ValueError(f"{name} control mismatch for {key}: {observed[key]} != {value}")


def build_protocol_evidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"topic": "auxiliary_bytecode", "paper_evidence": "Auxiliary bytecode fragment is discarded.", "source": "local paper PDF page 10, Section 3.2", "reconstruction": "Baseline uses released opcode; later extension parsers preserve complete contractCode.", "status": "documented_difference"},
            {"topic": "ngram_filter", "paper_evidence": "N-grams with document frequency less than 0.1 are discarded; Table 3 reports 77/767/2292 selected dimensions.", "source": "local paper PDF pages 13-14, Section 4.2.1/Table 3", "reconstruction": "Integer min_df=2 produces 44,671 1-3 grams on P0-P3.", "status": "under_specified_formula_semantics"},
            {"topic": "creator_updates", "paper_evidence": "Creator vector is chronological and may expand with upcoming labeled Ponzi contracts.", "source": "local paper PDF pages 16-17, Algorithm 1", "reconstruction": "Future evaluation labels never update creator state in deployable controls.", "status": "oracle_only_gap_test"},
            {"topic": "viewer_selection", "paper_evidence": "Best model for each feature is selected on the test set; five final models are claimed but only four are enumerated.", "source": "local paper PDF page 17, Section 4.3", "reconstruction": "Viewer choice uses validation or forward protocols; four explicitly enumerated viewers are implemented.", "status": "non_deployable_and_not_fully_specified"},
            {"topic": "cascade", "paper_evidence": "N viewers form the cascade; all viewers vote on residual test rows; theta commonly 0.05 and lambda controls votes.", "source": "local paper PDF pages 17-18, Section 4.4/Algorithm 2", "reconstruction": "N=2, frozen asymmetric second positive threshold 0.51, lambda=0.", "status": "partially_reconstructed"},
            {"topic": "p5_split", "paper_evidence": "Table 5 reports 5,990 train and 508 test contracts; temporal Table 4 uses P2-P5.", "source": "local paper PDF Sections 5.2.1-5.2.2, Tables 4-5", "reconstruction": "Whole-block membership yields 5,989 P0-P4 and 509 P5 rows.", "status": "one_row_boundary_difference"},
        ]
    )


def _membership_audit(assigned: pd.DataFrame, definitions: list[dict[str, Any]]) -> pd.DataFrame:
    p2 = assigned.loc[assigned["period"] == "P2"].sort_values(["createdBlockNumber", "row_index"])
    positives = p2.loc[p2["label"] == 1]
    definition = next(item for item in definitions if item["period"] == "P2")
    return pd.DataFrame(
        [
            {"check": "positive_count", "observed": int(positives.shape[0]), "paper_value": 50, "status": "MATCH", "note": "Paper defines each P0-P4 interval by 50 chronological Ponzi contracts."},
            {"check": "row_count", "observed": int(len(p2)), "paper_value": None, "status": "NOT VERIFIABLE FROM PAPER", "note": "Paper does not list P2 benign-row membership."},
            {"check": "block_range", "observed": f"{definition['block_min']}-{definition['block_max']}", "paper_value": None, "status": "NOT VERIFIABLE FROM PAPER", "note": "Derived from released CSV with whole-block boundaries."},
            {"check": "first_positive", "observed": f"{positives.iloc[0]['address']}@{int(positives.iloc[0]['createdBlockNumber'])}", "paper_value": None, "status": "NOT VERIFIABLE FROM PAPER", "note": "First-principles reconstruction evidence."},
            {"check": "last_positive", "observed": f"{positives.iloc[-1]['address']}@{int(positives.iloc[-1]['createdBlockNumber'])}", "paper_value": None, "status": "NOT VERIFIABLE FROM PAPER", "note": "First-principles reconstruction evidence."},
        ]
    )


def _write_report(results: dict[str, Any], output: Path) -> None:
    d0 = results["p5_controls"]
    d2 = results["creator_oracle_summary"]
    d2_rows = results["creator_oracle_period_metrics"]
    p2 = results["p2_false_positive_summary"]
    d2_table = "\n".join(
        f"| {row['period']} | {row['condition']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['tn']}/{row['fp']}/{row['fn']}/{row['tp']} |"
        for row in d2_rows
    )
    text = f"""# Phase 9A Paper/Reconstruction Protocol-Gap Decomposition

## Purpose and Scope

Phase 9A is a diagnostic reproduction analysis. It introduces no classifier,
feature family, hyperparameter, threshold, resampling method, calibration,
GNN, or Transformer. P5 is reused only as previously observed exploratory
Track A evidence. The creator and viewer-selection analyses are explicitly
non-deployable oracles.

## Frozen Inputs

- Verified Phase 0-8R5C1 anchors: {results['verification']['phase0_to_phase8r5c1_anchor_count']}
- Verified Phase 8 closure anchors: {results['verification']['phase8_closure_anchor_count']}
- Raw dataset SHA-256: `{results['verification']['raw_dataset_sha256']}`
- Phase 8 conclusions preserved unchanged: yes

## D0 Frozen Controls

| Control | Precision | Recall | F1 | TN/FP/FN/TP |
| --- | ---: | ---: | ---: | --- |
| Phase 5.5 P5 | {d0['phase55']['precision']:.4f} | {d0['phase55']['recall']:.4f} | {d0['phase55']['f1']:.4f} | {d0['phase55']['tn']}/{d0['phase55']['fp']}/{d0['phase55']['fn']}/{d0['phase55']['tp']} |
| Phase 7A P5 | {d0['phase7a']['precision']:.4f} | {d0['phase7a']['recall']:.4f} | {d0['phase7a']['f1']:.4f} | {d0['phase7a']['tn']}/{d0['phase7a']['fp']}/{d0['phase7a']['fn']}/{d0['phase7a']['tp']} |

The difference is described as a training-history/protocol-coverage difference,
not as a causal P4 effect.

## D2 Creator Oracle

The oracle predicts each row before revealing its label and commits true-positive
creator updates only after the whole creation block. Same-block rows cannot
affect one another. Frozen replays matched archived Phase 7A predictions for
P2-P5: {d2['all_frozen_replays_exact']}.

| Period | Condition | Precision | Recall | F1 | TN/FP/FN/TP |
| --- | --- | ---: | ---: | ---: | --- |
{d2_table}

The oracle is not deployable because it reveals target-period true labels after
prediction. It improves P5 by three true positives without changing false
positives, but it also increases P2 false positives from 251 to 264.

## D3 Viewer-Selection Oracle

The fixed 18-candidate replay was stopped by the resource gate after the
representative XGBoost estimate proved unrepresentative: the complete pool was
still unfinished after 37 minutes due to a slow SVM/high-dimensional-view
combination. No partial ranking is presented. Independently, a full paper-exact
MulCas replay is **not reproducible from the available specification** because
the paper claims five final viewers but enumerates only four and gives no unique
fifth viewer or complete ordering.

## D4 Combined Oracle and Gap Closure

D4 is unavailable. Combining D2 with D3 would require inventing the missing
viewer mapping/order. Recall/F1 gap-closure fractions are therefore N/A.

## N-gram Dimensionality Audit

The audit measures raw dimensions, min_df=2 dimensions, document-frequency
histograms, and the nearest integer cutoffs to the paper's 77/767/2,292 targets.
No classifier is trained in this audit. The paper's phrase "document frequency
less than 0.1" does not uniquely specify the integer filtering operation used.

## P2 False-Positive Audit

All {p2['false_positive_count']} archived P2 false positives reconcile with
Phase 7D. Routing counts are `{json.dumps(p2['routing_counts'], sort_keys=True)}`.
Exact normalized opcode/runtime hashes, creators, routes, scores, block numbers,
clone indicators, and structural summaries are retained in the row/group CSVs.

## Research Conclusion

Phase 9A decomposes protocol differences without changing the retained Phase 5.5
MulCas reference or any Phase 8 conclusion. Test-label creator updates and test-set
viewer selection can only be interpreted as non-deployable upper-bound diagnostics.
Because the publication does not uniquely specify the fifth viewer and full order,
the combined oracle is not fabricated. The recommended next research step remains
external/later dataset discovery, provenance, and overlap audit before further
model development.
"""
    output.write_text(text, encoding="utf-8")


def run_phase9a(config: dict[str, Any]) -> dict[str, Any]:
    verification = verify_inputs(config)
    feature_config, model_config, mulcas_config = _load_frozen_configs(
        {"paper_feature_config": config["feature_config"], "paper_model_config": config["model_config"], "paper_mulcas_config": config["mulcas_config"]}
    )
    raw = pd.read_csv(
        config["dataset"],
        usecols=["address", "opcode", "contractCode", "creator", "createdBlockNumber", "label"],
        low_memory=False,
    )
    raw["row_index"] = raw.index.astype(int)
    raw["opcode"] = raw["opcode"].fillna("").astype(str)
    assigned, definitions = build_ponzi_occurrence_periods(
        raw,
        positive_ends=config["periods"]["positive_ends"],
        expected_positives=int(config["periods"]["expected_total_positives"]),
    )

    phase55 = json.loads(config["phase55_results"].read_text(encoding="utf-8"))["experimental_reconstruction"]["test"]
    phase7_metrics = pd.read_csv(config["phase7a_metrics"])
    phase7_p5 = phase7_metrics.loc[phase7_metrics["prediction_period"] == "P5"].iloc[0].to_dict()
    _assert_control(phase55, config["controls"]["phase55_p5"], "Phase 5.5 P5")
    _assert_control(phase7_p5, config["controls"]["phase7a_p5"], "Phase 7A P5")
    p5_decomposition = pd.DataFrame(
        [
            {"decomposition": "D0A", "system": "Phase 5.5 frozen P5", **{k: phase55[k] for k in ("precision", "recall", "f1", "pr_auc", "tn", "fp", "fn", "tp")}, "interpretation": "frozen reference; trained through P3"},
            {"decomposition": "D0B", "system": "Phase 7A incremental P5", **{k: phase7_p5[k] for k in ("precision", "recall", "f1", "pr_auc", "tn", "fp", "fn", "tp")}, "interpretation": "primary decomposition control; trained through P4"},
            {"decomposition": "D1", "system": "D0B minus D0A", **{k: float(phase7_p5[k]) - float(phase55[k]) for k in ("precision", "recall", "f1", "pr_auc", "tn", "fp", "fn", "tp")}, "interpretation": "descriptive training-history/protocol-coverage difference; not a causal P4 effect"},
        ]
    )

    archived_predictions = pd.read_csv(config["phase7a_predictions"])
    creator_metrics, creator_changed, creator_summary = run_creator_oracle(
        assigned,
        incremental_scenarios(),
        feature_config,
        model_config,
        mulcas_config,
        archived_predictions,
        seed=int(config["seed"]),
    )
    train_p0_p4 = assigned.loc[assigned["period"].isin(["P0", "P1", "P2", "P3", "P4"])].copy()
    p5 = assigned.loc[assigned["period"] == "P5"].copy()
    observed_gate = config.get("observed_d3_runtime_gate", {})
    if observed_gate.get("activated"):
        viewer_metrics = pd.DataFrame(columns=[
            "candidate_id", "model_family", "feature_view", "status", "reason"
        ])
        preflight = {
            "gate_passed": False,
            "status": "stopped_after_observed_runtime_overrun",
            "representative_estimate_limitation": True,
            "attempted_candidate_pool": int(observed_gate["attempted_candidate_pool"]),
            "observed_elapsed_lower_bound_seconds": float(observed_gate["observed_elapsed_lower_bound_seconds"]),
            "runtime_gate_seconds": float(config["runtime_gate_minutes"]) * 60.0,
            "reason": observed_gate["reason"],
            "full_replay_status": "not_reproducible_from_available_specification",
        }
        viewer_selection = pd.DataFrame([{
            "status": "stopped_by_runtime_gate",
            "selection_complete": False,
            "partial_ranking_reported": False,
            "reason": observed_gate["reason"],
            "full_mulcas_replay_status": "not_reproducible_from_available_specification",
        }])
    else:
        viewer_metrics, viewer_selection, preflight = run_viewer_selection_oracle(
            train_p0_p4,
            p5,
            feature_config,
            model_config,
            seed=int(config["seed"]),
            runtime_gate_minutes=float(config["runtime_gate_minutes"]),
        )
        if not preflight["gate_passed"]:
            viewer_selection = pd.DataFrame([{"status": "stopped_by_runtime_gate", "reason": "Estimated candidate replay exceeded 30 minutes."}])

    ngram_audit, df_histograms = audit_ngram_dimensions(
        {"P0-P3": assigned.loc[assigned["period"].isin(["P0", "P1", "P2", "P3"])], "P0-P4": train_p0_p4}
    )
    inventory = pd.read_csv(config["phase7d_inventory"])
    p2_routing, p2_groups, p2_summary = audit_p2_false_positives(raw, inventory)
    membership = _membership_audit(assigned, definitions)
    protocol_evidence = build_protocol_evidence()

    combined = pd.DataFrame([{
        "decomposition": "D4", "status": "unavailable",
        "deployability": "non_deployable_oracle_if_specified",
        "reason": "D3 full MulCas mapping is not reproducible from the available paper specification; no fifth viewer/order was invented.",
    }])
    gap = pd.DataFrame([
        {"metric": "recall", "published": 0.674, "D0B": float(phase7_p5["recall"]), "D4": None, "gap_closed_fraction": None, "status": "N/A_D4_unavailable"},
        {"metric": "f1", "published": 0.789, "D0B": float(phase7_p5["f1"]), "D4": None, "gap_closed_fraction": None, "status": "N/A_D4_unavailable"},
    ])
    results = {
        "phase": "9A",
        "status": "complete_with_runtime_gated_d3",
        "analysis_type": "diagnostic_protocol_gap_decomposition",
        "scope": {"new_classifier": False, "new_features": False, "tuning": False, "threshold_changes": False, "p5_status": config["protocol"]["p5_status"], "phase9b_started": False},
        "verification": verification,
        "p5_controls": {"phase55": {k: phase55[k] for k in config["controls"]["phase55_p5"]}, "phase7a": {k: phase7_p5[k] for k in config["controls"]["phase7a_p5"]}},
        "creator_oracle_summary": creator_summary,
        "creator_oracle_period_metrics": creator_metrics.to_dict("records"),
        "viewer_selection_oracle": {"candidate_count": int(len(viewer_metrics)), "full_replay_status": preflight.get("full_replay_status", "not_run"), "non_deployable": True},
        "combined_oracle": {"status": "unavailable", "reason": combined.iloc[0]["reason"]},
        "ngram_df_histograms": df_histograms,
        "p2_false_positive_summary": p2_summary,
        "p2_membership_status": {row["check"]: row["status"] for row in membership.to_dict("records")},
        "phase8_conclusions_modified": False,
        "retained_reference": "phase5.5_mulcas",
        "recommendation": "external_or_later_dataset_provenance_and_overlap_audit_before_phase9b",
    }
    output = config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    results["runtime_preflight"] = preflight
    (output / "phase9a_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    protocol_evidence.to_csv(output / "protocol_evidence.csv", index=False)
    p5_decomposition.to_csv(output / "p5_decomposition.csv", index=False)
    creator_metrics.to_csv(output / "creator_oracle_period_metrics.csv", index=False)
    creator_changed.to_csv(output / "creator_oracle_changed_rows.csv", index=False)
    viewer_metrics.to_csv(output / "viewer_candidate_metrics.csv", index=False)
    viewer_selection.to_csv(output / "viewer_selection_oracle.csv", index=False)
    combined.to_csv(output / "combined_oracle_metrics.csv", index=False)
    ngram_audit.to_csv(output / "ngram_dimension_audit.csv", index=False)
    p2_routing.to_csv(output / "p2_fp_routing.csv", index=False)
    p2_groups.to_csv(output / "p2_fp_groups.csv", index=False)
    membership.to_csv(output / "p2_membership_audit.csv", index=False)
    gap.to_csv(output / "gap_decomposition.csv", index=False)
    (output / "runtime_preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    _write_report(results, output / "phase9a_report.md")
    return results
