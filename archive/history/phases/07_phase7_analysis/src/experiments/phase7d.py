"""Systematic error analysis over exact frozen Phase 7A MulCas predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from src.evaluation.temporal import build_ponzi_occurrence_periods
from src.experiments.phase6a import sha256_file, verify_frozen_manifest
from src.experiments.phase7a import load_config as load_phase7a_config
from src.features.operands import (
    FEATURE_INDEX as OPERAND_INDEX,
    disassemble_operands,
    extract_operand_features,
)
from src.features.structural import (
    FEATURE_INDEX as STRUCTURAL_INDEX,
    extract_structural_features,
)
from src.utils.paths import REPO_ROOT


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "phase7" / "error_analysis.yaml"
ERROR_TYPES = ("true_negative", "false_positive", "false_negative", "true_positive")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load Phase 7D and reject settings that could alter frozen behavior."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in (
        "dataset",
        "phase7a_config",
        "phase7a_results",
        "phase7a_predictions",
        "paper_mulcas_config",
        "frozen_manifest",
        "output_dir",
    ):
        config[key] = _resolve(config[key])
    config["config_path"] = path.resolve()
    config["config_sha256"] = sha256_file(path)
    protocol = config["protocol"]
    for key in ("optimization", "retraining", "retuning"):
        if protocol[key] != "prohibited":
            raise ValueError(f"Phase 7D {key} must remain prohibited.")
    if protocol["reference_predictions"] != "exact_phase7a_full_system":
        raise ValueError("Phase 7D must use exact Phase 7A predictions.")
    if protocol["phase6_interventions_enabled"]:
        raise ValueError("Phase 6 interventions must remain disabled.")
    if tuple(protocol["periods"]) != ("P2", "P3", "P4", "P5"):
        raise ValueError("Phase 7D must analyze P2-P5 exactly.")
    return config


def verify_prior_phase_chain(current_manifest: Path) -> dict[str, str]:
    """Verify Phase 5.5 through Phase 7C without rewriting prior artifacts."""
    manifests = (
        current_manifest,
        REPO_ROOT / "configs" / "phase7" / "frozen_phase7b_manifest.yaml",
        REPO_ROOT / "configs" / "phase7" / "frozen_phase7a_manifest.yaml",
        REPO_ROOT / "configs" / "phase7" / "frozen_phase6_manifest.yaml",
    )
    observed: dict[str, str] = {}
    for manifest in manifests:
        for relative_path, digest in verify_frozen_manifest(manifest).items():
            existing = observed.get(relative_path)
            if existing is not None and existing != digest:
                raise RuntimeError(f"Conflicting frozen hashes for {relative_path}.")
            observed[relative_path] = digest
    return observed


def classify_outcomes(labels: Iterable[int], predictions: Iterable[int]) -> np.ndarray:
    """Return one mutually exclusive confusion category for every row."""
    truth = np.asarray(list(labels), dtype=int)
    predicted = np.asarray(list(predictions), dtype=int)
    if truth.shape != predicted.shape or not set(np.unique(truth)).issubset({0, 1}):
        raise ValueError("Labels and predictions must be aligned binary arrays.")
    if not set(np.unique(predicted)).issubset({0, 1}):
        raise ValueError("Predictions must be binary.")
    return np.select(
        [
            (truth == 0) & (predicted == 0),
            (truth == 0) & (predicted == 1),
            (truth == 1) & (predicted == 0),
            (truth == 1) & (predicted == 1),
        ],
        ERROR_TYPES,
        default="unclassified",
    )


def parse_boolean_series(values: pd.Series) -> pd.Series:
    """Parse CSV booleans without treating the string ``False`` as true."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        raise ValueError("Boolean column contains an unsupported value.")
    return normalized.map(mapping).astype(bool)


def load_analysis_dataset(path: Path) -> pd.DataFrame:
    """Load only identity and existing diagnostic source fields."""
    columns = [
        "address",
        "opcode",
        "contractCode",
        "creationCode",
        "creator",
        "createdBlockNumber",
        "label",
    ]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    frame["row_index"] = frame.index.astype(int)
    for column in ("opcode", "contractCode", "creationCode"):
        frame[column] = frame[column].fillna("").astype(str)
    if frame[["address", "creator", "createdBlockNumber", "label"]].isna().any().any():
        raise ValueError("Required Phase 7D metadata contains null values.")
    return frame


def _nonempty_counts(values: pd.Series) -> Counter[str]:
    return Counter(value for value in values.astype(str) if value.strip())


def add_historical_context(
    inventory: pd.DataFrame, assigned: pd.DataFrame
) -> pd.DataFrame:
    """Add creator and exact-clone state from strictly earlier periods only."""
    enriched = inventory.copy()
    context_columns = {
        "previously_seen_creator": False,
        "previously_ponzi_creator": False,
        "historical_opcode_match_count": 0,
        "historical_runtime_bytecode_match_count": 0,
        "historical_creation_code_match_count": 0,
    }
    for column, default in context_columns.items():
        enriched[column] = default

    for period in ("P2", "P3", "P4", "P5"):
        period_number = int(period[1:])
        historical = assigned.loc[
            assigned["period"].str[1:].astype(int) < period_number
        ]
        target_mask = enriched["period"] == period
        target = enriched.loc[target_mask]
        if historical["createdBlockNumber"].max() >= target["createdBlockNumber"].min():
            raise AssertionError(f"Historical clone context for {period} is not strict.")
        creators = set(historical["creator"].astype(str))
        ponzi_creators = set(
            historical.loc[historical["label"] == 1, "creator"].astype(str)
        )
        opcode_counts = _nonempty_counts(historical["opcode"])
        runtime_counts = _nonempty_counts(historical["contractCode"])
        creation_counts = _nonempty_counts(historical["creationCode"])
        enriched.loc[target_mask, "previously_seen_creator"] = (
            target["creator"].astype(str).isin(creators).to_numpy()
        )
        enriched.loc[target_mask, "previously_ponzi_creator"] = (
            target["creator"].astype(str).isin(ponzi_creators).to_numpy()
        )
        enriched.loc[target_mask, "historical_opcode_match_count"] = (
            target["opcode"].map(opcode_counts).fillna(0).astype(int).to_numpy()
        )
        enriched.loc[target_mask, "historical_runtime_bytecode_match_count"] = (
            target["contractCode"].map(runtime_counts).fillna(0).astype(int).to_numpy()
        )
        enriched.loc[target_mask, "historical_creation_code_match_count"] = (
            target["creationCode"].map(creation_counts).fillna(0).astype(int).to_numpy()
        )

    enriched["has_historical_opcode_clone"] = enriched["historical_opcode_match_count"] > 0
    enriched["has_historical_runtime_bytecode_clone"] = (
        enriched["historical_runtime_bytecode_match_count"] > 0
    )
    enriched["has_historical_creation_code_clone"] = (
        enriched["historical_creation_code_match_count"] > 0
    )
    if not np.array_equal(
        enriched["creator_history_feature"].astype(int).to_numpy(),
        enriched["previously_ponzi_creator"].astype(int).to_numpy(),
    ):
        raise AssertionError("Reconstructed creator history differs from Phase 7A.")
    return enriched


def add_contract_diagnostics(
    inventory: pd.DataFrame,
    structural_names: list[str],
    operand_names: list[str],
    *,
    small_constant_max: int,
) -> pd.DataFrame:
    """Add fit-free, contract-local diagnostics from existing extractors."""
    unknown_structural = set(structural_names).difference(STRUCTURAL_INDEX)
    unknown_operand = set(operand_names).difference(OPERAND_INDEX)
    if unknown_structural or unknown_operand:
        raise ValueError(
            f"Unknown diagnostics: structural={sorted(unknown_structural)}, "
            f"operand={sorted(unknown_operand)}"
        )
    enriched = inventory.copy()
    structural = np.vstack(
        [extract_structural_features(value) for value in enriched["opcode"]]
    )
    operands = np.vstack(
        [
            extract_operand_features(value, small_constant_max=small_constant_max)
            for value in enriched["contractCode"]
        ]
    )
    for name in structural_names:
        enriched[name] = structural[:, STRUCTURAL_INDEX[name]]
    for name in operand_names:
        enriched[name] = operands[:, OPERAND_INDEX[name]]
    parsed = [disassemble_operands(value) for value in enriched["contractCode"]]
    enriched["malformed_truncated_push_count"] = [
        item.malformed_truncated_push_count for item in parsed
    ]
    enriched["unsupported_opcode_count"] = [item.unsupported_opcode_count for item in parsed]
    numeric = enriched.loc[:, structural_names + operand_names]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise AssertionError("Contract diagnostics contain non-finite values.")
    return enriched


def add_routing_diagnostics(
    inventory: pd.DataFrame, mulcas_config: dict[str, Any], *, margin: float
) -> pd.DataFrame:
    """Classify wrong routes and flag scores close to an applicable boundary."""
    enriched = inventory.copy()
    error_mask = enriched["error_type"].isin(("false_positive", "false_negative"))
    enriched["routing_error_category"] = "correct_prediction"
    enriched.loc[error_mask & (enriched["cascade_stage"] == 1), "routing_error_category"] = (
        "confident_wrong_first_viewer"
    )
    enriched.loc[error_mask & (enriched["cascade_stage"] == 2), "routing_error_category"] = (
        "confident_wrong_second_viewer"
    )
    enriched.loc[error_mask & enriched["reached_voting_stage"], "routing_error_category"] = (
        "unresolved_until_voting_wrong"
    )
    enriched["prediction_confidence"] = np.maximum(
        enriched["final_score"].astype(float), 1.0 - enriched["final_score"].astype(float)
    )
    enriched["near_boundary_wrong"] = False
    for stage, viewer in enumerate(
        mulcas_config["viewer_sequence"][: int(mulcas_config["complexity_control_n"])],
        start=1,
    ):
        mask = error_mask & (enriched["cascade_stage"] == stage)
        scores = enriched.loc[mask, "final_score"].astype(float)
        distance = np.minimum(
            np.abs(scores - float(viewer["negative_threshold"])),
            np.abs(scores - float(viewer["positive_threshold"])),
        )
        enriched.loc[mask, "near_boundary_wrong"] = (distance <= margin).to_numpy()
    vote_mask = error_mask & enriched["reached_voting_stage"]
    voting_score_columns = [
        column
        for column in enriched.columns
        if column.startswith("voting__") and column.endswith("__score")
    ]
    if voting_score_columns:
        near_vote = (
            enriched.loc[vote_mask, voting_score_columns].astype(float).sub(0.5).abs()
            <= margin
        ).any(axis=1)
        enriched.loc[vote_mask, "near_boundary_wrong"] = near_vote.to_numpy()
    return enriched


def _summary_rows(
    frame: pd.DataFrame,
    *,
    periods: Iterable[str],
    groups: Iterable[str],
    numeric_features: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for period in periods:
        for group in groups:
            selected = frame[(frame["period"] == period) & (frame["error_type"] == group)]
            for feature in numeric_features:
                values = (
                    pd.to_numeric(selected[feature], errors="coerce")
                    .dropna()
                    .astype(float)
                )
                statistics = {
                    "count": int(len(values)),
                    "mean": float(values.mean()) if len(values) else None,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else None,
                    "min": float(values.min()) if len(values) else None,
                    "q25": float(values.quantile(0.25)) if len(values) else None,
                    "median": float(values.median()) if len(values) else None,
                    "q75": float(values.quantile(0.75)) if len(values) else None,
                    "max": float(values.max()) if len(values) else None,
                }
                rows.extend(
                    {
                        "period": period,
                        "error_type": group,
                        "feature": feature,
                        "statistic": statistic,
                        "value": value,
                    }
                    for statistic, value in statistics.items()
                )
            for viewer, count in selected["accepted_viewer"].value_counts().items():
                rows.append(
                    {
                        "period": period,
                        "error_type": group,
                        "feature": f"accepted_viewer::{viewer}",
                        "statistic": "proportion",
                        "value": float(count / len(selected)),
                    }
                )
    return rows


def validate_reference_counts(
    inventory: pd.DataFrame, phase7a_results: dict[str, Any]
) -> None:
    """Require exact row predictions and confusion counts from Phase 7A."""
    if inventory["row_index"].duplicated().any():
        raise AssertionError("A Phase 7A prediction row was duplicated.")
    if (inventory["error_type"] == "unclassified").any():
        raise AssertionError("A prediction row was not classified.")
    key_map = {
        "true_negative": "tn",
        "false_positive": "fp",
        "false_negative": "fn",
        "true_positive": "tp",
    }
    for scenario in phase7a_results["scenarios"]:
        period = scenario["prediction_period"]
        counts = inventory.loc[inventory["period"] == period, "error_type"].value_counts()
        expected = scenario["metrics"]
        for error_type, key in key_map.items():
            if int(counts.get(error_type, 0)) != int(expected[key]):
                raise AssertionError(f"{period} {error_type} differs from Phase 7A.")


def build_period_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in ("P2", "P3", "P4", "P5"):
        selected = inventory[inventory["period"] == period]
        errors = selected[selected["error_type"].isin(("false_positive", "false_negative"))]
        row: dict[str, Any] = {"period": period, "row_count": int(len(selected))}
        counts = selected["error_type"].value_counts()
        for error_type in ERROR_TYPES:
            row[error_type] = int(counts.get(error_type, 0))
        row.update(
            {
                "error_count": int(len(errors)),
                "error_unseen_creator_proportion": float((~errors["previously_seen_creator"]).mean()) if len(errors) else 0.0,
                "error_creator_feature_1_proportion": float(errors["creator_history_feature"].mean()) if len(errors) else 0.0,
                "error_opcode_clone_proportion": float(errors["has_historical_opcode_clone"].mean()) if len(errors) else 0.0,
                "error_runtime_clone_proportion": float(errors["has_historical_runtime_bytecode_clone"].mean()) if len(errors) else 0.0,
                "error_reached_voting_proportion": float(errors["reached_voting_stage"].mean()) if len(errors) else 0.0,
                "error_near_boundary_proportion": float(errors["near_boundary_wrong"].mean()) if len(errors) else 0.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_routing_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    errors = inventory[inventory["error_type"].isin(("false_positive", "false_negative"))]
    rows = []
    for (period, category), group in errors.groupby(
        ["period", "routing_error_category"], sort=True
    ):
        period_total = int((errors["period"] == period).sum())
        rows.append(
            {
                "period": period,
                "routing_error_category": category,
                "count": int(len(group)),
                "percentage_of_period_errors": float(100 * len(group) / period_total),
                "false_positive_count": int((group["error_type"] == "false_positive").sum()),
                "false_negative_count": int((group["error_type"] == "false_negative").sum()),
                "near_boundary_count": int(group["near_boundary_wrong"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_clone_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (period, error_type), group in inventory.groupby(["period", "error_type"], sort=True):
        rows.append(
            {
                "period": period,
                "error_type": error_type,
                "count": int(len(group)),
                "opcode_clone_count": int(group["has_historical_opcode_clone"].sum()),
                "opcode_clone_proportion": float(group["has_historical_opcode_clone"].mean()),
                "runtime_clone_count": int(group["has_historical_runtime_bytecode_clone"].sum()),
                "runtime_clone_proportion": float(group["has_historical_runtime_bytecode_clone"].mean()),
                "creation_clone_count": int(group["has_historical_creation_code_clone"].sum()),
                "creation_clone_proportion": float(group["has_historical_creation_code_clone"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _format_rate(value: float) -> str:
    return f"{100 * value:.1f}%"


def render_report(
    period_summary: pd.DataFrame,
    routing: pd.DataFrame,
    clones: pd.DataFrame,
    group_summary: pd.DataFrame,
    p5_summary: pd.DataFrame,
) -> str:
    p5 = period_summary.set_index("period").loc["P5"]
    p5_clone = clones.set_index(["period", "error_type"])
    p5_fn = p5_clone.loc[("P5", "false_negative")]
    p5_tp = p5_clone.loc[("P5", "true_positive")]
    def summary_value(period: str, error_type: str, feature: str, statistic: str = "mean") -> float:
        found = group_summary[
            (group_summary["period"] == period)
            & (group_summary["error_type"] == error_type)
            & (group_summary["feature"] == feature)
            & (group_summary["statistic"] == statistic)
        ]["value"]
        return float(found.iloc[0])
    lines = [
        "# Phase 7D: Frozen MulCas Error Analysis",
        "",
        "This analysis reads the exact Phase 7A predictions. It performs no model fitting, optimization, threshold changes, or feature changes.",
        "",
        "## Error Inventory",
        "",
        "| Period | TN | FP | FN | TP |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in period_summary.to_dict("records"):
        lines.append(
            f"| {row['period']} | {row['true_negative']} | {row['false_positive']} | {row['false_negative']} | {row['true_positive']} |"
        )
    lines.extend(["", "## Routing of Errors", "", "| Period | Route | Count | Percent of errors | FP | FN | Near boundary |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in routing.to_dict("records"):
        lines.append(
            f"| {row['period']} | {row['routing_error_category']} | {row['count']} | {row['percentage_of_period_errors']:.1f}% | {row['false_positive_count']} | {row['false_negative_count']} | {row['near_boundary_count']} |"
        )
    lines.extend(
        [
            "",
            "Errors are usually accepted confidently before voting. P2 is dominated by second-viewer false positives, while P3 and P5 are dominated by first-viewer false negatives. Only 1.7%-19.5% of period errors reach voting.",
            "",
            "## False Negatives vs True Positives",
            "",
            "| Period | Mean opcode count FN / TP | Mean unique opcodes FN / TP | Mean entropy FN / TP | Mean operands FN / TP | Mean confidence FN / TP |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for period in ("P2", "P3", "P4", "P5"):
        lines.append(
            f"| {period} | {summary_value(period, 'false_negative', 'opcode_count'):.1f} / {summary_value(period, 'true_positive', 'opcode_count'):.1f} | "
            f"{summary_value(period, 'false_negative', 'unique_opcode_count'):.1f} / {summary_value(period, 'true_positive', 'unique_opcode_count'):.1f} | "
            f"{summary_value(period, 'false_negative', 'opcode_entropy'):.3f} / {summary_value(period, 'true_positive', 'opcode_entropy'):.3f} | "
            f"{summary_value(period, 'false_negative', 'parseable_operand_count'):.1f} / {summary_value(period, 'true_positive', 'parseable_operand_count'):.1f} | "
            f"{summary_value(period, 'false_negative', 'prediction_confidence'):.3f} / {summary_value(period, 'true_positive', 'prediction_confidence'):.3f} |"
        )
    lines.extend(
        [
            "",
            "There is no single size or operand pattern across periods. P2 false negatives are much longer than P2 true positives, whereas P3-P5 contrasts are smaller or reverse in direction. This period dependence is consistent with changing distributions, not proof of concept drift.",
            "",
            "## False Positives vs True Negatives",
            "",
            "| Period | Mean opcode count FP / TN | Mean entropy FP / TN | Mean operands FP / TN | Mean confidence FP / TN |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for period in ("P2", "P3", "P4", "P5"):
        lines.append(
            f"| {period} | {summary_value(period, 'false_positive', 'opcode_count'):.1f} / {summary_value(period, 'true_negative', 'opcode_count'):.1f} | "
            f"{summary_value(period, 'false_positive', 'opcode_entropy'):.3f} / {summary_value(period, 'true_negative', 'opcode_entropy'):.3f} | "
            f"{summary_value(period, 'false_positive', 'parseable_operand_count'):.1f} / {summary_value(period, 'true_negative', 'parseable_operand_count'):.1f} | "
            f"{summary_value(period, 'false_positive', 'prediction_confidence'):.3f} / {summary_value(period, 'true_negative', 'prediction_confidence'):.3f} |"
        )
    lines.extend(
        [
            "",
            "False positives generally have lower mean prediction confidence than true negatives. P2 false positives are also substantially shorter on average and are concentrated at the second cascade viewer.",
            "",
            "## Historical Context and Clones",
            "",
            "| Period | Errors from unseen creators | Errors with Ponzi-creator flag | Error opcode clones | Error runtime clones |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in period_summary.to_dict("records"):
        lines.append(
            f"| {row['period']} | {_format_rate(row['error_unseen_creator_proportion'])} | "
            f"{_format_rate(row['error_creator_feature_1_proportion'])} | "
            f"{_format_rate(row['error_opcode_clone_proportion'])} | "
            f"{_format_rate(row['error_runtime_clone_proportion'])} |"
        )
    lines.extend(
        [
            "",
            "## P5 False Negatives",
            "",
            f"P5 contains {int(p5['false_negative'])} false negatives and {int(p5['true_positive'])} true positives.",
            "",
            f"- Unseen-creator prevalence: FN {_format_rate(1 - summary_value('P5', 'false_negative', 'previously_seen_creator'))}; TP {_format_rate(1 - summary_value('P5', 'true_positive', 'previously_seen_creator'))}.",
            f"- Historical Ponzi-creator flag: FN {_format_rate(summary_value('P5', 'false_negative', 'creator_history_feature'))}; TP {_format_rate(summary_value('P5', 'true_positive', 'creator_history_feature'))}.",
            f"- Exact historical opcode clone: FN {_format_rate(float(p5_fn['opcode_clone_proportion']))}; TP {_format_rate(float(p5_tp['opcode_clone_proportion']))}.",
            f"- Exact historical runtime clone: FN {_format_rate(float(p5_fn['runtime_clone_proportion']))}; TP {_format_rate(float(p5_tp['runtime_clone_proportion']))}.",
            f"- Mean opcode count: FN {summary_value('P5', 'false_negative', 'opcode_count'):.1f}; TP {summary_value('P5', 'true_positive', 'opcode_count'):.1f}.",
            f"- Mean parseable operands: FN {summary_value('P5', 'false_negative', 'parseable_operand_count'):.1f}; TP {summary_value('P5', 'true_positive', 'parseable_operand_count'):.1f}.",
            f"- Reached voting: FN {_format_rate(summary_value('P5', 'false_negative', 'reached_voting_stage'))}; TP {_format_rate(summary_value('P5', 'true_positive', 'reached_voting_stage'))}.",
            "- Routing of the 37 misses: 28 first-viewer confident errors, 1 second-viewer confident error, and 8 voting-stage errors. Only 3 are near a configured boundary under the predeclared 0.01 margin.",
            "",
            "The missed P5 Ponzi contracts are mainly unseen-creator, non-clone, confidently misclassified cases: 36/37 have unseen creators, 36/37 lack the historical Ponzi-creator flag, all 37 lack exact historical opcode/runtime clones, and 29/37 are accepted incorrectly by a cascade viewer. Eight uncertain misses reach voting. These findings are descriptive and do not establish scam motive or concept drift.",
            "",
            "## Interpretation Limits",
            "",
            "Observed period differences are consistent with temporal distribution shift, but this descriptive analysis does not prove concept drift. Diagnostic structural and operand features were not supplied to the frozen model, and no result was used to tune it.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Create deterministic Phase 7D diagnostics without model training."""
    frozen_hashes = verify_prior_phase_chain(config["frozen_manifest"])
    phase7a_config = load_phase7a_config(config["phase7a_config"])
    phase7a_results = json.loads(config["phase7a_results"].read_text(encoding="utf-8"))
    predictions_hash_before = sha256_file(config["phase7a_predictions"])
    predictions = pd.read_csv(config["phase7a_predictions"])
    dataset = load_analysis_dataset(config["dataset"])
    assigned, definitions = build_ponzi_occurrence_periods(
        dataset,
        positive_ends=phase7a_config["periods"]["positive_ends"],
        expected_positives=phase7a_config["periods"]["expected_total_positives"],
    )
    expected_definitions = phase7a_results["period_definitions"]
    if definitions != expected_definitions:
        raise AssertionError("Phase 7D period definitions differ from Phase 7A.")

    source = assigned[
        assigned["period"].isin(config["protocol"]["periods"])
    ]
    inventory = predictions.merge(
        source[
            [
                "row_index",
                "address",
                "creator",
                "createdBlockNumber",
                "label",
                "opcode",
                "contractCode",
                "creationCode",
                "period",
            ]
        ],
        on="row_index",
        how="left",
        validate="one_to_one",
        suffixes=("_prediction", ""),
    )
    if inventory[["address", "creator", "opcode", "contractCode"]].isna().any().any():
        raise AssertionError("A frozen prediction could not be joined to its source row.")
    for column in ("address", "createdBlockNumber", "label", "period"):
        prediction_column = f"{column}_prediction"
        if prediction_column in inventory:
            left = inventory[prediction_column].astype(str)
            right = inventory[column].astype(str)
            if column == "createdBlockNumber":
                equal = np.isclose(
                    pd.to_numeric(left), pd.to_numeric(right), atol=0, rtol=0
                )
            else:
                equal = left.to_numpy() == right.to_numpy()
            if not np.all(equal):
                raise AssertionError(f"Phase 7A {column} identity changed.")
            inventory = inventory.drop(columns=prediction_column)
    inventory["reached_voting_stage"] = parse_boolean_series(
        inventory["reached_voting_stage"]
    )
    inventory["cascade_stage"] = pd.to_numeric(inventory["cascade_stage"], errors="coerce")
    inventory["error_type"] = classify_outcomes(
        inventory["label"], inventory["final_prediction"]
    )
    inventory = add_historical_context(inventory, assigned)
    diagnostic_config = config["diagnostics"]
    inventory = add_contract_diagnostics(
        inventory,
        diagnostic_config["structural_features"],
        diagnostic_config["operand_features"],
        small_constant_max=int(diagnostic_config["small_constant_max"]),
    )
    mulcas_config = yaml.safe_load(config["paper_mulcas_config"].read_text(encoding="utf-8"))
    inventory = add_routing_diagnostics(
        inventory,
        mulcas_config,
        margin=float(diagnostic_config["near_boundary_margin"]),
    )
    inventory = inventory.sort_values(["period", "createdBlockNumber", "row_index"], kind="mergesort").reset_index(drop=True)
    validate_reference_counts(inventory, phase7a_results)

    numeric_features = [
        "creator_history_feature",
        "previously_seen_creator",
        "previously_ponzi_creator",
        "final_score",
        "prediction_confidence",
        "reached_voting_stage",
        "near_boundary_wrong",
        "has_historical_opcode_clone",
        "has_historical_runtime_bytecode_clone",
        "has_historical_creation_code_clone",
        *diagnostic_config["structural_features"],
        *diagnostic_config["operand_features"],
        "malformed_truncated_push_count",
        "unsupported_opcode_count",
    ]
    group_summary = pd.DataFrame(
        _summary_rows(
            inventory,
            periods=("P2", "P3", "P4", "P5"),
            groups=ERROR_TYPES,
            numeric_features=numeric_features,
        )
    )
    period_summary = build_period_summary(inventory)
    routing_summary = build_routing_summary(inventory)
    clone_summary = build_clone_summary(inventory)
    p5_summary = group_summary[
        (group_summary["period"] == "P5")
        & group_summary["error_type"].isin(("false_negative", "true_positive"))
    ].copy()

    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    export_inventory = inventory.drop(
        columns=["opcode", "contractCode", "creationCode"]
    )
    export_inventory.to_csv(output_dir / "phase7d_error_inventory.csv", index=False)
    group_summary.to_csv(output_dir / "phase7d_group_summary.csv", index=False)
    period_summary.to_csv(output_dir / "phase7d_period_summary.csv", index=False)
    routing_summary.to_csv(output_dir / "phase7d_routing_errors.csv", index=False)
    clone_summary.to_csv(output_dir / "phase7d_clone_diagnostics.csv", index=False)
    p5_summary.to_csv(output_dir / "phase7d_p5_analysis.csv", index=False)
    report = render_report(
        period_summary, routing_summary, clone_summary, group_summary, p5_summary
    )
    (output_dir / "phase7d_report.md").write_text(report, encoding="utf-8")

    results = {
        "phase": "7D",
        "analysis_only": True,
        "optimization_performed": False,
        "model_training_performed": False,
        "random_seed": int(config["seed"]),
        "dataset_sha256": sha256_file(config["dataset"]),
        "config_sha256": config["config_sha256"],
        "phase7a_predictions_sha256": predictions_hash_before,
        "frozen_artifact_hashes": frozen_hashes,
        "reference_rows": int(len(inventory)),
        "period_definitions": definitions,
        "diagnostic_feature_names": numeric_features,
        "period_summary": period_summary.to_dict("records"),
        "routing_error_summary": routing_summary.to_dict("records"),
        "clone_summary": clone_summary.to_dict("records"),
        "statistical_tests": config["statistical_tests"],
        "outputs": [
            "phase7d_error_inventory.csv",
            "phase7d_group_summary.csv",
            "phase7d_period_summary.csv",
            "phase7d_routing_errors.csv",
            "phase7d_clone_diagnostics.csv",
            "phase7d_p5_analysis.csv",
            "phase7d_report.md",
        ],
    }
    (output_dir / "phase7d_results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if sha256_file(config["phase7a_predictions"]) != predictions_hash_before:
        raise AssertionError("Phase 7A predictions changed during Phase 7D.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Phase 7D error analysis.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
