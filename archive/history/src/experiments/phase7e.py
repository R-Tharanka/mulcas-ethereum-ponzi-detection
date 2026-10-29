"""Historical exact-clone sensitivity analysis for frozen Phase 7A predictions."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from src.evaluation.temporal import build_ponzi_occurrence_periods
from src.experiments.phase6a import sha256_file, verify_frozen_manifest
from src.experiments.phase7a import load_config as load_phase7a_config
from src.experiments.phase7d import (
    classify_outcomes,
    load_analysis_dataset,
    parse_boolean_series,
)
from src.utils.paths import REPO_ROOT


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "phase7" / "clone_sensitivity.yaml"
PERIODS = ("P2", "P3", "P4", "P5")
NOVELTY_GROUPS = (
    "exact_runtime_clone",
    "exact_opcode_clone_only",
    "exact_creation_code_clone_only",
    "no_exact_historical_clone",
)
METRIC_GROUPS = (
    "all_contracts",
    "historical_opcode_clone",
    "historical_runtime_clone",
    "any_exact_code_clone",
    "no_exact_historical_clone",
)
HEX_PATTERN = re.compile(r"^[0-9a-f]+$")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load Phase 7E and reject settings that could change frozen behavior."""
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
    for key in ("optimization", "retraining", "retuning", "fuzzy_matching"):
        if protocol[key] != "prohibited":
            raise ValueError(f"Phase 7E {key} must remain prohibited.")
    if protocol["reference_predictions"] != "exact_phase7a_full_system":
        raise ValueError("Phase 7E must reuse exact Phase 7A predictions.")
    if protocol["labels_used_for_clone_status"]:
        raise ValueError("Labels must not determine clone status.")
    if protocol["phase6_interventions_enabled"]:
        raise ValueError("Phase 6 interventions must remain disabled.")
    if tuple(protocol["periods"]) != PERIODS:
        raise ValueError("Phase 7E must analyze P2-P5 exactly.")
    if tuple(config["novelty_group_precedence"]) != NOVELTY_GROUPS:
        raise ValueError("Phase 7E novelty group precedence changed.")
    if tuple(config["metric_groups"]) != METRIC_GROUPS:
        raise ValueError("Phase 7E metric groups changed.")
    return config


def verify_prior_phase_chain(current_manifest: Path) -> dict[str, str]:
    """Verify Phase 5.5 through Phase 7D artifacts before analysis."""
    manifests = (
        current_manifest,
        REPO_ROOT / "configs" / "phase7" / "frozen_phase7c_manifest.yaml",
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


def normalize_opcode_sequence(value: object) -> str:
    """Normalize opcode names without fuzzy matching or operand interpretation."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).strip().upper().split())


def normalize_bytecode(value: object) -> str:
    """Normalize a valid hexadecimal byte sequence for exact comparison."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    if not text or len(text) % 2 or not HEX_PATTERN.fullmatch(text):
        return ""
    return text


def _match_metadata(
    historical: pd.DataFrame,
    target: pd.DataFrame,
    column: str,
    normalizer: Callable[[object], str],
    prefix: str,
) -> pd.DataFrame:
    """Count exact historical matches and their block extent for one code view."""
    history = pd.DataFrame(
        {
            "key": historical[column].map(normalizer),
            "block": historical["createdBlockNumber"].astype(int),
        }
    )
    history = history[history["key"] != ""]
    grouped = history.groupby("key", sort=False)["block"].agg(["size", "min", "max"])
    target_keys = target[column].map(normalizer)
    result = pd.DataFrame(index=target.index)
    result[f"historical_{prefix}_match_count"] = (
        target_keys.map(grouped["size"]).fillna(0).astype(int)
    )
    result[f"earliest_historical_{prefix}_match_block"] = target_keys.map(grouped["min"]).astype("Int64")
    result[f"latest_historical_{prefix}_match_block"] = target_keys.map(grouped["max"]).astype("Int64")
    return result


def build_clone_membership(assigned: pd.DataFrame) -> pd.DataFrame:
    """Assign exactly one novelty group using strictly earlier periods only."""
    required = {
        "row_index",
        "period",
        "createdBlockNumber",
        "opcode",
        "contractCode",
        "creationCode",
        "creator",
    }
    missing = required.difference(assigned.columns)
    if missing:
        raise ValueError(f"Missing clone columns: {sorted(missing)}")
    records: list[pd.DataFrame] = []
    period_numbers = assigned["period"].str[1:].astype(int)
    for period in PERIODS:
        number = int(period[1:])
        historical = assigned.loc[period_numbers < number]
        target = assigned.loc[assigned["period"] == period].copy()
        if historical.empty or target.empty:
            raise ValueError(f"Empty historical or target rows for {period}.")
        if int(historical["createdBlockNumber"].max()) >= int(target["createdBlockNumber"].min()):
            raise AssertionError(f"Clone history for {period} is not strictly earlier.")
        context = target[["row_index", "period"]].copy()
        context["previously_seen_creator"] = target["creator"].astype(str).isin(
            set(historical["creator"].astype(str))
        ).to_numpy()
        for source, normalizer, prefix in (
            ("opcode", normalize_opcode_sequence, "opcode"),
            ("contractCode", normalize_bytecode, "runtime_bytecode"),
            ("creationCode", normalize_bytecode, "creation_code"),
        ):
            metadata = _match_metadata(historical, target, source, normalizer, prefix)
            for column in metadata:
                context[column] = metadata[column].to_numpy()
        context["has_historical_opcode_clone"] = context["historical_opcode_match_count"] > 0
        context["has_historical_runtime_bytecode_clone"] = context["historical_runtime_bytecode_match_count"] > 0
        context["has_historical_creation_code_clone"] = context["historical_creation_code_match_count"] > 0
        context["has_any_exact_historical_clone"] = context[
            [
                "has_historical_opcode_clone",
                "has_historical_runtime_bytecode_clone",
                "has_historical_creation_code_clone",
            ]
        ].any(axis=1)
        earliest_columns = [
            "earliest_historical_opcode_match_block",
            "earliest_historical_runtime_bytecode_match_block",
            "earliest_historical_creation_code_match_block",
        ]
        latest_columns = [column.replace("earliest_", "latest_") for column in earliest_columns]
        context["earliest_historical_any_code_match_block"] = context[earliest_columns].min(axis=1).astype("Int64")
        context["latest_historical_any_code_match_block"] = context[latest_columns].max(axis=1).astype("Int64")
        context["novelty_group"] = np.select(
            [
                context["has_historical_runtime_bytecode_clone"],
                context["has_historical_opcode_clone"],
                context["has_historical_creation_code_clone"],
            ],
            NOVELTY_GROUPS[:3],
            default=NOVELTY_GROUPS[3],
        )
        records.append(context)
    membership = pd.concat(records, ignore_index=True)
    if membership["row_index"].duplicated().any():
        raise AssertionError("An evaluation row received multiple novelty groups.")
    if not set(membership["novelty_group"]).issubset(NOVELTY_GROUPS):
        raise AssertionError("Unknown novelty group generated.")
    return membership.sort_values(["period", "row_index"], kind="mergesort").reset_index(drop=True)


def safe_group_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute subgroup metrics and use None for mathematically undefined values."""
    count = int(len(frame))
    if not count:
        return {
            "row_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "pr_auc": None,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
    truth = frame["label"].to_numpy(dtype=int)
    predicted = frame["final_prediction"].to_numpy(dtype=int)
    scores = frame["final_score"].to_numpy(dtype=float)
    tn = int(((truth == 0) & (predicted == 0)).sum())
    fp = int(((truth == 0) & (predicted == 1)).sum())
    fn = int(((truth == 1) & (predicted == 0)).sum())
    tp = int(((truth == 1) & (predicted == 1)).sum())
    positive_count, negative_count = tp + fn, tn + fp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / positive_count if positive_count else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    both_classes = positive_count > 0 and negative_count > 0
    return {
        "row_count": count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "accuracy": float((tp + tn) / count),
        "balanced_accuracy": float((tp / positive_count + tn / negative_count) / 2) if both_classes else None,
        "precision": float(precision) if precision is not None else None,
        "recall": float(recall) if recall is not None else None,
        "f1": float(f1) if f1 is not None else None,
        "pr_auc": float(average_precision_score(truth, scores)) if both_classes else None,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def build_group_metrics(inventory: pd.DataFrame) -> pd.DataFrame:
    masks: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "all_contracts": lambda frame: pd.Series(True, index=frame.index),
        "historical_opcode_clone": lambda frame: frame["has_historical_opcode_clone"],
        "historical_runtime_clone": lambda frame: frame["has_historical_runtime_bytecode_clone"],
        "any_exact_code_clone": lambda frame: frame["has_any_exact_historical_clone"],
        "no_exact_historical_clone": lambda frame: ~frame["has_any_exact_historical_clone"],
    }
    rows = []
    for period in PERIODS:
        period_rows = inventory[inventory["period"] == period]
        for group in METRIC_GROUPS:
            selected = period_rows.loc[masks[group](period_rows)]
            rows.append({"period": period, "clone_group": group, **safe_group_metrics(selected)})
    return pd.DataFrame(rows)


def _recall(tp: int, fn: int) -> float | None:
    return float(tp / (tp + fn)) if tp + fn else None


def build_class_analysis(inventory: pd.DataFrame) -> pd.DataFrame:
    """Summarize positive recall and benign false-positive behavior."""
    rows = []
    for period in PERIODS:
        period_rows = inventory[inventory["period"] == period]
        positives = period_rows[period_rows["label"] == 1]
        negatives = period_rows[period_rows["label"] == 0]
        cloned_positive = positives[positives["has_any_exact_historical_clone"]]
        novel_positive = positives[~positives["has_any_exact_historical_clone"]]
        cloned_negative = negatives[negatives["has_any_exact_historical_clone"]]
        novel_negative = negatives[~negatives["has_any_exact_historical_clone"]]
        cloned_tp = int((cloned_positive["final_prediction"] == 1).sum())
        cloned_fn = int((cloned_positive["final_prediction"] == 0).sum())
        novel_tp = int((novel_positive["final_prediction"] == 1).sum())
        novel_fn = int((novel_positive["final_prediction"] == 0).sum())
        cloned_fp = int((cloned_negative["final_prediction"] == 1).sum())
        cloned_tn = int((cloned_negative["final_prediction"] == 0).sum())
        novel_fp = int((novel_negative["final_prediction"] == 1).sum())
        novel_tn = int((novel_negative["final_prediction"] == 0).sum())
        rows.append(
            {
                "period": period,
                "positive_count": int(len(positives)),
                "positive_opcode_clone_count": int(positives["has_historical_opcode_clone"].sum()),
                "positive_opcode_clone_percentage": float(100 * positives["has_historical_opcode_clone"].mean()),
                "positive_runtime_clone_count": int(positives["has_historical_runtime_bytecode_clone"].sum()),
                "positive_runtime_clone_percentage": float(100 * positives["has_historical_runtime_bytecode_clone"].mean()),
                "positive_any_clone_count": int(len(cloned_positive)),
                "positive_any_clone_percentage": float(100 * len(cloned_positive) / len(positives)),
                "cloned_ponzi_tp": cloned_tp,
                "cloned_ponzi_fn": cloned_fn,
                "recall_on_cloned_ponzi": _recall(cloned_tp, cloned_fn),
                "novel_ponzi_tp": novel_tp,
                "novel_ponzi_fn": novel_fn,
                "recall_on_novel_ponzi": _recall(novel_tp, novel_fn),
                "negative_count": int(len(negatives)),
                "negative_any_clone_count": int(len(cloned_negative)),
                "negative_any_clone_percentage": float(100 * len(cloned_negative) / len(negatives)),
                "cloned_benign_fp": cloned_fp,
                "cloned_benign_tn": cloned_tn,
                "novel_benign_fp": novel_fp,
                "novel_benign_tn": novel_tn,
                "cloned_benign_fp_rate": float(cloned_fp / len(cloned_negative)) if len(cloned_negative) else None,
                "novel_benign_fp_rate": float(novel_fp / len(novel_negative)) if len(novel_negative) else None,
            }
        )
    return pd.DataFrame(rows)


def build_creator_interaction(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    positives = inventory[inventory["label"] == 1]
    for period in PERIODS:
        period_rows = positives[positives["period"] == period]
        for creator_status, creator_seen in (("seen_creator", True), ("unseen_creator", False)):
            for clone_status, is_clone in (("historical_code_clone", True), ("novel_code", False)):
                selected = period_rows[
                    (period_rows["previously_seen_creator"] == creator_seen)
                    & (period_rows["has_any_exact_historical_clone"] == is_clone)
                ]
                tp = int((selected["final_prediction"] == 1).sum())
                fn = int((selected["final_prediction"] == 0).sum())
                rows.append(
                    {
                        "period": period,
                        "creator_status": creator_status,
                        "clone_status": clone_status,
                        "positive_count": int(len(selected)),
                        "tp": tp,
                        "fn": fn,
                        "recall": _recall(tp, fn),
                    }
                )
    return pd.DataFrame(rows)


def build_routing_interaction(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in PERIODS:
        for clone_status, is_clone in (("historical_code_clone", True), ("novel_code", False)):
            clone_rows = inventory[
                (inventory["period"] == period)
                & (inventory["has_any_exact_historical_clone"] == is_clone)
            ]
            for label_scope, selected in (
                ("all_contracts", clone_rows),
                ("ponzi_only", clone_rows[clone_rows["label"] == 1]),
                ("benign_only", clone_rows[clone_rows["label"] == 0]),
            ):
                count = int(len(selected))
                first = int((selected["cascade_stage"] == 1).sum())
                second = int((selected["cascade_stage"] == 2).sum())
                voting = int(selected["reached_voting_stage"].sum())
                errors = selected["error_type"].isin(("false_positive", "false_negative"))
                confident_wrong = int((errors & ~selected["reached_voting_stage"]).sum())
                uncertain_wrong = int((errors & selected["reached_voting_stage"]).sum())
                rows.append(
                    {
                        "period": period,
                        "clone_status": clone_status,
                        "label_scope": label_scope,
                        "row_count": count,
                        "first_viewer_acceptance_count": first,
                        "first_viewer_acceptance_rate": float(first / count) if count else None,
                        "second_viewer_acceptance_count": second,
                        "second_viewer_acceptance_rate": float(second / count) if count else None,
                        "voting_stage_count": voting,
                        "voting_stage_rate": float(voting / count) if count else None,
                        "confident_wrong_count": confident_wrong,
                        "confident_wrong_rate": float(confident_wrong / count) if count else None,
                        "uncertain_wrong_count": uncertain_wrong,
                        "uncertain_wrong_rate": float(uncertain_wrong / count) if count else None,
                    }
                )
    return pd.DataFrame(rows)


def validate_partition_counts(inventory: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Require clone/novel confusion counts to recover every full period."""
    if inventory["row_index"].duplicated().any() or set(inventory["novelty_group"]) - set(NOVELTY_GROUPS):
        raise AssertionError("Novelty membership is not one-to-one and exhaustive.")
    for period in PERIODS:
        full = metrics[(metrics["period"] == period) & (metrics["clone_group"] == "all_contracts")].iloc[0]
        clone = metrics[(metrics["period"] == period) & (metrics["clone_group"] == "any_exact_code_clone")].iloc[0]
        novel = metrics[(metrics["period"] == period) & (metrics["clone_group"] == "no_exact_historical_clone")].iloc[0]
        for key in ("row_count", "positive_count", "negative_count", "tn", "fp", "fn", "tp"):
            if int(clone[key]) + int(novel[key]) != int(full[key]):
                raise AssertionError(f"{period} clone partition does not sum for {key}.")


def _pct(value: float | None) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{100 * float(value):.1f}%"


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return JSON-safe records with undefined pandas metrics represented by null."""
    return json.loads(frame.to_json(orient="records"))


def render_report(
    metrics: pd.DataFrame,
    class_analysis: pd.DataFrame,
    creator: pd.DataFrame,
    routing: pd.DataFrame,
) -> str:
    lines = [
        "# Phase 7E: Historical Clone and Novelty Sensitivity",
        "",
        "This analysis reuses exact Phase 7A predictions. It performs no retraining, tuning, fuzzy matching, or model changes.",
        "",
        "## Exact Historical Clone Prevalence",
        "",
        "| Period | Any clone / all | Ponzi clone / Ponzi | Benign clone / benign |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in class_analysis.to_dict("records"):
        metric_rows = metrics[(metrics["period"] == row["period"])]
        all_count = int(metric_rows[metric_rows["clone_group"] == "all_contracts"].iloc[0]["row_count"])
        clone_count = int(metric_rows[metric_rows["clone_group"] == "any_exact_code_clone"].iloc[0]["row_count"])
        lines.append(
            f"| {row['period']} | {clone_count}/{all_count} ({100 * clone_count / all_count:.1f}%) | "
            f"{row['positive_any_clone_count']}/{row['positive_count']} ({row['positive_any_clone_percentage']:.1f}%) | "
            f"{row['negative_any_clone_count']}/{row['negative_count']} ({row['negative_any_clone_percentage']:.1f}%) |"
        )
    lines.extend(["", "## Ponzi Recall by Historical Novelty", "", "| Period | Cloned TP/FN | Recall cloned | Novel TP/FN | Recall novel |", "| --- | --- | ---: | --- | ---: |"])
    for row in class_analysis.to_dict("records"):
        lines.append(
            f"| {row['period']} | {row['cloned_ponzi_tp']}/{row['cloned_ponzi_fn']} | {_pct(row['recall_on_cloned_ponzi'])} | "
            f"{row['novel_ponzi_tp']}/{row['novel_ponzi_fn']} | {_pct(row['recall_on_novel_ponzi'])} |"
        )
    lines.extend(["", "## Benign False-Positive Behavior", "", "| Period | Cloned FP/TN | Clone FP rate | Novel FP/TN | Novel FP rate |", "| --- | --- | ---: | --- | ---: |"])
    for row in class_analysis.to_dict("records"):
        lines.append(
            f"| {row['period']} | {row['cloned_benign_fp']}/{row['cloned_benign_tn']} | {_pct(row['cloned_benign_fp_rate'])} | "
            f"{row['novel_benign_fp']}/{row['novel_benign_tn']} | {_pct(row['novel_benign_fp_rate'])} |"
        )
    p5_creator = creator[(creator["period"] == "P5") & (creator["positive_count"] > 0)]
    lines.extend(["", "## P5 Creator and Novelty Interaction", "", "| Creator status | Clone status | Ponzi | TP | FN | Recall |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in p5_creator.to_dict("records"):
        lines.append(
            f"| {row['creator_status']} | {row['clone_status']} | {row['positive_count']} | {row['tp']} | {row['fn']} | {_pct(row['recall'])} |"
        )
    p5_novel_ponzi = routing[
        (routing["period"] == "P5")
        & (routing["clone_status"] == "novel_code")
        & (routing["label_scope"] == "ponzi_only")
    ].iloc[0]
    p4 = class_analysis.set_index("period").loc["P4"]
    p5 = class_analysis.set_index("period").loc["P5"]
    lines.extend(
        [
            "",
            "## Routing and Interpretation",
            "",
            f"Among {int(p5_novel_ponzi['row_count'])} novel P5 Ponzi contracts, {int(p5_novel_ponzi['first_viewer_acceptance_count'])} were accepted at viewer 1, {int(p5_novel_ponzi['confident_wrong_count'])} were confidently misclassified before voting, and {int(p5_novel_ponzi['uncertain_wrong_count'])} were misclassified after reaching voting.",
            "",
            f"P4 contains {int(p4['positive_any_clone_count'])} cloned positives ({p4['positive_any_clone_percentage']:.1f}%) and achieves {_pct(p4['recall_on_novel_ponzi'])} recall even on novel positives. P5 contains only {int(p5['positive_any_clone_count'])} cloned positive ({p5['positive_any_clone_percentage']:.1f}%); its novel-positive recall is {_pct(p5['recall_on_novel_ponzi'])}.",
            "",
            "Historical code reuse is associated with easier detection where cloned Ponzi examples exist, but clone prevalence does not fully explain temporal variation. P2 and P3 have no cloned positives, P4 still performs strongly on novel positives, and subgroup sizes for clones are often very small. These are descriptive associations, not causal evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Generate deterministic clone sensitivity artifacts without model fitting."""
    frozen_hashes = verify_prior_phase_chain(config["frozen_manifest"])
    phase7a_config = load_phase7a_config(config["phase7a_config"])
    phase7a_results = json.loads(config["phase7a_results"].read_text(encoding="utf-8"))
    prediction_hash = sha256_file(config["phase7a_predictions"])
    predictions = pd.read_csv(config["phase7a_predictions"])
    predictions["reached_voting_stage"] = parse_boolean_series(predictions["reached_voting_stage"])
    source = load_analysis_dataset(config["dataset"])
    assigned, definitions = build_ponzi_occurrence_periods(
        source,
        positive_ends=phase7a_config["periods"]["positive_ends"],
        expected_positives=phase7a_config["periods"]["expected_total_positives"],
    )
    if definitions != phase7a_results["period_definitions"]:
        raise AssertionError("Phase 7E period definitions differ from Phase 7A.")
    membership = build_clone_membership(assigned)
    identity = assigned[assigned["period"].isin(PERIODS)][
        ["row_index", "address", "creator", "createdBlockNumber", "label", "period"]
    ]
    inventory = predictions.merge(membership, on=["row_index", "period"], validate="one_to_one")
    inventory = inventory.merge(identity, on=["row_index", "period"], validate="one_to_one", suffixes=("_prediction", ""))
    if len(inventory) != len(predictions):
        raise AssertionError("Phase 7A prediction membership is incomplete.")
    for column in ("address", "createdBlockNumber", "label"):
        prediction_column = f"{column}_prediction"
        if column == "createdBlockNumber":
            equal = np.isclose(inventory[prediction_column], inventory[column], atol=0, rtol=0)
        else:
            equal = inventory[prediction_column].astype(str).to_numpy() == inventory[column].astype(str).to_numpy()
        if not np.all(equal):
            raise AssertionError(f"Phase 7A {column} identity changed.")
        inventory = inventory.drop(columns=prediction_column)
    inventory["error_type"] = classify_outcomes(inventory["label"], inventory["final_prediction"])
    inventory["cascade_stage"] = pd.to_numeric(inventory["cascade_stage"], errors="coerce")
    inventory = inventory.sort_values(["period", "createdBlockNumber", "row_index"], kind="mergesort").reset_index(drop=True)

    group_metrics = build_group_metrics(inventory)
    class_analysis = build_class_analysis(inventory)
    creator_interaction = build_creator_interaction(inventory)
    routing_interaction = build_routing_interaction(inventory)
    validate_partition_counts(inventory, group_metrics)

    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    membership_export_columns = [
        "row_index", "address", "period", "createdBlockNumber", "creator", "label",
        "final_prediction", "final_score", "error_type", "previously_seen_creator",
        "novelty_group", "has_historical_opcode_clone", "has_historical_runtime_bytecode_clone",
        "has_historical_creation_code_clone", "has_any_exact_historical_clone",
        "historical_opcode_match_count", "historical_runtime_bytecode_match_count",
        "historical_creation_code_match_count", "earliest_historical_opcode_match_block",
        "latest_historical_opcode_match_block", "earliest_historical_runtime_bytecode_match_block",
        "latest_historical_runtime_bytecode_match_block", "earliest_historical_creation_code_match_block",
        "latest_historical_creation_code_match_block", "earliest_historical_any_code_match_block",
        "latest_historical_any_code_match_block", "accepted_viewer", "cascade_stage",
        "reached_voting_stage", "creator_history_feature",
    ]
    inventory[membership_export_columns].to_csv(output_dir / "phase7e_clone_membership.csv", index=False)
    group_metrics.to_csv(output_dir / "phase7e_group_metrics.csv", index=False)
    class_analysis.to_csv(output_dir / "phase7e_positive_analysis.csv", index=False)
    creator_interaction.to_csv(output_dir / "phase7e_creator_clone_interaction.csv", index=False)
    routing_interaction.to_csv(output_dir / "phase7e_routing_interaction.csv", index=False)
    report = render_report(group_metrics, class_analysis, creator_interaction, routing_interaction)
    (output_dir / "phase7e_report.md").write_text(report, encoding="utf-8")
    results = {
        "phase": "7E",
        "analysis_only": True,
        "model_training_performed": False,
        "optimization_performed": False,
        "fuzzy_matching_performed": False,
        "random_seed": int(config["seed"]),
        "dataset_sha256": sha256_file(config["dataset"]),
        "config_sha256": config["config_sha256"],
        "phase7a_predictions_sha256": prediction_hash,
        "frozen_artifact_hashes": frozen_hashes,
        "clone_definitions": config["clone_definitions"],
        "normalization": config["normalization"],
        "novelty_group_precedence": list(NOVELTY_GROUPS),
        "reference_rows": int(len(inventory)),
        "period_definitions": definitions,
        "group_metrics": _json_records(group_metrics),
        "class_analysis": _json_records(class_analysis),
        "creator_clone_interaction": _json_records(creator_interaction),
        "routing_interaction": _json_records(routing_interaction),
        "outputs": [
            "phase7e_clone_membership.csv",
            "phase7e_group_metrics.csv",
            "phase7e_positive_analysis.csv",
            "phase7e_creator_clone_interaction.csv",
            "phase7e_routing_interaction.csv",
            "phase7e_report.md",
        ],
    }
    (output_dir / "phase7e_results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if sha256_file(config["phase7a_predictions"]) != prediction_hash:
        raise AssertionError("Phase 7A predictions changed during Phase 7E.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Phase 7E clone sensitivity analysis.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
