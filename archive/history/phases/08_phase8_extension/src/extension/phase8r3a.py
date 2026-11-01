"""Frozen MulCas-GIN complementarity and hybrid-headroom analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import binary_classification_metrics
from src.experiments.phase6a import sha256_file, verify_frozen_manifest
from src.utils.paths import REPO_ROOT


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/extension/phase8r3a.yaml"
PERIODS = ("P2", "P3", "P4")
DISAGREEMENT_CATEGORIES = (
    "both_correct", "both_wrong", "mulcas_correct_gin_wrong", "gin_correct_mulcas_wrong"
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("mulcas_predictions", "novelty_creator_membership", "gin_predictions"):
        config["inputs"][key] = _resolve(config["inputs"][key])
    config["frozen_manifests"] = [_resolve(value) for value in config["frozen_manifests"]]
    config["performance_preflight_policy"] = _resolve(config["performance_preflight_policy"])
    config["output_dir"] = _resolve(config["output_dir"])
    config["config_path"] = path.resolve()
    config["config_sha256"] = sha256_file(path)
    if config["periods"] != list(PERIODS) or config["prohibited_period"] != "P5":
        raise ValueError("Phase 8R3A is restricted to P2-P4.")
    if config["inputs"]["gin_seeds"] != [7, 42, 123, 2026, 999]:
        raise ValueError("The frozen five-seed GIN policy changed.")
    if float(config["prediction_policy"]["threshold"]) != 0.5:
        raise ValueError("GIN ensemble threshold must remain 0.5.")
    if config["prediction_policy"]["threshold_optimization"]:
        raise ValueError("Threshold optimization is prohibited.")
    return config


def verify_frozen_inputs(config: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for manifest in config["frozen_manifests"]:
        observed.update(verify_frozen_manifest(manifest))
    return observed


def _read_frozen_rows(path: Path, count: int) -> pd.DataFrame:
    frame = pd.read_csv(path, nrows=count)
    if "period" in frame and set(frame["period"]) - set(PERIODS):
        raise AssertionError("A prohibited period was loaded.")
    return frame


def align_predictions(config: dict[str, Any]) -> pd.DataFrame:
    expected_count = 5141 + 289 + 150
    mulcas = _read_frozen_rows(config["inputs"]["mulcas_predictions"], expected_count)
    novelty = _read_frozen_rows(config["inputs"]["novelty_creator_membership"], expected_count)
    gin = pd.read_csv(config["inputs"]["gin_predictions"])
    gin = gin[
        gin["architecture_id"].eq(config["inputs"]["gin_architecture"])
        & gin["period"].isin(PERIODS)
    ].copy()
    if set(gin["period"]) != set(PERIODS) or gin["row_index"].nunique() != expected_count:
        raise AssertionError("GIN P2-P4 membership is incomplete.")
    if not np.array_equal(mulcas["row_index"].to_numpy(), novelty["row_index"].to_numpy()):
        raise AssertionError("Phase 7A and Phase 7E row identities differ.")
    for column in ("address", "period", "label"):
        if not np.array_equal(mulcas[column].to_numpy(), novelty[column].to_numpy()):
            raise AssertionError(f"Phase 7A and Phase 7E {column} values differ.")
    gin_identity = gin[["row_index", "period", "label"]].drop_duplicates()
    if len(gin_identity) != expected_count:
        raise AssertionError("GIN seed labels or periods are inconsistent.")
    identity_check = mulcas[["row_index", "period", "label"]].merge(
        gin_identity, on="row_index", suffixes=("_mulcas", "_gin"), validate="one_to_one"
    )
    if not (
        identity_check["period_mulcas"].eq(identity_check["period_gin"]).all()
        and identity_check["label_mulcas"].eq(identity_check["label_gin"]).all()
    ):
        raise AssertionError("MulCas and GIN labels or periods differ.")
    seeds = config["inputs"]["gin_seeds"]
    counts = gin.groupby("row_index")["seed"].agg(["count", "nunique"])
    if not counts.eq(len(seeds)).all().all() or set(gin["seed"]) != set(seeds):
        raise AssertionError("Every contract must have exactly five frozen GIN seeds.")
    probabilities = gin.pivot(index="row_index", columns="seed", values="probability")
    probabilities = probabilities.reindex(columns=seeds)
    probabilities.columns = [f"gin_probability_seed_{seed}" for seed in seeds]
    probabilities["gin_mean_probability"] = probabilities.mean(axis=1)
    probabilities["gin_ensemble_prediction"] = (
        probabilities["gin_mean_probability"] >= float(config["prediction_policy"]["threshold"])
    ).astype(int)
    for seed in seeds:
        probabilities[f"gin_prediction_seed_{seed}"] = (
            probabilities[f"gin_probability_seed_{seed}"] >= 0.5
        ).astype(int)
    mulcas_columns = [
        "row_index", "address", "period", "createdBlockNumber", "label", "final_prediction",
        "final_score", "accepted_viewer", "cascade_stage", "reached_voting_stage", "creator_history_feature",
    ]
    novelty_columns = [
        "row_index", "creator", "previously_seen_creator", "has_any_exact_historical_clone",
        "has_historical_opcode_clone", "has_historical_runtime_bytecode_clone", "novelty_group",
    ]
    aligned = mulcas[mulcas_columns].rename(columns={
        "final_prediction": "mulcas_prediction", "final_score": "mulcas_score",
        "creator_history_feature": "creator_feature_value",
    }).merge(novelty[novelty_columns], on="row_index", how="inner", validate="one_to_one")
    aligned = aligned.merge(probabilities.reset_index(), on="row_index", how="inner", validate="one_to_one")
    if len(aligned) != expected_count or aligned["row_index"].nunique() != expected_count:
        raise AssertionError("MulCas, GIN, and diagnostic memberships do not align exactly.")
    if not (aligned["period"].to_numpy() == mulcas["period"].to_numpy()).all():
        raise AssertionError("Aligned period order changed.")
    mulcas_correct = aligned["mulcas_prediction"].eq(aligned["label"])
    gin_correct = aligned["gin_ensemble_prediction"].eq(aligned["label"])
    aligned["disagreement_category"] = np.select(
        [mulcas_correct & gin_correct, ~mulcas_correct & ~gin_correct, mulcas_correct & ~gin_correct],
        DISAGREEMENT_CATEGORIES[:3], default=DISAGREEMENT_CATEGORIES[3],
    )
    aligned["mulcas_confidence"] = np.maximum(aligned["mulcas_score"], 1.0 - aligned["mulcas_score"])
    aligned["gin_ensemble_confidence"] = np.maximum(
        aligned["gin_mean_probability"], 1.0 - aligned["gin_mean_probability"]
    )
    aligned["absolute_score_disagreement"] = np.abs(
        aligned["mulcas_score"] - aligned["gin_mean_probability"]
    )
    lower = float(config["score_diagnostics"]["confidently_benign_maximum"])
    upper = float(config["score_diagnostics"]["strongly_positive_minimum"])
    aligned["mulcas_confidently_benign_gin_strongly_positive"] = (
        aligned["mulcas_score"].le(lower) & aligned["gin_mean_probability"].ge(upper)
    )
    aligned["gin_confidently_benign_mulcas_positive"] = (
        aligned["gin_mean_probability"].le(lower) & aligned["mulcas_prediction"].eq(1)
    )
    return aligned.sort_values("row_index", kind="mergesort").reset_index(drop=True)


def prediction_policies(frame: pd.DataFrame, seeds: list[int]) -> list[tuple[str, pd.Series]]:
    return [("five_seed_mean", frame["gin_ensemble_prediction"])] + [
        (f"seed_{seed}", frame[f"gin_prediction_seed_{seed}"]) for seed in seeds
    ]


def disagreement_summary(frame: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [(period, frame[frame["period"].eq(period)]) for period in PERIODS] + [("P2-P4", frame)]
    for policy, gin_prediction in prediction_policies(frame, seeds):
        temporary = frame.assign(_gin=gin_prediction)
        mulcas_correct = temporary["mulcas_prediction"].eq(temporary["label"])
        gin_correct = temporary["_gin"].eq(temporary["label"])
        temporary["_category"] = np.select(
            [mulcas_correct & gin_correct, ~mulcas_correct & ~gin_correct, mulcas_correct & ~gin_correct],
            DISAGREEMENT_CATEGORIES[:3], default=DISAGREEMENT_CATEGORIES[3],
        )
        for scope, _ in scopes:
            selected = temporary if scope == "P2-P4" else temporary[temporary["period"].eq(scope)]
            for category in DISAGREEMENT_CATEGORIES:
                count = int(selected["_category"].eq(category).sum())
                rows.append({"gin_policy": policy, "period": scope, "category": category,
                             "count": count, "percentage": count / len(selected)})
    return pd.DataFrame(rows)


def positive_complementarity(frame: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy, prediction in prediction_policies(frame, seeds):
        temporary = frame.assign(_gin=prediction)
        for period in (*PERIODS, "P2-P4"):
            selected = temporary if period == "P2-P4" else temporary[temporary["period"].eq(period)]
            positives = selected[selected["label"].eq(1)]
            m = positives["mulcas_prediction"].eq(1)
            g = positives["_gin"].eq(1)
            both, only_m, only_g, neither = int((m & g).sum()), int((m & ~g).sum()), int((~m & g).sum()), int((~m & ~g).sum())
            support = len(positives)
            rows.append({
                "gin_policy": policy, "period": period, "positive_support": support, "tp_by_both": both,
                "tp_only_mulcas": only_m, "tp_only_gin": only_g, "fn_by_both": neither,
                "mulcas_recall": float(m.mean()), "gin_recall": float(g.mean()),
                "oracle_positive_union_recall": (both + only_m + only_g) / support,
                "oracle_non_deployable": True,
            })
    return pd.DataFrame(rows)


def negative_complementarity(frame: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy, prediction in prediction_policies(frame, seeds):
        temporary = frame.assign(_gin=prediction)
        for period in (*PERIODS, "P2-P4"):
            selected = temporary if period == "P2-P4" else temporary[temporary["period"].eq(period)]
            negatives = selected[selected["label"].eq(0)]
            m = negatives["mulcas_prediction"].eq(0)
            g = negatives["_gin"].eq(0)
            rows.append({
                "gin_policy": policy, "period": period, "negative_support": len(negatives),
                "tn_by_both": int((m & g).sum()), "tn_only_mulcas": int((m & ~g).sum()),
                "tn_only_gin": int((~m & g).sum()), "fp_by_both": int((~m & ~g).sum()),
            })
    return pd.DataFrame(rows)


def oracle_headroom(frame: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy, prediction in prediction_policies(frame, seeds):
        temporary = frame.assign(_gin=prediction)
        for period in (*PERIODS, "P2-P4"):
            selected = temporary if period == "P2-P4" else temporary[temporary["period"].eq(period)]
            truth = selected["label"].to_numpy(dtype=int)
            mulcas = selected["mulcas_prediction"].to_numpy(dtype=int)
            gin = selected["_gin"].to_numpy(dtype=int)
            oracle = np.where(mulcas == truth, mulcas, np.where(gin == truth, gin, mulcas))
            metrics = binary_classification_metrics(truth, oracle, oracle.astype(float))
            rows.append({"gin_policy": policy, "period": period, "oracle": "per_row_perfect_selector",
                         "non_deployable": True, **metrics})
    return pd.DataFrame(rows)


def score_disagreement(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in PERIODS:
        for label in (0, 1):
            selected = frame[frame["period"].eq(period) & frame["label"].eq(label)]
            for name in ("mulcas_score", "gin_mean_probability", "absolute_score_disagreement"):
                values = selected[name]
                rows.append({
                    "period": period, "label": label, "measure": name, "count": len(values),
                    "mean": float(values.mean()), "std": float(values.std(ddof=0)),
                    "minimum": float(values.min()), "q25": float(values.quantile(0.25)),
                    "median": float(values.median()), "q75": float(values.quantile(0.75)),
                    "maximum": float(values.max()),
                    "mulcas_confidently_benign_gin_strongly_positive_count": int(
                        selected["mulcas_confidently_benign_gin_strongly_positive"].sum()
                    ),
                    "gin_confidently_benign_mulcas_positive_count": int(
                        selected["gin_confidently_benign_mulcas_positive"].sum()
                    ),
                })
    return pd.DataFrame(rows)


def novelty_creator_interaction(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [(period, frame[frame["period"].eq(period)]) for period in PERIODS] + [("P2-P4", frame)]
    for period, selected in scopes:
        for category in DISAGREEMENT_CATEGORIES:
            group = selected[selected["disagreement_category"].eq(category)]
            rows.append({
                "period": period, "disagreement_category": category, "row_count": len(group),
                "true_ponzi_count": int(group["label"].sum()),
                "historically_novel_count": int((~group["has_any_exact_historical_clone"].astype(bool)).sum()),
                "exact_clone_count": int(group["has_any_exact_historical_clone"].astype(bool).sum()),
                "unseen_creator_count": int((~group["previously_seen_creator"].astype(bool)).sum()),
                "seen_creator_count": int(group["previously_seen_creator"].astype(bool).sum()),
                "creator_feature_positive_count": int(group["creator_feature_value"].eq(1).sum()),
                "novel_true_ponzi_count": int((group["label"].eq(1) & ~group["has_any_exact_historical_clone"].astype(bool)).sum()),
                "unseen_creator_true_ponzi_count": int((group["label"].eq(1) & ~group["previously_seen_creator"].astype(bool)).sum()),
            })
    return pd.DataFrame(rows)


def routing_interaction(frame: pd.DataFrame) -> pd.DataFrame:
    rescued = frame[
        frame["label"].eq(1) & frame["mulcas_prediction"].eq(0) & frame["gin_ensemble_prediction"].eq(1)
    ].copy()
    rows: list[dict[str, Any]] = []
    for period in (*PERIODS, "P2-P4"):
        selected = rescued if period == "P2-P4" else rescued[rescued["period"].eq(period)]
        for viewer, group in selected.groupby("accepted_viewer", dropna=False, sort=True):
            rows.append({
                "period": period, "accepted_viewer": str(viewer), "rescued_ponzi_count": len(group),
                "cascade_stage_1_count": int(group["cascade_stage"].eq(1).sum()),
                "cascade_stage_2_count": int(group["cascade_stage"].eq(2).sum()),
                "voting_stage_count": int(group["reached_voting_stage"].astype(bool).sum()),
                "mulcas_confidently_benign_count": int(group["mulcas_score"].le(0.10).sum()),
                "mean_mulcas_score": float(group["mulcas_score"].mean()),
                "mean_gin_probability": float(group["gin_mean_probability"].mean()),
            })
    return pd.DataFrame(rows)


def hybrid_decision(frame: pd.DataFrame, positive: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    ensemble = positive[positive["gin_policy"].eq("five_seed_mean")].set_index("period")
    gate = config["hybrid_feasibility_gate"]
    difficult_recoveries = int(ensemble.loc[list(PERIODS), "tp_only_gin"].max())
    overall = ensemble.loc["P2-P4"]
    union_gain = float(overall["oracle_positive_union_recall"] - overall["mulcas_recall"])
    any_error = frame[
        frame["disagreement_category"].ne("both_correct")
    ]
    identical_failure_share = float(any_error["disagreement_category"].eq("both_wrong").mean())
    gin_only = frame[
        frame["label"].eq(1) & frame["mulcas_prediction"].eq(0) & frame["gin_ensemble_prediction"].eq(1)
    ]
    subgroup_recoveries = int((
        ~gin_only["has_any_exact_historical_clone"].astype(bool)
        | ~gin_only["previously_seen_creator"].astype(bool)
    ).sum())
    conditions = {
        "meaningful_period_recoveries": difficult_recoveries >= int(gate["minimum_gin_only_ponzi_recoveries_in_one_period"]),
        "material_overall_union_recall_gain": union_gain >= float(gate["minimum_absolute_overall_union_recall_gain"]),
        "errors_not_overwhelmingly_identical": identical_failure_share < float(gate["maximum_identical_failure_share_among_any_errors"]),
        "novel_or_unseen_creator_complementarity": subgroup_recoveries >= int(gate["minimum_novel_or_unseen_gin_only_ponzi_recoveries"]),
    }
    justified = all(conditions.values())
    return {
        "predeclared_conditions": conditions, "all_conditions_required": True,
        "maximum_gin_only_ponzi_recoveries_in_one_period": difficult_recoveries,
        "overall_union_recall_gain": union_gain,
        "identical_failure_share_among_any_errors": identical_failure_share,
        "novel_or_unseen_gin_only_ponzi_recoveries": subgroup_recoveries,
        "hybrid_experiment_justified": justified,
        "recommendation": (
            "Proceed to a small temporally trained Phase 8R3B hybrid/novelty-routing experiment."
            if justified else "Stop graph-hybrid research under the current evidence."
        ),
    }


def build_report(
    disagreement: pd.DataFrame, positive: pd.DataFrame, negative: pd.DataFrame, oracle: pd.DataFrame,
    routing: pd.DataFrame, decision: dict[str, Any],
) -> str:
    ensemble_disagreement = disagreement[disagreement["gin_policy"].eq("five_seed_mean")]
    table = ensemble_disagreement.pivot(index="period", columns="category", values="count").reindex([*PERIODS, "P2-P4"])
    disagreement_rows = [
        f"| {period} | {int(row.both_correct)} | {int(row.both_wrong)} | "
        f"{int(row.mulcas_correct_gin_wrong)} | {int(row.gin_correct_mulcas_wrong)} |"
        for period, row in table.iterrows()
    ]
    p = positive[positive["gin_policy"].eq("five_seed_mean")].set_index("period")
    positive_rows = [
        f"| {period} | {int(p.loc[period, 'tp_by_both'])} | {int(p.loc[period, 'tp_only_mulcas'])} | "
        f"{int(p.loc[period, 'tp_only_gin'])} | {int(p.loc[period, 'fn_by_both'])} | "
        f"{p.loc[period, 'mulcas_recall']:.4f} | {p.loc[period, 'oracle_positive_union_recall']:.4f} |"
        for period in (*PERIODS, "P2-P4")
    ]
    n = negative[negative["gin_policy"].eq("five_seed_mean")].set_index("period")
    negative_rows = [
        f"| {period} | {int(n.loc[period, 'tn_by_both'])} | {int(n.loc[period, 'tn_only_mulcas'])} | "
        f"{int(n.loc[period, 'tn_only_gin'])} | {int(n.loc[period, 'fp_by_both'])} |"
        for period in (*PERIODS, "P2-P4")
    ]
    o = oracle[(oracle["gin_policy"].eq("five_seed_mean"))].set_index("period")
    oracle_rows = [
        f"| {period} | {o.loc[period, 'precision']:.4f} | {o.loc[period, 'recall']:.4f} | "
        f"{o.loc[period, 'f1']:.4f} | {o.loc[period, 'balanced_accuracy']:.4f} | "
        f"{int(o.loc[period, 'tn'])}/{int(o.loc[period, 'fp'])}/{int(o.loc[period, 'fn'])}/{int(o.loc[period, 'tp'])} |"
        for period in (*PERIODS, "P2-P4")
    ]
    rescued = routing[routing["period"].eq("P2-P4")]
    return "\n".join([
        "# Phase 8R3A: Frozen MulCas-GIN Complementarity", "",
        "This is an analysis of saved P2-P4 predictions only. No model was trained, no threshold was tuned, and P5 was not loaded.", "",
        "## Ensemble Disagreement", "", "| Period | Both correct | Both wrong | MulCas only correct | GIN only correct |",
        "| --- | ---: | ---: | ---: | ---: |", *disagreement_rows, "",
        "## Positive Complementarity", "", "| Period | TP both | TP MulCas only | TP GIN only | FN both | MulCas recall | Oracle union recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |", *positive_rows, "",
        "## Negative Complementarity", "", "| Period | TN both | TN MulCas only | TN GIN only | FP both |",
        "| --- | ---: | ---: | ---: | ---: |", *negative_rows, "",
        "## Non-Deployable Oracle Ceiling", "", "| Period | Precision | Recall | F1 | Balanced accuracy | TN/FP/FN/TP |",
        "| --- | ---: | ---: | ---: | ---: | --- |", *oracle_rows, "",
        "These oracle values assume a perfect per-row selector and are not achievable model results.", "",
        "## Routing", "", f"GIN rescued {int(rescued['rescued_ponzi_count'].sum())} MulCas false negatives overall; "
        f"{int(rescued['mulcas_confidently_benign_count'].sum())} were confidently rejected by MulCas.", "",
        "## Hybrid Feasibility", "", *(f"- {name}: {value}" for name, value in decision["predeclared_conditions"].items()), "",
        f"Decision: **{'justified' if decision['hybrid_experiment_justified'] else 'not justified'}**.",
        decision["recommendation"], "",
        "The project-wide performance preflight rule is recorded in `docs/research_extension_performance_preflight.md`; "
        "the original Phase 8R0 protocol remains hash-frozen.", "",
    ])


def run(config: dict[str, Any]) -> dict[str, Any]:
    frozen = verify_frozen_inputs(config)
    aligned = align_predictions(config)
    seeds = config["inputs"]["gin_seeds"]
    disagreement = disagreement_summary(aligned, seeds)
    positive = positive_complementarity(aligned, seeds)
    negative = negative_complementarity(aligned, seeds)
    oracle = oracle_headroom(aligned, seeds)
    scores = score_disagreement(aligned)
    novelty = novelty_creator_interaction(aligned)
    routing = routing_interaction(aligned)
    decision = hybrid_decision(aligned, positive, config)
    output = config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        "aligned_predictions.csv": aligned,
        "disagreement_summary.csv": disagreement,
        "positive_complementarity.csv": positive,
        "negative_complementarity.csv": negative,
        "oracle_headroom.csv": oracle,
        "score_disagreement.csv": scores,
        "novelty_creator_interaction.csv": novelty,
        "routing_interaction.csv": routing,
    }
    for filename, frame in frames.items():
        frame.to_csv(output / filename, index=False)
    results = {
        "phase": "8R3A", "analysis_only": True, "models_trained": False,
        "threshold_tuned": False, "p5_loaded_or_evaluated": False,
        "gin_architecture": config["inputs"]["gin_architecture"], "gin_seeds": seeds,
        "gin_ensemble_policy": config["prediction_policy"]["gin_ensemble"],
        "gin_ensemble_threshold": 0.5, "aligned_row_count": len(aligned),
        "frozen_artifact_hashes": frozen, "config_sha256": config["config_sha256"],
        "oracle_results_non_deployable": True, "hybrid_feasibility": decision,
    }
    (output / "phase8r3a_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "phase8r3a_report.md").write_text(
        build_report(disagreement, positive, negative, oracle, routing, decision), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
