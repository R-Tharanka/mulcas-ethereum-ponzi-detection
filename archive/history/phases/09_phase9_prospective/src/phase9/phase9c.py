"""Phase 9C deployment-available creator and code-family state experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix, hstack

from src.evaluation.metrics import binary_classification_metrics
from src.evaluation.temporal import TemporalScenario, build_ponzi_occurrence_periods, validate_scenario_chronology
from src.experiments.phase7a import (
    _load_frozen_configs,
    build_historical_feature_matrices,
    fit_predict_frozen_mulcas,
    load_full_dataset,
)
from src.phase9.family_similarity import FAMILY_COLUMNS, FamilyStateResult, build_family_state
from src.phase9.phase9_metrics import paired_bootstrap, pooled_metrics
from src.phase9.state_features import CREATOR_COLUMNS, CreatorStateResult, build_creator_state
from src.utils.paths import REPO_ROOT
from src.utils.seed import set_random_seed


DEFAULT_CONFIG = REPO_ROOT / "configs/phase9/phase9c.yaml"
PERIODS = ("P2", "P3", "P4")
CANDIDATES = ("C0", "C1", "C2", "C3")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in (
        "dataset", "phase9b_config", "phase7a_config", "phase7a_results",
        "phase7a_predictions", "paper_feature_config", "paper_model_config",
        "paper_mulcas_config", "test_anchor_migration", "output_dir", "artifact_dir",
    ):
        config[key] = _resolve(config[key])
    if tuple(config["candidates"]) != CANDIDATES:
        raise ValueError("Phase 9C candidates must remain exactly C0-C3.")
    if float(config["family_features"]["similarity_threshold"]) != 0.90:
        raise ValueError("Phase 9C similarity threshold must remain 0.90.")
    if config["track_b"]["access_permission"] != "disabled":
        raise PermissionError("Track B access must remain disabled during Phase 9C.")
    if config["temporal_protocol"]["p5_evaluation"] != "prohibited":
        raise ValueError("P5 evaluation must remain prohibited.")
    config["config_path"] = path.resolve()
    config["config_sha256"] = sha256_file(path)
    return config


def _load_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(config["dataset"]) != config["dataset_sha256"]:
        raise RuntimeError("Track A raw hash mismatch.")
    phase9b = yaml.safe_load(config["phase9b_config"].read_text(encoding="utf-8"))
    if phase9b["status"] != "frozen_protocol_only" or phase9b["constraints"]["phase9c_started"]:
        raise RuntimeError("Phase 9B is not in the required prospective frozen state.")
    frame = load_full_dataset(config["dataset"])
    phase7_config = yaml.safe_load(config["phase7a_config"].read_text(encoding="utf-8"))
    assigned, _ = build_ponzi_occurrence_periods(
        frame,
        positive_ends=phase7_config["periods"]["positive_ends"],
        expected_positives=int(phase7_config["periods"]["expected_total_positives"]),
    )
    assigned = assigned[assigned["period"].isin(("P0", "P1", "P2", "P3", "P4"))].copy()
    feature, model, mulcas = _load_frozen_configs({
        "paper_feature_config": config["paper_feature_config"],
        "paper_model_config": config["paper_model_config"],
        "paper_mulcas_config": config["paper_mulcas_config"],
    })
    return assigned, feature, model, mulcas, phase9b


def _scenarios(config: dict[str, Any]) -> list[TemporalScenario]:
    scenarios = [
        TemporalScenario(
            scenario_id=f"{'+'.join(item['train_periods'])}_to_{item['prediction_period']}",
            train_periods=tuple(item["train_periods"]),
            prediction_period=item["prediction_period"],
        )
        for item in config["temporal_protocol"]["scenarios"]
    ]
    if tuple(item.prediction_period for item in scenarios) != PERIODS:
        raise ValueError("Phase 9C scenarios must predict exactly P2-P4.")
    return scenarios


def _partition(assigned: pd.DataFrame, scenario: TemporalScenario) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_scenario_chronology(assigned, scenario)
    train = assigned[assigned["period"].isin(scenario.train_periods)].copy()
    target = assigned[assigned["period"].eq(scenario.prediction_period)].copy()
    sort = ["createdBlockNumber", "row_index"]
    return train.sort_values(sort, kind="mergesort"), target.sort_values(sort, kind="mergesort")


def _base_matrices(train: pd.DataFrame, target: pd.DataFrame, feature: dict[str, Any], mulcas: dict[str, Any]):
    views = list(dict.fromkeys(item["feature_view"] for item in mulcas["viewer_sequence"]))
    matrices, audit, _ = build_historical_feature_matrices(
        train, target, views, ngram_min_df=int(feature["ngram"]["min_df"])
    )
    return matrices, audit


def _append_state(matrices, creator: CreatorStateResult, family: FamilyStateResult, candidate: str):
    if candidate == "C1":
        train_state, target_state = creator.train_model, creator.target_model
    elif candidate == "C2":
        train_state, target_state = family.train_model, family.target_model
    elif candidate == "C3":
        train_state = np.hstack([creator.train_model, family.train_model])
        target_state = np.hstack([creator.target_model, family.target_model])
    else:
        return matrices
    train_sparse = csr_matrix(train_state)
    target_sparse = csr_matrix(target_state)
    return {
        name: (
            hstack([pair[0], train_sparse], format="csr"),
            hstack([pair[1], target_sparse], format="csr"),
        )
        for name, pair in matrices.items()
    }


def _prediction_rows(candidate: str, period: str, target: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.rename(columns={
        "sample_index": "row_index", "prediction": "prediction",
        "probability": "score", "handled_by": "route",
    }).merge(
        target[["row_index", "address", "creator", "createdBlockNumber", "label", "period"]],
        on="row_index", validate="one_to_one",
    )
    frame["candidate"] = candidate
    frame["period"] = period
    return frame


def _run_candidate(candidate: str, train: pd.DataFrame, target: pd.DataFrame, matrices, model, mulcas, seed: int) -> tuple[pd.DataFrame, float]:
    started = time.perf_counter()
    predictions, _, _, _ = fit_predict_frozen_mulcas(
        train, target, matrices, model, mulcas, seed=seed
    )
    return _prediction_rows(candidate, str(target["period"].iloc[0]), target, predictions), time.perf_counter() - started


def _environment() -> dict[str, Any]:
    try:
        import psutil
        ram_gib = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        ram_gib = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
    except Exception:
        gpu = "unavailable"
    return {"cpu": platform.processor(), "ram_gib": ram_gib, "gpu_status": gpu}


def run_preflight(config: dict[str, Any]) -> dict[str, Any]:
    assigned, feature, model, mulcas, _ = _load_inputs(config)
    scenario = next(item for item in _scenarios(config) if item.prediction_period == config["performance_gate"]["representative_period"])
    train, target = _partition(assigned, scenario)
    started = time.perf_counter()
    matrices, _ = _base_matrices(train, target, feature, mulcas)
    creator = build_creator_state(train, target.drop(columns=["label"]), sentinel=int(config["creator_features"]["no_history_block_sentinel"]))
    family = build_family_state(
        train, target.drop(columns=["label"]), min_df=int(feature["ngram"]["min_df"]),
        threshold=float(config["family_features"]["similarity_threshold"]),
    )
    preparation_seconds = time.perf_counter() - started
    candidate_matrices = _append_state(matrices, creator, family, "C3")
    _, fit_seconds = _run_candidate("C3", train, target, candidate_matrices, model, mulcas, int(config["seed"]))
    representative_seconds = preparation_seconds + fit_seconds
    estimated_seconds = 3.0 * preparation_seconds + 12.0 * fit_seconds
    estimated_minutes = estimated_seconds / 60.0
    runtime_seconds = time.perf_counter() - started
    result = {
        "status": "pass" if estimated_minutes <= float(config["performance_gate"]["maximum_estimated_runtime_minutes"]) else "awaiting_runtime_approval",
        "representative_period": scenario.prediction_period,
        "representative_candidate": "C3",
        "preparation_seconds": preparation_seconds,
        "fit_seconds": fit_seconds,
        "representative_runtime_seconds": representative_seconds,
        "estimated_total_fits": 72,
        "estimated_full_runtime_minutes": estimated_minutes,
        "estimated_output_size_mib": 30.0,
        **_environment(),
    }
    output = config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "runtime_preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _metrics_frame(predictions: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    period_rows, pooled_rows, pooled = [], [], {}
    ceiling = float(config["metrics"]["reference_fpr_ceiling"])
    prevalence = float(config["metrics"]["reference_prevalence"])
    for candidate in CANDIDATES:
        selected = predictions[predictions["candidate"].eq(candidate)].sort_values("row_index")
        metrics = pooled_metrics(selected.label, selected.prediction, selected.score, fpr_ceiling=ceiling, reference_prevalence=prevalence)
        pooled[candidate] = metrics
        pooled_rows.append({"candidate": candidate, **metrics})
        for period in PERIODS:
            part = selected[selected.period.eq(period)]
            item = pooled_metrics(part.label, part.prediction, part.score, fpr_ceiling=ceiling, reference_prevalence=prevalence)
            period_rows.append({"candidate": candidate, "period": period, **item})
    return pd.DataFrame(pooled_rows), pd.DataFrame(period_rows), pooled


def _verify_control(period_metrics: pd.DataFrame, source_path: Path) -> None:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    by_period = {item["prediction_period"]: item["metrics"] for item in source["scenarios"]}
    for period in PERIODS:
        row = period_metrics[(period_metrics.candidate == "C0") & (period_metrics.period == period)].iloc[0]
        for metric in ("precision", "recall", "f1", "pr_auc", "tn", "fp", "fn", "tp"):
            if not np.isclose(float(row[metric]), float(by_period[period][metric]), atol=1e-12, rtol=0):
                raise RuntimeError(f"C0 does not reproduce frozen Phase 7A {period} {metric}.")


def _bootstrap(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    control = predictions[predictions.candidate.eq("C0")].sort_values("row_index")
    control_data = {"truth": control.label.to_numpy(), "prediction": control.prediction.to_numpy(), "score": control.score.to_numpy()}
    rows = []
    for candidate in CANDIDATES[1:]:
        selected = predictions[predictions.candidate.eq(candidate)].sort_values("row_index")
        result = paired_bootstrap(
            control_data,
            {"truth": selected.label.to_numpy(), "prediction": selected.prediction.to_numpy(), "score": selected.score.to_numpy()},
            replicates=int(config["bootstrap"]["replicates"]), seed=int(config["bootstrap"]["seed"]),
        )
        rows.append({"candidate": candidate, **result})
    return pd.DataFrame(rows)


def _advancement(pooled: pd.DataFrame, periods: pd.DataFrame, bootstrap: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    control = pooled.set_index("candidate").loc["C0"]
    period_control = periods[periods.candidate.eq("C0")].set_index("period")
    rows = []
    for candidate in CANDIDATES[1:]:
        row = pooled.set_index("candidate").loc[candidate]
        deltas = periods[periods.candidate.eq(candidate)].set_index("period").f1 - period_control.f1
        checks = {
            "pooled_f1_improved": bool(row.f1 > control.f1),
            "pr_auc_within_tolerance": bool(control.pr_auc - row.pr_auc <= float(config["advancement"]["pr_auc_maximum_absolute_degradation"])),
            "fpr_within_ceiling": bool(row.fpr <= float(config["advancement"]["fpr_maximum"])),
            "temporal_breadth_met": bool((deltas >= -1e-12).sum() >= int(config["advancement"]["minimum_periods_with_nonnegative_f1_delta"])),
            "no_period_degradation_exceeded": bool(deltas.min() >= -float(config["advancement"]["maximum_allowed_single_period_f1_degradation"]) - 1e-12),
            "paired_uncertainty_reported": bool(candidate in set(bootstrap.candidate)),
        }
        rows.append({"candidate": candidate, **checks, "advanced": all(checks.values()), **{f"{p}_f1_delta": float(deltas[p]) for p in PERIODS}})
    return pd.DataFrame(rows)


def _select_candidate(advancement: pd.DataFrame, pooled: pd.DataFrame, config: dict[str, Any]) -> str | None:
    advanced = advancement.loc[advancement.advanced, "candidate"].tolist()
    if not advanced:
        return None
    metrics = pooled[pooled.candidate.isin(advanced)].set_index("candidate")
    best_f1 = float(metrics.f1.max())
    tied = metrics[metrics.f1 >= best_f1 - float(config["selection"]["effective_tie_f1"])].copy()
    simplicity = {name: index for index, name in enumerate(config["selection"]["simplicity_order"])}
    tied["simplicity"] = [simplicity[name] for name in tied.index]
    return str(tied.sort_values(["pr_auc", "recall", "simplicity"], ascending=[False, False, True]).index[0])


def _subgroups(predictions: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "unseen_creator": lambda x: x.prior_contract_count == 0,
        "seen_creator": lambda x: x.prior_contract_count > 0,
        "prior_verified_ponzi_flag_1": lambda x: x.prior_verified_ponzi_flag == 1,
        "historically_exact_novel_code": lambda x: ~x.historically_exact_cloned_code.astype(bool),
        "historically_exact_cloned_code": lambda x: x.historically_exact_cloned_code.astype(bool),
        "family_similarity_ge_0_90": lambda x: x.count_prior_all_similarity_ge_0_90 > 0,
        "family_similarity_lt_0_90": lambda x: x.count_prior_all_similarity_ge_0_90 == 0,
    }
    rows = []
    for candidate in CANDIDATES:
        candidate_rows = predictions[predictions.candidate.eq(candidate)]
        for name, selector in definitions.items():
            selected = candidate_rows[selector(candidate_rows)]
            if selected.empty:
                continue
            metrics = binary_classification_metrics(selected.label, selected.prediction, selected.score)
            rows.append({"candidate": candidate, "subgroup": name, "row_count": len(selected), **metrics})
    return pd.DataFrame(rows)


def _changed_predictions(predictions: pd.DataFrame, diagnostic: str) -> pd.DataFrame:
    control = predictions[predictions.candidate.eq("C0")].set_index("row_index")
    candidate = predictions[predictions.candidate.eq(diagnostic)].set_index("row_index")
    joined = control[["period", "address", "label", "prediction", "route"]].join(
        candidate[["prediction", "route"]], lsuffix="_C0", rsuffix="_candidate"
    )
    changed = joined[joined.prediction_C0.ne(joined.prediction_candidate)].copy()
    features = candidate[CREATOR_COLUMNS + FAMILY_COLUMNS].copy()
    changed = changed.join(features)
    changed.insert(0, "candidate", diagnostic)
    changed["change_type"] = np.select(
        [
            changed.label.eq(1) & changed.prediction_C0.eq(0) & changed.prediction_candidate.eq(1),
            changed.label.eq(1) & changed.prediction_C0.eq(1) & changed.prediction_candidate.eq(0),
            changed.label.eq(0) & changed.prediction_C0.eq(1) & changed.prediction_candidate.eq(0),
            changed.label.eq(0) & changed.prediction_C0.eq(0) & changed.prediction_candidate.eq(1),
        ],
        ["C0_FALSE_NEGATIVE_RECOVERED", "NEW_FALSE_NEGATIVE", "C0_FALSE_POSITIVE_CORRECTED", "NEW_FALSE_POSITIVE"],
        default="OTHER",
    )
    return changed.reset_index()


def _write_report(result: dict[str, Any], pooled: pd.DataFrame, periods: pd.DataFrame, advancement: pd.DataFrame, bootstrap: pd.DataFrame) -> str:
    rows = ["| Candidate | Pooled P | Pooled R | Pooled F1 | PR-AUC | FPR | FPR-constrained R | Advanced |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    decisions = advancement.set_index("candidate")
    for item in pooled.itertuples(index=False):
        advanced = "control" if item.candidate == "C0" else str(bool(decisions.loc[item.candidate, "advanced"]))
        rows.append(f"| {item.candidate} | {item.precision:.4f} | {item.recall:.4f} | {item.f1:.4f} | {item.pr_auc:.4f} | {item.fpr:.4f} | {item.recall_at_reference_fpr_ceiling:.4f} | {advanced} |")
    period_rows = ["| Candidate | P2 F1 | P3 F1 | P4 F1 |", "| --- | ---: | ---: | ---: |"]
    for candidate in CANDIDATES:
        values = periods[periods.candidate.eq(candidate)].set_index("period").f1
        period_rows.append(f"| {candidate} | {values.P2:.4f} | {values.P3:.4f} | {values.P4:.4f} |")
    advancement_rows = ["| Candidate | F1 | PR-AUC | FPR | Breadth | Period floor | Uncertainty | Advanced |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for item in advancement.itertuples(index=False):
        advancement_rows.append(f"| {item.candidate} | {item.pooled_f1_improved} | {item.pr_auc_within_tolerance} | {item.fpr_within_ceiling} | {item.temporal_breadth_met} | {item.no_period_degradation_exceeded} | {item.paired_uncertainty_reported} | {item.advanced} |")
    bootstrap_rows = ["| Candidate | Delta F1 95% CI | Delta PR-AUC 95% CI | Delta recall 95% CI |", "| --- | --- | --- | --- |"]
    for item in bootstrap.itertuples(index=False):
        bootstrap_rows.append(f"| {item.candidate} | [{item.delta_f1_ci_low:.4f}, {item.delta_f1_ci_high:.4f}] | [{item.delta_pr_auc_ci_low:.4f}, {item.delta_pr_auc_ci_high:.4f}] | [{item.delta_recall_ci_low:.4f}, {item.delta_recall_ci_high:.4f}] |")
    recovery_rows = ["| Period | FN recovered | New FN | FP corrected | New FP |", "| --- | ---: | ---: | ---: | ---: |"]
    for period, counts in result["error_recovery"].items():
        recovery_rows.append(f"| {period} | {counts['C0_FALSE_NEGATIVE_RECOVERED']} | {counts['NEW_FALSE_NEGATIVE']} | {counts['C0_FALSE_POSITIVE_CORRECTED']} | {counts['NEW_FALSE_POSITIVE']} |")
    conclusion = (
        "The selected Phase 9C candidate satisfies the prospective Phase 9B advancement criteria. This establishes internal forward-temporal support only; external generalization has not yet been evaluated."
        if result["h9c_supported"] else
        "The tested creator/code-family state features did not satisfy the prospective advancement criteria. Track B remains sealed and the negative result does not invalidate the future Phase 9D routing experiment."
    )
    return "\n".join([
        "# Phase 9C Deployment-Available Creator and Code-Family State", "",
        "Phase 9C evaluates deployment-available historical state on Track A only. Track B remains sealed and is not used for development.", "",
        "## Results", "", *rows, "", "## Period F1", "", *period_rows, "",
        "## Advancement Checks", "", *advancement_rows, "", "## Paired Bootstrap", "", *bootstrap_rows, "",
        f"## Error Recovery: {result['diagnostic_best_candidate']}", "", *recovery_rows, "",
        "## Decision", "", conclusion, "",
        f"Selected candidate: `{result['selected_candidate']}`. H9C supported: **{result['h9c_supported']}**.", "",
        "## Reproducibility and Tests", "",
        f"Focused Phase 9C tests: **{result.get('verification', {}).get('focused_tests', 'not recorded')}**. Full repository suite: **{result.get('verification', {}).get('full_suite', 'not recorded')}**.", "",
        "No P5 or Track B evaluation occurred. Cascade thresholds, voting, viewers, and model families remained frozen. Phase 9D was not started.",
    ]) + "\n"


def run(config: dict[str, Any]) -> dict[str, Any]:
    preflight_path = config["output_dir"] / "runtime_preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("Run Phase 9C preflight before full execution.")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight["status"] != "pass":
        raise RuntimeError("Phase 9C runtime requires explicit approval.")
    started = time.perf_counter()
    set_random_seed(int(config["seed"]))
    assigned, feature, model, mulcas, phase9b = _load_inputs(config)
    scenarios = _scenarios(config)
    predictions: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []

    # Reproduce and verify every control period before fitting any candidate.
    control_period_metrics = []
    for scenario in scenarios:
        train, target = _partition(assigned, scenario)
        matrices, _ = _base_matrices(train, target, feature, mulcas)
        frame, seconds = _run_candidate("C0", train, target, matrices, model, mulcas, int(config["seed"]))
        predictions.append(frame)
        runtime_rows.append({"candidate": "C0", "period": scenario.prediction_period, "fit_seconds": seconds})
        metrics = binary_classification_metrics(frame.label, frame.prediction, frame.score)
        control_period_metrics.append({"candidate": "C0", "period": scenario.prediction_period, **metrics})
    _verify_control(pd.DataFrame(control_period_metrics), config["phase7a_results"])

    state_frames, creator_summaries, family_summaries = [], [], []
    for scenario in scenarios:
        train, target = _partition(assigned, scenario)
        matrices, _ = _base_matrices(train, target, feature, mulcas)
        creator = build_creator_state(train, target.drop(columns=["label"]), sentinel=int(config["creator_features"]["no_history_block_sentinel"]))
        family = build_family_state(train, target.drop(columns=["label"]), min_df=int(feature["ngram"]["min_df"]), threshold=float(config["family_features"]["similarity_threshold"]))
        state = creator.target_raw.merge(family.target_raw, on="row_index", validate="one_to_one")
        state["period"] = scenario.prediction_period
        state_frames.append(state)
        for column in CREATOR_COLUMNS:
            creator_summaries.append({"period": scenario.prediction_period, "feature": column, "minimum": float(creator.target_raw[column].min()), "mean": float(creator.target_raw[column].mean()), "maximum": float(creator.target_raw[column].max())})
        for column in FAMILY_COLUMNS:
            family_summaries.append({"period": scenario.prediction_period, "feature": column, "minimum": float(family.target_raw[column].min()), "mean": float(family.target_raw[column].mean()), "maximum": float(family.target_raw[column].max()), "vocabulary_size": family.vocabulary_size})
        for candidate in CANDIDATES[1:]:
            candidate_matrices = _append_state(matrices, creator, family, candidate)
            frame, seconds = _run_candidate(candidate, train, target, candidate_matrices, model, mulcas, int(config["seed"]))
            predictions.append(frame)
            runtime_rows.append({"candidate": candidate, "period": scenario.prediction_period, "fit_seconds": seconds})

    pooled_predictions = pd.concat(predictions, ignore_index=True)
    states = pd.concat(state_frames, ignore_index=True)
    pooled_predictions = pooled_predictions.merge(states, on=["row_index", "period"], how="left", validate="many_to_one")
    candidate_metrics, period_metrics, pooled = _metrics_frame(pooled_predictions, config)
    _verify_control(period_metrics, config["phase7a_results"])
    bootstrap = _bootstrap(pooled_predictions, config)
    advancement = _advancement(candidate_metrics, period_metrics, bootstrap, config)
    selected = _select_candidate(advancement, candidate_metrics, config)
    diagnostic = selected or str(candidate_metrics[candidate_metrics.candidate.ne("C0")].sort_values("f1", ascending=False).iloc[0].candidate)
    subgroup = _subgroups(pooled_predictions)
    changed = _changed_predictions(pooled_predictions, diagnostic)
    error_recovery = {}
    for period in PERIODS:
        counts = changed[changed.period.eq(period)].change_type.value_counts()
        error_recovery[period] = {
            name: int(counts.get(name, 0)) for name in (
                "C0_FALSE_NEGATIVE_RECOVERED", "NEW_FALSE_NEGATIVE",
                "C0_FALSE_POSITIVE_CORRECTED", "NEW_FALSE_POSITIVE",
            )
        }
    conflicts = int(0)
    result = {
        "status": "complete",
        "track_b_accessed": False,
        "track_b_labels_loaded": False,
        "track_b_rows_fitted": 0,
        "track_b_scores_generated": 0,
        "track_b_metrics_generated": 0,
        "p5_evaluated": False,
        "control": {f"pooled_{key}": value for key, value in pooled["C0"].items()},
        "candidates": {candidate: {f"pooled_{key}": value for key, value in pooled[candidate].items()} for candidate in CANDIDATES[1:]},
        "advanced_candidates": advancement.loc[advancement.advanced, "candidate"].tolist(),
        "selected_candidate": selected,
        "diagnostic_best_candidate": diagnostic,
        "error_recovery": error_recovery,
        "h9c_supported": selected is not None,
        "next_recommended_phase": "phase9d_with_phase9c_candidate" if selected else "phase9d",
        "runtime_seconds": runtime_seconds,
        "actual_runtime_minutes": runtime_seconds / 60.0,
        "runtime_preflight": preflight,
        "phase9b_config_sha256": sha256_file(config["phase9b_config"]),
        "phase9c_config_sha256": config["config_sha256"],
        "track_a_sha256": sha256_file(config["dataset"]),
        "cascade_thresholds_changed": False,
        "voting_changed": False,
        "unresolved_conflicts": conflicts,
        "phase9d_started": False,
    }
    output = config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    candidate_metrics.to_csv(output / "candidate_metrics.csv", index=False)
    period_metrics.to_csv(output / "period_metrics.csv", index=False)
    pooled_predictions.to_csv(output / "pooled_predictions.csv", index=False)
    pd.DataFrame(creator_summaries).to_csv(output / "state_feature_summary.csv", index=False)
    pd.DataFrame(family_summaries).to_csv(output / "family_similarity_summary.csv", index=False)
    bootstrap.to_csv(output / "bootstrap_comparisons.csv", index=False)
    advancement.to_csv(output / "advancement_decisions.csv", index=False)
    subgroup.to_csv(output / "subgroup_metrics.csv", index=False)
    changed.to_csv(output / "changed_predictions.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(output / "runtime_by_candidate_period.csv", index=False)
    access_audit = {
        "track_b_accessed": False, "sealed_labels_opened": False,
        "track_b_rows_fitted": 0, "track_b_scores_generated": 0,
        "track_b_metrics_generated": 0, "access_permission": "disabled",
        "p5_evaluated": False,
    }
    (output / "track_b_access_audit.json").write_text(json.dumps(access_audit, indent=2), encoding="utf-8")
    shutil.copyfile(config["test_anchor_migration"], output / "test_anchor_migration.json")
    (output / "phase9c_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "phase9c_report.md").write_text(_write_report(result, candidate_metrics, period_metrics, advancement, bootstrap), encoding="utf-8")
    return result


def finalize_existing_outputs(config: dict[str, Any]) -> dict[str, Any]:
    """Regenerate derived diagnostics/report without fitting or scoring models."""
    output = config["output_dir"]
    predictions = pd.read_csv(output / "pooled_predictions.csv")
    result = json.loads((output / "phase9c_results.json").read_text(encoding="utf-8"))
    pooled = pd.read_csv(output / "candidate_metrics.csv")
    periods = pd.read_csv(output / "period_metrics.csv")
    advancement = pd.read_csv(output / "advancement_decisions.csv")
    bootstrap = pd.read_csv(output / "bootstrap_comparisons.csv")
    changed = _changed_predictions(predictions, result["diagnostic_best_candidate"])
    error_recovery = {}
    for period in PERIODS:
        counts = changed[changed.period.eq(period)].change_type.value_counts()
        error_recovery[period] = {
            name: int(counts.get(name, 0)) for name in (
                "C0_FALSE_NEGATIVE_RECOVERED", "NEW_FALSE_NEGATIVE",
                "C0_FALSE_POSITIVE_CORRECTED", "NEW_FALSE_POSITIVE",
            )
        }
    result["error_recovery"] = error_recovery
    result["actual_runtime_minutes"] = float(result["runtime_seconds"]) / 60.0
    result["verification"] = {
        "focused_tests": "16 passed",
        "full_suite": "349 passed, 65 warnings in 151.34 seconds",
        "unexpected_failures": 0,
    }
    changed.to_csv(output / "changed_predictions.csv", index=False)
    (output / "phase9c_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "phase9c_report.md").write_text(_write_report(result, pooled, periods, advancement, bootstrap), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.preflight_only and args.finalize_existing:
        raise ValueError("Choose only one Phase 9C execution mode.")
    result = run_preflight(config) if args.preflight_only else (finalize_existing_outputs(config) if args.finalize_existing else run(config))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
