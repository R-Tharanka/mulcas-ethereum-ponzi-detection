"""Phase 9D out-of-time viewer calibration and fixed safer routing."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.evaluation.temporal import TemporalScenario, build_ponzi_occurrence_periods
from src.experiments.phase7a import (
    _load_frozen_configs,
    build_historical_feature_matrices,
    fit_predict_frozen_mulcas,
    load_full_dataset,
)
from src.phase9.phase9_metrics import paired_bootstrap, pooled_metrics
from src.utils.paths import REPO_ROOT
from src.utils.seed import set_random_seed


DEFAULT_CONFIG = REPO_ROOT / "configs/phase9/phase9d.yaml"
PERIODS = ("P2", "P3", "P4")
CANDIDATES = ("D0", "D1", "D2", "D3")
CASCADE_VIEWERS = (
    "xgboost_term_count",
    "xgboost_ngram_count_df_filtered",
)
VOTING_VIEWERS = (
    "xgboost_term_count",
    "xgboost_ngram_count_df_filtered",
    "xgboost_ngram_tfidf_df_filtered",
    "ridge_tfidf",
)
STREAMS = tuple(
    [("cascade", name) for name in CASCADE_VIEWERS]
    + [("voting", name) for name in VOTING_VIEWERS]
)


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
        "paper_mulcas_config", "output_dir",
    ):
        config[key] = _resolve(config[key])
    if tuple(config["candidates"]) != CANDIDATES:
        raise ValueError("Phase 9D candidates must remain exactly D0-D3.")
    if config["track_b"]["access_permission"] != "disabled":
        raise PermissionError("Track B access must remain disabled during Phase 9D.")
    if config["temporal_protocol"]["p5_evaluation"] != "prohibited":
        raise ValueError("P5 evaluation must remain prohibited.")
    if config["routing"]["threshold_search"] != "prohibited":
        raise ValueError("Phase 9D threshold search must remain prohibited.")
    if int(config["routing"]["positive_confirmation_votes"]) != 2:
        raise ValueError("Positive confirmation must remain fixed at two votes.")
    if int(config["routing"]["negative_recovery_votes"]) != 3:
        raise ValueError("Negative recovery must remain fixed at three votes.")
    config["config_path"] = path.resolve()
    config["config_sha256"] = sha256_file(path)
    return config


def _load_inputs(config: dict[str, Any]):
    if sha256_file(config["dataset"]) != config["dataset_sha256"]:
        raise RuntimeError("Track A raw hash mismatch.")
    protocol = yaml.safe_load(config["phase9b_config"].read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_protocol_only":
        raise RuntimeError("Phase 9B protocol is not frozen.")
    frame = load_full_dataset(config["dataset"])
    phase7 = yaml.safe_load(config["phase7a_config"].read_text(encoding="utf-8"))
    assigned, _ = build_ponzi_occurrence_periods(
        frame,
        positive_ends=phase7["periods"]["positive_ends"],
        expected_positives=int(phase7["periods"]["expected_total_positives"]),
    )
    assigned = assigned[assigned["period"].isin(("P0", "P1", "P2", "P3", "P4"))].copy()
    feature, model, mulcas = _load_frozen_configs({
        "paper_feature_config": config["paper_feature_config"],
        "paper_model_config": config["paper_model_config"],
        "paper_mulcas_config": config["paper_mulcas_config"],
    })
    return assigned, feature, model, mulcas, protocol


def _score_column(kind: str, viewer: str) -> str:
    return f"{kind}__{viewer}__score"


def _prediction_column(kind: str, viewer: str) -> str:
    return f"{kind}__{viewer}__prediction"


def _viewer_frame(target: pd.DataFrame, outputs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = target[["row_index", "address", "createdBlockNumber", "label", "period"]].copy()
    for kind, viewer in STREAMS:
        output = outputs[kind][viewer]
        rows[_score_column(kind, viewer)] = rows["row_index"].map(output.probability)
        rows[_prediction_column(kind, viewer)] = rows["row_index"].map(output.prediction)
    return rows


def _generate_p1_calibration(config: dict[str, Any], assigned, feature, model, mulcas):
    train = assigned[assigned.period.eq("P0")].sort_values(["createdBlockNumber", "row_index"])
    target = assigned[assigned.period.eq("P1")].sort_values(["createdBlockNumber", "row_index"])
    views = list(dict.fromkeys(item["feature_view"] for item in mulcas["viewer_sequence"]))
    started = time.perf_counter()
    matrices, audit, _ = build_historical_feature_matrices(
        train, target, views, ngram_min_df=int(feature["ngram"]["min_df"])
    )
    _, _, training_trace, outputs = fit_predict_frozen_mulcas(
        train, target, matrices, model, mulcas, seed=int(config["seed"])
    )
    return _viewer_frame(target, outputs), {
        "fit_periods": "P0",
        "calibration_period": "P1",
        "fit_rows": int(len(train)),
        "calibration_rows": int(len(target)),
        "fit_max_block": int(train.createdBlockNumber.max()),
        "calibration_min_block": int(target.createdBlockNumber.min()),
        "historical_only": bool(train.createdBlockNumber.max() < target.createdBlockNumber.min()),
        "feature_audit": audit,
        "cascade_training_trace": training_trace,
        "runtime_seconds": time.perf_counter() - started,
    }


def _clip_logit(values: np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def _fit_calibrators(frame: pd.DataFrame, config: dict[str, Any], target_period: str):
    labels = frame["label"].to_numpy(dtype=int)
    if np.unique(labels).size != 2:
        raise RuntimeError(f"Calibration data for {target_period} lacks both classes.")
    epsilon = float(config["calibration"]["probability_clip"])
    models: dict[tuple[str, str], LogisticRegression] = {}
    audit: list[dict[str, Any]] = []
    for kind, viewer in STREAMS:
        column = _score_column(kind, viewer)
        raw = frame[column].to_numpy(dtype=float)
        calibrator = LogisticRegression(
            C=float(config["calibration"]["logistic_c"]),
            solver=config["calibration"]["solver"],
            max_iter=int(config["calibration"]["max_iter"]),
            class_weight=config["calibration"]["class_weight"],
            random_state=int(config["seed"]),
        )
        calibrator.fit(_clip_logit(raw, epsilon), labels)
        calibrated = calibrator.predict_proba(_clip_logit(raw, epsilon))[:, 1]
        audit.append({
            "target_period": target_period,
            "calibration_period": str(frame["period"].iloc[0]),
            "stream": kind,
            "viewer": viewer,
            "rows": int(len(frame)),
            "positives": int(labels.sum()),
            "negatives": int(len(labels) - labels.sum()),
            "coefficient": float(calibrator.coef_[0, 0]),
            "intercept": float(calibrator.intercept_[0]),
            "raw_brier": float(brier_score_loss(labels, raw)),
            "calibrated_brier": float(brier_score_loss(labels, calibrated)),
            "raw_log_loss": float(log_loss(labels, np.clip(raw, epsilon, 1 - epsilon))),
            "calibrated_log_loss": float(log_loss(labels, np.clip(calibrated, epsilon, 1 - epsilon))),
        })
        models[(kind, viewer)] = calibrator
    return models, audit


def _calibrated_scores(frame: pd.DataFrame, models, config: dict[str, Any]):
    epsilon = float(config["calibration"]["probability_clip"])
    return {
        (kind, viewer): models[(kind, viewer)].predict_proba(
            _clip_logit(frame[_score_column(kind, viewer)].to_numpy(), epsilon)
        )[:, 1]
        for kind, viewer in STREAMS
    }


def _route(scores: dict[tuple[str, str], np.ndarray], mulcas: dict[str, Any]):
    size = len(next(iter(scores.values())))
    prediction = np.full(size, -1, dtype=int)
    probability = np.full(size, np.nan, dtype=float)
    route = np.full(size, "", dtype=object)
    unresolved = np.ones(size, dtype=bool)
    for viewer in mulcas["viewer_sequence"][: int(mulcas["complexity_control_n"])]:
        name = viewer["name"]
        values = scores[("cascade", name)]
        accepted = unresolved & (
            (values < float(viewer["negative_threshold"]))
            | (values > float(viewer["positive_threshold"]))
        )
        prediction[accepted] = (values[accepted] >= 0.5).astype(int)
        probability[accepted] = values[accepted]
        route[accepted] = name
        unresolved[accepted] = False
    vote_matrix = np.column_stack([scores[("voting", name)] for name in VOTING_VIEWERS])
    vote_count = (vote_matrix >= 0.5).sum(axis=1)
    prediction[unresolved] = (vote_count[unresolved] > float(mulcas["voting_threshold"])).astype(int)
    probability[unresolved] = vote_matrix[unresolved].max(axis=1)
    route[unresolved] = "vote"
    return prediction, probability, route, vote_count, vote_matrix


def _guard(base_prediction: np.ndarray, vote_count: np.ndarray, vote_matrix: np.ndarray, config):
    confirm = int(config["routing"]["positive_confirmation_votes"])
    recover = int(config["routing"]["negative_recovery_votes"])
    guarded = np.where(base_prediction == 1, vote_count >= confirm, vote_count >= recover).astype(int)
    routes = np.where(
        (base_prediction == 1) & (guarded == 0), "guard_rejected_positive",
        np.where(
            (base_prediction == 0) & (guarded == 1), "guard_recovered_negative",
            "guard_confirmed",
        ),
    )
    return guarded, vote_matrix.mean(axis=1), routes


def _candidate_rows(period_frame: pd.DataFrame, calibrators, config, mulcas):
    raw = {
        (kind, viewer): period_frame[_score_column(kind, viewer)].to_numpy(dtype=float)
        for kind, viewer in STREAMS
    }
    calibrated = _calibrated_scores(period_frame, calibrators, config)
    raw_pred, raw_score, raw_route, raw_votes, raw_vote_matrix = _route(raw, mulcas)
    cal_pred, cal_score, cal_route, cal_votes, cal_vote_matrix = _route(calibrated, mulcas)
    d2_pred, d2_score, d2_route = _guard(raw_pred, raw_votes, raw_vote_matrix, config)
    d3_pred, d3_score, d3_route = _guard(cal_pred, cal_votes, cal_vote_matrix, config)
    basics = period_frame[["row_index", "address", "createdBlockNumber", "label", "period"]]
    frames = []
    values = {
        "D0": (
            period_frame["final_prediction"].to_numpy(dtype=int),
            period_frame["final_score"].to_numpy(dtype=float),
            period_frame["accepted_viewer"].to_numpy(dtype=object),
            raw_votes,
        ),
        "D1": (cal_pred, cal_score, cal_route, cal_votes),
        "D2": (d2_pred, d2_score, d2_route, raw_votes),
        "D3": (d3_pred, d3_score, d3_route, cal_votes),
    }
    for candidate, (prediction, score, route, votes) in values.items():
        item = basics.copy()
        item["candidate"] = candidate
        item["prediction"] = prediction
        item["score"] = score
        item["route"] = route
        item["positive_votes"] = votes
        for kind, viewer in STREAMS:
            item[f"raw__{kind}__{viewer}"] = raw[(kind, viewer)]
            item[f"calibrated__{kind}__{viewer}"] = calibrated[(kind, viewer)]
        frames.append(item)
    return pd.concat(frames, ignore_index=True)


def _metrics(predictions: pd.DataFrame, config):
    pooled_rows, period_rows, lookup = [], [], {}
    ceiling = float(config["metrics"]["reference_fpr_ceiling"])
    prevalence = float(config["metrics"]["reference_prevalence"])
    for candidate in CANDIDATES:
        selected = predictions[predictions.candidate.eq(candidate)].sort_values("row_index")
        result = pooled_metrics(
            selected.label.to_numpy(), selected.prediction.to_numpy(), selected.score.to_numpy(),
            fpr_ceiling=ceiling, reference_prevalence=prevalence,
        )
        lookup[candidate] = result
        pooled_rows.append({"candidate": candidate, **result})
        for period in PERIODS:
            part = selected[selected.period.eq(period)]
            result_period = pooled_metrics(
                part.label.to_numpy(), part.prediction.to_numpy(), part.score.to_numpy(),
                fpr_ceiling=ceiling, reference_prevalence=prevalence,
            )
            period_rows.append({"candidate": candidate, "period": period, **result_period})
    return pd.DataFrame(pooled_rows), pd.DataFrame(period_rows), lookup


def _verify_control(period_metrics: pd.DataFrame, source_path: Path):
    source = json.loads(source_path.read_text(encoding="utf-8"))
    expected = {item["prediction_period"]: item["metrics"] for item in source["scenarios"]}
    for period in PERIODS:
        row = period_metrics[(period_metrics.candidate == "D0") & (period_metrics.period == period)].iloc[0]
        for metric in ("precision", "recall", "f1", "pr_auc", "tn", "fp", "fn", "tp"):
            if not np.isclose(float(row[metric]), float(expected[period][metric]), atol=1e-12, rtol=0):
                raise RuntimeError(f"D0 does not reproduce frozen {period} {metric}.")


def _bootstrap(predictions: pd.DataFrame, config):
    control = predictions[predictions.candidate.eq("D0")].sort_values("row_index")
    baseline = {
        "truth": control.label.to_numpy(), "prediction": control.prediction.to_numpy(),
        "score": control.score.to_numpy(),
    }
    rows = []
    for candidate in CANDIDATES[1:]:
        selected = predictions[predictions.candidate.eq(candidate)].sort_values("row_index")
        result = paired_bootstrap(
            baseline,
            {"truth": selected.label.to_numpy(), "prediction": selected.prediction.to_numpy(), "score": selected.score.to_numpy()},
            replicates=int(config["bootstrap"]["replicates"]), seed=int(config["bootstrap"]["seed"]),
        )
        rows.append({"candidate": candidate, **result})
    return pd.DataFrame(rows)


def _advancement(pooled, periods, bootstrap, config):
    control = pooled.set_index("candidate").loc["D0"]
    period_control = periods[periods.candidate.eq("D0")].set_index("period")
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


def _select(advancement, pooled, config):
    advanced = advancement.loc[advancement.advanced, "candidate"].tolist()
    if not advanced:
        return None
    metrics = pooled[pooled.candidate.isin(advanced)].set_index("candidate")
    best = float(metrics.f1.max())
    tied = metrics[metrics.f1 >= best - float(config["selection"]["effective_tie_f1"])].copy()
    simplicity = {name: index for index, name in enumerate(config["selection"]["simplicity_order"])}
    tied["simplicity"] = [simplicity[name] for name in tied.index]
    return str(tied.sort_values(["pr_auc", "recall", "simplicity"], ascending=[False, False, True]).index[0])


def _changed(predictions, diagnostic):
    control = predictions[predictions.candidate.eq("D0")].set_index("row_index")
    candidate = predictions[predictions.candidate.eq(diagnostic)].set_index("row_index")
    joined = control[["period", "address", "label", "prediction", "route", "positive_votes"]].join(
        candidate[["prediction", "route", "positive_votes"]], lsuffix="_D0", rsuffix="_candidate"
    )
    joined = joined[joined.prediction_D0.ne(joined.prediction_candidate)].copy()
    joined["change_type"] = np.select(
        [
            (joined.label == 1) & (joined.prediction_D0 == 0) & (joined.prediction_candidate == 1),
            (joined.label == 1) & (joined.prediction_D0 == 1) & (joined.prediction_candidate == 0),
            (joined.label == 0) & (joined.prediction_D0 == 1) & (joined.prediction_candidate == 0),
            (joined.label == 0) & (joined.prediction_D0 == 0) & (joined.prediction_candidate == 1),
        ],
        ["D0_FALSE_NEGATIVE_RECOVERED", "NEW_FALSE_NEGATIVE", "D0_FALSE_POSITIVE_CORRECTED", "NEW_FALSE_POSITIVE"],
        default="OTHER",
    )
    return joined.reset_index()


def _environment():
    try:
        import psutil
        ram = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        ram = None
    return {"cpu": platform.processor(), "ram_gib": ram, "gpu_required": False}


def _project_bootstrap_seconds(
    measured_seconds: float,
    measured_replicates: int,
    candidate_count: int,
    target_replicates: int,
) -> float:
    """Project the serial paired-bootstrap cost from a bounded benchmark."""
    if measured_seconds < 0 or measured_replicates <= 0 or candidate_count <= 0 or target_replicates <= 0:
        raise ValueError("Runtime projection inputs must be positive.")
    return float(measured_seconds * candidate_count * target_replicates / measured_replicates)


def run_preflight(config: dict[str, Any]):
    assigned, feature, model, mulcas, _ = _load_inputs(config)
    started = time.perf_counter()
    _, audit = _generate_p1_calibration(config, assigned, feature, model, mulcas)
    score_generation_seconds = time.perf_counter() - started

    # Benchmark the dominant uncertainty-analysis cost on the exact pooled
    # development rows. This is D0-vs-D0 timing only, not a candidate result.
    archived = pd.read_csv(config["phase7a_predictions"])
    pooled = archived[archived.period.isin(PERIODS)].sort_values("row_index")
    benchmark_replicates = 100
    baseline = {
        "truth": pooled.label.to_numpy(),
        "prediction": pooled.final_prediction.to_numpy(),
        "score": pooled.final_score.to_numpy(),
    }
    bootstrap_started = time.perf_counter()
    paired_bootstrap(
        baseline,
        baseline,
        replicates=benchmark_replicates,
        seed=int(config["bootstrap"]["seed"]),
    )
    bootstrap_benchmark_seconds = time.perf_counter() - bootstrap_started
    raw_projected_bootstrap_seconds = _project_bootstrap_seconds(
        bootstrap_benchmark_seconds,
        benchmark_replicates,
        len(CANDIDATES) - 1,
        int(config["bootstrap"]["replicates"]),
    )
    # Short bootstrap batches benefit disproportionately from warm caches and
    # omit CSV/report serialization, so retain a conservative whole-run margin.
    projection_safety_factor = 3.0
    projected_bootstrap_seconds = raw_projected_bootstrap_seconds * projection_safety_factor
    projected_total_seconds = score_generation_seconds * 1.25 + projected_bootstrap_seconds
    estimate = projected_total_seconds / 60.0
    result = {
        "status": "pass" if estimate <= float(config["performance_gate"]["maximum_estimated_runtime_minutes"]) else "approval_required",
        "representative_operation": "P0_to_P1_score_generation_plus_bounded_paired_bootstrap",
        "score_generation_runtime_seconds": score_generation_seconds,
        "bootstrap_benchmark_replicates": benchmark_replicates,
        "bootstrap_benchmark_runtime_seconds": bootstrap_benchmark_seconds,
        "bootstrap_projection_safety_factor": projection_safety_factor,
        "raw_projected_bootstrap_runtime_seconds": raw_projected_bootstrap_seconds,
        "projected_bootstrap_runtime_seconds": projected_bootstrap_seconds,
        "projected_total_runtime_seconds": projected_total_seconds,
        "estimated_full_runtime_minutes": estimate,
        "estimated_model_fits": 6,
        "estimated_output_size_mib": 20.0,
        **_environment(),
        "historical_only": audit["historical_only"],
    }
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    (config["output_dir"] / "runtime_preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _report(result, pooled, periods, advancement, bootstrap):
    lines = [
        "# Phase 9D Cascade Calibration and Safer Routing", "",
        "## Scope", "",
        "Phase 9D evaluates the frozen MulCas viewer scores on Track A P2-P4 only. "
        "Track B remains sealed, P5 is not evaluated, and no underlying viewer or static representation is changed.", "",
        "## Candidates", "",
        "- **D0:** exact frozen MulCas control.",
        "- **D1:** prior-period Platt calibration with frozen cascade thresholds and voting rule.",
        "- **D2:** fixed two-vote positive confirmation and three-vote negative recovery on raw scores.",
        "- **D3:** D1 calibration followed by the same fixed consensus guard.", "",
        "## Pooled Results", "",
        "| Candidate | Precision | Recall | F1 | PR-AUC | FPR | Recall at FPR ceiling |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pooled.itertuples():
        lines.append(f"| {row.candidate} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {row.pr_auc:.4f} | {row.fpr:.4f} | {row.recall_at_reference_fpr_ceiling:.4f} |")
    lines += ["", "## Period F1", "", "| Candidate | P2 | P3 | P4 |", "| --- | ---: | ---: | ---: |"]
    for candidate in CANDIDATES:
        values = periods[periods.candidate.eq(candidate)].set_index("period").f1
        lines.append(f"| {candidate} | {values['P2']:.4f} | {values['P3']:.4f} | {values['P4']:.4f} |")
    lines += ["", "## Prospective Decision", ""]
    for row in advancement.itertuples():
        lines.append(f"- **{row.candidate}:** {'ADVANCED' if row.advanced else 'NOT ADVANCED'}.")
    if result["h9d_supported"]:
        lines.append(f"\nH9D is supported internally; `{result['selected_candidate']}` satisfies every frozen Phase 9B advancement condition. External generalization has not been evaluated.")
    else:
        lines.append("\nH9D is not supported. The tested calibration and fixed routing guard did not satisfy every prospective Phase 9B advancement condition.")
    lines += [
        "", "## Integrity", "",
        "Track B labels were not opened, no Track B row was fitted or scored, and no Track B metric was generated. P5 was not evaluated. Phase 9E was not started.", "",
        "## Bootstrap", "",
        f"Paired contract-level bootstrap used {int(bootstrap.replicates.iloc[0])} replicates and seed {int(bootstrap.seed.iloc[0])}.",
    ]
    return "\n".join(lines) + "\n"


def run(config: dict[str, Any]):
    preflight_path = config["output_dir"] / "runtime_preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("Run Phase 9D preflight before full execution.")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight["status"] != "pass":
        raise RuntimeError("Phase 9D runtime requires explicit approval.")
    started = time.perf_counter()
    set_random_seed(int(config["seed"]))
    assigned, feature, model, mulcas, _ = _load_inputs(config)
    archived = pd.read_csv(config["phase7a_predictions"])
    archived = archived[archived.period.isin(PERIODS)].copy()
    p1, p1_audit = _generate_p1_calibration(config, assigned, feature, model, mulcas)
    calibration_sources = {
        "P2": p1,
        "P3": archived[archived.period.eq("P2")].copy(),
        "P4": archived[archived.period.eq("P3")].copy(),
    }
    prediction_frames, calibration_rows = [], []
    for period in PERIODS:
        source = calibration_sources[period]
        target = archived[archived.period.eq(period)].copy()
        if int(source.createdBlockNumber.max()) >= int(target.createdBlockNumber.min()):
            raise RuntimeError(f"Calibration chronology violation for {period}.")
        calibrators, audit = _fit_calibrators(source, config, period)
        calibration_rows.extend(audit)
        prediction_frames.append(_candidate_rows(target, calibrators, config, mulcas))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    pooled, periods, lookup = _metrics(predictions, config)
    _verify_control(periods, config["phase7a_results"])
    bootstrap = _bootstrap(predictions, config)
    advancement = _advancement(pooled, periods, bootstrap, config)
    selected = _select(advancement, pooled, config)
    diagnostic = selected or str(pooled[pooled.candidate.ne("D0")].sort_values("f1", ascending=False).iloc[0].candidate)
    changed = _changed(predictions, diagnostic)
    routing = predictions.groupby(["candidate", "period", "route"], dropna=False).size().reset_index(name="row_count")
    runtime = time.perf_counter() - started
    result = {
        "status": "complete",
        "track_b_accessed": False,
        "track_b_labels_loaded": False,
        "track_b_rows_fitted": 0,
        "track_b_scores_generated": 0,
        "track_b_metrics_generated": 0,
        "p5_evaluated": False,
        "control": {f"pooled_{key}": value for key, value in lookup["D0"].items()},
        "candidates": {name: {f"pooled_{key}": value for key, value in lookup[name].items()} for name in CANDIDATES[1:]},
        "advanced_candidates": advancement.loc[advancement.advanced, "candidate"].tolist(),
        "selected_candidate": selected,
        "diagnostic_best_candidate": diagnostic,
        "h9d_supported": selected is not None,
        "next_recommended_phase": "phase9e_with_phase9d_candidate" if selected else "phase9e",
        "phase9e_started": False,
        "underlying_viewers_changed": False,
        "threshold_search_performed": False,
        "calibration_method": config["calibration"]["method"],
        "routing_guard": {
            "positive_confirmation_votes": int(config["routing"]["positive_confirmation_votes"]),
            "negative_recovery_votes": int(config["routing"]["negative_recovery_votes"]),
        },
        "runtime_seconds": runtime,
        "actual_runtime_minutes": runtime / 60.0,
        "runtime_preflight": preflight,
        "p1_generation_audit": p1_audit,
        "phase9b_config_sha256": sha256_file(config["phase9b_config"]),
        "phase9d_config_sha256": config["config_sha256"],
        "track_a_sha256": sha256_file(config["dataset"]),
    }
    output = config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(output / "candidate_metrics.csv", index=False)
    periods.to_csv(output / "period_metrics.csv", index=False)
    predictions.to_csv(output / "pooled_predictions.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output / "calibration_audit.csv", index=False)
    routing.to_csv(output / "routing_summary.csv", index=False)
    bootstrap.to_csv(output / "bootstrap_comparisons.csv", index=False)
    advancement.to_csv(output / "advancement_decisions.csv", index=False)
    changed.to_csv(output / "changed_predictions.csv", index=False)
    access = {
        "track_b_accessed": False, "sealed_labels_opened": False,
        "track_b_rows_fitted": 0, "track_b_scores_generated": 0,
        "track_b_metrics_generated": 0, "access_permission": "disabled",
        "p5_evaluated": False,
    }
    (output / "track_b_access_audit.json").write_text(json.dumps(access, indent=2), encoding="utf-8")
    (output / "phase9d_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "phase9d_report.md").write_text(_report(result, pooled, periods, advancement, bootstrap), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_preflight(config) if args.preflight_only else run(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
