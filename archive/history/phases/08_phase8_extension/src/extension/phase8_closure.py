"""Deterministic evidence consolidation for the formal Phase 8 closure."""

from __future__ import annotations

import csv
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import yaml

from src.experiments.phase6a import sha256_file, verify_frozen_manifest
from src.utils.paths import REPO_ROOT


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "extension" / "phase8_closure.yaml"
METRICS = ("precision", "recall", "f1", "pr_auc", "balanced_accuracy")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["closure_date"] = str(config["closure_date"])
    config["frozen_manifests"] = [_resolve(item) for item in config["frozen_manifests"]]
    config["inputs"] = {key: _resolve(value) for key, value in config["inputs"].items()}
    config["output_dir"] = _resolve(config["output_dir"])
    config["docs_summary"] = _resolve(config["docs_summary"])
    return config


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row(rows: list[dict[str, str]], **wanted: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in wanted.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {wanted}, found {len(matches)}")
    return matches[0]


def _number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def verify_frozen_inputs(config: dict[str, Any]) -> dict[str, Any]:
    observed: dict[str, str] = {}
    for manifest in config["frozen_manifests"]:
        observed.update(verify_frozen_manifest(manifest))

    raw_path = REPO_ROOT / "data" / "raw" / "PonziDataset_20221114.csv"
    raw_hash = sha256_file(raw_path)
    expected_raw = "48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf"
    if raw_hash != expected_raw:
        raise ValueError(f"Raw dataset hash mismatch: {raw_hash}")

    checkpoints = _read_json(config["inputs"]["phase8r5b_checkpoints"])["checkpoints"]
    for entry in checkpoints:
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {entry['path']}")

    embeddings = _read_json(config["inputs"]["phase8r5c1_embeddings"])["embeddings"]
    for entry in embeddings:
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Embedding hash mismatch: {entry['path']}")

    return {
        "verified": True,
        "frozen_anchor_count": len(observed),
        "checkpoint_count": len(checkpoints),
        "embedding_artifact_count": len(embeddings),
        "raw_dataset_sha256": raw_hash,
        "anchor_fingerprint": hashlib.sha256(
            "".join(f"{key}:{observed[key]}\n" for key in sorted(observed)).encode("utf-8")
        ).hexdigest(),
    }


def _macro_entry(
    phase: str,
    intervention: str,
    candidate: str,
    source: dict[str, Any],
    criterion: str,
    eligible: bool,
    role: str,
    conclusion: str,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "intervention": intervention,
        "temporal_periods": "P2-P4",
        "candidate_system": candidate,
        **{f"macro_{metric}": source.get(metric) for metric in METRICS},
        "eligibility_criterion": criterion,
        "eligible": eligible,
        "selection_role": role,
        "conclusion": conclusion,
    }


def build_closure(config: dict[str, Any]) -> dict[str, Any]:
    verification = verify_frozen_inputs(config)
    inputs = config["inputs"]
    baseline = _read_json(inputs["phase55_mulcas"])
    phase7 = _read_json(inputs["phase7_summary"])
    r1 = _read_csv(inputs["phase8r1a_summary"])
    r2 = _read_csv(inputs["phase8r2b_summary"])
    r3a = _read_json(inputs["phase8r3a_results"])
    r3a_novelty = _read_csv(inputs["phase8r3a_novelty_interaction"])
    r3_p3 = _read_csv(inputs["phase8r3b_p3_metrics"])
    r3_p4 = _read_csv(inputs["phase8r3b_p4_metrics"])
    r4 = _read_csv(inputs["phase8r4b_summary"])
    r5 = _read_json(inputs["phase8r5c1_results"])

    phase55 = baseline["experimental_reconstruction"]["test"]
    published = baseline["published_mulcas"]
    r1_fallback = _row(r1, strategy_id="exponential_decay_0.75")
    r2_fallback = _row(r2, architecture_id="gin_l3_h128_d05")
    r4_fallback = _row(r4, candidate_id="logistic_regression__creation_full")

    mulcas_macro = r5["macro_metrics"]["frozen_mulcas"]
    contextual_macro = r5["macro_metrics"]["contextual"]
    noncontextual_macro = r5["macro_metrics"]["non_contextual"]
    phase7_p5_fn = phase7["phase7d"]["p5_false_negatives"]
    r3a_feasibility = r3a["hybrid_feasibility"]
    r3a_recoveries = _row(
        r3a_novelty, period="P2-P4", disagreement_category="gin_correct_mulcas_wrong"
    )
    if (
        phase7_p5_fn["without_exact_historical_opcode_clone"]
        != phase7_p5_fn["without_exact_historical_runtime_clone"]
    ):
        raise ValueError("P5 opcode/runtime no-clone counts do not agree")
    criterion = "macro precision >= 0.80"

    consolidated = [
        _macro_entry(
            "control", "Frozen temporal control", "Frozen MulCas", mulcas_macro,
            "reference only", False, "retained_reference",
            "Reference system; not a Phase 8 candidate.",
        ),
        _macro_entry(
            "8R1A", "Exponential temporal decay", "Decay alpha 0.75",
            {metric: _number(r1_fallback, f"macro_mean_{metric}") for metric in METRICS},
            criterion, False, "diagnostic_fallback", "H8R1 not supported.",
        ),
        _macro_entry(
            "8R2B", "Runtime CFG message passing", "GIN 3x128 dropout 0.5",
            {metric: _number(r2_fallback, f"macro_mean_{metric}") for metric in METRICS},
            criterion, False, "diagnostic_fallback", "H8R2 not supported.",
        ),
        {
            "phase": "8R3B", "intervention": "MulCas-GIN score fusion",
            "temporal_periods": "P3 selection; P4 confirmation", "candidate_system": "Hybrid A",
            **{f"macro_{metric}": None for metric in METRICS},
            "eligibility_criterion": "P3 precision >= 0.80", "eligible": False,
            "selection_role": "diagnostic_fallback", "conclusion": "H8R3B not supported.",
        },
        _macro_entry(
            "8R4B", "Creation-code representation", "Logistic Regression + creation_full",
            {metric: _number(r4_fallback, f"macro_mean_{metric}") for metric in METRICS},
            criterion, False, "diagnostic_fallback", "H8R4 not supported.",
        ),
        _macro_entry(
            "8R5C1", "Frozen self-supervised sequence embeddings", "Contextual seed mean",
            contextual_macro, criterion, False, "hypothesis_candidate",
            "H8R5 not supported by the frozen linear probe.",
        ),
        _macro_entry(
            "8R5C1", "Pretrained token-embedding control", "Non-contextual seed mean",
            noncontextual_macro, criterion, False, "scientific_control",
            "Control only; ineligible and not a replacement.",
        ),
    ]

    temporal = build_temporal_comparison(config)
    hypotheses = {
        "H8R1": {"status": "not_supported", "evidence_phase": "8R1A", "scope": "recency-aware training"},
        "H8R2": {"status": "not_supported", "evidence_phase": "8R2B", "scope": "runtime CFG and GIN under the predefined eligibility criterion"},
        "H8R3B": {"status": "not_supported", "evidence_phase": "8R3B", "scope": "lightweight learned MulCas-GIN score fusion"},
        "H8R4": {"status": "not_supported", "evidence_phase": "8R4B", "scope": "creation/deployment-bytecode representation"},
        "H8R5": {"status": "not_supported_by_frozen_linear_probe", "evidence_phase": "8R5C1", "scope": "temporally pretrained opcode-sequence representation; does not disprove all fine-tuning methods"},
    }
    construction_phases = {
        "8R0": "protocol_and_feasibility_only", "8R2A": "representation_construction_only",
        "8R3A": "saved_prediction_analysis_only", "8R4A": "representation_construction_only",
        "8R5A": "feasibility_only", "8R5B0": "systems_benchmark_only",
        "8R5B": "self_supervised_pretraining_only",
    }
    decisions = [
        {"decision": "retained_reference", "status": "phase5.5_mulcas", "rationale": "No Phase 8 candidate met its predefined temporal acceptance criterion."},
        {"decision": "phase8_static_model_search", "status": "closed", "rationale": "Multiple materially different static directions were tested without an accepted replacement."},
        {"decision": "phase8r5c2", "status": "not_executed_not_justified", "rationale": "The predefined Phase 8R5C1 fine-tuning gate failed."},
        {"decision": "track_a", "status": "previously_observed_exploratory_only", "rationale": "The 6,498-contract corpus and P5 have been heavily inspected."},
        {"decision": "track_b", "status": "not_established", "rationale": "No overlap-audited independent later dataset is available."},
        {"decision": "next_recommended_phase", "status": "phase8r6a_not_started", "rationale": "Audit external/later data provenance and overlap before more model development."},
        {"decision": "final_synthesis", "status": "pending", "rationale": "Notebook 08 remains pending and was not modified by closure."},
    ]
    return {
        "phase": "phase8_closure", "status": "complete", "closure_date": config["closure_date"],
        "closure_type": "documentation_evidence_consolidation_and_governance",
        "scope": config["scope"], "frozen_verification": verification,
        "retained_reference": "phase5.5_mulcas", "phase8_replacement_found": False,
        "retained_reference_p5": {key: phase55[key] for key in ("precision", "recall", "f1", "pr_auc", "tn", "fp", "fn", "tp")},
        "published_table5_reference": {key: published[key] for key in ("precision", "recall", "f1")},
        "published_gap_experimental_minus_published": {
            key: phase55[key] - published[key] for key in ("precision", "recall", "f1")
        },
        "hypotheses": hypotheses, "construction_and_analysis_phases": construction_phases,
        "consolidated_results": consolidated, "temporal_comparison": temporal,
        "phase7_failure_context": {
            "p5_false_negatives": phase7_p5_fn["count"],
            "p5_false_negatives_unseen_creator": phase7_p5_fn["unseen_creator"],
            "p5_false_negatives_without_exact_opcode_or_runtime_clone": phase7_p5_fn["without_exact_historical_opcode_clone"],
            "p5_false_negatives_confidently_rejected_by_viewer1": phase7_p5_fn["confident_wrong_first_viewer"],
            "phase8r3a_gin_only_ponzi_recoveries_p2_p4": int(r3a_recoveries["true_ponzi_count"]),
            "phase8r3a_gin_only_recoveries_novel_and_unseen": min(
                int(r3a_recoveries["novel_true_ponzi_count"]),
                int(r3a_recoveries["unseen_creator_true_ponzi_count"]),
                r3a_feasibility["novel_or_unseen_gin_only_ponzi_recoveries"],
            ),
        },
        "p5_status": "previously_observed_track_a_only", "track_a_status": "heavily_inspected_exploratory_only",
        "track_b_status": "not_established", "phase8r5c2_status": "not_justified_under_predefined_gate",
        "next_recommended_phase": "phase8r6a", "phase8r6a_status": "not_started",
        "rq6_answer": "The tested Phase 8 approaches did not produce a reliable improvement under the predefined temporal criteria. Period-specific diagnostic benefits did not generalize strongly enough across P2-P4 to replace frozen MulCas.",
        "research_decisions": decisions,
        "claims_not_made": ["concept_drift_proven", "novel_code_causes_errors", "all_sequence_fine_tuning_methods_disproven", "exact_paper_replication"],
        "phase7_closure_status": phase7["status"],
    }


def build_temporal_comparison(config: dict[str, Any]) -> list[dict[str, Any]]:
    inputs = config["inputs"]
    result: list[dict[str, Any]] = []

    for row in _read_csv(inputs["phase7a_period_metrics"]):
        if row["prediction_period"] in {"P2", "P3", "P4"}:
            result.append(_temporal_row("Frozen MulCas", row["prediction_period"], row, "retained_reference"))

    for row in _read_csv(inputs["phase8r2b_period_metrics"]):
        if row["architecture_id"] == "gin_l3_h128_d05":
            mapped = {metric: row[f"mean_{metric}"] for metric in METRICS}
            result.append(_temporal_row("GIN 3x128 dropout 0.5", row["prediction_period"], mapped, "diagnostic_fallback_seed_mean"))

    for row in _read_csv(inputs["phase8r4b_period_metrics"]):
        if row["candidate_id"] == "logistic_regression__creation_full":
            result.append(_temporal_row("LR + creation_full", row["period"], row, "diagnostic_fallback"))

    r5 = _read_json(inputs["phase8r5c1_results"])
    for row in r5["ensemble_metrics"]:
        system = "Contextual sequence ensemble" if row["representation"] == "contextual" else "Non-contextual sequence control"
        role = "fixed_three_seed_ensemble" if row["representation"] == "contextual" else "scientific_control_ensemble"
        result.append(_temporal_row(system, row["period"], row, role))
    return result


def _temporal_row(system: str, period: str, source: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "system": system, "period": period,
        **{metric: float(source[metric]) for metric in METRICS},
        "role": role,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(config: dict[str, Any], result: dict[str, Any]) -> None:
    output = config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "phase8_closure_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    hypothesis_rows = [
        {"hypothesis": key, **value} for key, value in result["hypotheses"].items()
    ]
    _write_csv(output / "phase8_hypothesis_status.csv", hypothesis_rows)
    _write_csv(output / "phase8_consolidated_results.csv", result["consolidated_results"])
    _write_csv(output / "phase8_temporal_comparison.csv", result["temporal_comparison"])
    _write_csv(output / "phase8_research_decisions.csv", result["research_decisions"])
    report = render_report(result)
    (output / "phase8_closure_report.md").write_text(report, encoding="utf-8")
    config["docs_summary"].write_text(report, encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    return str(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def render_report(result: dict[str, Any]) -> str:
    verification = result["frozen_verification"]
    hypothesis_lines = [
        f"| {key} | {value['scope']} | {value['status'].replace('_', ' ').upper()} | {value['evidence_phase']} |"
        for key, value in result["hypotheses"].items()
    ]
    consolidated_lines = [
        f"| {row['phase']} | {row['candidate_system']} | {_fmt(row['macro_precision'])} | "
        f"{_fmt(row['macro_recall'])} | {_fmt(row['macro_f1'])} | {_fmt(row['macro_pr_auc'])} | "
        f"{row['selection_role']} | {row['eligible']} |"
        for row in result["consolidated_results"]
    ]
    temporal_lines = [
        f"| {row['system']} | {row['period']} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | {_fmt(row['f1'])} | {_fmt(row['pr_auc'])} |"
        for row in result["temporal_comparison"]
    ]
    p5 = result["retained_reference_p5"]
    published = result["published_table5_reference"]
    gap = result["published_gap_experimental_minus_published"]
    return f"""# Phase 8 Static Research Extension Closure

## 1. Purpose

This document is the formal closure record for Phase 8R0 through Phase 8R5C1. It consolidates frozen evidence; it does not add an experiment or reinterpret an unsuccessful intervention as a success.

## 2. Scope

No model was trained, no threshold was changed, no P5 evaluation was run, and Phase 8R6A was not started. All metrics below were read from completed artifacts.

## 3. Frozen Inputs Verified

- Frozen artifact anchors verified: **{verification['frozen_anchor_count']}**
- Phase 8R5B checkpoints reverified: **{verification['checkpoint_count']}**
- Phase 8R5C1 embedding caches reverified: **{verification['embedding_artifact_count']}**
- Raw dataset SHA-256: `{verification['raw_dataset_sha256']}`
- Verification status: **PASS**

## 4. Phase 8 Hypothesis Register

| Hypothesis | Tested claim | Status | Evidence |
| --- | --- | --- | --- |
{chr(10).join(hypothesis_lines)}

Phase 8R0, 8R2A, 8R3A, 8R4A, 8R5A, 8R5B0, and 8R5B were protocol, construction, analysis, feasibility, systems, or representation-learning stages. They did not independently establish a supervised hypothesis result. Phase 8R5C2 was not executed because the predefined Phase 8R5C1 gate failed. This does not disprove every possible sequence fine-tuning method.

## 5. Results by Phase

| Phase | System | Macro P | Macro R | Macro F1 | Macro PR-AUC | Role | Eligible |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
{chr(10).join(consolidated_lines)}

Diagnostic fallbacks are preserved for interpretation only. None is a research winner.

## 6. P2-P4 Temporal Comparison

| System | Period | Precision | Recall | F1 | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(temporal_lines)}

P4 was consistently easier for several systems. P2 and P3 were much less stable, and some methods improved one period while degrading another. No Phase 8 method produced a reliable cross-period improvement under its predefined criterion. This is temporal instability and evidence consistent with distribution variation; it is not proof of concept drift.

## 7. Failure-Mode Synthesis

Phase 7 found 37 P5 false negatives: 36 involved unseen creators, all 37 lacked an exact historical opcode/runtime clone, and 28 were confidently rejected by MulCas viewer 1. These are descriptive associations, not causes.

Phase 8R3A showed genuine period-specific complementarity: the GIN recovered 28 P2-P4 Ponzi contracts missed by MulCas, all with historically novel code and unseen creators, with most of the benefit in P3. The learned Phase 8R3B fusion nevertheless predicted no P3 positives and degraded P4. Creation-code models showed weak positive-class complementarity. The contextual sequence representation underperformed frozen MulCas and its non-contextual control in macro F1; the non-contextual control's tiny macro-F1 increase remained ineligible because macro precision was only 0.5297.

## 8. Comparison With Frozen MulCas

No Phase 8 intervention replaces the frozen Phase 5.5 paper-aligned MulCas. Its P5 result remains P={_fmt(p5['precision'])}, R={_fmt(p5['recall'])}, F1={_fmt(p5['f1'])}, PR-AUC={_fmt(p5['pr_auc'])}, with TN/FP/FN/TP={p5['tn']}/{p5['fp']}/{p5['fn']}/{p5['tp']}.

P5 is no longer an untouched independent test after Phase 5.5 and Phase 7. Any future use must be labeled exploratory Track A evidence.

## 9. Published Benchmark Gap

| Metric | Published Table 5 | Frozen Phase 5.5 | Difference |
| --- | ---: | ---: | ---: |
| Precision | {_fmt(published['precision'])} | {_fmt(p5['precision'])} | {_fmt(gap['precision'])} |
| Recall | {_fmt(published['recall'])} | {_fmt(p5['recall'])} | {_fmt(gap['recall'])} |
| F1 | {_fmt(published['f1'])} | {_fmt(p5['f1'])} | {_fmt(gap['f1'])} |

This is not an exact replication comparison. The paper reports 5,990/508 rows while whole-block reconstruction gives 5,989/509; Table 5 trains through the 250th Ponzi while Phase 5.5 fits through the 200th and reserves P4 for validation; the paper's viewer selection appears test-based while this project uses validation/temporal protocols; and the N-gram dimensionality does not match exactly.

## 10. Why No Phase 8 Replacement Was Accepted

Recency weighting, runtime CFG aggregates and GINs, learned score fusion, creation-code views, and self-supervised opcode-sequence embeddings all failed their predefined temporal acceptance rules. Period-specific improvements were insufficient evidence of reliable P2-P4 generalization.

## 11. What Was Deliberately Not Executed

Phase 8R5C2 was not run because its gate failed. GAT, GraphSAGE, larger GINs, extra decay values, further threshold searches, additional fusion models, creation-code fusion, and arbitrary model shopping were not launched. Any future attempt requires a new preregistered rationale.

## 12. Track A / Track B Status

- **Track A:** the existing 6,498-contract corpus is heavily inspected and remains useful for exploratory analysis and controlled temporal experiments, but not untouched final-generalization claims.
- **Track B:** no genuinely external, later, independently labeled and overlap-audited benchmark has been established.

## 13. Updated RQ6 Answer

The tested Phase 8 approaches did not produce a reliable improvement under the predefined temporal criteria. Some methods showed period-specific or diagnostic benefits, especially GIN on P3, but no candidate generalized strongly enough across P2-P4 to replace frozen MulCas.

## 14. Limitations

P5 has been observed; the corpus ends in January 2019; no Track B benchmark exists; period imbalance is extreme; P2 offers sparse positive history for representation learning; CFG jump resolution is conservative; the sequence conclusion is linear-probe-specific; CUDA retraining is not guaranteed bit-identical; present-day generalization is unknown; and the paper's protocol and feature details remain under-specified.

## 15. Research-Governance Decision

The current Phase 8 static-model search is closed. Continuing to add classifiers, thresholds, graph variants, or post-hoc combinations on the same corpus is not justified by the current evidence. This does not mean improvement is impossible.

## 16. Recommended Next Phase

The next recommended phase is **Phase 8R6A: external/later dataset discovery, provenance, and overlap audit**. It is **not started**. New independent temporal coverage is more informative now than another unmotivated classifier search.

## 17. Reproducibility and Tests

The closure generator verifies the full frozen-manifest chain, raw data, nine MLM checkpoints, and 18 embedding caches. Closure outputs are deterministic and contain no training code. The complete repository test command is `python -m pytest -q`.

## 18. Final Phase 8 Closure Statement

Phase 8 is closed without an accepted replacement for the frozen Phase 5.5 MulCas reference. Recency-aware learning, runtime CFG/GIN, learned score fusion, creation-code features, and frozen self-supervised sequence embeddings did not produce a reliable improvement under their predefined temporal criteria. These negative results do not establish that improvement is impossible. They show that the tested static interventions did not resolve the observed temporal-generalization weakness on the current dataset. The next recommended research step is an external/later dataset provenance and overlap audit before further model development.
"""


def main() -> None:
    config = load_config()
    result = build_closure(config)
    write_outputs(config, result)
    print(json.dumps({
        "status": result["status"],
        "frozen_anchor_count": result["frozen_verification"]["frozen_anchor_count"],
        "replacement_found": result["phase8_replacement_found"],
        "output_dir": str(config["output_dir"]),
    }, indent=2))


if __name__ == "__main__":
    main()
