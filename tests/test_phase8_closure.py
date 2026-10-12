from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

import src.extension.phase8_closure as closure


def _result() -> dict:
    return closure.build_closure(closure.load_config())


def test_all_frozen_inputs_and_large_artifact_manifests_verify() -> None:
    verified = closure.verify_frozen_inputs(closure.load_config())
    assert verified["verified"] is True
    assert verified["frozen_anchor_count"] == 199
    assert verified["checkpoint_count"] == 9
    assert verified["embedding_artifact_count"] == 18
    assert verified["raw_dataset_sha256"] == (
        "48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf"
    )


def test_closure_scope_has_no_training_threshold_change_or_p5_evaluation() -> None:
    result = _result()
    assert result["scope"] == {
        "model_training": False,
        "new_experiments": False,
        "p5_evaluation": False,
        "threshold_tuning": False,
        "phase8r6a_started": False,
    }
    source = inspect.getsource(closure)
    assert ".fit(" not in source
    assert "LogisticRegression" not in source
    assert "torch" not in source


def test_retained_reference_and_published_gap_match_frozen_baseline() -> None:
    result = _result()
    assert result["retained_reference"] == "phase5.5_mulcas"
    assert result["phase8_replacement_found"] is False
    assert result["retained_reference_p5"] == {
        "precision": 0.90625,
        "recall": 0.453125,
        "f1": 0.6041666666666666,
        "pr_auc": 0.6624476140288026,
        "tn": 442,
        "fp": 3,
        "fn": 35,
        "tp": 29,
    }
    assert result["published_gap_experimental_minus_published"]["f1"] == pytest.approx(
        -0.1848333333333334
    )


def test_hypothesis_statuses_and_nonexecuted_branches_are_explicit() -> None:
    result = _result()
    assert result["hypotheses"]["H8R1"]["status"] == "not_supported"
    assert result["hypotheses"]["H8R2"]["status"] == "not_supported"
    assert result["hypotheses"]["H8R3B"]["status"] == "not_supported"
    assert result["hypotheses"]["H8R4"]["status"] == "not_supported"
    assert result["hypotheses"]["H8R5"]["status"] == "not_supported_by_frozen_linear_probe"
    assert result["phase8r5c2_status"] == "not_justified_under_predefined_gate"
    assert result["phase8r6a_status"] == "not_started"
    assert result["next_recommended_phase"] == "phase8r6a"


def test_diagnostic_fallbacks_are_not_winners_and_metrics_match_sources() -> None:
    result = _result()
    candidates = {row["candidate_system"]: row for row in result["consolidated_results"]}
    for candidate in (
        "Decay alpha 0.75",
        "GIN 3x128 dropout 0.5",
        "Hybrid A",
        "Logistic Regression + creation_full",
    ):
        assert candidates[candidate]["eligible"] is False
        assert candidates[candidate]["selection_role"] == "diagnostic_fallback"
    assert candidates["GIN 3x128 dropout 0.5"]["macro_f1"] == pytest.approx(
        0.4720768986273083
    )
    assert candidates["Contextual seed mean"]["macro_precision"] == pytest.approx(
        0.28414322378799445
    )


def test_track_wording_and_rq6_answer_preserve_proof_boundaries() -> None:
    result = _result()
    assert result["p5_status"] == "previously_observed_track_a_only"
    assert result["track_a_status"] == "heavily_inspected_exploratory_only"
    assert result["track_b_status"] == "not_established"
    assert "did not produce a reliable improvement" in result["rq6_answer"]
    assert "concept_drift_proven" in result["claims_not_made"]
    assert result["phase7_failure_context"] == {
        "p5_false_negatives": 37,
        "p5_false_negatives_unseen_creator": 36,
        "p5_false_negatives_without_exact_opcode_or_runtime_clone": 37,
        "p5_false_negatives_confidently_rejected_by_viewer1": 28,
        "phase8r3a_gin_only_ponzi_recoveries_p2_p4": 28,
        "phase8r3a_gin_only_recoveries_novel_and_unseen": 28,
    }


def test_closure_outputs_are_deterministic(tmp_path: Path) -> None:
    config = closure.load_config()
    config["output_dir"] = tmp_path / "out"
    config["docs_summary"] = tmp_path / "phase8_summary.md"
    result = closure.build_closure(config)
    closure.write_outputs(config, result)
    first = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(config["output_dir"].iterdir())
    }
    closure.write_outputs(config, result)
    second = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(config["output_dir"].iterdir())
    }
    assert first == second
    stored = json.loads((config["output_dir"] / "phase8_closure_results.json").read_text())
    assert stored["status"] == "complete"


def test_tracked_closure_artifacts_match_a_fresh_render(tmp_path: Path) -> None:
    config = closure.load_config()
    tracked_output = config["output_dir"]
    tracked_summary = config["docs_summary"]
    config["output_dir"] = tmp_path / "out"
    config["docs_summary"] = tmp_path / "phase8_summary.md"
    closure.write_outputs(config, closure.build_closure(config))
    for generated in sorted(config["output_dir"].iterdir()):
        assert generated.read_bytes() == (tracked_output / generated.name).read_bytes()
    assert config["docs_summary"].read_bytes() == tracked_summary.read_bytes()
