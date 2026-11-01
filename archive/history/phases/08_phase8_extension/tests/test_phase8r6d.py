from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import src.extension.phase8r6d as phase
from src.extension.phase8r6d import ProviderMetrics, RowFailure
from src.extension.sealed_track_b import load_sealed_labels
from src.utils.paths import REPO_ROOT


RAW_HASH = "48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf"


def _response(**updates):
    value = {
        "hash": "0x" + "1" * 40,
        "is_contract": True,
        "creation_status": "success",
        "creator_address_hash": "0x" + "2" * 40,
        "creation_transaction_hash": "0x" + "3" * 64,
    }
    value.update(updates)
    return value


def test_alternate_blockscout_success_schema_is_explicit(monkeypatch):
    value = _response(creation_transaction_hash=None, creation_tx_hash="0x" + "4" * 64)
    monkeypatch.setattr(phase, "_request_json", lambda *args, **kwargs: value)
    creation, fingerprint = phase.creation_lookup(value["hash"], "https://example.invalid/api/v2", "secret", 1, ProviderMetrics())
    assert creation["tx_hash"] == value["creation_tx_hash"]
    assert creation["creator"] == value["creator_address_hash"]
    assert fingerprint["creation_transaction_present"] is True


def test_null_creation_metadata_is_not_found_not_schema_drift(monkeypatch):
    value = _response(creator_address_hash=None, creation_transaction_hash=None)
    monkeypatch.setattr(phase, "_request_json", lambda *args, **kwargs: value)
    with pytest.raises(RowFailure) as caught:
        phase.creation_lookup(value["hash"], "https://example.invalid/api/v2", "secret", 1, ProviderMetrics())
    assert caught.value.category == "NOT_FOUND"
    assert caught.value.fingerprint["returned_identity_matches_target"] is True
    assert caught.value.fingerprint["creator_present"] is False


def test_incomplete_creation_metadata_is_unsupported(monkeypatch):
    value = _response(creator_address_hash=None)
    monkeypatch.setattr(phase, "_request_json", lambda *args, **kwargs: value)
    with pytest.raises(RowFailure) as caught:
        phase.creation_lookup(value["hash"], "https://example.invalid/api/v2", "secret", 1, ProviderMetrics())
    assert caught.value.category == "PROVIDER_SCHEMA_UNSUPPORTED"


def test_identity_mismatch_remains_strict(monkeypatch):
    value = _response(hash="0x" + "9" * 40)
    monkeypatch.setattr(phase, "_request_json", lambda *args, **kwargs: value)
    with pytest.raises(RowFailure) as caught:
        phase.creation_lookup("0x" + "1" * 40, "https://example.invalid/api/v2", "secret", 1, ProviderMetrics())
    assert caught.value.category == "IDENTITY_MISMATCH"


def test_isolated_schema_and_identity_failures_continue_but_systemic_stop():
    assert not phase.should_stop_systemic(["PROVIDER_SCHEMA_UNSUPPORTED"], "PROVIDER_SCHEMA_UNSUPPORTED")
    assert not phase.should_stop_systemic(["IDENTITY_MISMATCH"], "IDENTITY_MISMATCH")
    assert phase.should_stop_systemic(["PROVIDER_SCHEMA_UNSUPPORTED"] * 5, "PROVIDER_SCHEMA_UNSUPPORTED")
    history = ["SUCCESS"] * 45 + ["IDENTITY_MISMATCH"] * 5
    assert phase.should_stop_systemic(history, "IDENTITY_MISMATCH")


def test_resume_does_not_refetch_success_or_terminal_failures():
    candidates = pd.DataFrame({"address": ["a", "b", "c", "d", "e"]})
    statuses = pd.DataFrame({"address": ["a", "b", "c", "d", "e"], "status": ["SUCCESS", "INVALID_CONTRACT", "IDENTITY_MISMATCH", "NOT_FOUND", "NOT_ATTEMPTED"]})
    assert phase.pending_candidates(candidates, statuses).address.tolist() == ["e"]


def test_schema_fingerprint_contains_shape_not_values():
    value = _response()
    fingerprint = phase.schema_fingerprint(value, target=value["hash"])
    serialized = json.dumps(fingerprint)
    assert value["creation_transaction_hash"] not in serialized
    assert value["creator_address_hash"] not in serialized
    assert fingerprint["returned_identity_matches_target"] is True


def test_phase9b_protocol_is_prospective_and_model_free(tmp_path, monkeypatch):
    monkeypatch.setattr(phase, "REPO_ROOT", tmp_path)
    config = {
        "phase9b": {
            "config": "configs/phase9/phase9b.yaml",
            "protocol_doc": "docs/phase9_evaluation_protocol.md",
            "output_dir": "results/phase9/phase9b",
        }
    }
    result = phase.freeze_phase9b_protocol(config, {"sealed_ready": True, "track_b_primary_basis": "STRICT"})
    protocol = (tmp_path / "configs/phase9/phase9b.yaml").read_text(encoding="utf-8")
    assert result["status"] == "frozen"
    assert "phase9c_started: false" in protocol
    assert "track_b_performance_used: false" in protocol
    assert "replicates: 2000" in protocol
    assert "fpr_ceiling_basis: frozen_mulcas_pooled_forward_fpr" in protocol
    assert "predict_proba" not in protocol and "fit(" not in protocol


def test_track_a_hash_and_frozen_manifest_exist():
    raw = REPO_ROOT / "data/raw/PonziDataset_20221114.csv"
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == RAW_HASH
    assert len(pd.read_csv(raw, usecols=["address"])) == 6498
    assert (REPO_ROOT / "configs/extension/frozen_phase8r6c_manifest.yaml").is_file()


def test_generated_track_b_membership_does_not_expose_row_labels():
    for name in ("track_b_strict.csv", "track_b_provisional.csv", "track_b_all_unique_address.csv", "track_b_unique_code_sensitivity.csv"):
        path = REPO_ROOT / "results/extension/phase8r6d" / name
        assert path.is_file()
        assert "label" not in pd.read_csv(path, nrows=0).columns


def test_identity_mismatch_audit_preserves_strict_exclusion_without_raw_identity():
    audit = pd.read_csv(REPO_ROOT / "results/extension/phase8r6d/identity_mismatch_audit.csv")
    assert len(audit) == 2
    assert audit.status.eq("IDENTITY_MISMATCH").all()
    assert audit.provider_identity_matches_target.all()
    assert (~audit.receipt_identity_matches_target).all()
    assert audit.decision.eq("EXCLUDE_FROM_TRACK_B").all()
    assert "returned_address" not in audit.columns


def test_completed_run_is_idempotent_without_provider_credentials():
    paths = [
        REPO_ROOT / "results/extension/phase8r6d/phase8r6d_results.json",
        REPO_ROOT / "configs/phase9/phase9b.yaml",
        REPO_ROOT / "data/external/track_b/membership_manifest.json",
        REPO_ROOT / "data/external/track_b_sealed/labels.csv",
    ]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    result = phase.run(environ={})
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert result["status"] == "complete"
    assert before == after


def test_sealed_labels_remain_default_deny(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("address,label\n0x1,1\n", encoding="utf-8")
    with pytest.raises(PermissionError):
        load_sealed_labels(labels)


def test_phase8r6d_source_has_no_model_training_or_scoring():
    source = Path(phase.__file__).read_text(encoding="utf-8")
    forbidden = ("import sklearn", "import xgboost", "from src.models", ".fit(", "predict_proba", "load_sealed_labels")
    assert not any(token in source for token in forbidden)
    assert '"p5_evaluated": False' in source
