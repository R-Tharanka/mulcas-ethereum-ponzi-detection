"""Focused safety and gate tests for the Phase 8R6C provider continuation."""
from __future__ import annotations

import json
from urllib.error import HTTPError

import pandas as pd
import pytest
import yaml

from src.experiments.phase6a import sha256_file
from src.extension.phase8r6c_provider_resume import (
    GateFailure,
    Metrics,
    _http_category,
    _load_approved_resume_state,
    _safe_error_message,
    creation_lookup,
    creation_schema,
    enrich_one,
    provider_configuration,
    provider_stability_failure,
    rpc_health,
    run,
    successful_ledger_addresses,
)
from src.utils.paths import REPO_ROOT


def _address(char: str) -> str:
    return "0x" + char * 40


def test_credentials_are_never_serialized():
    secret = "do-not-serialize-this"
    state = provider_configuration({"ETH_RPC_URL": f"https://rpc.example/v2/{secret}", "BLOCKSCOUT_API_URL": "https://explorer.example/api/v2", "BLOCKSCOUT_API_KEY": secret})
    encoded = json.dumps(state)
    assert secret not in encoded
    assert state["ETH_RPC_URL"]["hostname"] == "rpc.example"
    assert state["credentials_serialized"] is False


def test_rpc_health_requires_mainnet(monkeypatch):
    replies = iter(["0xaa36a7", "0x10"])
    monkeypatch.setattr("src.extension.phase8r6c_provider_resume.rpc_call", lambda *args, **kwargs: next(replies))
    with pytest.raises(GateFailure, match="chain ID 1") as failure:
        rpc_health("https://rpc.example", 1, Metrics())
    assert failure.value.category == "WRONG_CHAIN"


@pytest.mark.parametrize(("code", "category"), [(401, "AUTH_FAILURE"), (403, "AUTH_FAILURE"), (429, "RATE_LIMITED")])
def test_auth_and_rate_limit_statuses_stop_distinctly(code, category):
    error = HTTPError("https://redacted.invalid/?apikey=do-not-serialize", code, "failure", None, None)
    assert _http_category(error) == category
    assert _safe_error_message(error) == f"HTTP {code}"
    assert "do-not-serialize" not in _safe_error_message(error)


def test_provider_schema_mismatch_stops():
    with pytest.raises(GateFailure) as failure:
        creation_schema("https://eth.blockscout.com/not-an-api")
    assert failure.value.category == "SCHEMA_MISMATCH"


@pytest.mark.parametrize("transaction_field", ["creation_transaction_hash", "creation_tx_hash"])
def test_blockscout_v2_schema_is_explicit_and_checked(monkeypatch, transaction_field):
    target = _address("1")
    response = {"hash": target, "creator_address_hash": _address("2"), transaction_field: "0xabc"}
    monkeypatch.setattr("src.extension.phase8r6c_provider_resume._request_json", lambda *args, **kwargs: response)
    row = creation_lookup(target, "https://eth.blockscout.com/api/v2", "secret", 1, Metrics())
    assert row["address"] == target and row["tx_hash"] == "0xabc"


def test_blockscout_v2_rejects_unknown_creation_hash_schema(monkeypatch):
    target = _address("1")
    monkeypatch.setattr(
        "src.extension.phase8r6c_provider_resume._request_json",
        lambda *args, **kwargs: {"hash": target, "creator_address_hash": _address("2"), "transaction_hash": "0xabc"},
    )
    with pytest.raises(GateFailure) as failure:
        creation_lookup(target, "https://eth.blockscout.com/api/v2", "secret", 1, Metrics())
    assert failure.value.category == "SCHEMA_MISMATCH"


def test_invalid_contract_is_data_status_not_provider_instability():
    summary = {"failures": [{"status": "INVALID_CONTRACT"}]}
    assert provider_stability_failure(summary) is None


@pytest.mark.parametrize("status", ["AUTH_FAILURE", "RATE_LIMITED", "RPC_FAILURE", "SCHEMA_MISMATCH", "IDENTITY_MISMATCH"])
def test_provider_failures_invalidate_stability(status):
    assert provider_stability_failure({"failures": [{"status": status}]}) == status


def test_contract_identity_mismatch_stops(monkeypatch):
    target = _address("1")
    monkeypatch.setattr("src.extension.phase8r6c_provider_resume.creation_lookup", lambda *args, **kwargs: {"address": _address("2"), "creator": _address("3"), "tx_hash": "0xabc", "block_number": None})
    with pytest.raises(GateFailure) as failure:
        enrich_one(target, "rpc", "https://explorer/api/v2", "key", 1, Metrics())
    assert failure.value.category == "IDENTITY_MISMATCH"


def test_missing_environment_creates_audited_stop_without_changing_parent(tmp_path):
    parent = REPO_ROOT / "results/extension/phase8r6c"
    tracked = [parent / "phase8r6c_results.json", parent / "phase8r6c_report.md", parent / "provider_preflight.json"]
    before = [sha256_file(path) for path in tracked]
    config = yaml.safe_load((REPO_ROOT / "configs/extension/phase8r6c_provider_resume.yaml").read_text(encoding="utf-8"))
    config["output_dir"] = tmp_path.as_posix()
    config_path = tmp_path / "provider_resume.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = run(config_path=config_path, environ={})
    after = [sha256_file(path) for path in tracked]
    assert before == after
    assert result["decision"] == "A" and result["full_resume_occurred"] is False
    assert result["track_a_modified"] is False
    output = tmp_path
    ledger = json.loads((output / "ledger_audit.json").read_text(encoding="utf-8"))
    assert ledger["source_modified"] is False and len(ledger["source_sha256"]) == 64


def test_resumable_ledger_recognizes_only_verified_success(tmp_path):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(f"address,status\n{_address('1')},SUCCESS\n{_address('2')},RATE_LIMITED\n", encoding="utf-8")
    assert successful_ledger_addresses(ledger) == {_address("1")}


def test_approved_resume_reuses_success_and_terminal_invalid_rows(tmp_path):
    candidates = pd.DataFrame([
        {"address": _address("1"), "label": 1, "sources": "one", "provenance": "A"},
        {"address": _address("2"), "label": 0, "sources": "two", "provenance": "B"},
        {"address": _address("3"), "label": 1, "sources": "three", "provenance": "B"},
    ])
    pd.DataFrame([{"address": _address("1"), "status": "SUCCESS", "created_block": 10}]).to_csv(tmp_path / "enriched_contracts.csv", index=False)
    pd.DataFrame([{"address": _address("1"), "status": "SUCCESS"}]).to_csv(tmp_path / "enrichment_status.csv", index=False)
    (tmp_path / "stability_100.json").write_text(json.dumps({"failures": [{"address": _address("2"), "status": "INVALID_CONTRACT", "error_type": "GateFailure"}]}), encoding="utf-8")
    enriched, status = _load_approved_resume_state(tmp_path, candidates)
    assert set(enriched.address) == {_address("1")}
    assert dict(zip(status.address, status.status)) == {_address("1"): "SUCCESS", _address("2"): "INVALID_CONTRACT"}
    assert _address("3") not in set(status.address)


def test_no_model_training_scoring_or_sealed_label_access():
    source = (REPO_ROOT / "src/extension/phase8r6c_provider_resume.py").read_text(encoding="utf-8")
    forbidden = ("import sklearn", "import xgboost", "from src.models", "load_sealed_labels")
    assert all(value not in source for value in forbidden)
    assert "read_csv(REPO_ROOT / \"data/external/track_b_sealed" not in source
    summary = json.loads((REPO_ROOT / "results/extension/phase8r6c/provider_resume/resume_summary.json").read_text(encoding="utf-8"))
    assert summary["models_trained"] is False and summary["detectors_evaluated"] is False

