"""Focused safety and gate tests for the Phase 8R6C provider continuation."""
from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

from src.experiments.phase6a import sha256_file
from src.extension.phase8r6c_provider_resume import (
    GateFailure,
    Metrics,
    _http_category,
    creation_lookup,
    creation_schema,
    enrich_one,
    provider_configuration,
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
    error = HTTPError("https://redacted.invalid", code, "failure", None, None)
    assert _http_category(error) == category


def test_provider_schema_mismatch_stops():
    with pytest.raises(GateFailure) as failure:
        creation_schema("https://eth.blockscout.com/not-an-api")
    assert failure.value.category == "SCHEMA_MISMATCH"


def test_blockscout_v2_schema_is_explicit_and_checked(monkeypatch):
    target = _address("1")
    monkeypatch.setattr("src.extension.phase8r6c_provider_resume._request_json", lambda *args, **kwargs: {"hash": target, "creator_address_hash": _address("2"), "creation_tx_hash": "0xabc"})
    row = creation_lookup(target, "https://eth.blockscout.com/api/v2", "secret", 1, Metrics())
    assert row["address"] == target and row["tx_hash"] == "0xabc"


def test_contract_identity_mismatch_stops(monkeypatch):
    target = _address("1")
    monkeypatch.setattr("src.extension.phase8r6c_provider_resume.creation_lookup", lambda *args, **kwargs: {"address": _address("2"), "creator": _address("3"), "tx_hash": "0xabc", "block_number": None})
    with pytest.raises(GateFailure) as failure:
        enrich_one(target, "rpc", "https://explorer/api/v2", "key", 1, Metrics())
    assert failure.value.category == "IDENTITY_MISMATCH"


def test_missing_environment_creates_audited_stop_without_changing_parent():
    parent = REPO_ROOT / "results/extension/phase8r6c"
    tracked = [parent / "phase8r6c_results.json", parent / "phase8r6c_report.md", parent / "provider_preflight.json"]
    before = [sha256_file(path) for path in tracked]
    result = run(environ={})
    after = [sha256_file(path) for path in tracked]
    assert before == after
    assert result["decision"] == "A" and result["full_resume_occurred"] is False
    assert result["track_a_modified"] is False
    output = parent / "provider_resume"
    ledger = json.loads((output / "ledger_audit.json").read_text(encoding="utf-8"))
    assert ledger["source_modified"] is False and len(ledger["source_sha256"]) == 64


def test_resumable_ledger_recognizes_only_verified_success(tmp_path):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(f"address,status\n{_address('1')},SUCCESS\n{_address('2')},RATE_LIMITED\n", encoding="utf-8")
    assert successful_ledger_addresses(ledger) == {_address("1")}


def test_no_model_training_scoring_or_sealed_label_access():
    source = (REPO_ROOT / "src/extension/phase8r6c_provider_resume.py").read_text(encoding="utf-8")
    forbidden = ("import sklearn", "import xgboost", "from src.models", "load_sealed_labels")
    assert all(value not in source for value in forbidden)
    assert "read_csv(REPO_ROOT / \"data/external/track_b_sealed" not in source
    summary = json.loads((REPO_ROOT / "results/extension/phase8r6c/provider_resume/resume_summary.json").read_text(encoding="utf-8"))
    assert summary["models_trained"] is False and summary["detectors_evaluated"] is False

