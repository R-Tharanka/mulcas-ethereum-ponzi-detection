"""Provider gating, separation, schema, and safety tests for Phase 8R6C."""
from __future__ import annotations

import hashlib,json
from pathlib import Path
from urllib.error import HTTPError
import pandas as pd
import pytest
import yaml

from src.experiments.phase6a import sha256_file,verify_frozen_manifest
from src.extension.artifact_resolver import resolve_artifact
from src.extension.external_rpc import retrieve_with_retry
from src.extension.phase8r6c import CONFIG,SCHEMA,provider_state
from src.extension.sealed_track_b import load_sealed_labels
from src.extension.successor_migration import verify_manifest_with_successor_migrations
from src.extension.track_b_builder import classify,enrich_static_identities,primary_membership
from src.utils.paths import REPO_ROOT

OUT=REPO_ROOT/"results/extension/phase8r6c"


def _address(char:str)->str:return "0x"+char*40


def test_all_phase8r6b_anchors_and_internal_dataset_remain_frozen():
    observed=verify_manifest_with_successor_migrations(REPO_ROOT/"configs/extension/frozen_phase8r6b_manifest.yaml")
    assert len(observed)==31
    raw=REPO_ROOT/"data/raw/PonziDataset_20221114.csv"
    assert sha256_file(raw)=="48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf"
    assert len(pd.read_csv(raw))==6498


def test_provider_configuration_uses_env_presence_only_and_serializes_no_secret(monkeypatch):
    cfg=yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for name in cfg["required_provider_environment"]:monkeypatch.setenv(name,"super-secret-value")
    state=provider_state(cfg)
    assert state["all_required_configured"] and state["credentials_serialized"] is False
    assert "super-secret-value" not in json.dumps(state)


def test_rate_limit_retries_stop_and_never_become_not_found(monkeypatch):
    def limited(*args,**kwargs):raise HTTPError("https://example.invalid",429,"Too Many Requests",None,None)
    sleeps=[];monkeypatch.setattr("src.extension.external_rpc.blockscout_creation",limited)
    cfg={"retry_policy":{"maximum_attempts":3,"base_backoff_seconds":1,"timeout_seconds":1},"providers":{"creation":{"endpoint":"x"},"rpc":{"endpoint":"y"}}}
    data,attempts,error=retrieve_with_retry(_address("1"),cfg,sleep=sleeps.append)
    assert data is None and attempts==3 and "429" in error
    assert sleeps==[1.0,2.0] and "not found" not in error.lower()


def test_resumed_status_covers_every_candidate_without_refetch_claims():
    status=pd.read_csv(OUT/"enrichment_status.csv")
    assert len(status)==1490 and status.address.is_unique
    assert (status.status=="RATE_LIMITED").sum()==8
    assert (status.status=="NOT_ATTEMPTED_PROVIDER_UNAVAILABLE").sum()==1482
    assert not status.status.isin(["NOT_FOUND","INVALID_CONTRACT","SUCCESS"]).any()


def test_normalized_schema_and_feature_compatibility_reconcile():
    enriched=pd.read_csv(OUT/"enriched_contracts.csv")
    assert enriched.columns.tolist()==SCHEMA and enriched.empty
    compatibility=pd.read_csv(OUT/"schema_compatibility.csv")
    assert {"MULCAS_READY","STATIC_RUNTIME_READY","CREATION_CODE_READY"}.issubset(set(compatibility.field_or_status))
    assert compatibility.available_count.sum()==0
    result=json.loads((OUT/"phase8r6c_results.json").read_text(encoding="utf-8"))
    assert result["mulcas_ready_count"]==result["static_runtime_ready_count"]==result["creation_code_ready_count"]==0


def test_temporal_and_independence_boundary_is_exact_and_creation_missing_is_uncertainty():
    base=pd.DataFrame({"address":[_address("1"),_address("2")],"runtime_hash":["r1","r2"],"opcode_hash":["o1","o2"],"creation_hash":["c1",None],"created_block":[7136486,7136487]})
    refs={"address":set(),"runtime_hash":set(),"opcode_hash":set(),"creation_hash":set()}
    result=classify(base,refs,7136486)
    assert result.temporal_status.tolist()==["PRE_EXISTING_PERIOD","LATER"]
    assert result.independence_status.tolist()==["STRICT_CODE_NEW","PROVISIONALLY_STRICT_INDEPENDENT"]


def test_only_A_or_B_can_enter_and_track_a_track_b_are_separate():
    base={"label":1,"address_new":True,"runtime_new":True,"opcode_new":True,"temporal_status":"LATER","unresolved_conflict":False,
      "runtime_hash":"r","opcode_hash":"o","creation_hash":None,"receipt_address_verified":True,"block_consistency_verified":True,"timestamp_consistency_verified":True}
    rows=pd.DataFrame([{**base,"address":_address("1"),"provenance":"A"},{**base,"address":_address("2"),"provenance":"B"},{**base,"address":_address("3"),"provenance":"D"}])
    selected,_=primary_membership(rows)
    assert set(selected.provenance)=={"A","B"}
    assert json.loads((OUT/"phase8r6c_results.json").read_text())["track_a_modified"] is False
    sealed_labels=REPO_ROOT/"data/external/track_b_sealed/labels.csv"
    assert sealed_labels.is_file()
    with pytest.raises(PermissionError,match="explicit final_evaluation=True is required"):
        load_sealed_labels(sealed_labels)


def test_sealed_labels_are_default_deny(tmp_path):
    labels=tmp_path/"labels.csv";labels.write_text("address,label\n"+_address("1")+",1\n",encoding="utf-8")
    with pytest.raises(PermissionError):load_sealed_labels(labels)
    assert len(load_sealed_labels(labels,final_evaluation=True))==1


def test_outputs_explicitly_await_provider_and_do_not_score():
    result=json.loads((OUT/"phase8r6c_results.json").read_text(encoding="utf-8"))
    assert result["status"]=="awaiting_provider_configuration"
    assert result["next_recommended_phase"]=="await_provider_approval"
    assert result["models_trained"] is False and result["detectors_evaluated"] is False
    assert result["zero_count_interpretation"].endswith("not evidence of absence")
    preflight=json.loads((OUT/"provider_preflight.json").read_text())
    assert preflight["requests"]==0 and preflight["rate_limit_events"]==0
    assert set(preflight["provider_configuration"]["missing_required"])=={"BLOCKSCOUT_API_URL","BLOCKSCOUT_API_KEY","ETH_RPC_URL"}


def test_portability_audit_and_resolver_require_hash_equality(tmp_path):
    audit=pd.read_csv(OUT/"portability_audit.csv")
    assert len(audit)==27 and (audit.status=="verified").all()
    root=tmp_path/"artifacts";candidate=root/"phase/a.bin";candidate.parent.mkdir(parents=True);candidate.write_bytes(b"x")
    good=hashlib.sha256(b"x").hexdigest();recorded=Path("Z:/old/artifacts/phase/a.bin")
    assert resolve_artifact(recorded,good,(root,)).relocated
    with pytest.raises(FileNotFoundError):resolve_artifact(recorded,"0"*64,(root,))


def test_expected_outputs_exist_and_no_model_imports():
    expected={"phase8r6c_results.json","provider_preflight.json","stability_batch.json","enrichment_status.csv","enriched_contracts.csv","schema_compatibility.csv","temporal_classification.csv","runtime_overlap.csv","opcode_overlap.csv","creation_overlap.csv","label_conflicts.csv","benign_quality.csv","track_b_strict.csv","track_b_provisional.csv","track_b_excluded.csv","manual_review_triage.csv","track_b_readiness.csv","portability_audit.csv","phase8r6c_report.md"}
    assert expected=={p.name for p in OUT.iterdir() if p.is_file()}
    source=(REPO_ROOT/"src/extension/phase8r6c.py").read_text(encoding="utf-8")
    assert all(term not in source for term in ["import sklearn","import xgboost","from src.models","from src.baselines"])
