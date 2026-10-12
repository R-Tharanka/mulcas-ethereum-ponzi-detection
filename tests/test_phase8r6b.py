"""Phase 8R6B data-construction, sealing, and portability tests."""
from __future__ import annotations

import hashlib,json
from pathlib import Path
import pandas as pd
import pytest

from src.extension.artifact_resolver import resolve_artifact
from src.extension.external_enrichment import enrich
from src.extension.external_normalization import digest,normalize_address,normalize_hex
from src.extension.external_rpc import rpc_deployment_bundle
from src.extension.track_b_builder import classify,enrich_static_identities,primary_membership,seal
from src.experiments.phase6a import verify_frozen_manifest
from src.utils.paths import REPO_ROOT


def _address(char:str)->str:return "0x"+char*40


def test_frozen_phase8r6a_manifest_verifies_without_modifying_prior_work():
    observed=verify_frozen_manifest(REPO_ROOT/"configs/extension/frozen_phase8r6a_manifest.yaml")
    assert len(observed)==28
    assert observed["data/raw/PonziDataset_20221114.csv"]=="48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf"


def test_rpc_bundle_maps_receipt_to_correct_address_and_checks_block(monkeypatch):
    address=_address("1");tx="0x"+"a"*64
    def fake(url,payload,timeout):
        return [{"id":1,"result":"0x6000"},{"id":2,"result":{"blockNumber":"0x64","from":_address("2"),"input":"0x6060"}},
                {"id":3,"result":{"blockNumber":"0x64","from":_address("2"),"contractAddress":address}},
                {"id":4,"result":{"number":"0x64","timestamp":"0x3e8"}}]
    monkeypatch.setattr("src.extension.external_rpc._json_request",fake)
    bundle=rpc_deployment_bundle(address,tx,100,"unused",1)
    assert bundle["timestamp"]==1000 and bundle["runtime_bytecode"]=="0x6000"


def test_enrichment_ledger_is_resumable(monkeypatch,tmp_path):
    calls=[]
    def fake(address,config):calls.append(address);return ({"address":address,"created_block":8},1,None)
    monkeypatch.setattr("src.extension.external_enrichment.retrieve_with_retry",fake)
    monkeypatch.setattr("src.extension.external_enrichment.time.sleep",lambda _:None)
    cfg={"audit_date":"2026-10-13","retry_policy":{"inter_address_delay_seconds":0}}
    candidates=pd.DataFrame({"address":[_address("1"),_address("2")],"label":[1,0]})
    ledger=tmp_path/"ledger.csv";enrich(candidates,cfg,ledger);enrich(candidates,cfg,ledger)
    assert calls==candidates.address.tolist()
    assert len(pd.read_csv(ledger))==2


def test_static_hash_and_opcode_derivation_are_deterministic():
    rows=pd.DataFrame({"runtime_bytecode":["0x6001600201"],"creation_code":[None]})
    a=enrich_static_identities(rows);b=enrich_static_identities(rows)
    assert a.loc[0,"runtime_hash"]==digest(normalize_hex("0x6001600201"))==b.loc[0,"runtime_hash"]
    assert a.loc[0,"opcode"]=="PUSH1 PUSH1 ADD"
    assert bool(a.loc[0,"disassembly_success"])


def test_missing_creation_is_provisional_not_strict_proof():
    row=enrich_static_identities(pd.DataFrame({"address":[_address("1")],"runtime_bytecode":["0x6000"],"creation_code":[None],"created_block":[8]}))
    refs={"address":set(),"runtime_hash":set(),"opcode_hash":set(),"creation_hash":set()}
    result=classify(row,refs,7)
    assert result.loc[0,"independence_status"]=="PROVISIONALLY_STRICT_INDEPENDENT"
    assert result.loc[0,"temporal_status"]=="LATER"


def test_primary_membership_enforces_all_rules_and_excludes_class_d():
    base={"address":_address("1"),"label":1,"provenance":"B","address_new":True,"runtime_new":True,"opcode_new":True,
      "temporal_status":"LATER","unresolved_conflict":False,"runtime_hash":"r","opcode_hash":"o","creation_hash":None,
      "receipt_address_verified":True,"block_consistency_verified":True,"timestamp_consistency_verified":True}
    rows=pd.DataFrame([base,{**base,"address":_address("2"),"provenance":"D"},{**base,"address":_address("3"),"temporal_status":"UNKNOWN"}])
    selected,excluded=primary_membership(rows)
    assert selected.address.tolist()==[_address("1")]
    assert "eligible_provenance" in excluded.loc[excluded.address==_address("2"),"exclusion_reasons"].iloc[0]
    assert "verified_later" in excluded.loc[excluded.address==_address("3"),"exclusion_reasons"].iloc[0]


def test_label_conflicts_are_not_majority_resolved():
    rows=pd.DataFrame({"address":[_address("1")],"label":[1],"provenance":["A"],"address_new":[True],"runtime_new":[True],"opcode_new":[True],"temporal_status":["LATER"],"unresolved_conflict":[True],"runtime_hash":["r"],"opcode_hash":["o"],"creation_hash":["c"],"receipt_address_verified":[True],"block_consistency_verified":[True],"timestamp_consistency_verified":[True]})
    selected,excluded=primary_membership(rows)
    assert selected.empty and "no_unresolved_conflict" in excluded.iloc[0].exclusion_reasons


def test_sealed_metadata_excludes_labels_and_manifest_hashes_reproduce(tmp_path):
    selected=pd.DataFrame({"address":[_address("1")],"label":[1],"sources":["x"],"provenance":["A"],"created_block":[8],"timestamp":["2020-01-01T00:00:00+00:00"],"creator":[_address("2")],"runtime_hash":["r"],"opcode_hash":["o"],"creation_hash":["c"],"strictness":["STRICT"]})
    metadata=tmp_path/"public/metadata.csv";labels=tmp_path/"sealed/labels.csv";manifest_path=tmp_path/"public/membership_manifest.json"
    manifest=seal(selected,metadata,labels,manifest_path)
    assert "label" not in pd.read_csv(metadata).columns and "label" in pd.read_csv(labels).columns
    assert hashlib.sha256(metadata.read_bytes()).hexdigest()==manifest["metadata_file_hash"]
    assert hashlib.sha256(labels.read_bytes()).hexdigest()==manifest["sealed_label_file_hash"]


def test_artifact_relocation_requires_exact_hash(tmp_path):
    root=tmp_path/"artifacts";candidate=root/"phase/file.bin";candidate.parent.mkdir(parents=True);candidate.write_bytes(b"safe")
    expected=hashlib.sha256(b"safe").hexdigest();recorded=Path("Z:/obsolete/artifacts/phase/file.bin")
    resolved=resolve_artifact(recorded,expected,(root,))
    assert resolved.relocated and Path(resolved.resolved_path)==candidate.resolve()
    with pytest.raises(FileNotFoundError):resolve_artifact(recorded,"0"*64,(root,))


def test_phase_has_no_model_or_detector_imports():
    source=(REPO_ROOT/"src/extension/phase8r6b.py").read_text(encoding="utf-8")
    assert "sklearn" not in source and "xgboost" not in source and "MulCas" not in source


def test_gate_stopped_outputs_do_not_claim_track_b_or_expose_labels():
    output=REPO_ROOT/"results/extension/phase8r6b"
    result=json.loads((output/"phase8r6b_results.json").read_text(encoding="utf-8"))
    assert result["status"]=="stopped_performance_gate"
    assert result["models_trained"] is False and result["detectors_evaluated"] is False
    assert result["track_b_established"] is False and result["sealed_ready"] is False
    assert result["next_recommended_phase"]=="phase8r6c_manual_verification"
    candidate=pd.read_csv(output/"track_b_candidate.csv")
    assert "label" not in candidate.columns
    assert not (REPO_ROOT/"data/external/track_b_sealed/labels.csv").exists()
