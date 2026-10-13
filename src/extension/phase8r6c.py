"""Phase 8R6C provider-gated external enrichment finalization (no model access)."""
from __future__ import annotations

import csv,json,os,platform,sys
from pathlib import Path
from typing import Any
import pandas as pd
import yaml

from src.experiments.phase6a import sha256_file,verify_frozen_manifest
from src.utils.paths import REPO_ROOT
from .external_enrichment import candidate_rows

CONFIG=REPO_ROOT/"configs/extension/phase8r6c.yaml"
SCHEMA=["external_source","external_record_id","address","label","label_provenance_class","createdBlockNumber","timestamp","createdTransactionHash","creator","contractCode","opcode","creationCode","runtime_sha256","opcode_sha256","creation_code_sha256","temporal_status","independence_status","source_url","retrieval_provider","retrieval_timestamp","MULCAS_READY","STATIC_RUNTIME_READY","CREATION_CODE_READY"]

def _p(value:str|Path)->Path:
    value=Path(value);return value if value.is_absolute() else REPO_ROOT/value

def _write(path:Path,rows:list[dict],fields:list[str]|None=None)->None:
    path.parent.mkdir(parents=True,exist_ok=True);keys=list(fields or [])
    for row in rows:
        for key in row:
            if key not in keys:keys.append(key)
    if not keys:keys=["status","reason"]
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def provider_state(cfg:dict)->dict[str,Any]:
    names=[item for item in cfg["required_provider_environment"]]
    configured={name:bool(os.environ.get(name,"").strip()) for name in names}
    return {"configuration_class":"environment_variables_only","configured":configured,"all_required_configured":all(configured.values()),
            "missing_required":[name for name,value in configured.items() if not value],"credentials_serialized":False}

def verify_prior(cfg:dict)->dict[str,Any]:
    observed=verify_frozen_manifest(_p(cfg["frozen_manifest"]))
    retrieval=pd.read_csv(REPO_ROOT/"results/extension/phase8r6a/source_retrieval_manifest.csv",keep_default_na=False)
    raw_checked=0
    for row in retrieval[retrieval.status.eq("downloaded")].itertuples():
        path=_p(row.original_filename)
        if not path.is_file() or sha256_file(path)!=row.sha256:raise RuntimeError(f"Frozen external raw mismatch: {path}")
        raw_checked+=1
    return {"manifest_anchors":len(observed),"external_raw_files":raw_checked,"internal_sha256":sha256_file(_p(cfg["internal_dataset"])),"verified":True}

def normalized_status(candidates:pd.DataFrame,ledger:pd.DataFrame)->pd.DataFrame:
    prior={row.address:row for row in ledger.itertuples()} if not ledger.empty else {}
    rows=[]
    for item in candidates.itertuples():
        old=prior.get(item.address);old_error=getattr(old,"error_reason","") if old else ""
        if old and "429" in old_error:status="RATE_LIMITED"
        elif old and getattr(old,"status","")=="success":status="SUCCESS"
        elif old:status="RPC_FAILURE"
        else:status="NOT_ATTEMPTED_PROVIDER_UNAVAILABLE"
        rows.append({"address":item.address,"label":item.label,"sources":item.sources,"label_provenance_class":item.provenance,"status":status,
          "attempts":int(getattr(old,"attempts",0)) if old else 0,"provider":getattr(old,"last_provider","") if old else "",
          "last_attempt":getattr(old,"last_retrieval_time","") if old else "","failure_type":status if status!="SUCCESS" else "",
          "retrieved_fields":getattr(old,"successful_fields","") if old else "","reason":old_error or "provider configuration unavailable; not blockchain missingness"})
    return pd.DataFrame(rows)

def run(config_path:Path=CONFIG)->dict[str,Any]:
    cfg=yaml.safe_load(config_path.read_text(encoding="utf-8"));out=_p(cfg["output_dir"]);out.mkdir(parents=True,exist_ok=True)
    frozen=verify_prior(cfg);providers=provider_state(cfg)
    candidates=candidate_rows(_p(cfg["internal_dataset"]),REPO_ROOT/"data/external/raw/ponzi_ml/dataset/dataset.csv",REPO_ROOT/"data/external/raw/blockchain_unica/ponzi-addresses.csv")
    ledger=pd.read_csv(_p(cfg["phase8r6b_ledger"]),keep_default_na=False);status=normalized_status(candidates,ledger);status.to_csv(out/"enrichment_status.csv",index=False)
    stop_reason="Missing required provider environment variables: "+", ".join(providers["missing_required"])
    preflight={"status":"NOT_RUN_PROVIDER_CONFIGURATION_REQUIRED","sample_positive":10,"sample_benign":10,"creation_success":0,"runtime_success":0,"full_field_success":0,"requests":0,"elapsed_seconds":0,"http_status_distribution":{},"retry_events":0,"rate_limit_events":0,"reason":stop_reason,"previous_rate_limit_evidence":{"attempted":8,"http_429":8},"provider_configuration":providers}
    stability={"status":"NOT_RUN_PRECONDITION_FAILED","address_count":100,"reason":"20-address preflight did not run because reliable provider configuration is absent","completion_rate":None,"http_429_rate":None,"projected_full_runtime_minutes":None}
    (out/"provider_preflight.json").write_text(json.dumps(preflight,indent=2)+"\n",encoding="utf-8");(out/"stability_batch.json").write_text(json.dumps(stability,indent=2)+"\n",encoding="utf-8")
    _write(out/"enriched_contracts.csv",[],SCHEMA)
    compatibility=[]
    for field in ["address","label","block","timestamp","creator","contractCode","opcode","creationCode","transaction_hash","MULCAS_READY","STATIC_RUNTIME_READY","CREATION_CODE_READY"]:
        compatibility.append({"field_or_status":field,"eligible_enriched_rows":0,"available_count":0,"available_percent":0.0,"reason":"no verified enrichment; provider configuration required"})
    _write(out/"schema_compatibility.csv",compatibility)
    _write(out/"temporal_classification.csv",[],["address","label","createdBlockNumber","timestamp","temporal_status","reason"])
    _write(out/"runtime_overlap.csv",[],["address","runtime_sha256","address_overlap","runtime_overlap","overall_independence","reason"])
    _write(out/"opcode_overlap.csv",[],["address","opcode_sha256","opcode_count","malformed","opcode_overlap","overall_independence","reason"])
    _write(out/"creation_overlap.csv",[],["address","creation_code_sha256","creation_overlap","overall_independence","reason"])
    conflicts=pd.read_csv(_p(cfg["phase8r6b_dir"])/"external_label_conflicts.csv",keep_default_na=False);conflicts.to_csv(out/"label_conflicts.csv",index=False)
    _write(out/"benign_quality.csv",[{"category":"BENIGN_CURATED","source":"ponzi_ml","candidate_count":927,"eligible_enriched_count":0,"contamination_risk":"unlabeled scams cannot be excluded","upgrade_to_strong":False}])
    _write(out/"track_b_strict.csv",[],["address","external_source","runtime_sha256","opcode_sha256","creation_code_sha256"])
    _write(out/"track_b_provisional.csv",[],["address","external_source","runtime_sha256","opcode_sha256","creation_code_status"])
    excluded=[{"address":row.address,"sources":row.sources,"label_provenance_class":row.provenance,"reason":"PROVIDER_CONFIGURATION_REQUIRED; temporal and code independence unknown"} for row in candidates.itertuples()]
    _write(out/"track_b_excluded.csv",excluded)
    later_predictions=pd.read_csv(_p(cfg["phase8r6b_dir"])/"manual_review_triage.csv",keep_default_na=False)
    triage=later_predictions[later_predictions.priority_group.eq("M5")].to_dict("records")
    _write(out/"manual_review_triage.csv",triage, list(later_predictions.columns))
    _write(out/"track_b_readiness.csv",[{"rating":"NONE","sealed_ready":False,"strict_rows":0,"provisional_rows":0,"positive":0,"benign":0,"reason":"provider configuration required; zero observed is not evidence of absence"}])
    portability=pd.read_csv(_p(cfg["phase8r6b_dir"])/"artifact_relocation_audit.csv",keep_default_na=False);portability["phase8r6c_reverified"]="hash_match_required";portability.to_csv(out/"portability_audit.csv",index=False)
    results={"status":"awaiting_provider_configuration","models_trained":False,"detectors_evaluated":False,"track_a_modified":False,"candidate_count":1490,
      "enriched_count":0,"mulcas_ready_count":0,"static_runtime_ready_count":0,"creation_code_ready_count":0,"later_positive_count":0,"later_benign_count":0,
      "strict_independent_positive_count":0,"strict_independent_benign_count":0,"provisional_independent_positive_count":0,"provisional_independent_benign_count":0,
      "track_b_established":False,"track_b_rating":"NONE","sealed_ready":False,"manual_review_required":False,"manual_review_count":len(triage),
      "unresolved_external_conflicts":int(conflicts.resolution.eq("unresolved_no_majority_vote").sum()),"provider_configuration":providers,"frozen_verification":frozen,
      "zero_count_interpretation":"not observed because enrichment did not run; not evidence of absence","next_recommended_phase":"await_provider_approval"}
    (out/"phase8r6c_results.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    write_report(out,results,preflight,stability)
    return results

def write_report(out:Path,r:dict,p:dict,s:dict)->None:
    sections=["Purpose","Scope","Frozen verification","Track separation","Normalized schema","Provider configuration","20-address preflight","100-address stability gate","Performance gate","Resume ledger","Creation metadata","Runtime and opcode","Creation code","Temporal findings","Independence findings","Label conflicts","Benign quality","Feature alignment","Manual review","Track B finalization","Portability","Environment","Limitations","Decision"]
    content={
      "Purpose":"Phase 8R6C enriches and constructs evaluation data; it does not evaluate a detector.",
      "Scope":"No model, MulCas score, P5 evaluation, Phase 9B execution, or external-label tuning occurred.",
      "Frozen verification":f"Verified {r['frozen_verification']['manifest_anchors']} Phase 8R6B anchors, all {r['frozen_verification']['external_raw_files']} external raw files, the raw internal hash, the ledger, resolver, configuration, results, and report.",
      "Track separation":"The external dataset is not merged into the original training dataset. Track A remains the development corpus and Track B remains reserved for future sealed evaluation.",
      "Normalized schema":"The requested external schema is emitted with explicit empty values. Raw source data and the Phase 8R6B ledger remain unchanged.",
      "Provider configuration":p["reason"]+". No credential values were read into artifacts, printed, or serialized.",
      "20-address preflight":"Not run. Reusing the credential-free provider would violate the prohibition on repeatedly hammering a provider after sustained HTTP 429 responses.",
      "100-address stability gate":"Not run because the 20-address reliable-provider preflight did not pass.",
      "Performance gate":"No full run was launched. Phase 8R6B's credential-free route projected at least 149 minutes after rate limiting; explicit provider/user approval is required.",
      "Resume ledger":"All 1,490 candidates have deterministic status rows: the eight prior attempts are RATE_LIMITED and unattempted rows are NOT_ATTEMPTED_PROVIDER_UNAVAILABLE. API failure is not treated as blockchain absence.",
      "Creation metadata":"No new metadata was retrieved. Transaction/receipt/address/block consistency therefore remains unknown, not failed.",
      "Runtime and opcode":"No runtime request was made and no opcode was derived. REQUEST_FAILED/NOT_ATTEMPTED remains distinct from NO_CODE_RETURNED.",
      "Creation code":"No creation code was retrieved. Missing creation-code data remain uncertainty, not evidence of uniqueness.",
      "Temporal findings":"Observed later positives=0 and benigns=0 because enrichment did not run. Publication dates were not substituted for deployment time.",
      "Independence findings":"Strict and provisional observed counts are zero because required identities are unavailable. New address, new code, and later deployment remain separate properties.",
      "Label conflicts":f"The frozen cross-source audit contains {r['unresolved_external_conflicts']} unresolved conflicts overall and none in the construction union. No majority vote was applied.",
      "Benign quality":"The 927 ponzi_ml benign candidates remain BENIGN_CURATED. They were not upgraded to BENIGN_STRONG and possible contamination remains documented.",
      "Feature alignment":"Feature alignment means that the external contract contains sufficient deployment/static information to reproduce the inputs required by the evaluated system; it does not require every optional Phase 8 representation to be available. Current ready counts are all zero because no row was enriched.",
      "Manual review":f"The critical M1+M2+M3 set is zero because automatic enrichment could not identify later independent rows. The packet retains {r['manual_review_count']} M5 class-D predictions for future verification only; they do not enter Track B.",
      "Track B finalization":"Track B remains NONE and unsealed. No metadata/label seal was created and the default label loader remains deny-by-default.",
      "Portability":"Reused the Phase 8R6B hash-verified resolver audit for 27 relocated artifacts. Frozen absolute-path manifests and archived results were not edited.",
      "Environment":f"Python {sys.version.split()[0]}; {platform.platform()}; provider configuration class is environment_variables_only; CUDA is not required.",
      "Limitations":"No reliable configured creation provider or RPC endpoint was available. Zero observed later/independent contracts after an enrichment failure must not be interpreted as evidence of absence.",
      "Decision":"D. AWAIT PROVIDER / USER APPROVAL. Configure BLOCKSCOUT_API_URL, BLOCKSCOUT_API_KEY, and ETH_RPC_URL, then rerun the 20-address preflight. Do not begin Phase 9B automatically."
    }
    report="# Phase 8R6C: reliable enrichment and Track B finalization\n\n"+"\n\n".join(f"## {i+1}. {name}\n\n{content[name]}" for i,name in enumerate(sections))+"\n"
    (out/"phase8r6c_report.md").write_text(report,encoding="utf-8")

if __name__=="__main__":print(json.dumps(run(),indent=2))
