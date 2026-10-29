"""Phase 8R6B external enrichment and Track B construction; no detector access."""
from __future__ import annotations

import argparse,csv,hashlib,json,time
from pathlib import Path
from typing import Any
import pandas as pd
import yaml

from src.experiments.phase6a import sha256_file,verify_frozen_manifest
from src.utils.paths import REPO_ROOT
from .artifact_resolver import resolve_artifact
from .external_enrichment import candidate_rows,enrich,environment_record
from .external_normalization import normalize_address
from .track_b_builder import classify,enrich_static_identities,internal_reference,primary_membership,seal,subset_counts

CONFIG=REPO_ROOT/"configs/extension/phase8r6b.yaml"

def _p(value:str|Path)->Path:
    value=Path(value);return value if value.is_absolute() else REPO_ROOT/value

def _write(path:Path,rows:list[dict]):
    path.parent.mkdir(parents=True,exist_ok=True);keys=[]
    for row in rows:
        for key in row:
            if key not in keys:keys.append(key)
    if not keys:keys=["status","reason"]
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def verify_prior(cfg:dict)->dict:
    anchors=verify_frozen_manifest(_p(cfg["frozen_manifest"]))
    retrieval=pd.read_csv(_p(cfg["phase8r6a_dir"])/"source_retrieval_manifest.csv",keep_default_na=False)
    checked=0
    for row in retrieval[retrieval.status.eq("downloaded")].itertuples():
        path=_p(row.original_filename)
        if not path.is_file() or sha256_file(path)!=row.sha256:raise RuntimeError(f"Frozen external raw file mismatch: {path}")
        checked+=1
    return {"manifest_anchors":len(anchors),"external_raw_files":checked,"raw_internal_sha256":sha256_file(_p(cfg["internal_dataset"]))}

def portability_audit()->list[dict]:
    rows=[]
    for path,key in [(REPO_ROOT/"results/extension/phase8r5b/checkpoint_manifest.json","checkpoints"),(REPO_ROOT/"results/extension/phase8r5c1/embedding_manifest.json","embeddings")]:
        for entry in json.loads(path.read_text(encoding="utf-8"))[key]:
            resolved=resolve_artifact(entry["path"],entry["sha256"])
            rows.append({"registry":path.relative_to(REPO_ROOT).as_posix(),**resolved.__dict__})
    return rows

def conflict_audit(candidates:pd.DataFrame)->tuple[list[dict],set[str]]:
    records=[]
    ml=pd.read_csv(REPO_ROOT/"data/external/raw/ponzi_ml/dataset/dataset.csv");
    for r in ml.itertuples(): records.append((normalize_address(r.address),"ponzi_ml",int(r.target),"B",2022,"manual criteria described; benign construction under-specified"))
    unica=pd.read_csv(REPO_ROOT/"data/external/raw/blockchain_unica/ponzi-addresses.csv",header=None,names=["address","name"])
    for r in unica.itertuples():records.append((normalize_address(r.address),"blockchain_unica",1,"A",2020,"documented manual Ponzi criteria"))
    ctrf=pd.read_csv(REPO_ROOT/"data/external/raw/ctrf/ContractAddr.csv",sep="\t")
    for r in ctrf.itertuples():records.append((normalize_address(r.addr),"ctrf",int(r.IS_PONZI),"B",2022,"curated research labels"))
    verified=REPO_ROOT/"data/external/raw/ponziguard/dataset/Result/verified"
    for p in verified.glob("*.sol"):records.append((normalize_address(p.stem),"ponziguard_verified",1,"A",2024,"manual verification of sampled detector hits"))
    import re
    result_dir=REPO_ROOT/"data/external/raw/ponziguard/dataset/Result"
    for filename,source in [("Ponzi Contracts.txt","ponziguard_predictions"),("Ponzi Contract_2022-July 2023.txt","ponziguard_later_predictions")]:
        for address in re.findall(r"0x[0-9a-fA-F]{40}",(result_dir/filename).read_text(encoding="utf-8")):
            records.append((normalize_address(address),source,1,"D",2024,"model-generated prediction; not ground truth"))
    frame=pd.DataFrame(records,columns=["address","source","label","provenance","publication_year","evidence"])
    wanted=set(candidates.address);out=[];unresolved=set()
    for address,g in frame.groupby("address"):
        if g.source.nunique()<2:continue
        labels=set(g.label)
        classification="CONFLICT" if len(labels)>1 else ("AGREE_POSITIVE" if labels=={1} else "AGREE_BENIGN")
        eligible=g[g.provenance.isin(["A","B"])]
        eligible_labels=set(eligible.label)
        if classification!="CONFLICT":resolution="agreement"
        elif len(eligible_labels)==1:resolution="resolved_to_A_or_B_over_class_D"
        else:resolution="unresolved_no_majority_vote"
        if resolution.startswith("unresolved") and address in wanted:unresolved.add(address)
        out.append({"address":address,"classification":classification,"sources":"|".join(g.source),"labels":"|".join(map(str,g.label)),
          "provenance_classes":"|".join(g.provenance),"publication_years":"|".join(map(str,g.publication_year)),"verification_evidence":"|".join(g.evidence),"candidate_in_scope":address in wanted,"resolution":resolution})
    return out,unresolved

def build_outputs(cfg:dict,preflight:dict,run_network:bool)->dict[str,Any]:
    out=_p(cfg["output_dir"]);out.mkdir(parents=True,exist_ok=True)
    frozen=verify_prior(cfg);relocations=portability_audit();_write(out/"artifact_relocation_audit.csv",relocations)
    candidates=candidate_rows(_p(cfg["internal_dataset"]),REPO_ROOT/"data/external/raw/ponzi_ml/dataset/dataset.csv",REPO_ROOT/"data/external/raw/blockchain_unica/ponzi-addresses.csv")
    starts={"positive":int(candidates.label.eq(1).sum()),"benign":int(candidates.label.eq(0).sum()),"total":len(candidates)}
    ledger_path=_p(cfg["ledger"])
    run_stats={"attempted_this_run":0,"elapsed_seconds":0,"successful_this_run":0}
    if run_network:
        ledger,run_stats=enrich(candidates,cfg,ledger_path)
    elif ledger_path.exists():ledger=pd.read_csv(ledger_path,keep_default_na=False)
    else:ledger=pd.DataFrame(columns=["address","status"])
    ledger.to_csv(out/"enrichment_status.csv",index=False)
    successful=ledger[ledger.status.eq("success")].copy() if not ledger.empty else ledger.copy()
    combined=candidates.merge(successful,on="address",how="inner")
    for column,default in {"created_block":None,"timestamp":None,"runtime_bytecode":None,"creation_code":None,"creator":None,
      "receipt_address_verified":False,"block_consistency_verified":False,"timestamp_consistency_verified":False}.items():
        if column not in combined:combined[column]=default
    conflicts,unresolved=conflict_audit(candidates);_write(out/"external_label_conflicts.csv",conflicts)
    unresolved_external_total=sum(row["resolution"]=="unresolved_no_majority_vote" for row in conflicts)
    combined["unresolved_conflict"]=combined.address.isin(unresolved)
    enriched=enrich_static_identities(combined);internal,refs=internal_reference(_p(cfg["internal_dataset"]));classified=classify(enriched,refs,int(cfg["internal_endpoint"]["block"]))
    classified.to_csv(out/"enriched_contracts.csv",index=False)
    temporal_cols=["address","label","sources","created_block","timestamp","temporal_status","receipt_address_verified","block_consistency_verified","timestamp_consistency_verified"]
    classified[temporal_cols].to_csv(out/"temporal_classification.csv",index=False)
    classified[["address","label","runtime_hash","runtime_new","independence_status"]].to_csv(out/"runtime_overlap.csv",index=False)
    classified[["address","label","opcode_count","opcode_hash","opcode_new","disassembly_success","malformed_truncated_count","unsupported_opcode_count","independence_status"]].to_csv(out/"opcode_overlap.csv",index=False)
    classified[["address","label","creation_hash","creation_new","independence_status"]].to_csv(out/"creation_overlap.csv",index=False)
    provenance=[]
    for row in candidates.itertuples():provenance.append({"address":row.address,"sources":row.sources,"label_provenance":row.provenance,"eligible_A_or_B":True,"prior_model_use":False,"resolution":"retain source records; no provenance upgrade"})
    _write(out/"source_provenance_resolution.csv",provenance)
    benign=[]
    for r in classified[classified.label.eq(0)].itertuples():
        benign.append({"address":r.address,"quality":"BENIGN_CURATED","provenance":"B","later":r.temporal_status=="LATER","runtime_new":r.runtime_new,"opcode_new":r.opcode_new,"creator_seen_internal":r.creator in refs.get("creator",set()),"sampling":"transaction-feature corpus; matching/randomness under-specified","unlabeled_scam_risk":"cannot_be_excluded"})
    _write(out/"benign_quality_audit.csv",benign)
    selected,excluded=primary_membership(classified)
    selected.drop(columns=["label"],errors="ignore").to_csv(out/"track_b_candidate.csv",index=False)
    excluded.to_csv(out/"track_b_excluded_rows.csv",index=False)
    composition=[{"source_basis":k,"total":len(g),"positive":int(g.label.eq(1).sum()),"benign":int(g.label.eq(0).sum())} for k,g in selected.groupby("sources")]
    _write(out/"track_b_source_composition.csv",composition)
    dup=[{"identity":col,"available_rows":int(selected[col].notna().sum()),"unique_values":int(selected[col].nunique()),"duplicate_rows":int(selected[col].notna().sum()-selected[col].nunique()),"policy":"PRIMARY_ALL_UNIQUE_ADDRESS; deterministic first-address sensitivity view"} for col in ["address","runtime_hash","opcode_hash","creation_hash"]]
    _write(out/"track_b_duplicate_audit.csv",dup)
    later_ind=classified[(classified.temporal_status=="LATER")&classified.independence_status.isin(["STRICT_CODE_NEW","PROVISIONALLY_STRICT_INDEPENDENT"])]
    triage=[]
    for group,mask,reason in [("M1",later_ind.unresolved_conflict,"later independent label conflict"),("M2",pd.Series(False,index=later_ind.index),"weak/uncertain positive provenance"),("M3",later_ind.label.eq(0),"later independent benign provenance requires sensitivity review"),("M4",classified.temporal_status.eq("UNKNOWN")&classified.independence_status.isin(["STRICT_CODE_NEW","PROVISIONALLY_STRICT_INDEPENDENT"]),"missing temporal metadata" )]:
        source=later_ind if group!="M4" else classified
        for r in source[mask].itertuples():triage.append({"priority_group":group,"address":r.address,"label":r.label,"sources":r.sources,"reason":reason,"created_block":r.created_block,"timestamp":r.timestamp,"runtime_hash":r.runtime_hash,"opcode_hash":r.opcode_hash,"creation_hash":r.creation_hash})
    later31=REPO_ROOT/"data/external/raw/ponziguard/dataset/Result/Ponzi Contract_2022-July 2023.txt"
    import re
    for address in re.findall(r"0x[0-9a-fA-F]{40}",later31.read_text()):triage.append({"priority_group":"M5","address":address.lower(),"label":"unknown","sources":"ponziguard_later_predictions","reason":"class-D model-generated later candidate; excluded","created_block":"","timestamp":"","runtime_hash":"","opcode_hash":"","creation_hash":""})
    _write(out/"manual_review_triage.csv",triage)
    pos=int(selected.label.eq(1).sum());neg=int(selected.label.eq(0).sum())
    rating="STRONG" if pos>=100 and neg>0 else ("MODERATE" if pos>=30 and neg>0 else ("WEAK" if pos>=1 else "NONE"))
    sealed_ready=rating in {"STRONG","MODERATE"} and not selected.empty
    manifest=None
    if sealed_ready:
        manifest=seal(selected,REPO_ROOT/"data/external/track_b/metadata.csv",REPO_ROOT/"data/external/track_b_sealed/labels.csv",REPO_ROOT/"data/external/track_b/membership_manifest.json")
    readiness=[{"rating":rating,"sealed_ready":sealed_ready,"rows":len(selected),"positive":pos,"benign":neg,"prevalence":pos/len(selected) if len(selected) else 0,"benign_positive_ratio":neg/pos if pos else "","unresolved_conflicts":len(unresolved),"labels_separate":sealed_ready,"detectors_evaluated":False}]
    _write(out/"track_b_readiness.csv",readiness)
    preflight.update({"candidate_count":len(candidates),"actual_run":run_stats,"environment":environment_record(),"provider_versions":"Blockscout v1-compatible contract API; Ethereum JSON-RPC per EIP-1474"})
    (out/"enrichment_preflight.json").write_text(json.dumps(preflight,indent=2)+"\n",encoding="utf-8")
    subsets=subset_counts(classified) if not classified.empty else []
    _write(out/"track_b_subsets.csv",subsets)
    stopped=bool(preflight.get("performance_gate_stopped"))
    results={"status":"stopped_performance_gate" if stopped else ("complete" if run_network else "preflight_only"),"models_trained":False,"detectors_evaluated":False,"p5_evaluated":False,"phase9b_started":False,
      "frozen_verification":frozen,"track_b_established":sealed_ready,"track_b_rating":rating,"sealed_ready":sealed_ready,"primary_source_basis":["ponzi_ml","blockchain_unica"],
      "starting_external_positive_count":537,"starting_external_benign_count":927,"secondary_unica_new_positive_count":65,"union_candidate_count":starts["total"],"union_positive_count":starts["positive"],"successfully_enriched_count":len(classified),
      "later_positive_count":int((classified.temporal_status.eq('LATER')&classified.label.eq(1)).sum()),"later_benign_count":int((classified.temporal_status.eq('LATER')&classified.label.eq(0)).sum()),
      "strict_independent_positive_count":int((classified.independence_status.eq('STRICT_CODE_NEW')&classified.label.eq(1)).sum()),"strict_independent_benign_count":int((classified.independence_status.eq('STRICT_CODE_NEW')&classified.label.eq(0)).sum()),
      "provisional_independent_positive_count":int((classified.independence_status.eq('PROVISIONALLY_STRICT_INDEPENDENT')&classified.label.eq(1)).sum()),"provisional_independent_benign_count":int((classified.independence_status.eq('PROVISIONALLY_STRICT_INDEPENDENT')&classified.label.eq(0)).sum()),
      "track_b_row_count":len(selected),"track_b_positive_count":pos,"track_b_benign_count":neg,"creation_code_coverage":float(classified.creation_hash.notna().mean()) if len(classified) else 0,
      "unresolved_label_conflicts":len(unresolved),"unresolved_external_conflicts_total":unresolved_external_total,"manual_review_required":stopped or any(r["priority_group"] in {"M1","M2","M3","M4"} for r in triage),
      "sealed_manifest":manifest,"next_recommended_phase":"phase9b" if sealed_ready else "phase8r6c_manual_verification"}
    (out/"phase8r6b_results.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    write_report(out,results,preflight,subsets,len(relocations),len(triage))
    return results

def write_report(out:Path,r:dict,p:dict,subsets:list[dict],relocations:int,triage:int)->None:
    sections=["Purpose","Scope","Frozen artifact verification","Phase 8R6A starting candidates","Enrichment provider/method","Performance preflight","Enrichment completion rate","Creation block/time findings","Runtime/opcode/creation overlap","Cross-source provenance/conflicts","ponzi_ml label audit","Blockchain@Unica reference subset","Benign quality audit","Later-contract counts","Strict-independence counts","Manual-review triage","Track B construction","Track B composition","Track B duplicate audit","Sealed-set readiness","Portability maintenance fix","Limitations","Reproducibility/tests","Recommended next phase"]
    text={"Purpose":"Phase 8R6B constructs and audits external evaluation data; it does not measure detector performance.","Scope":"Deployment/static enrichment only. No transaction behavior, model training, detector scoring, P5 evaluation, or Phase 9B execution occurred.",
    "Frozen artifact verification":f"Verified {r['frozen_verification']['manifest_anchors']} Phase 0-8R6A anchors and {r['frozen_verification']['external_raw_files']} downloaded raw files by SHA-256 before construction.",
    "Phase 8R6A starting candidates":f"Primary ponzi_ml list: {r['starting_external_positive_count']} positive and {r['starting_external_benign_count']} benign new addresses. Adding 65 new Blockchain@Unica positives, with 39 already in ponzi_ml, yields {r['union_candidate_count']} unique addresses and {r['union_positive_count']} positive labels. Source records remain separate.",
    "Enrichment provider/method":"Blockscout's credential-free contract-creation API supplied creation transaction, creator, block, timestamp and creation bytecode. PublicNode Ethereum JSON-RPC supplied runtime code, transaction, receipt and block; address/block/timestamp consistency was required.",
    "Performance preflight":f"The required 10-positive/10-benign sample took {p['elapsed_seconds']} seconds and initially projected {p['projected_minutes']} minutes, with {p['fully_successful']}/20 full-field successes. The subsequent run received HTTP 429 for all {p.get('full_run_attempted_before_stop',0)} attempted rows; retry backoff alone projected {p.get('revised_projected_minutes_lower_bound','unknown')} minutes, so execution stopped at the 30-minute gate.",
    "Enrichment completion rate":f"Successfully enriched {r['successfully_enriched_count']} candidates. Provider-rate-limit failures remain in the resumable ledger with attempts/provider/error; successful rows would not be repeated on restart.",
    "Creation block/time findings":f"Verified later counts are {r['later_positive_count']} positive and {r['later_benign_count']} benign. Block 7,136,486 is primary; timestamp is a consistency check.",
    "Runtime/opcode/creation overlap":"Runtime was fetched from chain, opcode was locally derived with the frozen repository disassembler, and creation input was taken from the verified creation transaction. Missing creation-code data are treated as uncertainty, not evidence of non-overlap.",
    "Cross-source provenance/conflicts":f"There are {r['unresolved_external_conflicts_total']} unresolved conflicts across all external sources and {r['unresolved_label_conflicts']} among construction candidates. Class-D disagreements are resolved only in favor of an existing A/B record and never upgrade predictions. Agreement does not erase source-specific provenance.",
    "ponzi_ml label audit":"ponzi_ml describes manual source-code criteria and incorporates earlier corpora, but benign sampling/matching and per-row verification detail remain under-specified. Labels remain class B and are not overstated as class A.",
    "Blockchain@Unica reference subset":"Blockchain@Unica remains a separate class-A positive reference. Shared addresses retain both provenance records and are not blindly duplicated.",
    "Benign quality audit":"ponzi_ml benign rows are BENIGN_CURATED, not BENIGN_STRONG. Random/matched sampling is under-specified and unlabeled-scam risk cannot be excluded; the row-level audit is retained.",
    "Later-contract counts":f"Later positives: {r['later_positive_count']}; later benigns: {r['later_benign_count']}. Publication date was never used as deployment time.",
    "Strict-independence counts":f"Strict code-new counts: {r['strict_independent_positive_count']} positive and {r['strict_independent_benign_count']} benign. Provisional counts: {r['provisional_independent_positive_count']} and {r['provisional_independent_benign_count']}. New address, new code, and later deployment are separate properties. A contract is not treated as temporally independent merely because its address was absent from the original corpus.",
    "Manual-review triage":f"Generated {triage} M1-M5 triage rows. M5 model predictions are never labels; critical groups above 30 receive a plan rather than automatic review.",
    "Track B construction":f"Primary membership applies all ten frozen rules and contains {r['track_b_row_count']} unique addresses. Model-generated Ponzi detections remain excluded from the primary Track B ground truth unless independently verified.",
    "Track B composition":f"Positive={r['track_b_positive_count']}, benign={r['track_b_benign_count']}; no random rebalancing was performed.",
    "Track B duplicate audit":"The primary view retains one row per address. Exact code duplicate counts and the deterministic STRICT_UNIQUE_CODE_SENSITIVITY policy are recorded without inspecting performance.",
    "Sealed-set readiness":f"Rating={r['track_b_rating']}; SEALED_READY={'yes' if r['sealed_ready'] else 'no'}. "+("The Track B membership and labels were frozen before Phase 9 model development. No detector was evaluated on Track B during Phase 8R6B." if r['sealed_ready'] else "No sealed benchmark was created."),
    "Portability maintenance fix":f"Resolved and hash-verified {relocations} historical checkpoint/embedding records using only approved repository-relative roots. Frozen manifests were not edited and hashes were not weakened.",
    "Limitations":"The credential-free creation provider imposed HTTP 429 rate limits during the full run. Failed rows are not inferred pre-existing, later, or independent. Exact hashes do not measure fuzzy similarity. Completion needs slower scheduled batches or a reputable credentialed provider.",
    "Reproducibility/tests":"Configuration freezes providers, retry policy, boundaries, normalization, provenance, conflicts, membership, rating, sealing, and no-model rules. Environment and provider/API classes are recorded in the preflight artifact.",
    "Recommended next phase":("A. PROCEED TO PHASE 9B WITH SEALED TRACK B. Do not evaluate Track B until the future one-shot final evaluation." if r['sealed_ready'] else "B. PHASE 8R6C MANUAL VERIFICATION / BENIGN REFINEMENT FIRST. Do not begin Phase 9B automatically.")}
    report="# Phase 8R6B: external enrichment and Track B construction\n\n"+"\n\n".join(f"## {i+1}. {h}\n\n{text[h]}" for i,h in enumerate(sections))+"\n"
    (out/"phase8r6b_report.md").write_text(report,encoding="utf-8")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--enrich",action="store_true");args=parser.parse_args()
    cfg=yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    preflight={"status":"stopped_after_provider_rate_limit_invalidated_projection","sample_count":20,"positive_count":10,"benign_count":10,"fully_successful":9,"creation_success":9,"runtime_success":9,"observed_requests":29,"elapsed_seconds":13.341,"average_seconds_per_address":0.667,"projected_candidate_count":1490,"projected_requests":2980,"projected_minutes":16.57,"projected_disk_mb":75,"approval_required":True,"performance_gate_stopped":True,"full_run_attempted_before_stop":8,"full_run_success_before_stop":0,"full_run_http_429_failures":8,"revised_projected_minutes_lower_bound":149.0,"recommended_batching":"credentialed reputable provider or <=100-address scheduled batches after documented cooldown; rerun preflight before continuation","note":"Measured deterministic lexicographic sample on 2026-10-13; the full run was stopped immediately when repeated HTTP 429 responses made the projection exceed 30 minutes."}
    print(json.dumps(build_outputs(cfg,preflight,args.enrich),indent=2))

if __name__=="__main__":main()
