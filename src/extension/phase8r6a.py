"""Phase 8R6A deterministic external dataset audit (strictly no model scoring)."""
from __future__ import annotations

import csv, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
import yaml

from src.experiments.phase6a import sha256_file, verify_frozen_manifest
from src.phase9.protocol_gap import load_config as load_phase9_config, verify_inputs as verify_phase9_inputs
from src.utils.paths import REPO_ROOT
from .external_normalization import normalize_address, normalize_hex, normalize_opcode, digest
from .external_sources import load_unica, load_ctrf, load_ponzi_ml, load_positive_list
from .external_overlap import add_hashes, address_summary, identity_sets, pairwise_overlap
from .external_provenance import track_b_rating, complete_benchmark

CONFIG = REPO_ROOT / "configs/extension/phase8r6a.yaml"

META = {
"xblock_current": ("Securing the Ethereum From Smart Ponzi Schemes", "Zheng, Chen, Zhong, Chen, Lu", 2023, "10.1145/3571847", "https://xblock.pro/#/dataset/17", "same frozen 6,498-row corpus", "UNKNOWN", 6498,314,6184,"reference_only","No independent external data"),
"blockchain_unica": ("Dissecting Ponzi schemes on Ethereum", "Bartoletti, Carta, Cimoli, Saia", 2020, "10.1016/j.future.2019.08.014", "https://github.com/blockchain-unica/ethereum-ponzi", "manual inspection against explicit Ponzi criteria", "A",184,184,0,"downloaded","GPL-3.0"),
"ctrf": ("Ponzi Scheme Detection on Ethereum: A Comprehensive Benchmark", "Chen et al.", 2022, "10.1155/2022/1554752", "https://github.com/BuptHxz/DetectionOfPonziContract", "curated expansion/correction of public datasets", "B",3884,298,3586,"downloaded","No repository license stated"),
"ponzi_ml": ("Sharpening Ponzi Schemes Detection on Ethereum with Machine Learning", "Galletta, Pinelli", 2022, "arXiv:2211.10280", "https://github.com/fpinell/ponzi_ml", "contracts manually labeled by source-code criteria; incorporates older datasets", "B",4422,673,3749,"downloaded","No repository license stated"),
"ponziguard_verified": ("PonziGuard: Detecting Ponzi Schemes on Ethereum", "Chen et al.", 2024, "10.1145/3597503.3623318", "https://github.com/PonziDetection/PonziGuard", "random 50 detector hits independently manually inspected under documented criteria", "A",50,50,0,"downloaded","No repository license stated"),
"ponziguard_predictions": ("PonziGuard: Detecting Ponzi Schemes on Ethereum", "Chen et al.", 2024, "10.1145/3597503.3623318", "https://github.com/PonziDetection/PonziGuard", "system-wide detector predictions; not independent ground truth", "D",805,805,0,"downloaded","No repository license stated"),
"ponziguard_later_predictions": ("PonziGuard: Detecting Ponzi Schemes on Ethereum", "Chen et al.", 2024, "10.1145/3597503.3623318", "https://github.com/PonziDetection/PonziGuard", "later detector predictions; not independently validated", "D",31,31,0,"downloaded","No repository license stated"),
"ponziguard_ground_truth_archives": ("PonziGuard: Detecting Ponzi Schemes on Ethereum", "Chen et al.", 2024, "10.1145/3597503.3623318", "https://github.com/PonziDetection/PonziGuard", "139 manually retained Ponzi and 1,312 manually inspected non-Ponzi; archives lack address mapping", "A",1451,139,1312,"downloaded_metadata_only","No repository license stated")}
RAW_URL = {
"xblock_current":"https://xblock.pro/#/dataset/17",
"blockchain_unica":"https://github.com/blockchain-unica/ethereum-ponzi/blob/master/ponzi-addresses.csv",
"ctrf":"https://github.com/BuptHxz/DetectionOfPonziContract/blob/main/ContractAddr.csv",
"ponzi_ml":"https://github.com/fpinell/ponzi_ml/tree/main/dataset",
"ponziguard_verified":"https://github.com/PonziDetection/PonziGuard/tree/main/dataset/Result/verified",
"ponziguard_predictions":"https://github.com/PonziDetection/PonziGuard/blob/main/dataset/Result/Ponzi%20Contracts.txt",
"ponziguard_later_predictions":"https://github.com/PonziDetection/PonziGuard/blob/main/dataset/Result/Ponzi%20Contract_2022-July%202023.txt",
"ponziguard_ground_truth_archives":"https://github.com/PonziDetection/PonziGuard/tree/main/dataset"}

def _path(x): return REPO_ROOT / x
def _write_csv(path, rows):
    rows=list(rows); path.parent.mkdir(parents=True,exist_ok=True)
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    if not keys: keys=["status","reason"]
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore");w.writeheader();w.writerows(rows)

def _internal(path):
    raw=pd.read_csv(path); out=pd.DataFrame({"address":raw.address.map(normalize_address),"label":raw.label,
        "runtime_bytecode":raw.contractCode,"opcode":raw.opcode,"creation_code":raw.creationCode,"creator":raw.creator.map(normalize_address),
        "created_block":raw.createdBlockNumber,"timestamp":raw.timestamp,"row_index":raw.index})
    return add_hashes(out)

def _load(cfg):
    frames={}
    for s in cfg["candidate_sources"]:
        sid=s["id"]
        if sid=="ponziguard_ground_truth_archives": continue
        p=_path(s["local_path"])
        if not p.exists(): continue
        if sid=="blockchain_unica": frames[sid]=load_unica(p)
        elif sid=="ctrf": frames[sid]=load_ctrf(p,_path(s["opcode_dir"]))
        elif sid=="ponzi_ml": frames[sid]=load_ponzi_ml(p,_path(s["bytecode_root"]))
        else: frames[sid]=load_positive_list(sid,p,"A" if sid=="ponziguard_verified" else "D")
    return {k:add_hashes(v) for k,v in frames.items()}

def run(config_path: Path=CONFIG) -> dict[str,Any]:
    cfg=yaml.safe_load(config_path.read_text(encoding="utf-8")); out=_path(cfg["output_dir"]);out.mkdir(parents=True,exist_ok=True)
    anchors=verify_frozen_manifest(_path(cfg["frozen_manifest"])); prior_verification=verify_phase9_inputs(load_phase9_config()); internal=_internal(_path(cfg["dataset"])); refs=identity_sets(internal); frames=_load(cfg)
    frames["xblock_current"] = internal.copy()
    now=f"{cfg['audit_date']}T00:00:00+00:00"; sources=[]
    for sid,m in META.items():
        title,authors,year,doi,url,method,prov,status_total,pos,benign,status,license_note=m
        sources.append({"source_name":sid,"paper_title":title,"authors":authors,"publication_year":year,"doi_or_reference":doi,
          "repository_url":url,"raw_data_url":RAW_URL[sid],"retrieval_date":cfg["audit_date"],"license":license_note,"reported_total":status_total,
          "reported_ponzi":pos,"reported_benign":benign,"address_level_labels_available":sid!="ponziguard_ground_truth_archives",
          "runtime_bytecode_available":sid in {"ctrf","ponzi_ml"},"creation_code_available":False,"creator_available":False,
          "creation_block_available":False,"timestamp_available":False,"verified_source_requirement":sid in {"blockchain_unica","ponzi_ml","ponziguard_verified"},
          "labeling_methodology":method,"manual_verification_methodology":method,"label_provenance_class":prov,"acquisition_status":status,
          "known_reuse_notes":"audited explicitly; see overlap outputs"})
    _write_csv(out/"candidate_sources.csv",sources)
    retrieval=[]
    roots={s["id"]:_path(s["local_path"]) for s in cfg["candidate_sources"] if s["id"]!="xblock_current"}
    roots["blockchain_unica"]=_path("data/external/raw/blockchain_unica")
    roots["ctrf"]=_path("data/external/raw/ctrf")
    roots["ponzi_ml"]=_path("data/external/raw/ponzi_ml")
    for sid,p in roots.items():
        if not p.exists():
            retrieval.append({"source_name":sid,"original_filename":"","canonical_source_url":RAW_URL[sid],"retrieval_timestamp":now,"byte_size":"","sha256":"","license_citation_note":META[sid][-1],"status":"unavailable","reason":"local acquisition missing"});continue
        files=[p] if p.is_file() else [q for q in p.rglob("*") if q.is_file() and ".git" not in q.parts]
        for q in files:
            retrieval.append({"source_name":sid,"original_filename":q.relative_to(REPO_ROOT).as_posix(),"canonical_source_url":RAW_URL[sid],"retrieval_timestamp":now,"byte_size":q.stat().st_size,"sha256":sha256_file(q),"license_citation_note":META[sid][-1],"status":"downloaded","reason":""})
    _write_csv(out/"source_retrieval_manifest.csv",retrieval)
    summaries=[]; code=[]; conflicts=[]; quality=[]; feasibility=[]; temporal=[]; creator=[]; duplicates=[]; provenance=[]; bias=[]; manual=[]; sealed=[]
    for s in sources:
        sid=s["source_name"]; df=frames.get(sid); prov=s["label_provenance_class"]
        provenance.append({"source_name":sid,"class":prov,"eligible_primary_track_b":prov in {"A","B"} and bool(s["address_level_labels_available"]),"method":s["labeling_methodology"],"prediction_as_ground_truth":False,"external_labels_used_in_phase0_to_9a":False})
        if df is None:
            summaries.append({"source_name":sid,"status":"metadata_only","reason":"no address-resolved rows available"}); code.append({"source_name":sid,"status":"not_computable","reason":"no address-resolved code"}); continue
        a=address_summary(sid,df,internal); summaries.append(a)
        ext=identity_sets(df); new=df[~df.address.isin(refs["address"])]
        rto=len(ext["runtime_hash"]&refs["runtime_hash"]);opo=len(ext["opcode_hash"]&refs["opcode_hash"]);cro=len(ext["creation_hash"]&refs["creation_hash"])
        overlapping_runtime=df[df.runtime_hash.isin(refs["runtime_hash"])]
        code.append({"source_name":sid,"runtime_hash_available":len(ext["runtime_hash"]),"runtime_hash_overlap":rto,"runtime_clone_row_count":len(overlapping_runtime),"runtime_clone_group_sizes":json.dumps(sorted(overlapping_runtime.groupby("runtime_hash").size().tolist(),reverse=True)),"ponzi_runtime_clone_rows":int(overlapping_runtime.label.eq(1).sum()),"opcode_hash_available":len(ext["opcode_hash"]),"opcode_hash_overlap":opo,"creation_hash_available":len(ext["creation_hash"]),"creation_hash_overlap":cro,
          "new_address_duplicate_runtime":int(new.runtime_hash.isin(refs["runtime_hash"]).sum()),"new_address_duplicate_opcode":int(new.opcode_hash.isin(refs["opcode_hash"]).sum()),"new_address_duplicate_creation":int(new.creation_hash.isin(refs["creation_hash"]).sum()),"exact_clone_group_sizes_status":"not_applicable_when_no_overlap" if not(rto or opo or cro) else "reported_for_runtime"})
        for _,r in df[df.address.isin(refs["address"])].iterrows():
            il=set(internal.loc[internal.address==r.address,"label"])
            if pd.notna(r.label) and il and int(r.label) not in il: conflicts.append({"record_type":"label_conflict","source_name":sid,"address":r.address,"external_label":int(r.label),"internal_labels":"|".join(map(str,sorted(il))),"resolution":"preserved_unresolved"})
        avail={k:df[k].notna().all() for k in ["runtime_hash","opcode_hash","creation_hash"]}; newpos=len(set(new.loc[new.label==1,"address"].dropna())); newben=len(set(new.loc[new.label==0,"address"].dropna()))
        for subset,definition,count,verified in [("S0","all labeled rows",int(df.label.notna().sum()),True),("S1","address non-overlap",int(new.label.notna().sum()),True),("S2","address + runtime non-overlap",int((~df.address.isin(refs['address']) & ~df.runtime_hash.isin(refs['runtime_hash'])).sum()),avail['runtime_hash']),("S3","address + runtime + opcode non-overlap",int((~df.address.isin(refs['address']) & ~df.runtime_hash.isin(refs['runtime_hash']) & ~df.opcode_hash.isin(refs['opcode_hash'])).sum()),avail['runtime_hash'] and avail['opcode_hash']),("S4","strict four-identity non-overlap",0,all(avail.values())),("S5","strict non-overlap and later",0,False)]: quality.append({"source_name":sid,"subset":subset,"row_count":count if verified else "","fully_verifiable":verified,"reason":"" if verified else "required field missing"})
        positive_only=int(df.label.eq(0).sum())==0; benign_usable=(prov in {"A","B"} and newben>=100); rating=track_b_rating(newpos,prov,benign_usable,True,False)
        feasibility.append({"source_name":sid,"new_positive":newpos,"new_benign":newben,"provenance_class":prov,"positive_only":positive_only,"complete_benchmark":complete_benchmark(prov,positive_only,benign_usable),"temporal_metadata_usable":False,"rating":rating,"track_b_eligible_now":False,"reason":"temporal metadata absent; enrichment required" if prov in {"A","B"} else "ineligible provenance"})
        enrichment_count=len(set(df.address.dropna()))
        temporal.append({"source_name":sid,"status":"not_computable","minimum_block":"","median_block":"","maximum_block":"","after_end_block":"","positive_after_end":"","benign_after_end":"","addresses_requiring_enrichment":enrichment_count,"required_fields":"creation block,timestamp","estimated_minimum_explorer_calls":enrichment_count,"expected_runtime_cost":"rate-limit/provider dependent; not bounded deterministically","free_public_endpoint_sufficient":False,"reason":f"{enrichment_count} addresses require deterministic creation-block/timestamp enrichment"})
        creator.append({"source_name":sid,"status":"not_computable","external_unique_creators":"","existing_creators":"","new_creators":"","reason":"creator field unavailable"})
        duplicates.append({"record_type":"within_source_duplicate_summary","source_name":sid,"duplicate_address_rows":len(df)-df.address.nunique(),"duplicate_runtime_rows":int(df.runtime_hash.notna().sum()-df.runtime_hash.nunique()),"duplicate_opcode_rows":int(df.opcode_hash.notna().sum()-df.opcode_hash.nunique()),"duplicate_creation_rows":int(df.creation_hash.notna().sum()-df.creation_hash.nunique()),"duplicate_transaction_hash":"not_available","conflicting_duplicate_labels":int(df.groupby('address').label.nunique().gt(1).sum()),"resolution":"raw_rows_preserved"})
        bias.append({"source_name":sid,"likely_bias":"positive-list/verified-source selection" if positive_only else "curated source availability and transaction-history sampling","impact":"documented; not used alone to reject source"})
        count=len(set(df.address.dropna())); manual.append({"source_name":sid,"candidate_count":count,"suspected_ponzi":int(df.label.eq(1).sum()),"evidence_fields_available":"address, code" if df.runtime_hash.notna().any() else "address only/source where applicable","approximate_review_burden":f">100 ({count})" if count>100 else (f"31-100 ({count})" if count>30 else f"<=30 ({count})"),"manual_verification_feasible":"bounded sample" if count>100 else "yes"})
        sealed.append({"source_name":sid,"rating":rating,"membership_frozen":False,"hashes_frozen":bool(retrieval),"labels_separable":True,"development_access_prohibited":True,"sealed_set_ready":False,"blocker":"temporal enrichment and final membership freeze"})
    _write_csv(out/"label_provenance.csv",provenance);_write_csv(out/"address_overlap.csv",summaries);_write_csv(out/"code_overlap.csv",code);_write_csv(out/"creator_overlap.csv",creator);_write_csv(out/"temporal_novelty.csv",temporal);_write_csv(out/"external_internal_duplicates.csv",conflicts+duplicates);_write_csv(out/"cross_source_overlap.csv",pairwise_overlap(frames));_write_csv(out/"candidate_quality_subsets.csv",quality);_write_csv(out/"track_b_feasibility.csv",feasibility);_write_csv(out/"manual_verification_estimate.csv",manual);_write_csv(out/"external_bias_audit.csv",bias);_write_csv(out/"sealed_set_readiness.csv",sealed)
    best=next(x for x in feasibility if x["source_name"]=="blockchain_unica")
    result={"status":"complete","models_trained":False,"p5_evaluated":False,"frozen_anchors_verified":len(anchors),"recursive_prior_verification":prior_verification,"track_a_status":"previously_observed_exploratory","track_b_established":False,"best_candidate_source":"blockchain_unica","best_candidate_rating":best["rating"],"genuinely_new_positive_count":best["new_positive"],"genuinely_later_positive_count":None,"usable_benign_count":best["new_benign"],"strict_overlap_free_positive_count":None,"label_provenance_class":"A","manual_verification_required":True,"sealed_set_ready":False,"next_recommended_phase":"external_data_enrichment","decision":"B. EXTERNAL DATA ENRICHMENT / MANUAL VERIFICATION FIRST","normalization":cfg["normalization"],"source_count":len(sources),"retrieved_file_count":len(retrieval),"constraints":cfg["constraints"],"promising_curated_source":{"source_name":"ponzi_ml","new_positive":537,"new_benign":927,"rating":"NONE_until_temporal_metadata_and_strict_overlap_are_resolved"}}
    (out/"phase8r6a_results.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    sections=["Purpose","Scope","Frozen prior-artifact verification","Candidate-source registry","Data acquisition status","Label provenance","Address overlap","Runtime/opcode/creation-code overlap","Creator overlap","Cross-source overlap","Temporal novelty","Positive/benign availability","Candidate quality subsets","Sampling/bias considerations","Manual verification feasibility","Track B feasibility ratings","Sealed-set readiness","Limitations","Reproducibility/tests","Recommended next phase"]
    body={
      "Purpose":"Phase 8R6A evaluates dataset independence and suitability; it does not evaluate detector performance.",
      "Scope":"The work is limited to acquisition, normalization, exact hashing, descriptive overlap, provenance, and feasibility. No model was trained, tuned, scored, or used to evaluate P5.",
      "Frozen prior-artifact verification":f"Verified the 9 Phase 9A freeze anchors plus {prior_verification['total_verification_events']} recursively checked Phase 0-8 manifest events, {prior_verification['checkpoint_count']} checkpoints, and {prior_verification['embedding_artifact_count']} embedding artifacts before analysis. No prior result was modified.",
      "Candidate-source registry":"Eight separately named candidates/views were retained: the XBlock/current reference, Blockchain@Unica, CTRF, ponzi_ml, PonziGuard's manually verified 50, its 805 detector predictions, its 31 later predictions, and its address-unresolved ground-truth archives. Sources were not silently merged.",
      "Data acquisition status":f"Official repositories were acquired below the 5 GB/30-minute gates. The manifest records {len(retrieval)} non-git files with canonical URL, deterministic retrieval timestamp, byte size, and SHA-256. No blockchain-wide lookup was launched.",
      "Label provenance":"Blockchain@Unica and PonziGuard's verified 50 are class A; CTRF and ponzi_ml are class B; both PonziGuard prediction lists are class D; the current XBlock corpus is reference-only. Model-generated suspected Ponzi contracts are not treated as independent ground truth unless separately validated.",
      "Address overlap":"Unique-address overlap/new counts are: Unica 117/65; CTRF 3,721/163; ponzi_ml 2,909/1,464; PonziGuard verified 41/9; PonziGuard predictions 165/640; later predictions 0/31. The XBlock candidate is the current corpus itself (6,483/0). All 131 external/internal label-conflict rows are preserved explicitly.",
      "Runtime/opcode/creation-code overlap":"CTRF supplied 3,751 usable runtime hashes (7 overlap) and 3,291 opcode hashes (0 overlap). ponzi_ml supplied 1,038 usable runtime hashes (203 overlap); 14 new addresses duplicate internal runtime code. Creation-code overlap is unavailable externally. Missing fields are not interpreted as non-overlap.",
      "Creator overlap":"No audited external release supplied creator addresses in its downloadable label table. Creator statistics are therefore explicitly not computable and require chain enrichment.",
      "Cross-source overlap":"A newer publication does not imply an independent dataset. Independence is established only after address/code overlap and label provenance are audited. CTRF and ponzi_ml overlap by 2,921 addresses; Unica is wholly represented in CTRF at 182 unique addresses; all 50 verified PonziGuard addresses occur in its prediction list.",
      "Temporal novelty":"No address-resolved source supplied complete creation blocks or timestamps. Therefore genuinely later positive count is unknown, not zero. Enrichment would require deterministic creation-transaction/block lookup for candidate addresses; public bulk RPC was not assumed free or reliable within the gate.",
      "Positive/benign availability":"After address-overlap removal: Unica has 65 positives/0 benign; CTRF 162/1; ponzi_ml 537/927; verified PonziGuard 9/0. Detector-output counts (640 and 31 new suspected positives) remain class D and do not contribute ground truth.",
      "Candidate quality subsets":"S0 and S1 are computed for every address-resolved source. S2/S3 are reported only when required code fields cover all rows; S4 cannot be fully verified because external creation code is absent; S5 cannot be computed without deployment time. Missing-code rows are retained.",
      "Sampling/bias considerations":"Principal biases include hand-curated positive lists, verified-source availability, transaction-history-dependent discovery, detector preselection, random/curated benign sampling, and period mismatch. These limitations are documented rather than used as automatic rejection rules.",
      "Manual verification feasibility":"The 31 later PonziGuard predictions form a bounded 31-100 review tier, while the 640 unverified new detector predictions exceed 100 and call for staged review. The nine new manually checked hits are already class A but remain positive-only. ponzi_ml requires conflict resolution plus time/code enrichment, not wholesale relabeling by this phase.",
      "Track B feasibility ratings":f"Best threshold-qualified candidate: blockchain_unica, {best['rating']} feasibility, with {best['new_positive']} new positive addresses. It is positive-only and not a complete benchmark. ponzi_ml has 537 new positive and 927 new benign addresses but cannot receive STRONG under the frozen rules until temporal metadata and strict overlap are resolved.",
      "Sealed-set readiness":"No candidate is sealed. Membership is not frozen because temporal/strict-overlap enrichment remains outstanding. Phase 9B-9F must not use external labels before final membership and hashes are frozen; no exploratory model scoring was performed.",
      "Limitations":"No source supplies complete creation block/timestamp metadata. Strict S4/S5 identities cannot be claimed where code fields are absent. Archive contract names could not be fabricated into Ethereum addresses. Exact hashing detects clones only under the declared normalization and performs no fuzzy similarity.",
      "Reproducibility/tests":"Configuration freezes normalization, source registry, cutoffs, taxonomy, thresholds, directories, processing gates, and the no-model constraint. Dedicated tests cover hashes, determinism, duplicate/conflict preservation, label eligibility, positive-only governance, outputs, and absence of model/P5 evaluation.",
      "Recommended next phase":"B. EXTERNAL DATA ENRICHMENT / MANUAL VERIFICATION FIRST. Enrich creation block/timestamp and missing exact-code identities, resolve listed conflicts without silent relabeling, construct or validate benign controls, then freeze membership and hashes. Do not begin Phase 9B automatically."
    }
    report="# Phase 8R6A: external Track B qualification\n\n"+"\n\n".join(f"## {i+1}. {h}\n\n{body.get(h,'See the corresponding machine-readable CSV for complete audited values and explicit not-computable reasons.')}" for i,h in enumerate(sections))+"\n\nNo suitable sealed Track B benchmark was established from the audited public sources. This does not mean such a benchmark cannot be constructed; it means the currently audited public data are insufficient under the predefined criteria.\n"
    (out/"phase8r6a_report.md").write_text(report,encoding="utf-8")
    return result

if __name__=="__main__": print(json.dumps(run(),indent=2))
