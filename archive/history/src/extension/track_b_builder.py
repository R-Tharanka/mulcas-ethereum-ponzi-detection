"""Strict, model-free Track B construction from verified enrichment records."""
from __future__ import annotations

import hashlib, json
from pathlib import Path
import pandas as pd

from src.features.operands import disassemble_opcode_names, disassemble_operands
from .external_normalization import digest, normalize_address, normalize_hex, normalize_opcode
from .external_overlap import add_hashes, identity_sets


def enrich_static_identities(rows: pd.DataFrame) -> pd.DataFrame:
    out=rows.copy()
    for col in ("runtime_bytecode","creation_code"):
        if col not in out: out[col]=None
    out["normalized_runtime"]=out.runtime_bytecode.map(normalize_hex)
    out["runtime_hash"]=out.normalized_runtime.map(digest)
    out["normalized_creation"]=out.creation_code.map(normalize_hex)
    out["creation_hash"]=out.normalized_creation.map(digest)
    opcodes=[];counts=[];malformed=[];unsupported=[]
    for value in out.runtime_bytecode:
        parsed=disassemble_operands(value); names=disassemble_opcode_names(value)
        opcodes.append(" ".join(names) if names else None);counts.append(len(names))
        malformed.append(parsed.malformed_truncated_push_count);unsupported.append(parsed.unsupported_opcode_count)
    out["opcode"]=opcodes;out["opcode_count"]=counts;out["malformed_truncated_count"]=malformed;out["unsupported_opcode_count"]=unsupported
    out["disassembly_success"]=out.opcode.notna();out["opcode_hash"]=out.opcode.map(lambda x:digest(normalize_opcode(x)))
    return out


def internal_reference(path: Path) -> tuple[pd.DataFrame,dict[str,set]]:
    raw=pd.read_csv(path); frame=pd.DataFrame({"address":raw.address.map(normalize_address),"label":raw.label,
      "runtime_bytecode":raw.contractCode,"opcode":raw.opcode,"creation_code":raw.creationCode,
      "creator":raw.creator.map(normalize_address)})
    frame=add_hashes(frame);refs=identity_sets(frame);refs["creator"]=set(frame.creator.dropna());return frame,refs


def classify(rows: pd.DataFrame, refs: dict[str,set], end_block: int) -> pd.DataFrame:
    out=rows.copy();out["address_new"]=~out.address.isin(refs["address"])
    out["runtime_new"]=out.runtime_hash.notna() & ~out.runtime_hash.isin(refs["runtime_hash"])
    out["opcode_new"]=out.opcode_hash.notna() & ~out.opcode_hash.isin(refs["opcode_hash"])
    out["creation_new"]=out.creation_hash.notna() & ~out.creation_hash.isin(refs["creation_hash"])
    out["temporal_status"]=out.created_block.map(lambda x:"LATER" if pd.notna(x) and int(x)>end_block else ("PRE_EXISTING_PERIOD" if pd.notna(x) else "UNKNOWN"))
    def status(r):
        if not r.address_new:return "ADDRESS_OVERLAP"
        if pd.isna(r.runtime_hash):return "STRICT_CODE_STATUS_UNKNOWN"
        if not r.runtime_new:return "NEW_ADDRESS_DUP_RUNTIME"
        if pd.isna(r.opcode_hash):return "STRICT_CODE_STATUS_UNKNOWN"
        if not r.opcode_new:return "NEW_ADDRESS_DUP_OPCODE"
        if pd.notna(r.creation_hash) and not r.creation_new:return "NEW_ADDRESS_DUP_CREATION"
        return "STRICT_CODE_NEW" if pd.notna(r.creation_hash) else "PROVISIONALLY_STRICT_INDEPENDENT"
    out["independence_status"]=out.apply(status,axis=1)
    return out


def subset_counts(rows: pd.DataFrame) -> list[dict]:
    masks={"B0":rows.label.notna(),"B1":rows.address_new,"B2":rows.address_new&rows.runtime_new,
      "B3":rows.address_new&rows.runtime_new&rows.opcode_new,
      "B4":rows.independence_status.isin(["STRICT_CODE_NEW","PROVISIONALLY_STRICT_INDEPENDENT"]),
      "B5":rows.independence_status.isin(["STRICT_CODE_NEW","PROVISIONALLY_STRICT_INDEPENDENT"])&rows.temporal_status.eq("LATER")}
    eligible=rows.provenance.map(lambda x:bool(set(str(x).split("|"))&{"A","B"})) & ~rows.get("unresolved_conflict",False)
    masks["B6"]=masks["B5"]&eligible
    return [{"subset":name,"total":int(mask.sum()),"positive":int((mask&rows.label.eq(1)).sum()),"benign":int((mask&rows.label.eq(0)).sum())} for name,mask in masks.items()]


def primary_membership(rows: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    eligible_provenance=rows.provenance.map(lambda x:bool(set(str(x).split("|"))&{"A","B"}))
    rules={"known_label":rows.label.notna(),"eligible_provenance":eligible_provenance,"new_address":rows.address_new,
      "runtime_new":rows.runtime_new,"opcode_new":rows.opcode_new,"verified_later":rows.temporal_status.eq("LATER"),
      "no_unresolved_conflict":~rows.unresolved_conflict,"identity_fields":rows.runtime_hash.notna()&rows.opcode_hash.notna(),
      "verified_metadata":rows.receipt_address_verified.map(lambda x:str(x).lower()=="true")&rows.block_consistency_verified.map(lambda x:str(x).lower()=="true")&rows.timestamp_consistency_verified.map(lambda x:str(x).lower()=="true")}
    mask=pd.Series(True,index=rows.index)
    for value in rules.values():mask&=value
    selected=rows[mask].drop_duplicates("address").copy()
    selected["strictness"]=selected.creation_hash.map(lambda x:"STRICT" if pd.notna(x) else "PROVISIONAL")
    excluded=rows[~mask].copy()
    excluded["exclusion_reasons"]=["|".join(name for name,value in rules.items() if not bool(value.loc[i])) for i in excluded.index]
    return selected,excluded


def seal(selected: pd.DataFrame, metadata_path: Path, labels_path: Path, manifest_path: Path) -> dict:
    metadata_path.parent.mkdir(parents=True,exist_ok=True);labels_path.parent.mkdir(parents=True,exist_ok=True)
    public_columns=["address","sources","provenance","created_block","timestamp","creator","runtime_hash","opcode_hash","creation_hash","strictness"]
    metadata=selected[public_columns].sort_values("address"); labels=selected[["address","label"]].sort_values("address")
    metadata.to_csv(metadata_path,index=False);labels.to_csv(labels_path,index=False)
    file_hash=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    membership_hash=hashlib.sha256("\n".join(metadata.address).encode()).hexdigest()
    manifest={"row_count":len(selected),"positive_count":int(selected.label.eq(1).sum()),"benign_count":int(selected.label.eq(0).sum()),
      "membership_hash":membership_hash,"metadata_file_hash":file_hash(metadata_path),"sealed_label_file_hash":file_hash(labels_path),
      "temporal_range":{"minimum_block":int(selected.created_block.min()),"maximum_block":int(selected.created_block.max()),"minimum_timestamp":selected.timestamp.min(),"maximum_timestamp":selected.timestamp.max()},
      "address_unique":selected.address.is_unique,"runtime_unique_count":selected.runtime_hash.nunique(),"opcode_unique_count":selected.opcode_hash.nunique(),
      "provenance_summary":selected.provenance.value_counts().to_dict(),"source_composition":selected.sources.value_counts().to_dict(),
      "strict_count":int(selected.strictness.eq("STRICT").sum()),"provisional_count":int(selected.strictness.eq("PROVISIONAL").sum()),
      "models_evaluated":False,"labels_exposed_to_development":False}
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8");return manifest
