"""Exact identity overlap calculations; no fuzzy similarity or model use."""
from __future__ import annotations

from itertools import combinations
import pandas as pd

from .external_normalization import digest, normalize_address, normalize_hex, normalize_opcode


def add_hashes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["address"] = out["address"].map(normalize_address)
    out["runtime_hash"] = out.get("runtime_bytecode", pd.Series(index=out.index, dtype=object)).map(lambda x: digest(normalize_hex(x)))
    out["opcode_hash"] = out.get("opcode", pd.Series(index=out.index, dtype=object)).map(lambda x: digest(normalize_opcode(x)))
    out["creation_hash"] = out.get("creation_code", pd.Series(index=out.index, dtype=object)).map(lambda x: digest(normalize_hex(x)))
    return out


def identity_sets(df: pd.DataFrame) -> dict[str, set]:
    return {c: set(df[c].dropna()) for c in ("address", "runtime_hash", "opcode_hash", "creation_hash")}


def address_summary(source: str, df: pd.DataFrame, internal: pd.DataFrame) -> dict:
    ext = set(df.address.dropna()); cur = set(internal.address.dropna())
    ip = set(internal.loc[internal.label == 1, "address"]); ib = set(internal.loc[internal.label == 0, "address"])
    ep = set(df.loc[df.label == 1, "address"].dropna()); eb = set(df.loc[df.label == 0, "address"].dropna())
    return {"source_name": source, "candidate_total_rows": len(df), "candidate_unique_addresses": len(ext),
            "positive_addresses": len(ep), "benign_addresses": len(eb),
            "unknown_label_addresses": len(set(df.loc[df.label.isna(), "address"].dropna())),
            "exact_address_overlap": len(ext & cur), "overlap_pct_candidate": round(100*len(ext&cur)/len(ext), 6) if ext else 0,
            "overlap_pct_current": round(100*len(ext&cur)/len(cur), 6),
            "positive_address_overlap": len(ep & ip), "benign_address_overlap": len(eb & ib),
            "external_ponzi_internal_benign": len(ep & ib), "external_benign_internal_ponzi": len(eb & ip),
            "genuinely_new_unique_addresses": len(ext-cur), "duplicate_address_rows": len(df)-len(ext),
            "conflicting_internal_duplicate_labels": int(df.groupby("address", dropna=True).label.nunique().gt(1).sum()),
            "deduplicated_unambiguous_addresses": int(df.dropna(subset=["address"]).groupby("address").label.nunique(dropna=False).le(1).sum())}


def pairwise_overlap(frames: dict[str, pd.DataFrame]) -> list[dict]:
    rows=[]
    for a,b in combinations(sorted(frames),2):
        sa,sb=identity_sets(frames[a]),identity_sets(frames[b])
        rows.append({"source_a":a,"source_b":b,**{f"{k}_overlap":len(sa[k]&sb[k]) for k in sa}})
    return rows
