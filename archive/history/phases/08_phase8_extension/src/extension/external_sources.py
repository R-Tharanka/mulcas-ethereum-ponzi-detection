"""Source-specific readers that preserve raw labels and missing values."""
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

from .external_normalization import normalize_address, normalize_hex, normalize_label

ADDRESS_IN_TEXT = re.compile(r"0x[0-9a-fA-F]{40}")


def _frame(source: str, addresses, labels, provenance: str) -> pd.DataFrame:
    rows = []
    for i, (raw_address, raw_label) in enumerate(zip(addresses, labels)):
        rows.append({"source_name": source, "source_record_id": str(i),
                     "address_original": str(raw_address), "address": normalize_address(raw_address),
                     "label": normalize_label(raw_label), "label_text_original": str(raw_label),
                     "label_provenance_class": provenance, "runtime_bytecode": None,
                     "creation_code": None, "opcode": None, "creator": None,
                     "created_block": None, "timestamp": None})
    return pd.DataFrame(rows)


def load_unica(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, names=["address", "name"])
    out = _frame("blockchain_unica", raw.address, [1] * len(raw), "A")
    out["source_record_id"] = raw["name"].astype(str)
    return out


def load_ctrf(path: Path, opcode_dir: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t")
    out = _frame("ctrf", raw.addr, raw.IS_PONZI, "B")
    for i, address in enumerate(out.address):
        candidate = opcode_dir / f"{address}.txt"
        if not candidate.exists():
            continue
        lines = candidate.read_text(encoding="utf-8", errors="ignore").splitlines()
        out.at[i, "runtime_bytecode"] = normalize_hex(lines[0]) if lines else None
        ops = [m.group(1) for line in lines[1:] if (m := re.match(r"^\s*\d+:\s+([A-Za-z0-9_]+)", line))]
        out.at[i, "opcode"] = " ".join(ops) or None
    return out


def load_ponzi_ml(path: Path, bytecode_root: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    out = _frame("ponzi_ml", raw.address, raw.target, "B")
    files = {normalize_address(p.name.split(".")[0]): p for p in bytecode_root.rglob("*.contract.bin")}
    for i, address in enumerate(out.address):
        p = files.get(address)
        if p:
            out.at[i, "runtime_bytecode"] = normalize_hex(p.read_text(encoding="utf-8", errors="ignore"))
    return out


def load_positive_list(source: str, path: Path, provenance: str) -> pd.DataFrame:
    if path.is_dir():
        addresses = [p.stem for p in sorted(path.glob("*.sol"))]
    else:
        addresses = ADDRESS_IN_TEXT.findall(path.read_text(encoding="utf-8", errors="ignore"))
    return _frame(source, addresses, [1] * len(addresses), provenance)
