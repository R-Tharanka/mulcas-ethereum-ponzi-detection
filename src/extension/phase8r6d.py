"""Phase 8R6D external enrichment, Track B sealing, and Phase 9B protocol freeze.

This module is intentionally model-free. Provider credentials remain in process
memory and are never returned, logged, or written to an artifact.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd
import yaml

from src.utils.paths import REPO_ROOT
from .external_enrichment import candidate_rows
from .external_normalization import normalize_address
from .phase8r6b import conflict_audit
from .successor_migration import verify_manifest_with_successor_migrations
from .track_b_builder import internal_reference, seal


CONFIG = REPO_ROOT / "configs/extension/phase8r6d.yaml"
SUCCESS_PROVIDER = "authenticated_blockscout_and_ethereum_rpc"


class RowFailure(RuntimeError):
    def __init__(self, category: str, message: str, fingerprint: dict[str, Any] | None = None):
        super().__init__(message)
        self.category = category
        self.fingerprint = fingerprint


@dataclass
class ProviderMetrics:
    requests: int = 0
    retries: int = 0
    latencies: list[float] = field(default_factory=list)
    http_statuses: Counter[str] = field(default_factory=Counter)

    def record(self, elapsed: float, status: int | str) -> None:
        self.requests += 1
        self.latencies.append(elapsed)
        self.http_statuses[str(status)] += 1


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _csv_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def verify_frozen_inputs(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = REPO_ROOT / config["frozen_manifest"]
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    verified = len(verify_manifest_with_successor_migrations(manifest_path))

    registry = manifest["external_source_registry"]
    registry_path = REPO_ROOT / registry["manifest_path"]
    if sha256_file(registry_path) != registry["manifest_sha256"]:
        raise RuntimeError("Frozen external source registry changed")
    rows = pd.read_csv(registry_path, keep_default_na=False)
    downloaded = rows[rows.status.eq("downloaded")]
    for row in downloaded.itertuples(index=False):
        path = REPO_ROOT / row.original_filename
        if not path.is_file() or sha256_file(path) != row.sha256:
            raise RuntimeError(f"Frozen external raw file mismatch: {row.original_filename}")
    if len(downloaded) != int(registry["downloaded_file_count_verified"]):
        raise RuntimeError("Frozen external source count mismatch")
    return {"direct_anchor_count": verified, "external_raw_file_count": len(downloaded)}


def _request_json(url: str, payload: Any | None, timeout: float, metrics: ProviderMetrics) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json", "User-Agent": "mulcas-phase8r6d/1.0"})
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
            metrics.record(time.perf_counter() - started, response.status)
            return value
    except HTTPError as exc:
        metrics.record(time.perf_counter() - started, exc.code)
        raise
    except (URLError, TimeoutError):
        metrics.record(time.perf_counter() - started, "TRANSPORT_ERROR")
        raise


def _shape(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "null", "empty": True}
    if isinstance(value, dict):
        return {"type": "object", "empty": not value, "keys": sorted(value)}
    if isinstance(value, list):
        return {"type": "list", "empty": not value, "item_types": sorted({type(item).__name__ for item in value})}
    if isinstance(value, str):
        return {"type": "string", "empty": not bool(value.strip())}
    return {"type": type(value).__name__, "empty": False}


def schema_fingerprint(value: Any, *, http_status: int = 200, target: str | None = None) -> dict[str, Any]:
    fields = sorted(value) if isinstance(value, dict) else []
    fingerprint: dict[str, Any] = {
        "http_status": http_status,
        "top_level_type": type(value).__name__,
        "top_level_fields": fields,
        "field_shapes": {key: _shape(item) for key, item in value.items()} if isinstance(value, dict) else {},
        "expected_fields_present": {
            key: key in fields
            for key in ("hash", "is_contract", "creation_status", "creator_address_hash", "creation_transaction_hash", "creation_tx_hash", "creation_block_number")
        },
    }
    if isinstance(value, dict):
        fingerprint["is_contract"] = value.get("is_contract") if isinstance(value.get("is_contract"), bool) else None
        fingerprint["creation_status"] = value.get("creation_status") if isinstance(value.get("creation_status"), str) else None
        fingerprint["returned_identity_matches_target"] = normalize_address(value.get("hash")) == normalize_address(target) if target else None
        fingerprint["creator_present"] = bool(value.get("creator_address_hash"))
        fingerprint["creation_transaction_present"] = bool(value.get("creation_transaction_hash") or value.get("creation_tx_hash"))
    canonical = json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fingerprint["fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
    return fingerprint


def _creation_schema(endpoint: str) -> str:
    path = urlparse(endpoint).path.rstrip("/")
    if path.endswith("/api/v2"):
        return "blockscout_v2"
    if path.endswith("/api"):
        return "blockscout_etherscan_compatible"
    raise RowFailure("PROVIDER_SCHEMA_UNSUPPORTED", "Creation-provider URL has an unsupported API path")


def creation_lookup(address: str, endpoint: str, key: str, timeout: float, metrics: ProviderMetrics) -> tuple[dict[str, Any], dict[str, Any]]:
    target = normalize_address(address)
    schema = _creation_schema(endpoint)
    if schema == "blockscout_v2":
        url = endpoint.rstrip("/") + "/addresses/" + target
        if key:
            url += "?" + urlencode({"apikey": key})
        value = _request_json(url, None, timeout, metrics)
        fingerprint = schema_fingerprint(value, target=target)
        if not isinstance(value, dict) or "hash" not in value:
            raise RowFailure("PROVIDER_SCHEMA_UNSUPPORTED", "Blockscout address response is not an identifiable object", fingerprint)
        returned = normalize_address(value.get("hash"))
        if returned != target:
            raise RowFailure("IDENTITY_MISMATCH", "Creation-provider identity does not match target", fingerprint)
        tx_hash = value.get("creation_transaction_hash") or value.get("creation_tx_hash")
        creator = normalize_address(value.get("creator_address_hash"))
        if not tx_hash and not creator:
            raise RowFailure("NOT_FOUND", "Creation metadata is unavailable for this address", fingerprint)
        if not tx_hash or not creator:
            raise RowFailure("PROVIDER_SCHEMA_UNSUPPORTED", "Creation metadata is incomplete and cannot be interpreted safely", fingerprint)
        return {
            "address": returned,
            "creator": creator,
            "tx_hash": str(tx_hash).lower(),
            "block_number": value.get("creation_block_number"),
        }, fingerprint

    query = {"module": "contract", "action": "getcontractcreation", "contractaddresses": target, "apikey": key}
    value = _request_json(endpoint + "?" + urlencode(query), None, timeout, metrics)
    fingerprint = schema_fingerprint(value, target=target)
    if not isinstance(value, dict) or str(value.get("status")) != "1" or not isinstance(value.get("result"), list) or not value["result"]:
        raise RowFailure("PROVIDER_SCHEMA_UNSUPPORTED", "Compatible creation response lacks a successful result list", fingerprint)
    row = value["result"][0]
    if not isinstance(row, dict) or not {"contractAddress", "contractCreator", "txHash"}.issubset(row):
        raise RowFailure("PROVIDER_SCHEMA_UNSUPPORTED", "Compatible creation row lacks required identity fields", fingerprint)
    returned = normalize_address(row.get("contractAddress"))
    if returned != target:
        raise RowFailure("IDENTITY_MISMATCH", "Creation-provider identity does not match target", fingerprint)
    return {"address": returned, "creator": normalize_address(row["contractCreator"]), "tx_hash": str(row["txHash"]).lower(), "block_number": row.get("blockNumber")}, fingerprint


def rpc_call(endpoint: str, method: str, params: list[Any], timeout: float, metrics: ProviderMetrics) -> Any:
    value = _request_json(endpoint, {"jsonrpc": "2.0", "id": metrics.requests + 1, "method": method, "params": params}, timeout, metrics)
    if not isinstance(value, dict) or value.get("error") or "result" not in value:
        raise RowFailure("RPC_FAILURE", f"Invalid JSON-RPC response for {method}")
    return value["result"]


def enrich_one(address: str, rpc_endpoint: str, creation_endpoint: str, key: str, timeout: float, metrics: ProviderMetrics) -> tuple[dict[str, Any], dict[str, Any]]:
    from .track_b_builder import enrich_static_identities

    target = normalize_address(address)
    creation, fingerprint = creation_lookup(target, creation_endpoint, key, timeout, metrics)
    tx = rpc_call(rpc_endpoint, "eth_getTransactionByHash", [creation["tx_hash"]], timeout, metrics)
    receipt = rpc_call(rpc_endpoint, "eth_getTransactionReceipt", [creation["tx_hash"]], timeout, metrics)
    if not isinstance(tx, dict) or not isinstance(receipt, dict):
        raise RowFailure("RPC_FAILURE", "Creation transaction or receipt is unavailable", fingerprint)
    if normalize_address(receipt.get("contractAddress")) != target:
        raise RowFailure("IDENTITY_MISMATCH", "Receipt contract address does not match target", fingerprint)
    try:
        tx_block = int(tx["blockNumber"], 16)
        receipt_block = int(receipt["blockNumber"], 16)
    except (KeyError, TypeError, ValueError) as exc:
        raise RowFailure("RPC_FAILURE", "Creation block number is unavailable", fingerprint) from exc
    if tx_block != receipt_block or (creation.get("block_number") not in (None, "") and int(creation["block_number"]) != tx_block):
        raise RowFailure("IDENTITY_MISMATCH", "Creation block numbers do not reconcile", fingerprint)
    block = rpc_call(rpc_endpoint, "eth_getBlockByNumber", [hex(tx_block), False], timeout, metrics)
    runtime = rpc_call(rpc_endpoint, "eth_getCode", [target, "latest"], timeout, metrics)
    historical = rpc_call(rpc_endpoint, "eth_getCode", [target, hex(tx_block)], timeout, metrics)
    if not isinstance(block, dict) or not block.get("timestamp"):
        raise RowFailure("RPC_FAILURE", "Creation block timestamp is unavailable", fingerprint)
    if runtime in (None, "0x", "0x0"):
        raise RowFailure("INVALID_CONTRACT", "Address returned empty runtime bytecode", fingerprint)
    static = enrich_static_identities(pd.DataFrame([{"runtime_bytecode": runtime, "creation_code": tx.get("input")}])).iloc[0]
    if not bool(static.disassembly_success):
        raise RowFailure("INVALID_CONTRACT", "Deterministic opcode derivation failed", fingerprint)
    timestamp_unix = int(block["timestamp"], 16)
    return {
        "address": target,
        "creator": creation["creator"],
        "creation_transaction_hash": creation["tx_hash"],
        "created_block": tx_block,
        "timestamp_unix": timestamp_unix,
        "timestamp": datetime.fromtimestamp(timestamp_unix, tz=timezone.utc).isoformat(),
        "runtime_bytecode": runtime,
        "opcode": static.opcode,
        "creation_code": tx.get("input"),
        "runtime_hash": static.runtime_hash,
        "opcode_hash": static.opcode_hash,
        "creation_hash": static.creation_hash,
        "historical_query_supported": historical is not None,
        "receipt_address_verified": True,
        "block_consistency_verified": True,
        "timestamp_consistency_verified": True,
    }, fingerprint


def failure_category(exc: BaseException) -> str:
    if isinstance(exc, RowFailure):
        return exc.category
    if isinstance(exc, HTTPError):
        if exc.code in (401, 403):
            return "AUTH_FAILURE"
        if exc.code == 429:
            return "RATE_LIMITED"
        if exc.code == 404:
            return "NOT_FOUND"
    return "RPC_FAILURE"


def should_stop_systemic(history: list[str], category: str, *, consecutive: int = 5, window_size: int = 50, fraction: float = 0.10) -> bool:
    if category not in {"PROVIDER_SCHEMA_UNSUPPORTED", "IDENTITY_MISMATCH"}:
        return False
    trailing = 0
    for value in reversed(history):
        if value != category:
            break
        trailing += 1
    if trailing >= consecutive:
        return True
    recent = history[-window_size:]
    return len(recent) >= window_size and recent.count(category) / len(recent) >= fraction


def _load_parent_state(parent: Path, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = pd.read_csv(parent / "enriched_contracts.csv", keep_default_na=False)
    enriched["address"] = enriched.address.map(normalize_address)
    enriched = enriched[enriched.address.notna()].drop_duplicates("address", keep="last")
    for name in ("created_block", "timestamp_unix"):
        if name in enriched:
            enriched[name] = pd.to_numeric(enriched[name], errors="coerce")
    for name in ("label", "sources", "provenance"):
        enriched = enriched.drop(columns=name, errors="ignore")
    enriched = enriched.merge(candidates[["address", "label", "sources", "provenance"]], on="address", how="inner", validate="one_to_one")
    enriched["status"] = "SUCCESS"

    parent_status = pd.read_csv(parent / "enrichment_status.csv", keep_default_na=False)
    parent_status["address"] = parent_status.address.map(normalize_address)
    indexed = parent_status.drop_duplicates("address", keep="last").set_index("address")
    records = []
    for row in candidates.itertuples(index=False):
        if row.address in indexed.index:
            old = indexed.loc[row.address]
            records.append({"address": row.address, "label": row.label, "sources": row.sources, "provenance": row.provenance, "status": str(old.status).upper(), "attempts": int(float(old.get("attempts", 0) or 0)), "error_type": old.get("error_type", ""), "failure_reason": old.get("failure_reason", ""), "schema_fingerprint_sha256": old.get("schema_fingerprint_sha256", ""), "sanitized_identity_metadata_json": old.get("sanitized_identity_metadata_json", ""), "requests": int(float(old.get("requests", 0) or 0)), "retrieval_provider": old.get("retrieval_provider", ""), "retrieval_timestamp": old.get("retrieval_timestamp", "")})
        else:
            records.append({"address": row.address, "label": row.label, "sources": row.sources, "provenance": row.provenance, "status": "NOT_ATTEMPTED", "attempts": 0, "error_type": "", "failure_reason": "", "schema_fingerprint_sha256": "", "sanitized_identity_metadata_json": "", "requests": 0, "retrieval_provider": "", "retrieval_timestamp": ""})
    status = pd.DataFrame(records)
    status.loc[status.address.isin(set(enriched.address)), "status"] = "SUCCESS"
    return enriched, status


def _checkpoint(out: Path, candidates: pd.DataFrame, enriched: pd.DataFrame, statuses: pd.DataFrame) -> None:
    _csv_write(enriched.drop_duplicates("address", keep="last").sort_values("address"), out / "enriched_contracts_final.csv")
    indexed = statuses.drop_duplicates("address", keep="last").set_index("address")
    rows = []
    for candidate in candidates.itertuples(index=False):
        row = indexed.loc[candidate.address].to_dict() if candidate.address in indexed.index else {}
        rows.append({"address": candidate.address, "label": candidate.label, "sources": candidate.sources, "provenance": candidate.provenance, "status": row.get("status", "NOT_ATTEMPTED"), "attempts": row.get("attempts", 0), "error_type": row.get("error_type", ""), "failure_reason": row.get("failure_reason", ""), "schema_fingerprint_sha256": row.get("schema_fingerprint_sha256", ""), "sanitized_identity_metadata_json": row.get("sanitized_identity_metadata_json", ""), "requests": row.get("requests", 0), "retrieval_provider": row.get("retrieval_provider", ""), "retrieval_timestamp": row.get("retrieval_timestamp", "")})
    _csv_write(pd.DataFrame(rows), out / "enrichment_final_status.csv")


def _rating(positive: int, benign: int) -> str:
    if positive >= 100 and benign > 0:
        return "STRONG"
    if positive >= 30 and benign > 0:
        return "MODERATE"
    return "WEAK" if positive else "NONE"


def _count(candidates: pd.DataFrame, addresses: set[str], stage: str) -> dict[str, Any]:
    rows = candidates[candidates.address.isin(addresses)]
    return {"stage": stage, "total": len(rows), "positive": int(rows.label.eq(1).sum()), "benign": int(rows.label.eq(0).sum())}


def _matches(internal: pd.DataFrame, identity: str) -> dict[str, str]:
    available = internal.dropna(subset=[identity])
    return {
        value: "|".join(f"{index}:{row.address}:{int(row.label)}" for index, row in group.sort_values("address").iterrows())
        for value, group in available.groupby(identity)
    }


def _bool_series(frame: pd.DataFrame, name: str, default: bool = False) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index)
    return frame[name].astype(str).str.lower().eq("true")


def finalize_track_b(enriched: pd.DataFrame, statuses: pd.DataFrame, candidates: pd.DataFrame, config: dict[str, Any], out: Path) -> dict[str, Any]:
    rows = enriched.drop_duplicates("address", keep="last").copy()
    for name in ("created_block", "timestamp", "creator", "runtime_bytecode", "opcode", "creation_code", "creation_transaction_hash", "runtime_hash", "opcode_hash", "creation_hash"):
        if name not in rows:
            rows[name] = None
    rows["source"] = rows.sources
    rows["source_record_id"] = rows.sources.astype(str) + ":" + rows.address.astype(str)
    rows["label_provenance_class"] = rows.provenance

    conflict_rows, unresolved = conflict_audit(candidates)
    conflicts = pd.DataFrame(conflict_rows)
    if conflicts.empty:
        conflicts = pd.DataFrame(columns=["address", "classification", "sources", "labels", "provenance_classes", "publication_years", "verification_evidence", "candidate_in_scope", "resolution"])
    rows["unresolved_conflict"] = rows.address.isin(unresolved)
    rows["eligible_ab"] = rows.provenance.map(lambda value: bool(set(str(value).split("|")) & {"A", "B"}))
    verified = _bool_series(rows, "receipt_address_verified", True) & _bool_series(rows, "block_consistency_verified", True) & _bool_series(rows, "timestamp_consistency_verified", True)
    rows["MULCAS_READY"] = rows.eligible_ab & verified & rows[["address", "created_block", "timestamp", "creator", "runtime_bytecode", "opcode"]].notna().all(axis=1)
    rows["STATIC_RUNTIME_READY"] = rows.runtime_bytecode.notna() & rows.opcode.notna()
    rows["CREATION_CODE_READY"] = rows.creation_code.notna()

    internal, refs = internal_reference(REPO_ROOT / config["internal_dataset"])
    rows["address_new"] = ~rows.address.isin(refs["address"])
    rows["runtime_new"] = rows.runtime_hash.notna() & ~rows.runtime_hash.isin(refs["runtime_hash"])
    rows["opcode_new"] = rows.opcode_hash.notna() & ~rows.opcode_hash.isin(refs["opcode_hash"])
    rows["creation_new"] = rows.creation_hash.notna() & ~rows.creation_hash.isin(refs["creation_hash"])
    boundary = int(config["temporal_boundary"]["block"])
    rows["temporal_status"] = rows.created_block.map(lambda value: "LATER" if pd.notna(value) and int(value) > boundary else ("PRE_EXISTING_PERIOD" if pd.notna(value) else "UNKNOWN"))
    known_overlap = (~rows.address_new) | (rows.runtime_hash.notna() & ~rows.runtime_new) | (rows.opcode_hash.notna() & ~rows.opcode_new) | (rows.creation_hash.notna() & ~rows.creation_new)
    rows["independence_classification"] = "UNKNOWN"
    rows.loc[known_overlap, "independence_classification"] = "NOT_INDEPENDENT"
    strict_independent = rows.address_new & rows.runtime_new & rows.opcode_new & rows.creation_hash.notna() & rows.creation_new
    provisional_independent = rows.address_new & rows.runtime_new & rows.opcode_new & rows.creation_hash.isna()
    rows.loc[strict_independent, "independence_classification"] = "STRICT_INDEPENDENT"
    rows.loc[provisional_independent, "independence_classification"] = "PROVISIONAL_INDEPENDENT"
    rows["prior_model_use"] = False

    base = rows.eligible_ab & rows.address_new & rows.temporal_status.eq("LATER") & rows.runtime_hash.notna() & rows.opcode_hash.notna() & rows.runtime_new & rows.opcode_new & ~rows.unresolved_conflict & rows.MULCAS_READY & ~rows.prior_model_use
    strict = rows[base & rows.creation_hash.notna() & rows.creation_new].copy()
    provisional = rows[base & rows.creation_hash.isna()].copy()
    strict["strictness"] = "STRICT"
    provisional["strictness"] = "PROVISIONAL"
    selected = pd.concat([strict, provisional], ignore_index=True).drop_duplicates("address").sort_values("address")

    aliases = {
        "created_block": "createdBlockNumber",
        "creation_transaction_hash": "createdTransactionHash",
        "runtime_bytecode": "contractCode",
        "creation_code": "creationCode",
        "runtime_hash": "runtime_sha256",
        "opcode_hash": "opcode_sha256",
        "creation_hash": "creation_code_sha256",
    }
    for source, target in aliases.items():
        rows[target] = rows[source]
    rows["retrieval_provider"] = SUCCESS_PROVIDER
    _csv_write(rows.sort_values("address"), out / "enriched_contracts_final.csv")

    temporal_columns = ["address", "label", "provenance", "created_block", "timestamp", "temporal_status"]
    _csv_write(rows[temporal_columns].sort_values("address"), out / "temporal_classification.csv")
    runtime_audit = rows[["address", "label", "runtime_hash", "runtime_new", "independence_classification"]].copy()
    runtime_audit["internal_matches"] = runtime_audit.runtime_hash.map(_matches(internal, "runtime_hash")).fillna("")
    opcode_audit = rows[["address", "label", "opcode_hash", "opcode_new", "independence_classification"]].copy()
    opcode_audit["internal_matches"] = opcode_audit.opcode_hash.map(_matches(internal, "opcode_hash")).fillna("")
    creation_audit = rows[["address", "label", "creation_hash", "creation_new", "independence_classification"]].copy()
    creation_audit["creation_overlap_status"] = creation_audit.apply(lambda row: "CREATION_OVERLAP_UNKNOWN" if pd.isna(row.creation_hash) else ("CREATION_CODE_NEW" if row.creation_new else "CREATION_CODE_OVERLAP"), axis=1)
    creation_audit["internal_matches"] = creation_audit.creation_hash.map(_matches(internal, "creation_hash")).fillna("")
    _csv_write(runtime_audit.sort_values("address"), out / "runtime_overlap.csv")
    _csv_write(opcode_audit.sort_values("address"), out / "opcode_overlap.csv")
    _csv_write(creation_audit.sort_values("address"), out / "creation_overlap.csv")
    independence_columns = ["address", "label", "provenance", "address_new", "runtime_new", "opcode_new", "creation_new", "temporal_status", "unresolved_conflict", "independence_classification"]
    _csv_write(rows[independence_columns].sort_values("address"), out / "independence_classification.csv")

    provenance = rows[["address", "label", "sources", "provenance", "eligible_ab", "unresolved_conflict", "prior_model_use"]].copy()
    provenance["decision"] = provenance.apply(lambda row: "EXCLUDE_UNRESOLVED_CONFLICT" if row.unresolved_conflict else ("ELIGIBLE_A_B" if row.eligible_ab else "EXCLUDE_NON_A_B"), axis=1)
    _csv_write(provenance.sort_values("address"), out / "provenance_filter.csv")
    conflicts["successfully_enriched"] = conflicts.address.map(normalize_address).isin(set(rows.address)) if not conflicts.empty else False
    _csv_write(conflicts.sort_values("address"), out / "label_conflicts.csv")
    identity_columns = ["address", "status", "failure_reason", "schema_fingerprint_sha256", "sanitized_identity_metadata_json"]
    for column in identity_columns:
        if column not in statuses:
            statuses[column] = ""
    _csv_write(statuses.loc[statuses.status.eq("IDENTITY_MISMATCH"), identity_columns].sort_values("address"), out / "identity_mismatch_audit.csv")

    funnel = [_count(candidates, set(candidates.address), "starting_candidates")]
    attempted = set(statuses.loc[~statuses.status.eq("NOT_ATTEMPTED"), "address"])
    funnel.append(_count(candidates, attempted, "attempted"))
    masks = [
        (pd.Series(True, index=rows.index), "successfully_enriched"),
        (rows.eligible_ab, "eligible_A_B_provenance"),
        (rows.eligible_ab & rows.address_new, "new_address"),
        (rows.eligible_ab & rows.address_new & rows.temporal_status.eq("LATER"), "later_deployment"),
        (rows.eligible_ab & rows.address_new & rows.temporal_status.eq("LATER") & rows.runtime_new, "runtime_new"),
        (rows.eligible_ab & rows.address_new & rows.temporal_status.eq("LATER") & rows.runtime_new & rows.opcode_new, "opcode_new"),
        (rows.eligible_ab & rows.address_new & rows.temporal_status.eq("LATER") & rows.runtime_new & rows.opcode_new & (rows.creation_new | rows.creation_hash.isna()), "creation_new_or_unknown"),
        (rows.eligible_ab & rows.address_new & rows.temporal_status.eq("LATER") & rows.runtime_new & rows.opcode_new & (rows.creation_new | rows.creation_hash.isna()) & ~rows.unresolved_conflict, "no_unresolved_conflict"),
        (base, "MULCAS_READY"),
    ]
    for mask, stage in masks:
        funnel.append(_count(candidates, set(rows.loc[mask, "address"]), stage))
    funnel.append(_count(candidates, set(strict.address), "TRACK_B_STRICT"))
    funnel.append(_count(candidates, set(provisional.address), "TRACK_B_PROVISIONAL"))
    _csv_write(pd.DataFrame(funnel), out / "track_b_filter_funnel.csv")

    benign = rows[rows.label.eq(0)][["address", "temporal_status", "runtime_new", "opcode_new", "creation_new", "sources", "provenance"]].copy()
    benign["quality"] = "BENIGN_CURATED"
    benign["sampling_limitation"] = "sampling_and_matching_are_under_specified"
    benign["unlabeled_scam_limitation"] = "cannot_be_excluded"
    _csv_write(benign.sort_values("address"), out / "benign_quality.csv")

    ambiguous_benign = set()
    if not conflicts.empty:
        ambiguous_benign = set(conflicts.loc[conflicts.classification.eq("CONFLICT") & conflicts.resolution.eq("resolved_to_A_or_B_over_class_D"), "address"].map(normalize_address))
    triage: list[dict[str, Any]] = []
    independent_later = rows[rows.temporal_status.eq("LATER") & rows.independence_classification.isin(["STRICT_INDEPENDENT", "PROVISIONAL_INDEPENDENT"])]
    for row in independent_later[independent_later.unresolved_conflict].itertuples():
        triage.append({"priority_group": "M1", "address": row.address, "label": row.label, "sources": row.sources, "reason": "later independent unresolved A/B conflict"})
    for row in independent_later[~independent_later.eligible_ab].itertuples():
        triage.append({"priority_group": "M2", "address": row.address, "label": row.label, "sources": row.sources, "reason": "later independent uncertain provenance"})
    for row in independent_later[independent_later.label.eq(0) & independent_later.address.isin(ambiguous_benign)].itertuples():
        triage.append({"priority_group": "M3", "address": row.address, "label": row.label, "sources": row.sources, "reason": "later independent benign label conflicts with weaker evidence"})
    failed_ab = statuses[~statuses.status.isin(["SUCCESS", "NOT_ATTEMPTED"]) & statuses.provenance.map(lambda value: bool(set(str(value).split("|")) & {"A", "B"}))]
    for row in failed_ab.itertuples():
        triage.append({"priority_group": "M4", "address": row.address, "label": row.label, "sources": row.sources, "reason": f"missing critical enrichment field: {row.status}"})
    old_triage = REPO_ROOT / "results/extension/phase8r6b/manual_review_triage.csv"
    if old_triage.exists():
        old = pd.read_csv(old_triage, keep_default_na=False)
        for row in old[old.priority_group.eq("M5")].itertuples():
            triage.append({"priority_group": "M5", "address": normalize_address(row.address), "label": "", "sources": getattr(row, "sources", "ponziguard_predictions"), "reason": "class-D prediction; verification only"})
    triage_frame = pd.DataFrame(triage, columns=["priority_group", "address", "label", "sources", "reason"]).drop_duplicates(["priority_group", "address"])
    _csv_write(triage_frame.sort_values(["priority_group", "address"]), out / "manual_review_triage.csv")
    critical_review = int(triage_frame.priority_group.isin(["M1", "M2", "M3"]).sum())
    if 0 < critical_review <= 30:
        _csv_write(triage_frame[triage_frame.priority_group.isin(["M1", "M2", "M3"])], out / "manual_review_packet.csv")

    all_unique = selected.drop_duplicates("address").sort_values("address")
    unique_code = all_unique.sort_values("address").drop_duplicates(["runtime_hash", "opcode_hash", "creation_hash"], keep="first")
    public_membership_columns = [
        "address", "sources", "provenance", "created_block", "timestamp", "creator",
        "runtime_hash", "opcode_hash", "creation_hash", "strictness",
    ]
    _csv_write(strict[public_membership_columns].sort_values("address"), out / "track_b_strict.csv")
    _csv_write(provisional.reindex(columns=public_membership_columns).sort_values("address"), out / "track_b_provisional.csv")
    _csv_write(all_unique[public_membership_columns], out / "track_b_all_unique_address.csv")
    _csv_write(unique_code[public_membership_columns], out / "track_b_unique_code_sensitivity.csv")
    duplicate_rows: list[dict[str, Any]] = []
    for identity in ("address", "runtime_hash", "opcode_hash", "creation_hash"):
        available = all_unique.dropna(subset=[identity])
        duplicate_rows.append({"record_type": "SUMMARY", "identity": identity, "identity_value": "", "group_size": "", "addresses": "", "available_rows": len(available), "unique_values": available[identity].nunique(), "duplicate_rows": len(available) - available[identity].nunique()})
        for value, group in available.groupby(identity):
            if len(group) > 1:
                duplicate_rows.append({"record_type": "CLONE_GROUP", "identity": identity, "identity_value": value, "group_size": len(group), "addresses": "|".join(sorted(group.address)), "available_rows": "", "unique_values": "", "duplicate_rows": ""})
    duplicate_rows.append({"record_type": "VIEW", "identity": "TRACK_B_ALL_UNIQUE_ADDRESS", "identity_value": "", "group_size": len(all_unique), "addresses": "", "available_rows": len(all_unique), "unique_values": len(all_unique), "duplicate_rows": 0})
    duplicate_rows.append({"record_type": "VIEW", "identity": "TRACK_B_UNIQUE_CODE_SENSITIVITY", "identity_value": "", "group_size": len(unique_code), "addresses": "", "available_rows": len(unique_code), "unique_values": len(unique_code), "duplicate_rows": len(all_unique) - len(unique_code)})
    _csv_write(pd.DataFrame(duplicate_rows), out / "track_b_duplicate_audit.csv")

    schema_rows = []
    fields = [
        ("address", rows.address.notna()), ("label", rows.label.notna()), ("eligible_A_B_label", rows.eligible_ab),
        ("createdBlockNumber", rows.created_block.notna()), ("timestamp", rows.timestamp.notna()), ("creator", rows.creator.notna()),
        ("contractCode", rows.runtime_bytecode.notna()), ("opcode", rows.opcode.notna()), ("creationCode", rows.creation_code.notna()),
        ("createdTransactionHash", rows.creation_transaction_hash.notna()), ("MULCAS_READY", rows.MULCAS_READY),
        ("STATIC_RUNTIME_READY", rows.STATIC_RUNTIME_READY), ("CREATION_CODE_READY", rows.CREATION_CODE_READY),
    ]
    for field, mask in fields:
        count = int(mask.sum())
        schema_rows.append({"field_or_status": field, "successful_enrichment_count": len(rows), "available_count": count, "available_percent": 100 * count / len(rows) if len(rows) else 0})
    _csv_write(pd.DataFrame(schema_rows), out / "schema_compatibility.csv")

    strict_positive, strict_benign = int(strict.label.eq(1).sum()), int(strict.label.eq(0).sum())
    provisional_positive, provisional_benign = int(provisional.label.eq(1).sum()), int(provisional.label.eq(0).sum())
    strict_rating = _rating(strict_positive, strict_benign)
    combined_rating = _rating(strict_positive + provisional_positive, strict_benign + provisional_benign)
    primary_basis = "STRICT" if strict_rating in {"STRONG", "MODERATE"} else "STRICT_PLUS_PROVISIONAL"
    primary_rating = strict_rating if primary_basis == "STRICT" else combined_rating
    scientifically_ready = primary_rating in {"STRONG", "MODERATE"} and critical_review == 0
    readiness = pd.DataFrame([
        {"subset": "STRICT", "rating": strict_rating, "rows": len(strict), "positive": strict_positive, "benign": strict_benign, "scientifically_ready": strict_rating in {"STRONG", "MODERATE"} and critical_review == 0},
        {"subset": "STRICT_PLUS_PROVISIONAL", "rating": combined_rating, "rows": len(all_unique), "positive": strict_positive + provisional_positive, "benign": strict_benign + provisional_benign, "scientifically_ready": scientifically_ready},
    ])
    _csv_write(readiness, out / "track_b_readiness.csv")
    sealed_manifest = None
    if scientifically_ready:
        seal_rows = strict if primary_basis == "STRICT" else all_unique
        sealed_manifest = seal(seal_rows, REPO_ROOT / "data/external/track_b/metadata.csv", REPO_ROOT / "data/external/track_b_sealed/labels.csv", REPO_ROOT / "data/external/track_b/membership_manifest.json")

    status_counts = statuses.status.value_counts().to_dict()
    temporal = rows.groupby(["temporal_status", "label"]).size().to_dict()
    return {
        "successful_enrichment_count": len(rows),
        "invalid_contract_count": int(status_counts.get("INVALID_CONTRACT", 0)),
        "identity_mismatch_count": int(status_counts.get("IDENTITY_MISMATCH", 0)),
        "schema_unsupported_count": int(status_counts.get("PROVIDER_SCHEMA_UNSUPPORTED", 0)),
        "not_found_count": int(status_counts.get("NOT_FOUND", 0)),
        "provider_failure_count": int(sum(status_counts.get(name, 0) for name in ("AUTH_FAILURE", "RATE_LIMITED", "RPC_FAILURE"))),
        "mulcas_ready_count": int(rows.MULCAS_READY.sum()),
        "static_runtime_ready_count": int(rows.STATIC_RUNTIME_READY.sum()),
        "creation_code_ready_count": int(rows.CREATION_CODE_READY.sum()),
        "later_positive_count": int(temporal.get(("LATER", 1), 0)),
        "later_benign_count": int(temporal.get(("LATER", 0), 0)),
        "pre_existing_positive_count": int(temporal.get(("PRE_EXISTING_PERIOD", 1), 0)),
        "pre_existing_benign_count": int(temporal.get(("PRE_EXISTING_PERIOD", 0), 0)),
        "unknown_positive_count": int(temporal.get(("UNKNOWN", 1), 0)),
        "unknown_benign_count": int(temporal.get(("UNKNOWN", 0), 0)),
        "runtime_new_positive_count": int((rows.runtime_new & rows.label.eq(1)).sum()),
        "runtime_new_benign_count": int((rows.runtime_new & rows.label.eq(0)).sum()),
        "opcode_new_positive_count": int((rows.opcode_new & rows.label.eq(1)).sum()),
        "opcode_new_benign_count": int((rows.opcode_new & rows.label.eq(0)).sum()),
        "creation_new_count": int(rows.creation_new.sum()),
        "creation_unknown_count": int(rows.creation_hash.isna().sum()),
        "strict_independent_positive_count": int((rows.independence_classification.eq("STRICT_INDEPENDENT") & rows.label.eq(1)).sum()),
        "strict_independent_benign_count": int((rows.independence_classification.eq("STRICT_INDEPENDENT") & rows.label.eq(0)).sum()),
        "provisional_independent_positive_count": int((rows.independence_classification.eq("PROVISIONAL_INDEPENDENT") & rows.label.eq(1)).sum()),
        "provisional_independent_benign_count": int((rows.independence_classification.eq("PROVISIONAL_INDEPENDENT") & rows.label.eq(0)).sum()),
        "track_b_strict_positive_count": strict_positive,
        "track_b_strict_benign_count": strict_benign,
        "track_b_provisional_positive_count": provisional_positive,
        "track_b_provisional_benign_count": provisional_benign,
        "strict_rating": strict_rating,
        "strict_plus_provisional_rating": combined_rating,
        "track_b_rating": primary_rating,
        "track_b_primary_basis": primary_basis,
        "track_b_established": scientifically_ready,
        "sealed_ready": scientifically_ready,
        "sealed_manifest": sealed_manifest,
        "critical_manual_review_count": critical_review,
        "manual_review_count": len(triage_frame),
        "unresolved_conflict_count": len(unresolved),
    }


def freeze_phase9b_protocol(config: dict[str, Any], track_b: dict[str, Any]) -> dict[str, Any]:
    phase9 = config["phase9b"]
    config_path = REPO_ROOT / phase9["config"]
    doc_path = REPO_ROOT / phase9["protocol_doc"]
    output_dir = REPO_ROOT / phase9["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    reference_fpr = 269 / 5430
    prevalence = 314 / 6498
    protocol = {
        "phase": "9B",
        "status": "frozen_protocol_only",
        "research_question": "To what extent can deployment-available historical state, code-family similarity, and calibrated cascade routing improve forward temporal Ponzi detection under a leakage-safe prospective protocol, and does the final frozen system generalize to a sealed external later-time benchmark?",
        "development_data": {
            "dataset": "Track A only",
            "periods": ["P2", "P3", "P4"],
            "expanding_predictions": ["P0+P1 -> P2", "P0-P2 -> P3", "P0-P3 -> P4"],
            "p5_status": "previously_observed_exploratory_only_not_for_selection",
            "track_b_labels_available_to_development": False,
        },
        "primary_metrics": ["pooled_prequential_pr_auc", "pooled_prequential_f1", "recall_at_reference_fpr_ceiling"],
        "secondary_metrics": ["balanced_accuracy", "roc_auc", "per_period_precision", "per_period_recall", "per_period_f1", "prevalence_standardized_precision"],
        "prevalence_standardization": {
            "reference_prevalence_numerator": 314,
            "reference_prevalence_denominator": 6498,
            "reference_prevalence": prevalence,
            "formula": "PPV(pi)=(TPR*pi)/(TPR*pi+FPR*(1-pi))",
            "sole_selection_criterion": False,
        },
        "frozen_mulcas_reference": {
            "pooled_forward_periods": ["P2", "P3", "P4"],
            "tn": 5161,
            "fp": 269,
            "fn": 84,
            "tp": 66,
            "fpr": reference_fpr,
            "fpr_ceiling": reference_fpr,
            "fpr_ceiling_basis": "frozen_mulcas_pooled_forward_fpr",
        },
        "uncertainty": {
            "method": "contract_level_paired_bootstrap_within_pooled_forward_predictions",
            "replicates": 2000,
            "seed": 42,
            "report": ["delta_f1_ci", "delta_pr_auc_ci", "delta_recall_ci_where_applicable"],
            "track_b_used_for_selection": False,
        },
        "advancement_rule": {
            "pooled_f1_minimum_delta": 0.0,
            "pooled_f1_relation": "strictly_greater_than_frozen_mulcas",
            "pr_auc_maximum_absolute_degradation": 0.01,
            "fpr_maximum": reference_fpr,
            "temporal_breadth": {
                "minimum_periods_with_nonnegative_f1_delta": 2,
                "maximum_allowed_single_period_f1_degradation": 0.02,
            },
            "paired_uncertainty_required": True,
        },
        "planned_interventions": {
            "phase9c": "deployment_available_creator_and_family_state",
            "phase9d": "cascade_calibration_and_safer_routing",
            "phase9e": "selector_and_semantic_static_features",
            "phase9f": "combine_only_independently_supported_components",
        },
        "prohibited_reopenings": ["arbitrary_model_zoo", "GIN", "Transformer", "creation_code_model", "unrestricted_threshold_search"],
        "track_b_final_evaluation": {
            "status": "sealed_available" if track_b["sealed_ready"] else "not_available",
            "primary_basis": track_b.get("track_b_primary_basis"),
            "labels_inaccessible_during": ["Phase 9C", "Phase 9D", "Phase 9E", "Phase 9F"],
            "one_shot_only_after_phase9f_system_freeze": True,
            "tuning_after_evaluation": False,
            "metrics": ["precision", "recall", "f1", "pr_auc", "balanced_accuracy", "confusion_matrix"],
            "strict_provisional_distinction_required": True,
            "inadequate_or_positive_only_rule": "do_not_report_full_precision_recall_f1_as_if_a_complete_benchmark",
        },
        "constraints": {
            "models_trained": False,
            "detectors_scored": False,
            "p5_evaluated": False,
            "phase9c_started": False,
            "track_b_performance_used": False,
        },
    }
    config_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    protocol_result = {
        "status": "frozen",
        "protocol_config": phase9["config"],
        "protocol_config_sha256": sha256_file(config_path),
        "track_b_status_at_freeze": protocol["track_b_final_evaluation"]["status"],
        "models_trained": False,
        "detectors_evaluated": False,
        "p5_evaluated": False,
        "track_b_predictions_loaded": False,
        "track_b_labels_loaded": False,
        "phase9c_started": False,
    }
    _json_write(output_dir / "phase9b_protocol.json", protocol_result)
    text = f"""# Phase 9B Prospective Evaluation Protocol

## Purpose

Phase 9B freezes the evaluation rules before Phase 9C model development. It does not train, calibrate, select, or score a detector.

## RQ7

{protocol['research_question']}

## Development Boundary

Phase 9 development uses Track A only. Forward predictions are `P0+P1 -> P2`, `P0-P2 -> P3`, and `P0-P3 -> P4`. P5 is previously observed and is excluded from model selection. Track B labels remain inaccessible throughout Phases 9C-9F.

## Metrics And Selection

Primary metrics are pooled prequential PR-AUC, pooled prequential F1, and recall at the frozen MulCas pooled FPR ceiling `{reference_fpr:.12f}`. Secondary metrics include balanced accuracy, ROC-AUC, per-period precision/recall/F1, and prevalence-standardized precision at `{prevalence:.12f}`.

A candidate advances only when pooled F1 is strictly higher, pooled PR-AUC degrades by no more than `0.01`, FPR does not exceed the frozen MulCas ceiling, at least two of three periods have nonnegative F1 change, no period loses more than `0.02` F1, and paired uncertainty is reported. Paired contract-level bootstrap uses 2,000 replicates and seed 42.

## Planned Interventions

Only mechanism-driven Phases 9C-9F are preregistered: deployment-available creator/family state, cascade calibration and safer routing, selector/semantic static features, and combination only of independently supported components. GIN, Transformer, creation-code model, unrestricted threshold search, and arbitrary model-zoo searches remain closed.

## External Evaluation

Track B status at freeze: **{protocol['track_b_final_evaluation']['status']}**. A final Phase 9F system must be frozen before one external evaluation can access sealed labels. No tuning is permitted afterward. Strict and provisional subsets must remain distinct.

## Integrity Statement

This protocol was frozen without loading Track B detector predictions or performance. No detector was trained or scored, and Phase 9C was not started.
"""
    doc_path.write_text(text, encoding="utf-8")
    (output_dir / "phase9b_report.md").write_text(text, encoding="utf-8")
    return protocol_result


def _secrets(environ: dict[str, str]) -> dict[str, str]:
    required = {"rpc": "ETH_RPC_URL", "creation": "BLOCKSCOUT_API_URL", "key": "BLOCKSCOUT_API_KEY"}
    values = {name: environ.get(env_name, "").strip() for name, env_name in required.items()}
    if not all(values.values()):
        raise RuntimeError("PROVIDER CONFIGURATION INVALID")
    _creation_schema(values["creation"])
    return values


def _load_state(out: Path, parent: Path, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if (out / "enriched_contracts_final.csv").exists() and (out / "enrichment_final_status.csv").exists():
        enriched = pd.read_csv(out / "enriched_contracts_final.csv", keep_default_na=False)
        statuses = pd.read_csv(out / "enrichment_final_status.csv", keep_default_na=False)
        enriched["address"] = enriched.address.map(normalize_address)
        statuses["address"] = statuses.address.map(normalize_address)
        return enriched.drop_duplicates("address", keep="last"), statuses.drop_duplicates("address", keep="last")
    return _load_parent_state(parent, candidates)


def pending_candidates(candidates: pd.DataFrame, statuses: pd.DataFrame) -> pd.DataFrame:
    terminal = {"SUCCESS", "INVALID_CONTRACT", "IDENTITY_MISMATCH", "NOT_FOUND", "PROVIDER_SCHEMA_UNSUPPORTED"}
    completed = set(statuses.loc[statuses.status.isin(terminal), "address"])
    return candidates[~candidates.address.isin(completed)].copy()


def run(config_path: Path = CONFIG, environ: dict[str, str] | None = None) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frozen = verify_frozen_inputs(config)
    out = REPO_ROOT / config["output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    track_a = REPO_ROOT / config["internal_dataset"]
    if sha256_file(track_a) != config["track_a"]["expected_sha256"] or len(pd.read_csv(track_a, usecols=["address"])) != int(config["track_a"]["expected_rows"]):
        raise RuntimeError("Track A integrity verification failed")
    candidates = candidate_rows(track_a, REPO_ROOT / config["ponzi_ml_source"], REPO_ROOT / config["unica_source"])
    if len(candidates) != 1490 or not candidates.address.is_unique:
        raise RuntimeError("Frozen candidate population mismatch")

    parent = REPO_ROOT / config["parent_provider_dir"]
    enriched, statuses = _load_state(out, parent, candidates)
    existing_result_path = out / "phase8r6d_results.json"
    if existing_result_path.exists():
        existing_result = json.loads(existing_result_path.read_text(encoding="utf-8"))
        unfinished = statuses.status.isin(["NOT_ATTEMPTED", "PROVIDER_SCHEMA_FAILURE"]).any()
        if existing_result.get("status") == "complete" and not unfinished:
            if existing_result.get("track_a_sha256") != config["track_a"]["expected_sha256"]:
                raise RuntimeError("Completed Phase 8R6D result has the wrong Track A anchor")
            return existing_result
    secrets = _secrets(dict(os.environ) if environ is None else environ)
    starting_successes = int(statuses.status.eq("SUCCESS").sum())
    todo = pending_candidates(candidates, statuses)
    parent_runtime = json.loads((parent / "full_enrichment_runtime.json").read_text(encoding="utf-8"))
    prior_total_minutes = float(parent_runtime["actual_total_minutes"])
    preflight_requests = int(json.loads((parent / "preflight_20.json").read_text(encoding="utf-8")).get("requests", 0))
    stability_requests = int(json.loads((parent / "stability_100.json").read_text(encoding="utf-8")).get("requests", 0))
    prior_request_total = preflight_requests + stability_requests + int(parent_runtime.get("requests_this_resume", 0))

    existing_audit = out / "schema_variant_audit.csv"
    audit_records = pd.read_csv(existing_audit, keep_default_na=False).to_dict("records") if existing_audit.exists() else []
    variant_counts: Counter[str] = Counter()
    variant_examples: dict[str, dict[str, Any]] = {}
    for record in audit_records:
        variant_counts[str(record["fingerprint_sha256"])] += int(record.get("count", 0) or 0)
        try:
            variant_examples[str(record["fingerprint_sha256"])] = json.loads(record["sanitized_fingerprint_json"])
        except (json.JSONDecodeError, TypeError):
            pass

    metrics = ProviderMetrics()
    history: list[str] = []
    transport_consecutive = 0
    processed = 0
    started = time.perf_counter()
    stop_category: str | None = None
    stop_reason = ""
    provider = config["provider"]
    timeout = float(provider["request_timeout_seconds"])
    checkpoint_interval = int(provider["checkpoint_interval"])
    max_attempts = int(provider["maximum_attempts"])

    for item in todo.itertuples(index=False):
        attempts = 0
        request_start = metrics.requests
        outcome: dict[str, Any] | None = None
        fingerprint: dict[str, Any] | None = None
        category = "RPC_FAILURE"
        error_type = ""
        failure_reason = ""
        while attempts < max_attempts:
            attempts += 1
            try:
                outcome, fingerprint = enrich_one(item.address, secrets["rpc"], secrets["creation"], secrets["key"], timeout, metrics)
                category = "SUCCESS"
                break
            except (HTTPError, URLError, TimeoutError, RowFailure, ValueError, KeyError, TypeError) as exc:
                category = failure_category(exc)
                error_type = type(exc).__name__
                failure_reason = str(exc) if isinstance(exc, RowFailure) else (f"HTTP {exc.code}" if isinstance(exc, HTTPError) else type(exc).__name__)
                fingerprint = exc.fingerprint if isinstance(exc, RowFailure) else None
                retryable = category in {"RATE_LIMITED", "RPC_FAILURE"}
                if retryable and attempts < max_attempts:
                    metrics.retries += 1
                    time.sleep(float(provider["backoff_seconds"]) * attempts)
                    continue
                break

        timestamp = datetime.now(timezone.utc).isoformat()
        if fingerprint:
            fingerprint_hash = fingerprint["fingerprint_sha256"]
            variant_counts[fingerprint_hash] += 1
            variant_examples[fingerprint_hash] = fingerprint
        if outcome is not None:
            outcome.update({"label": item.label, "sources": item.sources, "provenance": item.provenance, "status": "SUCCESS", "retrieval_provider": SUCCESS_PROVIDER, "retrieval_timestamp": timestamp})
            enriched = pd.concat([enriched, pd.DataFrame([outcome])], ignore_index=True).drop_duplicates("address", keep="last")

        identity_metadata = ""
        if category == "IDENTITY_MISMATCH" and fingerprint:
            identity_metadata = json.dumps({
                "returned_identity_matches_target": fingerprint.get("returned_identity_matches_target"),
                "creator_present": fingerprint.get("creator_present"),
                "creation_transaction_present": fingerprint.get("creation_transaction_present"),
            }, sort_keys=True, separators=(",", ":"))
        record = {"address": item.address, "label": item.label, "sources": item.sources, "provenance": item.provenance, "status": category, "attempts": attempts, "error_type": error_type, "failure_reason": failure_reason, "schema_fingerprint_sha256": fingerprint.get("fingerprint_sha256", "") if fingerprint else "", "sanitized_identity_metadata_json": identity_metadata, "requests": metrics.requests - request_start, "retrieval_provider": SUCCESS_PROVIDER, "retrieval_timestamp": timestamp}
        statuses = pd.concat([statuses, pd.DataFrame([record])], ignore_index=True).drop_duplicates("address", keep="last")
        history.append(category)
        processed += 1

        if category in {"RATE_LIMITED", "RPC_FAILURE"}:
            transport_consecutive += 1
        else:
            transport_consecutive = 0
        if category == "AUTH_FAILURE":
            stop_category, stop_reason = category, "Provider authentication failed"
        elif should_stop_systemic(history, category, consecutive=int(provider["systemic_consecutive_failures"]), window_size=int(provider["systemic_window_size"]), fraction=float(provider["systemic_window_fraction"])):
            stop_category, stop_reason = category, "Systemic provider schema or identity failure threshold reached"
        elif transport_consecutive >= int(provider["sustained_transport_failures"]):
            stop_category, stop_reason = category, "Sustained provider transport failure threshold reached"

        checkpoint_due = processed % checkpoint_interval == 0 or processed == len(todo) or stop_category is not None
        if checkpoint_due:
            _checkpoint(out, candidates, enriched, statuses)
            audit = [
                {"fingerprint_sha256": key, "count": count, "sanitized_fingerprint_json": json.dumps(variant_examples[key], sort_keys=True, separators=(",", ":"))}
                for key, count in sorted(variant_counts.items())
            ]
            _csv_write(pd.DataFrame(audit, columns=["fingerprint_sha256", "count", "sanitized_fingerprint_json"]), out / "schema_variant_audit.csv")
            elapsed_minutes = (time.perf_counter() - started) / 60
            remaining = len(todo) - processed
            seconds_per_address = elapsed_minutes * 60 / processed if processed else 0
            projected_total = prior_total_minutes + elapsed_minutes + remaining * seconds_per_address / 60
            progress = {"processed_phase8r6d": processed, "remaining_phase8r6d": remaining, "successful_total": len(enriched), "elapsed_phase8r6d_minutes": round(elapsed_minutes, 3), "projected_total_minutes": round(projected_total, 3), "requests": metrics.requests, "retries": metrics.retries}
            print(json.dumps(progress), flush=True)
            if processed >= checkpoint_interval and projected_total > float(provider["approved_total_minutes"]):
                stop_category = "RUNTIME_LIMIT_EXCEEDED"
                stop_reason = f"Projected total runtime {projected_total:.2f} minutes exceeds approved {float(provider['approved_total_minutes']):.2f} minutes"
        if stop_category is not None:
            break

    _checkpoint(out, candidates, enriched, statuses)
    elapsed_minutes = (time.perf_counter() - started) / 60
    cumulative_minutes = prior_total_minutes + elapsed_minutes
    remaining_unattempted = int(statuses.status.eq("NOT_ATTEMPTED").sum()) + int(statuses.status.eq("PROVIDER_SCHEMA_FAILURE").sum())
    complete = stop_category is None and remaining_unattempted == 0
    runtime = {
        "starting_successful_rows": starting_successes,
        "resumed_addresses": processed,
        "additional_runtime_minutes": elapsed_minutes,
        "cumulative_runtime_minutes": cumulative_minutes,
        "approved_total_minutes": float(provider["approved_total_minutes"]),
        "requests_phase8r6d": metrics.requests,
        "requests_cumulative": prior_request_total + metrics.requests,
        "retries_phase8r6d": metrics.retries,
        "http_status_distribution": dict(metrics.http_statuses),
        "average_request_latency_seconds": statistics.fmean(metrics.latencies) if metrics.latencies else None,
        "remaining_unattempted": remaining_unattempted,
        "stop_category": stop_category,
    }
    _json_write(out / "full_enrichment_runtime.json", runtime)

    if complete:
        counts = finalize_track_b(enriched, statuses, candidates, config, out)
        phase9b = freeze_phase9b_protocol(config, counts)
        if counts["track_b_established"]:
            decision = "A. TRACK B READY + PHASE 9B FROZEN"
            next_phase = "phase9c"
        elif 0 < counts["critical_manual_review_count"] <= 30:
            decision = "B. TRACK B NEEDS <=30 CRITICAL MANUAL REVIEWS + PHASE 9B FROZEN"
            next_phase = "bounded_manual_review"
        else:
            decision = "C. TRACK B INADEQUATE + PHASE 9B FROZEN"
            next_phase = "phase9c_without_track_b"
        status_name = "complete"
    else:
        status_counts = statuses.status.value_counts().to_dict()
        counts = {
            "successful_enrichment_count": int(status_counts.get("SUCCESS", 0)),
            "invalid_contract_count": int(status_counts.get("INVALID_CONTRACT", 0)),
            "identity_mismatch_count": int(status_counts.get("IDENTITY_MISMATCH", 0)),
            "schema_unsupported_count": int(status_counts.get("PROVIDER_SCHEMA_UNSUPPORTED", 0)),
            "not_found_count": int(status_counts.get("NOT_FOUND", 0)),
            "provider_failure_count": int(sum(status_counts.get(name, 0) for name in ("AUTH_FAILURE", "RATE_LIMITED", "RPC_FAILURE"))),
            "mulcas_ready_count": len(enriched), "static_runtime_ready_count": len(enriched), "creation_code_ready_count": int(enriched.creation_code.notna().sum()),
            "later_positive_count": 0, "later_benign_count": 0, "runtime_new_positive_count": 0, "runtime_new_benign_count": 0,
            "opcode_new_positive_count": 0, "opcode_new_benign_count": 0, "strict_independent_positive_count": 0, "strict_independent_benign_count": 0,
            "provisional_independent_positive_count": 0, "provisional_independent_benign_count": 0, "track_b_strict_positive_count": 0,
            "track_b_strict_benign_count": 0, "track_b_provisional_positive_count": 0, "track_b_provisional_benign_count": 0,
            "track_b_rating": "NONE", "track_b_established": False, "sealed_ready": False, "critical_manual_review_count": 0,
        }
        phase9b = {"status": "not_frozen_due_to_provider_blocker"}
        decision = "D. PROVIDER/SYSTEMIC SCHEMA FAILURE"
        next_phase = "resume_phase8r6d"
        status_name = "stopped"

    result = {
        "status": status_name,
        "decision": decision,
        "stop_reason": stop_reason,
        "models_trained": False,
        "detectors_evaluated": False,
        "p5_evaluated": False,
        "track_a_modified": False,
        "candidate_count": len(candidates),
        "attempted_count": int((~statuses.status.eq("NOT_ATTEMPTED")).sum()),
        **counts,
        "phase9b_protocol_frozen": phase9b.get("status") == "frozen",
        "actual_runtime_minutes": elapsed_minutes,
        "cumulative_runtime_minutes": cumulative_minutes,
        "provider_requests_phase8r6d": metrics.requests,
        "provider_requests_cumulative": prior_request_total + metrics.requests,
        "provider_retries_phase8r6d": metrics.retries,
        "frozen_inputs_verified": frozen,
        "track_a_rows": int(config["track_a"]["expected_rows"]),
        "track_a_sha256": sha256_file(track_a),
        "next_recommended_phase": next_phase,
    }
    _json_write(out / "phase8r6d_results.json", result)
    report = f"""# Phase 8R6D Final Report

## Scope And Integrity

Phase 8R6D resumed the frozen provider ledger, completed external static-data auditing where provider health allowed, and {'froze' if result['phase9b_protocol_frozen'] else 'did not freeze'} the prospective Phase 9B protocol. No detector was trained or scored, P5 was not evaluated, Track B labels were not used for development, and Track A was not modified.

## Enrichment

- Candidates: {len(candidates)}
- Attempted: {result['attempted_count']}
- Successful: {counts['successful_enrichment_count']}
- Invalid contracts: {counts['invalid_contract_count']}
- Identity mismatches: {counts['identity_mismatch_count']}
- Unsupported schemas: {counts['schema_unsupported_count']}
- Creation metadata not found: {counts.get('not_found_count', 0)}
- Provider failures: {counts['provider_failure_count']}
- Provider requests in Phase 8R6D / cumulative: {metrics.requests} / {prior_request_total + metrics.requests}
- Provider retries in Phase 8R6D: {metrics.retries}
- Additional runtime: {elapsed_minutes:.2f} minutes
- Cumulative runtime: {cumulative_minutes:.2f} minutes

## Feature Alignment

MULCAS_READY: {counts['mulcas_ready_count']}. STATIC_RUNTIME_READY: {counts['static_runtime_ready_count']}. CREATION_CODE_READY: {counts['creation_code_ready_count']}.

## Track B

Later positives/benigns: {counts['later_positive_count']} / {counts['later_benign_count']}. Strict Track B positives/benigns: {counts['track_b_strict_positive_count']} / {counts['track_b_strict_benign_count']}. Provisional positives/benigns: {counts['track_b_provisional_positive_count']} / {counts['track_b_provisional_benign_count']}. Rating: **{counts['track_b_rating']}**. Sealed: **{counts['sealed_ready']}**. Critical manual reviews: {counts['critical_manual_review_count']}.

## Decision

**{decision}**

Next recommended phase/action: `{next_phase}`. Phase 9C was not started.
"""
    (out / "phase8r6d_report.md").write_text(report, encoding="utf-8")

    if sha256_file(track_a) != config["track_a"]["expected_sha256"]:
        raise RuntimeError("Track A changed during Phase 8R6D")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
