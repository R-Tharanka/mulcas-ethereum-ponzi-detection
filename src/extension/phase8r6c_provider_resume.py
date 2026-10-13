"""Audited provider validation and bounded Phase 8R6C continuation.

Secret values are confined to request construction and are never returned,
logged, or written. This module does not import model-development code.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd
import yaml

from src.experiments.phase6a import sha256_file
from src.utils.paths import REPO_ROOT
from .external_enrichment import candidate_rows, deterministic_sample
from .external_normalization import normalize_address
from .track_b_builder import classify, enrich_static_identities, internal_reference, primary_membership, seal

CONFIG = REPO_ROOT / "configs/extension/phase8r6c_provider_resume.yaml"
PARENT_FILES = ("phase8r6c_results.json", "phase8r6c_report.md", "provider_preflight.json")
PROVIDER_STABILITY_FAILURES = {"AUTH_FAILURE", "RATE_LIMITED", "RPC_FAILURE", "SCHEMA_MISMATCH", "IDENTITY_MISMATCH"}


class GateFailure(RuntimeError):
    def __init__(self, gate: str, category: str, message: str):
        super().__init__(message)
        self.gate, self.category = gate, category


@dataclass
class Metrics:
    requests: int = 0
    latencies: list[float] = field(default_factory=list)
    http_statuses: dict[str, int] = field(default_factory=dict)
    retries: int = 0

    def record(self, elapsed: float, status: int | str) -> None:
        self.requests += 1
        self.latencies.append(elapsed)
        key = str(status)
        self.http_statuses[key] = self.http_statuses.get(key, 0) + 1


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_host(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return urlparse(value).hostname
    except ValueError:
        return None


def provider_configuration(environ: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    rpc, creation, key = (env.get("ETH_RPC_URL", "").strip(), env.get("BLOCKSCOUT_API_URL", "").strip(), env.get("BLOCKSCOUT_API_KEY", "").strip())
    return {
        "ETH_RPC_URL": {"configured": bool(rpc), "hostname": _safe_host(rpc), "provider_class": "ethereum_json_rpc", "authentication_required": True},
        "BLOCKSCOUT_API_URL": {"configured": bool(creation), "hostname": _safe_host(creation), "provider_class": "blockscout_creation_api", "authentication_required": True},
        "BLOCKSCOUT_API_KEY": {"configured": bool(key), "hostname": None, "provider_class": "api_credential", "authentication_required": True},
        "credentials_serialized": False,
    }


def _request_json(url: str, payload: Any | None, timeout: float, metrics: Metrics, headers: dict[str, str] | None = None) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json", "User-Agent": "mulcas-phase8r6c-resume/1.0", **(headers or {})})
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            metrics.record(time.perf_counter() - started, response.status)
            return json.loads(raw)
    except HTTPError as exc:
        metrics.record(time.perf_counter() - started, exc.code)
        raise
    except (URLError, TimeoutError):
        metrics.record(time.perf_counter() - started, "TRANSPORT_ERROR")
        raise


def rpc_call(endpoint: str, method: str, params: list[Any], timeout: float, metrics: Metrics) -> Any:
    response = _request_json(endpoint, {"jsonrpc": "2.0", "id": metrics.requests + 1, "method": method, "params": params}, timeout, metrics)
    if not isinstance(response, dict) or response.get("error") or "result" not in response:
        raise GateFailure("rpc", "RPC_FAILURE", f"Invalid JSON-RPC response for {method}")
    return response["result"]


def creation_schema(endpoint: str) -> str:
    path = urlparse(endpoint).path.rstrip("/")
    if path.endswith("/api/v2"):
        return "blockscout_v2"
    if path.endswith("/api"):
        return "blockscout_etherscan_compatible"
    raise GateFailure("creation_provider", "SCHEMA_MISMATCH", "BLOCKSCOUT_API_URL must end in /api/v2 or /api")


def creation_lookup(address: str, endpoint: str, key: str, timeout: float, metrics: Metrics) -> dict[str, Any]:
    schema = creation_schema(endpoint)
    if schema == "blockscout_v2":
        url = endpoint.rstrip("/") + "/addresses/" + normalize_address(address)
        if key:
            url += "?" + urlencode({"apikey": key})
        value = _request_json(url, None, timeout, metrics)
        if not isinstance(value, dict) or "creator_address_hash" not in value:
            raise GateFailure("creation_provider", "SCHEMA_MISMATCH", "Blockscout v2 address response lacks creator_address_hash")
        tx_hash = value.get("creation_transaction_hash") or value.get("creation_tx_hash")
        if not tx_hash:
            raise GateFailure("creation_provider", "SCHEMA_MISMATCH", "Blockscout v2 address response lacks a supported creation transaction hash field")
        return {"address": normalize_address(value.get("hash") or address), "creator": normalize_address(value["creator_address_hash"]), "tx_hash": str(tx_hash).lower(), "block_number": value.get("creation_block_number")}
    query = {"module": "contract", "action": "getcontractcreation", "contractaddresses": normalize_address(address), "apikey": key}
    value = _request_json(endpoint + "?" + urlencode(query), None, timeout, metrics)
    if not isinstance(value, dict) or str(value.get("status")) != "1" or not isinstance(value.get("result"), list) or not value["result"]:
        raise GateFailure("creation_provider", "SCHEMA_MISMATCH", "Compatible creation response lacks a successful result list")
    row = value["result"][0]
    required = {"contractAddress", "contractCreator", "txHash"}
    if not required.issubset(row):
        raise GateFailure("creation_provider", "SCHEMA_MISMATCH", "Compatible creation row lacks required identity fields")
    return {"address": normalize_address(row["contractAddress"]), "creator": normalize_address(row["contractCreator"]), "tx_hash": str(row["txHash"]).lower(), "block_number": row.get("blockNumber")}


def _http_category(exc: BaseException) -> str:
    if isinstance(exc, HTTPError) and exc.code in (401, 403):
        return "AUTH_FAILURE"
    if isinstance(exc, HTTPError) and exc.code == 429:
        return "RATE_LIMITED"
    if isinstance(exc, HTTPError) and exc.code == 404:
        return "NOT_FOUND"
    if isinstance(exc, GateFailure):
        return exc.category
    return "RPC_FAILURE"


def _safe_error_message(exc: BaseException) -> str:
    if isinstance(exc, GateFailure):
        return str(exc)
    if isinstance(exc, HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, (URLError, TimeoutError)):
        return type(exc).__name__
    return type(exc).__name__


def rpc_health(endpoint: str, timeout: float, metrics: Metrics) -> dict[str, Any]:
    started = time.perf_counter()
    chain = rpc_call(endpoint, "eth_chainId", [], timeout, metrics)
    block = rpc_call(endpoint, "eth_blockNumber", [], timeout, metrics)
    chain_id = int(chain, 16)
    if chain_id != 1:
        raise GateFailure("rpc_health", "WRONG_CHAIN", f"Expected Ethereum mainnet chain ID 1, observed {chain_id}")
    return {"status": "PASS", "chain_id": chain_id, "latest_block": int(block, 16), "mainnet_verified": True, "elapsed_seconds": time.perf_counter() - started, "requests": 2, "http_status_distribution": dict(metrics.http_statuses)}


def enrich_one(address: str, endpoint: str, creation_endpoint: str, key: str, timeout: float, metrics: Metrics) -> dict[str, Any]:
    creation = creation_lookup(address, creation_endpoint, key, timeout, metrics)
    target = normalize_address(address)
    if creation["address"] != target:
        raise GateFailure("identity", "IDENTITY_MISMATCH", "Creation provider contract address does not match target")
    tx = rpc_call(endpoint, "eth_getTransactionByHash", [creation["tx_hash"]], timeout, metrics)
    receipt = rpc_call(endpoint, "eth_getTransactionReceipt", [creation["tx_hash"]], timeout, metrics)
    if not isinstance(tx, dict) or not isinstance(receipt, dict):
        raise GateFailure("identity", "RPC_FAILURE", "Creation transaction or receipt is unavailable")
    receipt_address = normalize_address(receipt.get("contractAddress"))
    if receipt_address != target:
        raise GateFailure("identity", "IDENTITY_MISMATCH", "Receipt contractAddress does not match target")
    tx_block, receipt_block = int(tx["blockNumber"], 16), int(receipt["blockNumber"], 16)
    if tx_block != receipt_block or (creation.get("block_number") not in (None, "") and int(creation["block_number"]) != tx_block):
        raise GateFailure("identity", "IDENTITY_MISMATCH", "Creation block numbers do not reconcile")
    block = rpc_call(endpoint, "eth_getBlockByNumber", [hex(tx_block), False], timeout, metrics)
    runtime = rpc_call(endpoint, "eth_getCode", [target, "latest"], timeout, metrics)
    historical = rpc_call(endpoint, "eth_getCode", [target, hex(tx_block)], timeout, metrics)
    if not isinstance(block, dict) or not block.get("timestamp"):
        raise GateFailure("identity", "RPC_FAILURE", "Creation block timestamp is unavailable")
    if runtime in (None, "0x", "0x0"):
        raise GateFailure("runtime", "INVALID_CONTRACT", "Known contract returned empty runtime code")
    static = enrich_static_identities(pd.DataFrame([{"runtime_bytecode": runtime, "creation_code": tx.get("input")}])).iloc[0]
    if not static.disassembly_success:
        raise GateFailure("runtime", "INVALID_CONTRACT", "Deterministic opcode derivation failed")
    timestamp_unix = int(block["timestamp"], 16)
    return {"address": target, "creator": creation["creator"], "creation_transaction_hash": creation["tx_hash"], "created_block": tx_block, "timestamp_unix": timestamp_unix, "timestamp": datetime.fromtimestamp(timestamp_unix, tz=timezone.utc).isoformat(), "runtime_bytecode": runtime, "creation_code": tx.get("input"), "opcode": static.opcode, "runtime_hash": static.runtime_hash, "opcode_hash": static.opcode_hash, "creation_hash": static.creation_hash, "historical_query_supported": historical is not None, "receipt_address_verified": True, "block_consistency_verified": True, "timestamp_consistency_verified": True}


def _batch(addresses: pd.DataFrame, secrets: dict[str, str], cfg: dict[str, Any], metrics: Metrics) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows, failures, started = [], [], time.perf_counter()
    for item in addresses.itertuples(index=False):
        try:
            row = enrich_one(item.address, secrets["rpc"], secrets["creation"], secrets["key"], float(cfg["provider"]["request_timeout_seconds"]), metrics)
            row.update({"label": item.label, "sources": item.sources, "provenance": item.provenance, "status": "SUCCESS"})
            rows.append(row)
        except (HTTPError, URLError, TimeoutError, GateFailure, ValueError, KeyError, TypeError) as exc:
            category = _http_category(exc)
            failures.append({"address": item.address, "status": category, "error_type": type(exc).__name__})
            if category in {"AUTH_FAILURE", "RATE_LIMITED", "SCHEMA_MISMATCH", "IDENTITY_MISMATCH"} and sum(x["status"] == category for x in failures) >= int(cfg["provider"]["sustained_failure_count"]):
                break
    elapsed = time.perf_counter() - started
    latency = metrics.latencies
    summary = {"attempted": len(rows) + len(failures), "full_successes": len(rows), "partial_successes": 0, "failures": failures, "requests": metrics.requests, "elapsed_seconds": elapsed, "mean_request_latency_seconds": statistics.fmean(latency) if latency else None, "median_request_latency_seconds": statistics.median(latency) if latency else None, "http_status_distribution": dict(metrics.http_statuses), "retries": metrics.retries}
    return pd.DataFrame(rows), summary


def _empty_gate(status: str, reason: str) -> dict[str, Any]:
    return {"status": status, "reason": reason, "requests": 0, "elapsed_seconds": 0}


def provider_stability_failure(summary: dict[str, Any]) -> str | None:
    failures = [row["status"] for row in summary.get("failures", []) if row.get("status") in PROVIDER_STABILITY_FAILURES]
    return failures[-1] if failures else None


def _readiness_counts(rows: pd.DataFrame) -> dict[str, int]:
    if rows.empty:
        return {"mulcas_ready_count": 0, "static_runtime_ready_count": 0, "creation_code_ready_count": 0}
    required = ("address", "label", "created_block", "timestamp", "creator", "runtime_bytecode", "opcode")
    mulcas_ready = rows.loc[:, list(required)].notna().all(axis=1) if all(name in rows for name in required) else pd.Series(False, index=rows.index)
    static_ready = rows.runtime_bytecode.notna() & rows.opcode.notna() if all(name in rows for name in ("runtime_bytecode", "opcode")) else pd.Series(False, index=rows.index)
    creation_ready = rows.creation_code.notna() if "creation_code" in rows else pd.Series(False, index=rows.index)
    return {
        "mulcas_ready_count": int(mulcas_ready.sum()),
        "static_runtime_ready_count": int(static_ready.sum()),
        "creation_code_ready_count": int(creation_ready.sum()),
    }


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _load_approved_resume_state(out: Path, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched_path = out / "enriched_contracts.csv"
    enriched = pd.read_csv(enriched_path, keep_default_na=False) if enriched_path.exists() else pd.DataFrame()
    if not enriched.empty:
        enriched["address"] = enriched.address.map(normalize_address)
        enriched = enriched[enriched.address.notna()].drop_duplicates("address", keep="last")
        for column in ("created_block", "timestamp_unix"):
            if column in enriched:
                enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
        for column in ("label", "sources", "provenance"):
            if column in enriched:
                enriched = enriched.drop(columns=column)
        enriched = enriched.merge(candidates[["address", "label", "sources", "provenance"]], on="address", how="inner", validate="one_to_one")
        enriched["status"] = "SUCCESS"

    status_path = out / "enrichment_status.csv"
    status = pd.read_csv(status_path, keep_default_na=False) if status_path.exists() else pd.DataFrame()
    records: dict[str, dict[str, Any]] = {}
    if not status.empty and "address" in status:
        for row in status.to_dict("records"):
            address = normalize_address(row.get("address"))
            if address:
                records[address] = row
    for row in enriched.to_dict("records"):
        records[row["address"]] = {"address": row["address"], "status": "SUCCESS", "attempts": 1, "error_type": "", "requests": 6}

    stability_path = out / "stability_100.json"
    if stability_path.exists():
        stability = json.loads(stability_path.read_text(encoding="utf-8"))
        for failure in stability.get("failures", []):
            address = normalize_address(failure.get("address"))
            if address and address not in records:
                records[address] = {"address": address, "status": failure.get("status", "RPC_FAILURE"), "attempts": 1, "error_type": failure.get("error_type", ""), "requests": 6}

    labels = candidates.set_index("address")[["label", "sources", "provenance"]].to_dict("index")
    normalized = []
    for address, row in records.items():
        if address not in labels:
            continue
        normalized.append({
            "address": address,
            **labels[address],
            "status": str(row.get("status", "")).upper(),
            "attempts": int(float(row.get("attempts") or 0)),
            "error_type": row.get("error_type", ""),
            "requests": int(float(row.get("requests") or 0)),
            "retrieval_provider": "authenticated_blockscout_and_ethereum_rpc",
            "retrieval_timestamp": row.get("retrieval_timestamp", ""),
        })
    return enriched.reset_index(drop=True), pd.DataFrame(normalized)


def _checkpoint_approved_resume(out: Path, candidates: pd.DataFrame, enriched: pd.DataFrame, status: pd.DataFrame) -> None:
    successful = enriched.drop_duplicates("address", keep="last").sort_values("address") if not enriched.empty else enriched
    _atomic_csv(successful, out / "enriched_contracts.csv")
    indexed = status.drop_duplicates("address", keep="last").set_index("address") if not status.empty else pd.DataFrame()
    rows = []
    for candidate in candidates.itertuples(index=False):
        if not indexed.empty and candidate.address in indexed.index:
            row = indexed.loc[candidate.address].to_dict()
            rows.append({"address": candidate.address, **row})
        else:
            rows.append({"address": candidate.address, "label": candidate.label, "sources": candidate.sources, "provenance": candidate.provenance, "status": "NOT_ATTEMPTED", "attempts": 0, "error_type": "", "requests": 0, "retrieval_provider": "", "retrieval_timestamp": ""})
    _atomic_csv(pd.DataFrame(rows), out / "enrichment_status.csv")


def _rating(positive: int, benign: int) -> str:
    if positive >= 100 and benign > 0:
        return "STRONG"
    if positive >= 30 and benign > 0:
        return "MODERATE"
    return "WEAK" if positive else "NONE"


def _match_map(frame: pd.DataFrame, identity: str) -> dict[str, str]:
    available = frame.dropna(subset=[identity])
    return {
        value: "|".join(f"{idx}:{row.address}:{int(row.label)}" for idx, row in group.sort_values("address").iterrows())
        for value, group in available.groupby(identity)
    }


def _count_stage(candidates: pd.DataFrame, addresses: set[str], stage: str) -> dict[str, Any]:
    rows = candidates[candidates.address.isin(addresses)]
    return {"stage": stage, "total": len(rows), "positive": int(rows.label.eq(1).sum()), "benign": int(rows.label.eq(0).sum())}


def _finalize_approved_track_b(enriched: pd.DataFrame, statuses: pd.DataFrame, candidates: pd.DataFrame, cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    rows = enriched.copy().drop_duplicates("address", keep="last")
    for name, default in (("receipt_address_verified", True), ("block_consistency_verified", True), ("timestamp_consistency_verified", True)):
        if name not in rows:
            rows[name] = default
    for name in ("created_block", "timestamp", "creator", "runtime_bytecode", "opcode", "creation_code", "creation_transaction_hash"):
        if name not in rows:
            rows[name] = None

    conflict_path = REPO_ROOT / "results/extension/phase8r6b/external_label_conflicts.csv"
    conflicts = pd.read_csv(conflict_path, keep_default_na=False)
    unresolved = set(conflicts.loc[conflicts.resolution.eq("unresolved_no_majority_vote") & conflicts.candidate_in_scope.astype(str).str.lower().eq("true"), "address"].map(normalize_address))
    rows["unresolved_conflict"] = rows.address.isin(unresolved)
    eligible_ab = rows.provenance.map(lambda value: bool(set(str(value).split("|")) & {"A", "B"}))
    verified_chronology = rows.receipt_address_verified.astype(str).str.lower().eq("true") & rows.block_consistency_verified.astype(str).str.lower().eq("true") & rows.timestamp_consistency_verified.astype(str).str.lower().eq("true")
    rows["MULCAS_READY"] = eligible_ab & verified_chronology & rows[["address", "created_block", "timestamp", "creator", "runtime_bytecode", "opcode"]].notna().all(axis=1)
    rows["STATIC_RUNTIME_READY"] = rows.runtime_bytecode.notna() & rows.opcode.notna()
    rows["CREATION_CODE_READY"] = rows.creation_code.notna()

    internal, refs = internal_reference(REPO_ROOT / cfg["internal_dataset"])
    classified = classify(rows, refs, int(cfg["temporal_boundary_block"]))
    classified["eligible_ab"] = eligible_ab.values
    classified["prior_model_use"] = False
    classified["independence_classification"] = "UNKNOWN"
    known_overlap = (~classified.address_new) | (classified.runtime_hash.notna() & ~classified.runtime_new) | (classified.opcode_hash.notna() & ~classified.opcode_new) | (classified.creation_hash.notna() & ~classified.creation_new)
    classified.loc[known_overlap, "independence_classification"] = "NOT_INDEPENDENT"
    strict_mask = classified.address_new & classified.runtime_new & classified.opcode_new & classified.creation_hash.notna() & classified.creation_new
    provisional_mask = classified.address_new & classified.runtime_new & classified.opcode_new & classified.creation_hash.isna()
    classified.loc[strict_mask, "independence_classification"] = "STRICT_INDEPENDENT"
    classified.loc[provisional_mask, "independence_classification"] = "PROVISIONAL_INDEPENDENT"

    base = classified.eligible_ab & classified.address_new & classified.temporal_status.eq("LATER") & classified.runtime_hash.notna() & classified.opcode_hash.notna() & classified.runtime_new & classified.opcode_new & ~classified.unresolved_conflict & classified.MULCAS_READY & ~classified.prior_model_use
    strict = classified[base & classified.creation_hash.notna() & classified.creation_new].copy()
    provisional = classified[base & classified.creation_hash.isna()].copy()
    strict["strictness"] = "STRICT"
    provisional["strictness"] = "PROVISIONAL"
    selected = pd.concat([strict, provisional], ignore_index=True).drop_duplicates("address").sort_values("address")

    aliases = {"created_block": "createdBlockNumber", "creation_transaction_hash": "createdTransactionHash", "runtime_bytecode": "contractCode", "creation_code": "creationCode", "runtime_hash": "runtime_sha256", "opcode_hash": "opcode_sha256", "creation_hash": "creation_code_sha256"}
    for source, target in aliases.items():
        classified[target] = classified[source]
    classified["label_provenance"] = classified.provenance
    classified["retrieval_provider"] = "authenticated_blockscout_and_ethereum_rpc"
    classified["retrieval_timestamp"] = datetime.now(timezone.utc).isoformat()
    _atomic_csv(classified.sort_values("address"), out / "enriched_contracts.csv")

    temporal_columns = ["address", "label", "created_block", "timestamp", "temporal_status"]
    _atomic_csv(classified[temporal_columns].sort_values("address"), out / "temporal_classification.csv")
    runtime_matches = _match_map(internal, "runtime_hash")
    opcode_matches = _match_map(internal, "opcode_hash")
    creation_matches = _match_map(internal, "creation_hash")
    runtime_audit = classified[["address", "label", "runtime_hash", "runtime_new", "independence_classification"]].copy()
    runtime_audit["internal_matches"] = runtime_audit.runtime_hash.map(runtime_matches).fillna("")
    opcode_audit = classified[["address", "label", "opcode_hash", "opcode_new", "opcode_count", "malformed_truncated_count", "unsupported_opcode_count", "independence_classification"]].copy()
    opcode_audit["internal_matches"] = opcode_audit.opcode_hash.map(opcode_matches).fillna("")
    creation_audit = classified[["address", "label", "creation_hash", "creation_new", "independence_classification"]].copy()
    creation_audit["creation_overlap_status"] = creation_audit.apply(lambda row: "CREATION_OVERLAP_UNKNOWN" if pd.isna(row.creation_hash) else ("CREATION_CODE_NEW" if row.creation_new else "CREATION_CODE_OVERLAP"), axis=1)
    creation_audit["internal_matches"] = creation_audit.creation_hash.map(creation_matches).fillna("")
    _atomic_csv(runtime_audit.sort_values("address"), out / "runtime_overlap.csv")
    _atomic_csv(opcode_audit.sort_values("address"), out / "opcode_overlap.csv")
    _atomic_csv(creation_audit.sort_values("address"), out / "creation_overlap.csv")
    independence_columns = ["address", "label", "provenance", "address_new", "runtime_new", "opcode_new", "creation_new", "temporal_status", "unresolved_conflict", "independence_classification"]
    _atomic_csv(classified[independence_columns].sort_values("address"), out / "independence_classification.csv")

    funnel = []
    all_addresses = set(candidates.address)
    success_addresses = set(classified.address)
    funnel.append(_count_stage(candidates, all_addresses, "starting_candidate_rows"))
    funnel.append(_count_stage(candidates, success_addresses, "successfully_enriched"))
    masks = [
        (classified.eligible_ab, "A_B_provenance"),
        (classified.eligible_ab & classified.address_new, "new_address"),
        (classified.eligible_ab & classified.address_new & classified.temporal_status.eq("LATER"), "later_deployment"),
        (classified.eligible_ab & classified.address_new & classified.temporal_status.eq("LATER") & classified.runtime_new, "runtime_new"),
        (classified.eligible_ab & classified.address_new & classified.temporal_status.eq("LATER") & classified.runtime_new & classified.opcode_new, "opcode_new"),
        (classified.eligible_ab & classified.address_new & classified.temporal_status.eq("LATER") & classified.runtime_new & classified.opcode_new & (classified.creation_hash.isna() | classified.creation_new), "creation_new_or_unavailable"),
        (classified.eligible_ab & classified.address_new & classified.temporal_status.eq("LATER") & classified.runtime_new & classified.opcode_new & (classified.creation_hash.isna() | classified.creation_new) & ~classified.unresolved_conflict, "no_unresolved_A_B_conflict"),
        (base, "MULCAS_READY"),
    ]
    for mask, name in masks:
        funnel.append(_count_stage(candidates, set(classified.loc[mask, "address"]), name))
    funnel.append(_count_stage(candidates, set(strict.address), "TRACK_B_STRICT"))
    funnel.append(_count_stage(candidates, set(provisional.address), "TRACK_B_PROVISIONAL"))
    funnel.append(_count_stage(candidates, set(selected.address), "TRACK_B_STRICT_PLUS_PROVISIONAL"))
    _atomic_csv(pd.DataFrame(funnel), out / "track_b_filter_funnel.csv")

    conflicts = conflicts.copy()
    conflicts["successfully_enriched"] = conflicts.address.map(normalize_address).isin(success_addresses)
    _atomic_csv(conflicts, out / "label_conflicts.csv")
    benign = classified[classified.label.eq(0)][["address", "temporal_status", "runtime_new", "opcode_new", "creation_new"]].copy()
    benign["quality"] = "BENIGN_CURATED"
    benign["provenance"] = "B"
    benign["unlabeled_scam_limitation"] = "cannot_be_excluded"
    _atomic_csv(benign.sort_values("address"), out / "benign_quality.csv")

    triage = []
    independent_later = classified[classified.temporal_status.eq("LATER") & classified.independence_classification.isin(["STRICT_INDEPENDENT", "PROVISIONAL_INDEPENDENT"])]
    for row in independent_later[independent_later.unresolved_conflict].itertuples():
        triage.append({"priority_group": "M1", "address": row.address, "label": row.label, "sources": row.sources, "reason": "later independent unresolved A/B conflict"})
    for row in independent_later[~independent_later.eligible_ab].itertuples():
        triage.append({"priority_group": "M2", "address": row.address, "label": row.label, "sources": row.sources, "reason": "later independent provenance uncertainty"})
    for row in independent_later[independent_later.label.eq(0)].itertuples():
        triage.append({"priority_group": "M3", "address": row.address, "label": row.label, "sources": row.sources, "reason": "later independent curated-benign ambiguity"})
    failed_positive = statuses[~statuses.status.eq("SUCCESS") & statuses.label.eq(1)]
    for row in failed_positive.itertuples():
        triage.append({"priority_group": "M4", "address": row.address, "label": row.label, "sources": row.sources, "reason": f"missing critical enrichment field: {row.status}"})
    previous_triage = REPO_ROOT / "results/extension/phase8r6b/manual_review_triage.csv"
    if previous_triage.exists():
        old = pd.read_csv(previous_triage, keep_default_na=False)
        for row in old[old.priority_group.eq("M5")].itertuples():
            triage.append({"priority_group": "M5", "address": normalize_address(row.address), "label": "unknown", "sources": getattr(row, "sources", "ponziguard_predictions"), "reason": "class-D model prediction; verification only"})
    triage_frame = pd.DataFrame(triage, columns=["priority_group", "address", "label", "sources", "reason"]).drop_duplicates(["priority_group", "address"])
    _atomic_csv(triage_frame.sort_values(["priority_group", "address"]), out / "manual_review_triage.csv")
    critical_review = int(triage_frame.priority_group.isin(["M1", "M2", "M3"]).sum())
    review_packet_created = critical_review <= 30 and critical_review > 0
    if review_packet_created:
        _atomic_csv(triage_frame[triage_frame.priority_group.isin(["M1", "M2", "M3"])], out / "manual_review_packet.csv")

    _atomic_csv(strict, out / "track_b_strict.csv")
    _atomic_csv(provisional, out / "track_b_provisional.csv")
    _atomic_csv(selected, out / "track_b_all_unique_address.csv")
    sensitivity = strict.sort_values("address").drop_duplicates(["runtime_hash", "opcode_hash", "creation_hash"], keep="first")
    _atomic_csv(sensitivity, out / "track_b_strict_unique_code_sensitivity.csv")
    duplicate_rows = []
    for identity in ("address", "runtime_hash", "opcode_hash", "creation_hash"):
        duplicate_rows.append({"record_type": "SUMMARY", "identity": identity, "hash": "", "group_size": "", "addresses": "", "available_rows": int(selected[identity].notna().sum()), "unique_values": int(selected[identity].nunique()), "duplicate_rows": int(selected[identity].notna().sum() - selected[identity].nunique())})
        for value, group in selected.dropna(subset=[identity]).groupby(identity):
            if len(group) > 1:
                duplicate_rows.append({"record_type": "CLONE_GROUP", "identity": identity, "hash": value, "group_size": len(group), "addresses": "|".join(sorted(group.address)), "available_rows": "", "unique_values": "", "duplicate_rows": ""})
    _atomic_csv(pd.DataFrame(duplicate_rows), out / "track_b_duplicate_audit.csv")

    schema = []
    schema_fields = [("address", classified.address.notna()), ("label", classified.label.notna()), ("eligible_A_B_label", classified.eligible_ab), ("createdBlockNumber", classified.created_block.notna()), ("timestamp", classified.timestamp.notna()), ("creator", classified.creator.notna()), ("contractCode", classified.runtime_bytecode.notna()), ("opcode", classified.opcode.notna()), ("creationCode", classified.creation_code.notna()), ("createdTransactionHash", classified.creation_transaction_hash.notna()), ("MULCAS_READY", classified.MULCAS_READY), ("STATIC_RUNTIME_READY", classified.STATIC_RUNTIME_READY), ("CREATION_CODE_READY", classified.CREATION_CODE_READY)]
    for field, mask in schema_fields:
        count = int(mask.sum())
        schema.append({"field_or_status": field, "candidate_count": len(candidates), "available_count": count, "available_percent": 100 * count / len(candidates) if len(candidates) else 0})
    _atomic_csv(pd.DataFrame(schema), out / "schema_compatibility.csv")

    strict_positive, strict_benign = int(strict.label.eq(1).sum()), int(strict.label.eq(0).sum())
    provisional_positive, provisional_benign = int(provisional.label.eq(1).sum()), int(provisional.label.eq(0).sum())
    strict_rating = _rating(strict_positive, strict_benign)
    combined_rating = _rating(strict_positive + provisional_positive, strict_benign + provisional_benign)
    scientifically_ready = strict_rating in {"STRONG", "MODERATE"} and critical_review == 0
    readiness = pd.DataFrame([
        {"subset": "STRICT", "rating": strict_rating, "rows": len(strict), "positive": strict_positive, "benign": strict_benign, "sealed_ready": scientifically_ready},
        {"subset": "STRICT_PLUS_PROVISIONAL", "rating": combined_rating, "rows": len(selected), "positive": strict_positive + provisional_positive, "benign": strict_benign + provisional_benign, "sealed_ready": scientifically_ready},
    ])
    _atomic_csv(readiness, out / "track_b_readiness.csv")
    sealed_manifest = None
    if scientifically_ready:
        sealed_manifest = seal(selected, REPO_ROOT / "data/external/track_b/metadata.csv", REPO_ROOT / "data/external/track_b_sealed/labels.csv", REPO_ROOT / "data/external/track_b/membership_manifest.json")

    status_counts = statuses.status.value_counts().to_dict()
    temporal = classified.groupby(["temporal_status", "label"]).size().to_dict()
    return {
        **_readiness_counts(classified),
        "invalid_contract_count": int(status_counts.get("INVALID_CONTRACT", 0)),
        "failed_count": int(len(statuses) - status_counts.get("SUCCESS", 0)),
        "later_positive_count": int(temporal.get(("LATER", 1), 0)),
        "later_benign_count": int(temporal.get(("LATER", 0), 0)),
        "pre_existing_positive_count": int(temporal.get(("PRE_EXISTING_PERIOD", 1), 0)),
        "pre_existing_benign_count": int(temporal.get(("PRE_EXISTING_PERIOD", 0), 0)),
        "unknown_positive_count": int(temporal.get(("UNKNOWN", 1), 0)),
        "unknown_benign_count": int(temporal.get(("UNKNOWN", 0), 0)),
        "runtime_new_positive_count": int((classified.runtime_new & classified.label.eq(1)).sum()),
        "runtime_new_benign_count": int((classified.runtime_new & classified.label.eq(0)).sum()),
        "opcode_new_positive_count": int((classified.opcode_new & classified.label.eq(1)).sum()),
        "opcode_new_benign_count": int((classified.opcode_new & classified.label.eq(0)).sum()),
        "creation_new_count": int(classified.creation_new.sum()),
        "creation_unknown_count": int(classified.creation_hash.isna().sum()),
        "strict_independent_positive_count": int((classified.independence_classification.eq("STRICT_INDEPENDENT") & classified.label.eq(1)).sum()),
        "strict_independent_benign_count": int((classified.independence_classification.eq("STRICT_INDEPENDENT") & classified.label.eq(0)).sum()),
        "provisional_independent_positive_count": int((classified.independence_classification.eq("PROVISIONAL_INDEPENDENT") & classified.label.eq(1)).sum()),
        "provisional_independent_benign_count": int((classified.independence_classification.eq("PROVISIONAL_INDEPENDENT") & classified.label.eq(0)).sum()),
        "track_b_strict_positive_count": strict_positive,
        "track_b_strict_benign_count": strict_benign,
        "track_b_provisional_positive_count": provisional_positive,
        "track_b_provisional_benign_count": provisional_benign,
        "track_b_strict_count": len(strict),
        "track_b_provisional_count": len(provisional),
        "strict_rating": strict_rating,
        "combined_rating": combined_rating,
        "track_b_rating": strict_rating,
        "track_b_established": scientifically_ready,
        "sealed_ready": scientifically_ready,
        "sealed_manifest": sealed_manifest,
        "manual_review_required": critical_review > 0,
        "critical_manual_review_count": critical_review,
        "manual_review_count": len(triage_frame),
        "review_packet_created": review_packet_created,
        "unresolved_conflict_count": len(unresolved),
    }


def successful_ledger_addresses(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ledger = pd.read_csv(path, keep_default_na=False)
    if "status" not in ledger or "address" not in ledger:
        return set()
    return {normalize_address(value) for value in ledger.loc[ledger.status.astype(str).str.upper().eq("SUCCESS"), "address"] if normalize_address(value)}


def successful_ledger_rows(path: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    ledger = pd.read_csv(path, keep_default_na=False)
    if "status" not in ledger or "address" not in ledger:
        return pd.DataFrame()
    rows = ledger[ledger.status.astype(str).str.upper().eq("SUCCESS")].copy()
    if rows.empty:
        return rows
    rows["address"] = rows.address.map(normalize_address)
    labels = candidates[["address", "label", "sources", "provenance"]]
    rows = rows.drop(columns=[name for name in ("label", "sources", "provenance") if name in rows]).merge(labels, on="address", how="inner", validate="one_to_one")
    rows["status"] = "SUCCESS"
    return rows


def _finalize_track_b(enriched: pd.DataFrame, cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    rows = enriched.copy()
    for name, default in (("unresolved_conflict", False), ("receipt_address_verified", True), ("block_consistency_verified", True), ("timestamp_consistency_verified", True)):
        if name not in rows:
            rows[name] = default
    rows["MULCAS_READY"] = rows.apply(lambda r: pd.notna(r.get("address")) and pd.notna(r.get("label")) and pd.notna(r.get("created_block")) and pd.notna(r.get("timestamp")) and pd.notna(r.get("creator")) and pd.notna(r.get("runtime_bytecode")) and pd.notna(r.get("opcode")), axis=1)
    rows["STATIC_RUNTIME_READY"] = rows.runtime_bytecode.notna() & rows.opcode.notna()
    rows["CREATION_CODE_READY"] = rows.creation_code.notna()
    _, refs = internal_reference(REPO_ROOT / cfg["internal_dataset"])
    classified = classify(rows, refs, int(cfg["temporal_boundary_block"]))
    selected, excluded = primary_membership(classified)
    strict = selected[selected.strictness.eq("STRICT")]
    provisional = selected[selected.strictness.eq("PROVISIONAL")]
    rows.to_csv(out / "enriched_contracts.csv", index=False)
    classified[["address", "label", "created_block", "timestamp", "temporal_status"]].to_csv(out / "temporal_classification.csv", index=False)
    classified[["address", "runtime_hash", "runtime_new", "independence_status"]].to_csv(out / "runtime_overlap.csv", index=False)
    classified[["address", "opcode_hash", "opcode_new", "independence_status"]].to_csv(out / "opcode_overlap.csv", index=False)
    classified[["address", "creation_hash", "creation_new", "independence_status"]].to_csv(out / "creation_overlap.csv", index=False)
    strict.to_csv(out / "track_b_strict.csv", index=False)
    provisional.to_csv(out / "track_b_provisional.csv", index=False)
    availability = []
    for field in ("address", "label", "created_block", "timestamp", "creator", "runtime_bytecode", "opcode", "creation_code", "creation_transaction_hash", "MULCAS_READY", "STATIC_RUNTIME_READY", "CREATION_CODE_READY"):
        count = int(rows[field].fillna(False).astype(bool).sum()) if field in {"MULCAS_READY", "STATIC_RUNTIME_READY", "CREATION_CODE_READY"} else int(rows[field].notna().sum())
        availability.append({"field_or_status": field, "total_eligible_rows": len(rows), "available_count": count, "available_percent": 100 * count / len(rows) if len(rows) else 0})
    pd.DataFrame(availability).to_csv(out / "schema_compatibility.csv", index=False)
    positive = int(selected.label.eq(1).sum())
    benign = int(selected.label.eq(0).sum())
    rating = "STRONG" if positive >= 100 and benign else ("MODERATE" if positive >= 30 and benign else ("WEAK" if positive else "NONE"))
    sealed_ready = rating in {"STRONG", "MODERATE"}
    readiness = {"rating": rating, "sealed_ready": sealed_ready, "strict_rows": len(strict), "provisional_rows": len(provisional), "positive": positive, "benign": benign}
    pd.DataFrame([readiness]).to_csv(out / "track_b_readiness.csv", index=False)
    if sealed_ready:
        seal(selected, REPO_ROOT / "data/external/track_b/metadata.csv", REPO_ROOT / "data/external/track_b_sealed/labels.csv", REPO_ROOT / "data/external/track_b/membership_manifest.json")
    return {"mulcas_ready_count": int(rows.MULCAS_READY.sum()), "static_runtime_ready_count": int(rows.STATIC_RUNTIME_READY.sum()), "creation_code_ready_count": int(rows.CREATION_CODE_READY.sum()), "later_positive_count": int((classified.temporal_status.eq("LATER") & classified.label.eq(1)).sum()), "later_benign_count": int((classified.temporal_status.eq("LATER") & classified.label.eq(0)).sum()), "track_b_strict_count": len(strict), "track_b_provisional_count": len(provisional), "track_b_rating": rating, "sealed_ready": sealed_ready, "excluded_count": len(excluded)}


def _decision(category: str) -> tuple[str, str]:
    if category == "RATE_LIMITED":
        return "B", "PROVIDER VALID BUT RATE-LIMITED"
    return "A", "PROVIDER CONFIGURATION INVALID"


def run(config_path: Path = CONFIG, environ: dict[str, str] | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    parent = REPO_ROOT / cfg["parent_results"]
    parent_hashes_before = {name: sha256_file(parent / name) for name in PARENT_FILES}
    env = os.environ if environ is None else environ
    configuration = provider_configuration(env)
    _json_write(out / "provider_configuration_status.json", configuration)
    required = [name for name in ("ETH_RPC_URL", "BLOCKSCOUT_API_URL", "BLOCKSCOUT_API_KEY") if not configuration[name]["configured"]]
    gates: dict[str, Any] = {}
    enriched = pd.DataFrame()
    final_counts = {"mulcas_ready_count": 0, "static_runtime_ready_count": 0, "creation_code_ready_count": 0, "later_positive_count": 0, "later_benign_count": 0, "track_b_strict_count": 0, "track_b_provisional_count": 0, "track_b_rating": "NONE", "sealed_ready": False}
    full_resume = False
    category = "MISSING_CONFIGURATION"
    reason = "Missing required environment variables: " + ", ".join(required) if required else ""
    if not required:
        secrets = {"rpc": env["ETH_RPC_URL"].strip(), "creation": env["BLOCKSCOUT_API_URL"].strip(), "key": env["BLOCKSCOUT_API_KEY"].strip()}
        timeout = float(cfg["provider"]["request_timeout_seconds"])
        try:
            creation_schema(secrets["creation"])
            rpc_metrics = Metrics()
            gates["rpc"] = rpc_health(secrets["rpc"], timeout, rpc_metrics)
            candidates = candidate_rows(REPO_ROOT / cfg["internal_dataset"], REPO_ROOT / cfg["ponzi_ml_source"], REPO_ROOT / cfg["unica_source"])
            ledger_path = REPO_ROOT / cfg["source_ledger"]
            completed_before = successful_ledger_addresses(ledger_path)
            prior_rows = successful_ledger_rows(ledger_path, candidates)
            available = candidates[~candidates.address.isin(completed_before)]
            internal = pd.read_csv(REPO_ROOT / cfg["internal_dataset"])
            known = internal.loc[internal.contractCode.astype(str).str.startswith("0x") & internal.contractCode.astype(str).ne("0x"), "address"].head(2).map(normalize_address).tolist()
            creation_metrics = Metrics()
            checked = [creation_lookup(address, secrets["creation"], secrets["key"], timeout, creation_metrics) for address in known]
            if any(row["address"] != address for row, address in zip(checked, known)):
                raise GateFailure("creation_provider", "IDENTITY_MISMATCH", "Creation health-check identity mismatch")
            gates["creation"] = {"status": "PASS", "addresses_checked": 2, "requests": creation_metrics.requests, "http_status_distribution": creation_metrics.http_statuses, "schema": creation_schema(secrets["creation"])}
            end_metrics = Metrics()
            end = enrich_one(known[0], secrets["rpc"], secrets["creation"], secrets["key"], timeout, end_metrics)
            gates["end_to_end"] = {"status": "PASS", "address_match": True, "receipt_match": True, "block_reconciled": True, "timestamp_available": True, "runtime_nonempty": True, "opcode_derived": bool(end["opcode"]), "historical_query_supported": end["historical_query_supported"], "requests": end_metrics.requests}
            preflight_sample = deterministic_sample(available, int(cfg["preflight"]["positive_count"]), int(cfg["preflight"]["benign_count"]))
            preflight_metrics = Metrics()
            preflight_rows, gates["preflight"] = _batch(preflight_sample, secrets, cfg, preflight_metrics)
            if len(preflight_rows) != len(preflight_sample):
                failure_category = gates["preflight"]["failures"][-1]["status"] if gates["preflight"]["failures"] else "RPC_FAILURE"
                raise GateFailure("preflight", failure_category, "20-address preflight did not fully succeed")
            gates["preflight"]["status"] = "PASS"
            remaining = available[~available.address.isin(preflight_sample.address)]
            stability_sample = deterministic_sample(remaining, int(cfg["stability"]["positive_count"]), int(cfg["stability"]["benign_count"]))
            stability_metrics = Metrics()
            stability_rows, gates["stability"] = _batch(stability_sample, secrets, cfg, stability_metrics)
            failure_category = provider_stability_failure(gates["stability"])
            if failure_category:
                raise GateFailure("stability", failure_category, "100-address stability gate did not fully succeed")
            gates["stability"]["status"] = "PASS"
            remaining_count = len(available) - len(preflight_sample) - len(stability_sample)
            per_address = gates["stability"]["elapsed_seconds"] / gates["stability"]["attempted"]
            projected_minutes = remaining_count * per_address / 60
            observed_minutes = (gates["preflight"]["elapsed_seconds"] + gates["stability"]["elapsed_seconds"]) / 60
            requests_per_address = gates["stability"]["requests"] / gates["stability"]["attempted"]
            gates["projection"] = {"status": "PASS" if projected_minutes <= float(cfg["performance_gate_minutes"]) else "STOP", "remaining_candidate_count": remaining_count, "estimated_remaining_requests": round(remaining_count * requests_per_address), "projected_remaining_minutes": projected_minutes, "projected_total_minutes": observed_minutes + projected_minutes, "expected_retry_overhead": gates["stability"]["retries"], "maximum_minutes": cfg["performance_gate_minutes"]}
            enriched = pd.concat([prior_rows, preflight_rows, stability_rows], ignore_index=True)
            final_counts.update(_readiness_counts(enriched))
            if projected_minutes > float(cfg["performance_gate_minutes"]):
                category, reason = "RUNTIME_OVER_30_MIN", "Observed stability gate projects full runtime above 30 minutes; awaiting user approval"
            else:
                # The continuation is authorized; resume only rows not already successful.
                todo = available[~available.address.isin(enriched.address)]
                full_metrics = Metrics()
                full_rows, full_summary = _batch(todo, secrets, cfg, full_metrics)
                gates["full_resume"] = full_summary
                enriched = pd.concat([enriched, full_rows], ignore_index=True).drop_duplicates("address", keep="last")
                full_resume = len(enriched) == len(candidates)
                category = "COMPLETE" if full_resume else (full_summary["failures"][-1]["status"] if full_summary["failures"] else "RPC_FAILURE")
                reason = "Full enrichment completed" if full_resume else "Full enrichment stopped on provider failure"
                if full_resume:
                    final_counts = _finalize_track_b(enriched, cfg, out)
        except (HTTPError, URLError, TimeoutError, GateFailure, ValueError, KeyError, TypeError) as exc:
            category = _http_category(exc)
            reason = _safe_error_message(exc)
    for name, filename in (("rpc", "rpc_health_check.json"), ("creation", "creation_provider_health_check.json"), ("end_to_end", "end_to_end_validation.json"), ("preflight", "preflight_20.json"), ("stability", "stability_100.json"), ("projection", "runtime_projection.json")):
        _json_write(out / filename, gates.get(name, _empty_gate("NOT_RUN", reason)))
    source_ledger = REPO_ROOT / cfg["source_ledger"]
    _json_write(out / "ledger_audit.json", {"source": cfg["source_ledger"], "source_sha256": sha256_file(source_ledger), "source_modified": False, "continuation_rows": len(enriched)})
    if not enriched.empty:
        enriched.to_csv(out / "enriched_contracts.csv", index=False)
        enriched[["address", "status", "created_block", "creation_transaction_hash"]].to_csv(out / "enrichment_status.csv", index=False)
    parent_hashes_after = {name: sha256_file(parent / name) for name in PARENT_FILES}
    if parent_hashes_before != parent_hashes_after:
        raise RuntimeError("Frozen Phase 8R6C evidence changed during continuation")
    code, label = ("C", "PROVIDER VALID; FULL RUNTIME >30 MIN — AWAIT USER APPROVAL") if category == "RUNTIME_OVER_30_MIN" else _decision(category)
    if category == "COMPLETE":
        if final_counts["sealed_ready"]:
            code, label = "D", "ENRICHMENT COMPLETE; TRACK B READY"
        else:
            code, label = "E", "ENRICHMENT COMPLETE; TRACK B INADEQUATE"
    summary = {"status": category, "decision": code, "decision_label": label, "reason": reason, "full_resume_occurred": full_resume, "enriched_count": len(enriched), **final_counts, "models_trained": False, "detectors_evaluated": False, "track_a_modified": False, "parent_results_preserved": True}
    _json_write(out / "resume_summary.json", summary)
    report = "# Phase 8R6C provider resume\n\nPhase 8R6C provider continuation validates and constructs evaluation data; it does not train or evaluate a detector.\n\n"
    report += f"Decision: **{code}. {label}**\n\nStatus: `{category}`. {reason}\n\nThe original Phase 8R6C evidence, Track A, and sealed-label guard remain unchanged. No secret values are serialized. Phase 9B was not started.\n"
    (out / "phase8r6c_provider_resume_report.md").write_text(report, encoding="utf-8")
    return summary


def run_approved_full(
    config_path: Path = CONFIG,
    environ: dict[str, str] | None = None,
    approved_total_minutes: float = 100.0,
    checkpoint_interval: int = 50,
) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    parent = REPO_ROOT / cfg["parent_results"]
    parent_hashes_before = {name: sha256_file(parent / name) for name in PARENT_FILES}
    env = os.environ if environ is None else environ
    configuration = provider_configuration(env)
    missing = [name for name in ("ETH_RPC_URL", "BLOCKSCOUT_API_URL", "BLOCKSCOUT_API_KEY") if not configuration[name]["configured"]]
    if missing:
        raise GateFailure("configuration", "MISSING_CONFIGURATION", "Required provider configuration is unavailable")
    secrets = {"rpc": env["ETH_RPC_URL"].strip(), "creation": env["BLOCKSCOUT_API_URL"].strip(), "key": env["BLOCKSCOUT_API_KEY"].strip()}
    creation_schema(secrets["creation"])

    raw_path = REPO_ROOT / cfg["internal_dataset"]
    expected_raw_hash = "48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf"
    if sha256_file(raw_path) != expected_raw_hash or len(pd.read_csv(raw_path, usecols=["address"])) != 6498:
        raise RuntimeError("Track A integrity verification failed")

    candidates = candidate_rows(raw_path, REPO_ROOT / cfg["ponzi_ml_source"], REPO_ROOT / cfg["unica_source"])
    if len(candidates) != 1490 or not candidates.address.is_unique:
        raise RuntimeError("Frozen candidate population mismatch")
    enriched, status = _load_approved_resume_state(out, candidates)
    terminal = {"SUCCESS", "INVALID_CONTRACT", "NOT_FOUND"}
    completed = set(status.loc[status.status.isin(terminal), "address"]) if not status.empty else set()
    todo = candidates[~candidates.address.isin(completed)].copy()
    starting_successes = len(enriched)

    preflight = json.loads((out / "preflight_20.json").read_text(encoding="utf-8"))
    stability = json.loads((out / "stability_100.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or stability.get("status") != "PASS":
        raise RuntimeError("Approved continuation requires passed bounded provider gates")
    prior_gate_seconds = float(preflight["elapsed_seconds"]) + float(stability["elapsed_seconds"])

    metrics = Metrics()
    started = time.perf_counter()
    processed = 0
    stop_category: str | None = None
    stop_reason = ""
    consecutive_provider_failures = 0
    max_attempts = int(cfg["provider"].get("maximum_attempts", 3))
    timeout = float(cfg["provider"]["request_timeout_seconds"])
    checkpoint_interval = max(1, int(checkpoint_interval))
    approved_total_minutes = float(approved_total_minutes)

    for item in todo.itertuples(index=False):
        attempts = 0
        request_start = metrics.requests
        outcome: dict[str, Any] | None = None
        category = "RPC_FAILURE"
        error_type = ""
        while attempts < max_attempts:
            attempts += 1
            try:
                outcome = enrich_one(item.address, secrets["rpc"], secrets["creation"], secrets["key"], timeout, metrics)
                category = "SUCCESS"
                error_type = ""
                break
            except (HTTPError, URLError, TimeoutError, GateFailure, ValueError, KeyError, TypeError) as exc:
                category = _http_category(exc)
                if category == "SCHEMA_MISMATCH":
                    category = "PROVIDER_SCHEMA_FAILURE"
                error_type = type(exc).__name__
                retryable = category in {"RPC_FAILURE", "RATE_LIMITED"}
                if retryable and attempts < max_attempts:
                    metrics.retries += 1
                    time.sleep(float(cfg["provider"].get("backoff_seconds", 1)) * attempts)
                    continue
                break

        timestamp = datetime.now(timezone.utc).isoformat()
        if outcome is not None:
            outcome.update({"label": item.label, "sources": item.sources, "provenance": item.provenance, "status": "SUCCESS", "retrieval_provider": "authenticated_blockscout_and_ethereum_rpc", "retrieval_timestamp": timestamp})
            enriched = pd.concat([enriched, pd.DataFrame([outcome])], ignore_index=True).drop_duplicates("address", keep="last")
            consecutive_provider_failures = 0
        elif category in {"AUTH_FAILURE", "RATE_LIMITED", "RPC_FAILURE", "PROVIDER_SCHEMA_FAILURE", "IDENTITY_MISMATCH"}:
            consecutive_provider_failures += 1
        else:
            consecutive_provider_failures = 0

        record = {"address": item.address, "label": item.label, "sources": item.sources, "provenance": item.provenance, "status": category, "attempts": attempts, "error_type": error_type, "requests": metrics.requests - request_start, "retrieval_provider": "authenticated_blockscout_and_ethereum_rpc", "retrieval_timestamp": timestamp}
        status = pd.concat([status, pd.DataFrame([record])], ignore_index=True).drop_duplicates("address", keep="last")
        processed += 1

        immediate_stop = category in {"AUTH_FAILURE", "PROVIDER_SCHEMA_FAILURE"}
        sustained_stop = category in {"RATE_LIMITED", "RPC_FAILURE", "IDENTITY_MISMATCH"} and consecutive_provider_failures >= int(cfg["provider"]["sustained_failure_count"])
        checkpoint_due = processed % checkpoint_interval == 0 or processed == len(todo) or immediate_stop or sustained_stop
        progress_due = processed % 5 == 0 or checkpoint_due
        if checkpoint_due:
            _checkpoint_approved_resume(out, candidates, enriched, status)
        if progress_due:
            elapsed = time.perf_counter() - started
            remaining = len(todo) - processed
            seconds_per_address = elapsed / processed if processed else 0.0
            projected_total = (prior_gate_seconds + elapsed + remaining * seconds_per_address) / 60
            progress = {"processed_this_resume": processed, "remaining_this_resume": remaining, "successful_total": len(enriched), "elapsed_minutes": round(elapsed / 60, 3), "projected_total_minutes": round(projected_total, 3), "requests": metrics.requests, "retries": metrics.retries}
            print(json.dumps(progress), flush=True)
            if processed >= checkpoint_interval and projected_total > approved_total_minutes:
                stop_category = "RUNTIME_LIMIT_EXCEEDED"
                stop_reason = f"Projected total runtime {projected_total:.2f} minutes exceeds approved {approved_total_minutes:.2f} minutes"
                break
        if immediate_stop or sustained_stop:
            stop_category = category
            stop_reason = "Provider health degraded during approved continuation"
            break

    _checkpoint_approved_resume(out, candidates, enriched, status)
    full_status = pd.read_csv(out / "enrichment_status.csv", keep_default_na=False)
    elapsed = time.perf_counter() - started
    actual_total_minutes = (prior_gate_seconds + elapsed) / 60
    completed_count = len(full_status) - int(full_status.status.eq("NOT_ATTEMPTED").sum())
    complete = stop_category is None and completed_count == len(candidates)

    runtime_record = {
        "approved_total_minutes": approved_total_minutes,
        "prior_bounded_gate_minutes": prior_gate_seconds / 60,
        "approved_resume_minutes": elapsed / 60,
        "actual_total_minutes": actual_total_minutes,
        "starting_successful_rows": starting_successes,
        "processed_this_resume": processed,
        "remaining_unattempted": int(full_status.status.eq("NOT_ATTEMPTED").sum()),
        "requests_this_resume": metrics.requests,
        "retries_this_resume": metrics.retries,
        "http_status_distribution": metrics.http_statuses,
        "average_request_latency_seconds": statistics.fmean(metrics.latencies) if metrics.latencies else None,
        "stopped_category": stop_category,
    }
    _json_write(out / "full_enrichment_runtime.json", runtime_record)

    if complete:
        final_counts = _finalize_approved_track_b(enriched, full_status, candidates, cfg, out)
        if final_counts["sealed_ready"]:
            decision = "A"
            decision_label = "TRACK B READY — PROCEED TO PHASE 9B"
            next_phase = "phase9b"
        elif final_counts["track_b_rating"] in {"STRONG", "MODERATE"} and final_counts["manual_review_required"]:
            decision = "B"
            decision_label = "TRACK B PROMISING — BOUNDED PHASE 8R6D MANUAL REVIEW REQUIRED"
            next_phase = "phase8r6d_manual_review"
        else:
            decision = "C"
            decision_label = "TRACK B INADEQUATE — PROCEED TO PHASE 9B WITHOUT TRACK B"
            next_phase = "phase9b_without_track_b"
        status_name = "complete"
    else:
        final_counts = {
            **_readiness_counts(enriched),
            "invalid_contract_count": int(full_status.status.eq("INVALID_CONTRACT").sum()),
            "failed_count": int((~full_status.status.isin(["SUCCESS", "NOT_ATTEMPTED"])).sum()),
            "later_positive_count": 0,
            "later_benign_count": 0,
            "strict_independent_positive_count": 0,
            "strict_independent_benign_count": 0,
            "provisional_independent_positive_count": 0,
            "provisional_independent_benign_count": 0,
            "track_b_strict_positive_count": 0,
            "track_b_strict_benign_count": 0,
            "track_b_provisional_positive_count": 0,
            "track_b_provisional_benign_count": 0,
            "track_b_rating": "NONE",
            "track_b_established": False,
            "sealed_ready": False,
            "manual_review_required": False,
            "critical_manual_review_count": 0,
        }
        decision = "D"
        decision_label = "PROVIDER/RUNTIME FAILURE — RESUME LATER"
        next_phase = "phase8r6c_provider_resume"
        status_name = "stopped"

    result = {
        "status": status_name,
        "decision": decision,
        "decision_label": decision_label,
        "stop_reason": stop_reason,
        "models_trained": False,
        "detectors_evaluated": False,
        "p5_evaluated": False,
        "track_a_modified": False,
        "candidate_count": len(candidates),
        "enriched_count": len(enriched),
        **final_counts,
        "actual_runtime_minutes": actual_total_minutes,
        "next_recommended_phase": next_phase,
        "parent_results_preserved": True,
    }
    _json_write(out / "full_enrichment_summary.json", result)
    report_lines = [
        "# Phase 8R6C final provider-resume report",
        "",
        "This continuation constructed and audited external static evaluation data only. No detector was trained, scored, or evaluated, and P5 was not evaluated.",
        "",
        f"Decision: **{decision}. {decision_label}**",
        "",
        f"Candidates: {len(candidates)}. Successful enrichments: {len(enriched)}. Invalid contracts: {final_counts['invalid_contract_count']}. Failed rows: {final_counts['failed_count']}.",
        f"Actual cumulative runtime: {actual_total_minutes:.2f} minutes. MULCAS_READY: {final_counts['mulcas_ready_count']}. STATIC_RUNTIME_READY: {final_counts['static_runtime_ready_count']}. CREATION_CODE_READY: {final_counts['creation_code_ready_count']}.",
        f"Later positives: {final_counts['later_positive_count']}. Later benigns: {final_counts['later_benign_count']}. Track B rating: {final_counts['track_b_rating']}. Sealed ready: {final_counts['sealed_ready']}.",
        "",
        "Track A remains separate and hash-verified. Provider credentials were neither printed nor serialized. Phase 9B was not started.",
    ]
    (out / "phase8r6c_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    parent_hashes_after = {name: sha256_file(parent / name) for name in PARENT_FILES}
    if parent_hashes_before != parent_hashes_after:
        raise RuntimeError("Frozen Phase 8R6C evidence changed during approved continuation")
    if sha256_file(raw_path) != expected_raw_hash:
        raise RuntimeError("Track A changed during approved continuation")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
