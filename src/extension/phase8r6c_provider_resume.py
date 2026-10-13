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


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
