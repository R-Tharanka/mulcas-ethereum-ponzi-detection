"""Credential-free Ethereum deployment/static metadata clients."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from .external_normalization import normalize_address


def _json_request(url: str, payload: object | None, timeout: float) -> object:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"User-Agent":"mulcas-phase8r6b/1.0", "Content-Type":"application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def blockscout_creation(address: str, endpoint: str, timeout: float) -> dict:
    url = endpoint + "?" + urlencode({"module":"contract","action":"getcontractcreation","contractaddresses":address})
    payload = _json_request(url, None, timeout)
    if not isinstance(payload, dict) or str(payload.get("status")) != "1" or not payload.get("result"):
        raise LookupError(f"creation lookup failed: {payload}")
    row = payload["result"][0]
    if normalize_address(row.get("contractAddress")) != normalize_address(address):
        raise ValueError("provider creation record maps to a different address")
    return row


def rpc_deployment_bundle(address: str, tx_hash: str, block_number: int,
                          endpoint: str, timeout: float) -> dict:
    block_hex = hex(block_number)
    calls = [
        {"jsonrpc":"2.0","method":"eth_getCode","params":[address,"latest"],"id":1},
        {"jsonrpc":"2.0","method":"eth_getTransactionByHash","params":[tx_hash],"id":2},
        {"jsonrpc":"2.0","method":"eth_getTransactionReceipt","params":[tx_hash],"id":3},
        {"jsonrpc":"2.0","method":"eth_getBlockByNumber","params":[block_hex,False],"id":4},
    ]
    payload = _json_request(endpoint, calls, timeout)
    by_id={item["id"]:item.get("result") for item in payload if isinstance(item,dict)}
    if any(by_id.get(i) is None for i in range(1,5)):
        raise LookupError("incomplete JSON-RPC deployment bundle")
    tx,receipt,block=by_id[2],by_id[3],by_id[4]
    if normalize_address(receipt.get("contractAddress")) != normalize_address(address):
        raise ValueError("receipt contract address mismatch")
    blocks={int(value,16) for value in (tx.get("blockNumber"),receipt.get("blockNumber"),block.get("number"))}
    if blocks != {block_number}:
        raise ValueError(f"creation block disagreement: {sorted(blocks)}")
    if normalize_address(tx.get("from")) != normalize_address(receipt.get("from")):
        raise ValueError("creator disagreement between transaction and receipt")
    return {"runtime_bytecode":by_id[1],"transaction":tx,"receipt":receipt,"block":block,
            "timestamp":int(block["timestamp"],16)}


def retrieve_with_retry(address: str, config: dict, sleep=time.sleep) -> tuple[dict | None, int, str | None]:
    retry=config["retry_policy"]; attempts=0; error=None
    while attempts < int(retry["maximum_attempts"]):
        attempts += 1
        try:
            creation=blockscout_creation(address,config["providers"]["creation"]["endpoint"],float(retry["timeout_seconds"]))
            block_number=int(creation["blockNumber"])
            bundle=rpc_deployment_bundle(address,creation["txHash"],block_number,config["providers"]["rpc"]["endpoint"],float(retry["timeout_seconds"]))
            provider_ts=int(creation["timestamp"]); rpc_ts=int(bundle["timestamp"])
            if provider_ts != rpc_ts:
                raise ValueError(f"timestamp disagreement: blockscout={provider_ts},rpc={rpc_ts}")
            tx=bundle["transaction"]
            return {"address":normalize_address(address),"creation_transaction_hash":creation["txHash"].lower(),
              "creator":normalize_address(creation["contractCreator"]),"created_block":block_number,
              "timestamp_unix":rpc_ts,"timestamp":datetime.fromtimestamp(rpc_ts,tz=timezone.utc).isoformat(),
              "runtime_bytecode":bundle["runtime_bytecode"],"creation_code":tx.get("input") or creation.get("creationBytecode"),
              "receipt_address_verified":True,"block_consistency_verified":True,"timestamp_consistency_verified":True,
              "provider":"Blockscout + PublicNode JSON-RPC"},attempts,None
        except (HTTPError,URLError,TimeoutError,LookupError,ValueError,KeyError,TypeError) as exc:
            error=f"{type(exc).__name__}: {exc}"
            if attempts < int(retry["maximum_attempts"]): sleep(float(retry["base_backoff_seconds"])*(2**(attempts-1)))
    return None,attempts,error
