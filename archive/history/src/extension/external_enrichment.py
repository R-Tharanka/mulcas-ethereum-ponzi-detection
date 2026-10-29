"""Candidate construction, resumable ledger, and deterministic preflight."""
from __future__ import annotations

import platform, sys, time
from pathlib import Path
import pandas as pd

from .external_normalization import normalize_address
from .external_rpc import retrieve_with_retry


def candidate_rows(internal_path: Path, ponzi_ml_path: Path, unica_path: Path) -> pd.DataFrame:
    internal={normalize_address(x) for x in pd.read_csv(internal_path).address}
    ml=pd.read_csv(ponzi_ml_path); ml["address"]=ml.address.map(normalize_address)
    ml=ml[~ml.address.isin(internal)].copy()
    # Preserve conflicting duplicate labels as unresolved instead of choosing a row.
    conflicts=set(ml.groupby("address").target.nunique().loc[lambda x:x>1].index)
    ml=ml[~ml.address.isin(conflicts)].sort_values("address").drop_duplicates("address")
    ml=ml[["address","target"]].rename(columns={"target":"label"})
    ml["sources"]="ponzi_ml"; ml["provenance"]="B"
    unica=pd.read_csv(unica_path,header=None,names=["address","name"]);unica["address"]=unica.address.map(normalize_address)
    unica=unica[~unica.address.isin(internal)].sort_values("address").drop_duplicates("address")
    for address in set(unica.address)&set(ml.address):
        ml.loc[ml.address==address,"sources"]="ponzi_ml|blockchain_unica"
        ml.loc[ml.address==address,"provenance"]="A|B"
    extra=unica[~unica.address.isin(set(ml.address))][["address"]].copy();extra["label"]=1;extra["sources"]="blockchain_unica";extra["provenance"]="A"
    return pd.concat([ml,extra],ignore_index=True).sort_values("address").reset_index(drop=True)


def deterministic_sample(candidates: pd.DataFrame, positives: int=10, benign: int=10) -> pd.DataFrame:
    return pd.concat([candidates[candidates.label==1].head(positives),candidates[candidates.label==0].head(benign)]).reset_index(drop=True)


def enrich(candidates: pd.DataFrame, config: dict, ledger_path: Path, limit: int | None=None) -> tuple[pd.DataFrame,dict]:
    ledger_path.parent.mkdir(parents=True,exist_ok=True)
    if ledger_path.exists(): ledger=pd.read_csv(ledger_path,keep_default_na=False)
    else: ledger=pd.DataFrame()
    completed=set(ledger.loc[ledger.status=="success","address"]) if not ledger.empty else set()
    work=candidates[~candidates.address.isin(completed)]
    if limit is not None: work=work.head(limit)
    start=time.perf_counter(); new=[]
    for idx,row in enumerate(work.itertuples(index=False),1):
        data,attempts,error=retrieve_with_retry(row.address,config)
        record={"address":row.address,"status":"success" if data else "failed","attempts":attempts,
                "last_provider":"Blockscout + PublicNode JSON-RPC","last_retrieval_time":config["audit_date"],"error_reason":error or "",
                "successful_fields":"creation_tx|creator|block|timestamp|runtime|creation_code" if data else ""}
        if data: record.update(data)
        new.append(record)
        snapshot=pd.concat([ledger,pd.DataFrame(new)],ignore_index=True).drop_duplicates("address",keep="last")
        snapshot.to_csv(ledger_path,index=False)
        time.sleep(float(config["retry_policy"]["inter_address_delay_seconds"]))
    final=pd.read_csv(ledger_path,keep_default_na=False) if ledger_path.exists() else ledger
    elapsed=time.perf_counter()-start
    return final,{"attempted_this_run":len(new),"elapsed_seconds":elapsed,"average_seconds_per_address":elapsed/len(new) if new else 0,
                  "successful_this_run":sum(r["status"]=="success" for r in new)}


def environment_record() -> dict:
    import numpy,pandas,yaml
    return {"python":sys.version.split()[0],"numpy":numpy.__version__,"pandas":pandas.__version__,"pyyaml":yaml.__version__,
            "operating_system":platform.platform(),"repository_root":str(Path.cwd().resolve())}
