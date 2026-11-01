"""Controlled normalized operand-derived static-feature experiments."""

from __future__ import annotations

import argparse, json, time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import binary_classification_metrics
from src.experiments.phase6a import _git_commit, build_feature_pair, fold_metadata, load_development_data, sha256_file, verify_frozen_manifest
from src.experiments.phase6b import baseline_parameters
from src.experiments.temporal_cv import make_positive_occurrence_folds
from src.features.developer import ChronologicalCreatorTransformer
from src.features.operands import FEATURE_GROUPS, FEATURE_NAMES, GROUPS, disassemble_opcode_names, disassemble_operands, feature_names_for_groups, transform_operand_features
from src.models.baselines import build_baseline_model, positive_class_scores
from src.models.run_baselines import load_baseline_config
from src.utils.paths import REPO_ROOT
from src.utils.seed import set_random_seed

DEFAULT_CONFIG_PATH=REPO_ROOT/"configs/experiments/phase6d2/operand_features.yaml"

def _resolve(value:str)->Path:
    path=Path(value); return path if path.is_absolute() else REPO_ROOT/path

def load_config(path:Path=DEFAULT_CONFIG_PATH)->dict[str,Any]:
    config=yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("dataset","split_membership","paper_feature_config","paper_model_config","paper_mulcas_config","frozen_phase55_manifest","frozen_phase6a_manifest","frozen_phase6b_manifest","frozen_phase6c_manifest","frozen_phase6d1_manifest","output_dir"): config[key]=_resolve(config[key])
    config["config_sha256"]=sha256_file(path)
    if config["selection"]["final_test_usage"]!="prohibited": raise ValueError("Phase 6D2 final-test usage must be prohibited")
    return config

def load_development_with_operands(config:dict[str,Any])->dict[str,pd.DataFrame]:
    partitions=load_development_data(config["dataset"],config["split_membership"])
    source=pd.read_csv(config["dataset"],usecols=[config["operand_source"]]); source["row_index"]=source.index.astype(int)
    return {name:frame.merge(source,on="row_index",how="left",validate="one_to_one") for name,frame in partitions.items()}

def build_operand_audit(config:dict[str,Any])->dict[str,Any]:
    frame=pd.read_csv(config["dataset"],usecols=["opcode",config["operand_source"]])
    widths=Counter(); no_operand=truncated=unsupported=invalid=empty=aligned=0
    for row in frame.itertuples(index=False):
        bytecode=getattr(row,config["operand_source"]); parsed=disassemble_operands(bytecode)
        empty+=parsed.null_or_empty; no_operand+=not parsed.operands; truncated+=parsed.malformed_truncated_push_count; unsupported+=parsed.unsupported_opcode_count
        for operand in parsed.operands: widths[len(operand)]+=1
        aligned+=disassemble_opcode_names(bytecode)==tuple(str(row.opcode).split())
    return {"feasible":aligned==len(frame),"authoritative_operand_source":config["operand_source"],"source_justification":"deployed runtime bytecode; deterministic historical-EVM disassembly exactly matches released opcode names", "contracts_parsed":len(frame),"opcode_alignment_contracts":aligned,"no_operand_contracts":no_operand,"null_or_empty_bytecode_contracts":empty,"invalid_hex_contracts":invalid,"malformed_truncated_push_count":truncated,"unsupported_opcode_count":unsupported,"metadata_removal_policy":config["metadata_policy"],"operand_width_distribution":{str(k):v for k,v in sorted(widths.items())}}

def _creator_pair(train:pd.DataFrame,later:pd.DataFrame)->tuple[csr_matrix,csr_matrix]:
    creator=ChronologicalCreatorTransformer(); return csr_matrix(creator.fit_transform(train)),csr_matrix(creator.transform(later[["creator","createdBlockNumber"]]))

def build_representation_pair(spec:dict[str,Any],family:str,train:pd.DataFrame,later:pd.DataFrame,baseline:tuple[Any,Any],operand:tuple[csr_matrix,csr_matrix],creator:tuple[csr_matrix,csr_matrix])->tuple[csr_matrix,csr_matrix,dict[str,Any]]:
    groups=tuple(spec["groups"]); offsets={}; cursor=0
    for group in GROUPS: offsets[group]=cursor; cursor+=len(FEATURE_GROUPS[group])
    indices=[offsets[group]+i for group in groups for i in range(len(FEATURE_GROUPS[group]))]
    train_operand=operand[0][:,indices] if indices else csr_matrix((len(train),0)); later_operand=operand[1][:,indices] if indices else csr_matrix((len(later),0)); scaler_type=None
    if groups and family in {"ridge","svm"}:
        scaler=StandardScaler(with_mean=False); train_operand=scaler.fit_transform(train_operand).tocsr(); later_operand=scaler.transform(later_operand).tocsr(); scaler_type="StandardScaler(with_mean=False), fold-training fit only"
    if spec["baseline"]:
        train_x,later_x=baseline; baseline_dimension=train_x.shape[1]
        if groups: train_x=hstack([train_x,train_operand],format="csr"); later_x=hstack([later_x,later_operand],format="csr")
    else:
        train_x=hstack([train_operand,creator[0]],format="csr"); later_x=hstack([later_operand,creator[1]],format="csr"); baseline_dimension=0
    if not issparse(train_x) or not issparse(later_x): raise AssertionError("Representations must remain sparse")
    return train_x.tocsr(),later_x.tocsr(),{"included_feature_groups":list(groups),"ordered_operand_feature_names":list(feature_names_for_groups(groups)),"operand_dimension":len(indices),"baseline_dimension":int(baseline_dimension),"combined_dimension":int(train_x.shape[1]),"representation_type":"scipy.sparse.csr_matrix","scaler_type":scaler_type,"scaler_fit_rows":len(train) if scaler_type else 0,"fit_requirements":"operand extractor fit-free; scaler and baseline vocabulary fit on fold-training only"}

def _evaluate(family:str,parameters:dict[str,Any],seed:int,train_x:Any,train_y:np.ndarray,later_x:Any,later_y:np.ndarray)->tuple[dict[str,Any],float]:
    started=time.perf_counter(); model=build_baseline_model(family,parameters,seed); model.fit(train_x,train_y); pred=np.asarray(model.predict(later_x),dtype=int); return binary_classification_metrics(later_y,pred,positive_class_scores(model,later_x)),time.perf_counter()-started

def _aggregate(runs:list[dict[str,Any]])->dict[str,float]:
    output={}
    for metric in ("precision","recall","f1","pr_auc","balanced_accuracy"):
        values=np.asarray([r[metric] for r in runs],dtype=float); output[f"cv_{metric}_mean"]=float(values.mean()); output[f"cv_{metric}_std"]=float(values.std())
    return output

def run_phase6d2(config:dict[str,Any])->dict[str,Any]:
    started=time.perf_counter(); seed=int(config["seed"]); set_random_seed(seed)
    manifests={name:verify_frozen_manifest(config[f"frozen_{name}_manifest"]) for name in ("phase55","phase6a","phase6b","phase6c","phase6d1")}
    audit=build_operand_audit(config)
    if not audit["feasible"]: raise RuntimeError("Operand extraction failed authoritative-source alignment")
    partitions=load_development_with_operands(config); training,validation=partitions["train"],partitions["validation"]; indexed=training.set_index("row_index",drop=False)
    folds=make_positive_occurrence_folds(training,config["temporal_cv"]["train_positive_ends"],config["temporal_cv"]["validation_positive_ends"]); fold_defs=[fold_metadata(f,indexed) for f in folds]
    min_df=int(yaml.safe_load(config["paper_feature_config"].read_text())["ngram"]["min_df"]); frozen=load_baseline_config(config["paper_model_config"])["models"]; parameters={f:baseline_parameters(f,frozen[f]) for f in ("ridge","svm","xgboost")}
    common={"hypothesis":config["hypothesis"],"git_commit":_git_commit(),"random_seed":seed,"dataset_sha256":sha256_file(config["dataset"]),"split_config_sha256":manifests["phase55"]["configs/paper_aligned_split.yaml"],"feature_config_sha256":manifests["phase55"]["configs/paper_aligned_features.yaml"],"experiment_config_sha256":config["config_sha256"]}
    records=[]; selected_by_family={}; fallbacks={}; small=int(config["small_constant_max_inclusive"])
    for family in ("ridge","svm","xgboost"):
        cache={}
        for fold in folds:
            train=indexed.loc[list(fold.train_indices)].reset_index(drop=True); later=indexed.loc[list(fold.validation_indices)].reset_index(drop=True)
            cache[fold.fold]=(train,later,build_feature_pair(config["baseline_features"][family],train,later,ngram_min_df=min_df)[:2],(transform_operand_features(train[config["operand_source"]],small_constant_max=small),transform_operand_features(later[config["operand_source"]],small_constant_max=small)),_creator_pair(train,later))
        family_records=[]
        for spec in config["representations"]:
            runs=[]
            for fold in folds:
                train,later,baseline,operand,creator=cache[fold.fold]; train_x,later_x,details=build_representation_pair(spec,family,train,later,baseline,operand,creator); metrics,elapsed=_evaluate(family,parameters[family],seed,train_x,train.label.to_numpy(int),later_x,later.label.to_numpy(int)); meta=fold_defs[fold.fold-1]
                runs.append({"fold_id":fold.fold,"model_family":family,"representation_id":spec["id"],"train_block_range":[meta["train_block_min"],meta["train_block_max"]],"cv_block_range":[meta["validation_block_min"],meta["validation_block_max"]],**details,**metrics,"runtime_seconds":elapsed})
            record={"experiment_id":f"phase6d2_{family}_{spec['id']}","model_family":family,"representation_id":spec["id"],**common,**runs[0],"temporal_cv_fold_metrics":runs,**_aggregate(runs),"selected_by_temporal_cv":False,"selection_reason":None}; records.append(record); family_records.append(record)
        eligible=[r for r in family_records if r["cv_precision_mean"]>=float(config["selection"]["minimum_cv_precision"])]; pool=eligible or family_records; selected=max(pool,key=lambda r:(r["cv_f1_mean"],r["cv_recall_mean"],r["cv_balanced_accuracy_mean"],-r["operand_dimension"],r["representation_id"])); fallback=not bool(eligible); selected["selected_by_temporal_cv"]=True; selected["selection_reason"]="maximum mean temporal-CV F1 subject to precision >= 0.80; ties recall, balanced accuracy, fewer operand features" if not fallback else "diagnostic fallback: no representation met mean temporal-CV precision >= 0.80"
        for r in family_records:
            if not r["selected_by_temporal_cv"]: r["selection_reason"]="not selected by predefined temporal-CV criterion"
        selected_by_family[family]=selected["representation_id"]; fallbacks[family]=fallback
    validations=[]
    for selected in [r for r in records if r["selected_by_temporal_cv"]]:
        family=selected["model_family"]; spec=next(x for x in config["representations"] if x["id"]==selected["representation_id"]); baseline=build_feature_pair(config["baseline_features"][family],training,validation,ngram_min_df=min_df)[:2]; operand=(transform_operand_features(training[config["operand_source"]],small_constant_max=small),transform_operand_features(validation[config["operand_source"]],small_constant_max=small)); creator=_creator_pair(training,validation); train_x,val_x,details=build_representation_pair(spec,family,training,validation,baseline,operand,creator); metrics,elapsed=_evaluate(family,parameters[family],seed,train_x,training.label.to_numpy(int),val_x,validation.label.to_numpy(int)); reference=config["reference_validation"]; deltas={key:metrics[key]-reference[key] for key in ("precision","recall","f1","pr_auc","balanced_accuracy","tn","fp","fn","tp")}; validations.append({"model_family":family,"representation_id":selected["representation_id"],**details,"frozen_baseline_metrics":reference,"operand_candidate_metrics":metrics,"metric_deltas":deltas,"runtime_seconds":elapsed})
    supported=any(v["operand_candidate_metrics"]["precision"]>=.8 and v["operand_candidate_metrics"]["f1"]>float(config["reference_validation"]["f1"]) for v in validations)
    return {"generated_at_utc":datetime.now(UTC).isoformat(),"phase":"6D2","hypothesis":config["hypothesis"],"protocol":{"intervention":"normalized deployed-runtime PUSH operand features only","creator_feature":"included in every representation","sampling":"none","class_weighting":"none","decision_thresholds":"native defaults","final_test_usage":"prohibited","mulcas_unchanged":True},"provenance":{**common,"frozen_manifests":manifests},"operand_audit":audit,"feature_schema":{"groups":{g:list(FEATURE_GROUPS[g]) for g in GROUPS},"ordered_feature_names":list(FEATURE_NAMES),"total_features":len(FEATURE_NAMES),"small_constant_max_inclusive":small,"raw_operand_identities_emitted":False},"temporal_cv_folds":fold_defs,"records":records,"selected_by_family":selected_by_family,"cv_precision_constraint_fallback":fallbacks,"validation_records":validations,"reference_validation":config["reference_validation"],"h6d2_supported":supported,"total_runtime_seconds":time.perf_counter()-started}

def save_results(results:dict[str,Any],output_dir:Path)->dict[str,Path]:
    output_dir.mkdir(parents=True,exist_ok=True); names={"json":"phase6d2_results.json","cv":"phase6d2_cv_runs.csv","candidates":"phase6d2_candidates.csv","validation":"phase6d2_validation.csv","schema":"phase6d2_feature_schema.json","audit":"phase6d2_operand_audit.json","report":"phase6d2_report.md","folds":"temporal_cv_folds.json"}; paths={k:output_dir/v for k,v in names.items()}; paths["json"].write_text(json.dumps(results,indent=2),encoding="utf-8"); pd.DataFrame([x for r in results["records"] for x in r["temporal_cv_fold_metrics"]]).to_csv(paths["cv"],index=False); pd.DataFrame([{k:v for k,v in r.items() if k!="temporal_cv_fold_metrics"} for r in results["records"]]).to_csv(paths["candidates"],index=False)
    flat=[]
    for r in results["validation_records"]:
        row={k:v for k,v in r.items() if k not in {"frozen_baseline_metrics","operand_candidate_metrics","metric_deltas"}}
        for prefix,key in (("baseline","frozen_baseline_metrics"),("candidate","operand_candidate_metrics"),("delta","metric_deltas")): row.update({f"{prefix}_{name}":value for name,value in r[key].items()})
        flat.append(row)
    pd.DataFrame(flat).to_csv(paths["validation"],index=False); paths["schema"].write_text(json.dumps(results["feature_schema"],indent=2),encoding="utf-8"); paths["audit"].write_text(json.dumps(results["operand_audit"],indent=2),encoding="utf-8"); paths["folds"].write_text(json.dumps(results["temporal_cv_folds"],indent=2),encoding="utf-8")
    rows=["| Model | Selection | CV precision | CV recall | CV F1 | Validation precision | Validation recall | Validation F1 |","| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"] ; by={v["model_family"]:v for v in results["validation_records"]}
    for r in results["records"]:
        if r["selected_by_temporal_cv"]:
            m=by[r["model_family"]]["operand_candidate_metrics"]; rows.append(f"| {r['model_family']} | {r['representation_id']} | {r['cv_precision_mean']:.6f} | {r['cv_recall_mean']:.6f} | {r['cv_f1_mean']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} |")
    a=results["operand_audit"]; paths["report"].write_text("# Phase 6D2: Normalized Operand Features\n\nAuthoritative source: `contractCode` deployed runtime bytecode. Historical-EVM disassembly matches all released opcode sequences. Full runtime bytes are preserved; no new metadata stripping policy was introduced.\n\n"+"\n".join(rows)+f"\n\n- Contracts parsed: {a['contracts_parsed']}\n- Truncated PUSH instructions: {a['malformed_truncated_push_count']}\n- Unsupported opcode bytes: {a['unsupported_opcode_count']}\n- H6D2 supported: `{str(results['h6d2_supported']).lower()}`\n",encoding="utf-8"); return paths

def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG_PATH); args=parser.parse_args(); config=load_config(args.config)
    for name,path in save_results(run_phase6d2(config),config["output_dir"]).items(): print(f"Wrote {name}: {path}")
if __name__=="__main__": main()
