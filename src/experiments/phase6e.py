"""Controlled Logistic Regression and LightGBM experiments for Phase 6E."""

from __future__ import annotations

import argparse, itertools, json, platform, time, warnings
from datetime import UTC,datetime
from pathlib import Path
from typing import Any

import gensim, lightgbm, numpy as np, pandas as pd, scipy, sklearn, yaml
from scipy.sparse import csr_matrix,hstack,issparse
from sklearn.exceptions import ConvergenceWarning

from src.evaluation.metrics import binary_classification_metrics
from src.experiments.phase6a import _git_commit,fold_metadata,load_development_data,sha256_file,verify_frozen_manifest
from src.experiments.temporal_cv import make_positive_occurrence_folds
from src.features.developer import ChronologicalCreatorTransformer
from src.features.opcode import make_opcode_term_count_vectorizer,make_opcode_tfidf_vectorizer,make_paper_aligned_ngram_count_vectorizer,make_paper_aligned_ngram_tfidf_vectorizer
from src.features.word2vec import IDFWeightedOpcodeWord2VecTransformer
from src.models.alternative import build_alternative_model
from src.models.baselines import positive_class_scores
from src.utils.paths import REPO_ROOT
from src.utils.seed import set_random_seed

DEFAULT_CONFIG_PATH=REPO_ROOT/"configs/experiments/phase6e/alternative_classifiers.yaml"

def _resolve(value:str)->Path:
    path=Path(value); return path if path.is_absolute() else REPO_ROOT/path

def load_config(path:Path=DEFAULT_CONFIG_PATH)->dict[str,Any]:
    config=yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("dataset","split_membership","paper_feature_config","paper_model_config","paper_mulcas_config","frozen_phase55_manifest","frozen_phase6a_manifest","frozen_phase6b_manifest","frozen_phase6c_manifest","frozen_phase6d1_manifest","frozen_phase6d2_manifest","output_dir"): config[key]=_resolve(config[key])
    config["config_sha256"]=sha256_file(path)
    if config["selection"]["final_test_usage"]!="prohibited": raise ValueError("Final-test usage must be prohibited")
    return config

def environment_versions()->dict[str,str]:
    return {"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"gensim":gensim.__version__,"lightgbm":lightgbm.__version__}

def build_feature_pair(feature:str,train:pd.DataFrame,later:pd.DataFrame,feature_config:dict[str,Any])->tuple[csr_matrix,csr_matrix,dict[str,Any]]:
    train_docs=train.opcode.astype(str).tolist(); later_docs=later.opcode.astype(str).tolist(); min_df=int(feature_config["ngram"]["min_df"])
    factories={"opcode_term_count":lambda:make_opcode_term_count_vectorizer(),"opcode_tfidf":lambda:make_opcode_tfidf_vectorizer(),"opcode_ngram_count_1_3_df_filtered":lambda:make_paper_aligned_ngram_count_vectorizer(min_df),"opcode_ngram_tfidf_1_3_df_filtered":lambda:make_paper_aligned_ngram_tfidf_vectorizer(min_df)}
    if feature in factories:
        transformer=factories[feature](); train_static=transformer.fit_transform(train_docs).tocsr(); later_static=transformer.transform(later_docs).tocsr(); vocabulary_size=len(transformer.vocabulary_)
    elif feature=="opcode_word2vec_idf_weighted":
        values=feature_config["word2vec"]; transformer=IDFWeightedOpcodeWord2VecTransformer(vector_size=int(values["vector_size"]),window=int(values["window"]),min_count=int(values["min_count"]),epochs=int(values["epochs"]),seed=int(values["seed"]),workers=int(values["workers"])); train_static=csr_matrix(transformer.fit_transform(train_docs)); later_static=csr_matrix(transformer.transform(later_docs)); vocabulary_size=transformer.vocabulary_size_
    else: raise ValueError(f"Unauthorized Phase 6E feature: {feature}")
    creator=ChronologicalCreatorTransformer(); train_creator=csr_matrix(creator.fit_transform(train)); later_creator=csr_matrix(creator.transform(later[["creator","createdBlockNumber"]])); train_x=hstack([train_static,train_creator],format="csr"); later_x=hstack([later_static,later_creator],format="csr")
    return train_x,later_x,{"feature_fit_rows":len(train),"feature_fit_block_max":int(train.createdBlockNumber.max()),"evaluation_block_min":int(later.createdBlockNumber.min()),"vocabulary_size":vocabulary_size,"combined_dimension":train_x.shape[1],"matrix_type":"scipy.sparse.csr_matrix"}

def candidate_parameters(config:dict[str,Any])->dict[str,list[dict[str,Any]]]:
    logistic=[{**config["logistic_regression"]["fixed"],"C":float(c)} for c in config["logistic_regression"]["C"]]
    light=[{**config["lightgbm"]["fixed"],"num_leaves":int(leaves),"learning_rate":float(rate),"n_estimators":int(trees)} for leaves,rate,trees in itertools.product(config["lightgbm"]["num_leaves"],config["lightgbm"]["learning_rate"],config["lightgbm"]["n_estimators"])]
    return {"logistic_regression":logistic,"lightgbm":light}

def evaluate(family:str,parameters:dict[str,Any],seed:int,train_x:Any,train_y:np.ndarray,later_x:Any,later_y:np.ndarray)->tuple[dict[str,Any],dict[str,Any]]:
    model=build_alternative_model(family,parameters,seed); fit_start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",ConvergenceWarning); model.fit(train_x,train_y)
    fit_runtime=time.perf_counter()-fit_start; predict_start=time.perf_counter(); predictions=np.asarray(model.predict(later_x),dtype=int); scores=positive_class_scores(model,later_x); prediction_runtime=time.perf_counter()-predict_start
    convergence=[str(item.message) for item in caught if issubclass(item.category,ConvergenceWarning)]; iterations=int(np.max(model.n_iter_)) if family=="logistic_regression" else int(parameters["n_estimators"])
    return binary_classification_metrics(later_y,predictions,scores),{"fit_runtime_seconds":fit_runtime,"prediction_runtime_seconds":prediction_runtime,"convergence_status":"converged" if not convergence else "warning","convergence_warnings":convergence,"iteration_count":iterations}

def _aggregate(runs:list[dict[str,Any]])->dict[str,float]:
    output={}
    for metric in ("precision","recall","f1","pr_auc","balanced_accuracy"):
        values=np.asarray([r[metric] for r in runs],dtype=float); output[f"cv_{metric}_mean"]=float(values.mean()); output[f"cv_{metric}_std"]=float(values.std())
    return output

def run_phase6e(config:dict[str,Any])->dict[str,Any]:
    started=time.perf_counter(); seed=int(config["seed"]); set_random_seed(seed); versions=environment_versions(); manifests={name:verify_frozen_manifest(config[f"frozen_{name}_manifest"]) for name in ("phase55","phase6a","phase6b","phase6c","phase6d1","phase6d2")}
    partitions=load_development_data(config["dataset"],config["split_membership"]); training,validation=partitions["train"],partitions["validation"]; indexed=training.set_index("row_index",drop=False); folds=make_positive_occurrence_folds(training,config["temporal_cv"]["train_positive_ends"],config["temporal_cv"]["validation_positive_ends"]); fold_defs=[fold_metadata(f,indexed) for f in folds]; feature_config=yaml.safe_load(config["paper_feature_config"].read_text())
    cache={}
    for fold in folds:
        train=indexed.loc[list(fold.train_indices)].reset_index(drop=True); later=indexed.loc[list(fold.validation_indices)].reset_index(drop=True)
        for feature in config["feature_representations"]: cache[(fold.fold,feature)]=(train,later,*build_feature_pair(feature,train,later,feature_config))
    common={"hypothesis":config["hypothesis"],"git_commit":_git_commit(),"random_seed":seed,"dataset_sha256":sha256_file(config["dataset"]),"split_config_sha256":manifests["phase55"]["configs/paper_aligned_split.yaml"],"feature_config_sha256":manifests["phase55"]["configs/paper_aligned_features.yaml"],"experiment_config_sha256":config["config_sha256"],"library_versions":versions}
    records=[]; selected_by_family={}; fallbacks={}
    for family,parameter_sets in candidate_parameters(config).items():
        family_records=[]
        for feature in config["feature_representations"]:
            for candidate_index,parameters in enumerate(parameter_sets):
                experiment_id=f"phase6e_{family}_{feature}_{candidate_index:02d}"; runs=[]
                for fold in folds:
                    train,later,train_x,later_x,audit=cache[(fold.fold,feature)]; metrics,timing=evaluate(family,parameters,seed,train_x,train.label.to_numpy(int),later_x,later.label.to_numpy(int)); meta=fold_defs[fold.fold-1]; positives=int(train.label.sum()); cv_positives=int(later.label.sum())
                    runs.append({"experiment_id":experiment_id,"classifier_family":family,"feature_representation":feature,"hyperparameters":parameters,"fold_id":fold.fold,"training_rows":len(train),"training_positive":positives,"training_negative":len(train)-positives,"cv_rows":len(later),"cv_positive":cv_positives,"cv_negative":len(later)-cv_positives,"train_block_range":[meta["train_block_min"],meta["train_block_max"]],"cv_block_range":[meta["validation_block_min"],meta["validation_block_max"]],**audit,**metrics,**timing,**common})
                record={"experiment_id":experiment_id,"classifier_family":family,"feature_representation":feature,"hyperparameters":parameters,"temporal_cv_fold_metrics":runs,**_aggregate(runs),"eligible":False,"selected_by_temporal_cv":False,"selection_reason":None}; records.append(record); family_records.append(record)
        floor=float(config["selection"]["minimum_cv_precision"]); eligible=[r for r in family_records if r["cv_precision_mean"]>=floor]
        for r in eligible:r["eligible"]=True
        pool=eligible or family_records
        def complexity(r:dict[str,Any])->tuple[float,...]:
            p=r["hyperparameters"]; return (-float(p["C"]),) if family=="logistic_regression" else (-float(p["num_leaves"]),-float(p["n_estimators"]))
        selected=max(pool,key=lambda r:(r["cv_f1_mean"],r["cv_recall_mean"],r["cv_balanced_accuracy_mean"],*complexity(r),r["feature_representation"])); fallback=not bool(eligible); selected["selected_by_temporal_cv"]=True; selected["selection_reason"]="maximum mean temporal-CV F1 subject to precision >= 0.80; ties recall, balanced accuracy, lower complexity" if not fallback else "diagnostic fallback: no candidate met mean temporal-CV precision >= 0.80"
        for r in family_records:
            if not r["selected_by_temporal_cv"]:r["selection_reason"]="not selected by predefined temporal-CV criterion"
        selected_by_family[family]=selected["experiment_id"]; fallbacks[family]=fallback
    full_cache={feature:build_feature_pair(feature,training,validation,feature_config) for feature in config["feature_representations"]}; validations=[]
    for selected in [r for r in records if r["selected_by_temporal_cv"]]:
        train_x,val_x,audit=full_cache[selected["feature_representation"]]; metrics,timing=evaluate(selected["classifier_family"],selected["hyperparameters"],seed,train_x,training.label.to_numpy(int),val_x,validation.label.to_numpy(int)); reference=config["reference_validation"]; deltas={key:metrics[key]-reference[key] for key in ("precision","recall","f1","pr_auc","balanced_accuracy","tn","fp","fn","tp")}; validations.append({"experiment_id":selected["experiment_id"],"classifier_family":selected["classifier_family"],"feature_representation":selected["feature_representation"],"hyperparameters":selected["hyperparameters"],**audit,**metrics,"metric_deltas":deltas,**timing})
    supported=any(not fallbacks[v["classifier_family"]] and v["precision"]>=.8 and v["f1"]>float(config["reference_validation"]["f1"]) for v in validations)
    return {"generated_at_utc":datetime.now(UTC).isoformat(),"phase":"6E","hypothesis":config["hypothesis"],"protocol":{"intervention":"classifier family only","features":"frozen Phase 5.5 views only","sampling":"none","class_weighting":"none","thresholds":"native defaults","final_test_usage":"prohibited","mulcas_unchanged":True},"provenance":{**common,"frozen_manifests":manifests},"environment_versions":versions,"temporal_cv_folds":fold_defs,"records":records,"selected_by_family":selected_by_family,"cv_precision_constraint_fallback":fallbacks,"validation_records":validations,"reference_validation":config["reference_validation"],"h6e_supported":supported,"total_runtime_seconds":time.perf_counter()-started}

def save_results(results:dict[str,Any],output_dir:Path)->dict[str,Path]:
    output_dir.mkdir(parents=True,exist_ok=True); names={"json":"phase6e_results.json","cv":"phase6e_cv_runs.csv","candidates":"phase6e_candidates.csv","validation":"phase6e_validation.csv","report":"phase6e_report.md","folds":"temporal_cv_folds.json","versions":"environment_versions.json"}; paths={k:output_dir/v for k,v in names.items()}; paths["json"].write_text(json.dumps(results,indent=2),encoding="utf-8"); pd.DataFrame([x for r in results["records"] for x in r["temporal_cv_fold_metrics"]]).to_csv(paths["cv"],index=False); pd.DataFrame([{k:v for k,v in r.items() if k!="temporal_cv_fold_metrics"} for r in results["records"]]).to_csv(paths["candidates"],index=False); pd.DataFrame([{**{k:v for k,v in r.items() if k!="metric_deltas"},**{f"delta_{k}":v for k,v in r["metric_deltas"].items()}} for r in results["validation_records"]]).to_csv(paths["validation"],index=False); paths["folds"].write_text(json.dumps(results["temporal_cv_folds"],indent=2),encoding="utf-8"); paths["versions"].write_text(json.dumps(results["environment_versions"],indent=2),encoding="utf-8")
    rows=["| Classifier | Feature | CV precision | CV recall | CV F1 | Validation precision | Validation recall | Validation F1 |","| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"] ; by={v["experiment_id"]:v for v in results["validation_records"]}
    for r in results["records"]:
        if r["selected_by_temporal_cv"]:
            v=by[r["experiment_id"]]; rows.append(f"| {r['classifier_family']} | {r['feature_representation']} | {r['cv_precision_mean']:.6f} | {r['cv_recall_mean']:.6f} | {r['cv_f1_mean']:.6f} | {v['precision']:.6f} | {v['recall']:.6f} | {v['f1']:.6f} |")
    paths["report"].write_text("# Phase 6E: Selected Alternative Classifiers\n\nOnly Logistic Regression and LightGBM were tested against frozen Phase 5.5 views. Final-test rows were not loaded.\n\n"+"\n".join(rows)+f"\n\n- LightGBM version: `{results['environment_versions']['lightgbm']}`\n- H6E supported: `{str(results['h6e_supported']).lower()}`\n",encoding="utf-8"); return paths

def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG_PATH); args=parser.parse_args(); config=load_config(args.config)
    for name,path in save_results(run_phase6e(config),config["output_dir"]).items():print(f"Wrote {name}: {path}")
if __name__=="__main__":main()
