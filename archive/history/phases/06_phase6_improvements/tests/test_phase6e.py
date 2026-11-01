"""Protocol tests for Phase 6E alternative classifiers."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import issparse

from src.experiments.phase6a import load_development_data,verify_frozen_manifest
from src.experiments.phase6e import DEFAULT_CONFIG_PATH,build_feature_pair,candidate_parameters,evaluate,load_config
from src.experiments.temporal_cv import make_positive_occurrence_folds
from src.models.alternative import build_alternative_model

def test_folds_reused_and_final_test_excluded()->None:
    config=load_config(DEFAULT_CONFIG_PATH); parts=load_development_data(config["dataset"],config["split_membership"]); assert set(parts)=={"train","validation"}; folds=make_positive_occurrence_folds(parts["train"],config["temporal_cv"]["train_positive_ends"],config["temporal_cv"]["validation_positive_ends"]); prior=json.loads((config["output_dir"].parent/"phase6d2/temporal_cv_folds.json").read_text()); indexed=parts["train"].set_index("row_index")
    for fold,expected in zip(folds,prior,strict=True):
        train=indexed.loc[list(fold.train_indices)]; later=indexed.loc[list(fold.validation_indices)]; assert train.createdBlockNumber.max()<later.createdBlockNumber.min(); assert len(train)==expected["train_rows"] and len(later)==expected["validation_rows"]

def test_allowed_features_are_exactly_frozen_views_and_fold_local_sparse()->None:
    config=load_config(DEFAULT_CONFIG_PATH); assert config["feature_representations"]==["opcode_term_count","opcode_tfidf","opcode_ngram_count_1_3_df_filtered","opcode_ngram_tfidf_1_3_df_filtered","opcode_word2vec_idf_weighted"]; assert all("structural" not in x and "operand" not in x for x in config["feature_representations"])
    parts=load_development_data(config["dataset"],config["split_membership"]); train=parts["train"].iloc[:100]; later=parts["train"].iloc[100:120]; feature_config=yaml.safe_load(config["paper_feature_config"].read_text()); train_x,later_x,audit=build_feature_pair("opcode_term_count",train,later,feature_config); assert issparse(train_x) and issparse(later_x); assert audit["feature_fit_rows"]==len(train) and audit["feature_fit_block_max"]<=audit["evaluation_block_min"]

def test_candidate_grids_and_imbalance_controls_are_fixed()->None:
    config=load_config(DEFAULT_CONFIG_PATH); candidates=candidate_parameters(config); assert {p["C"] for p in candidates["logistic_regression"]}=={.01,.1,1.,10.}; assert len(candidates["lightgbm"])==8
    for p in candidates["logistic_regression"]: assert p["solver"]=="liblinear" and p["penalty"]=="l2" and p["max_iter"]==2000 and p["class_weight"] is None
    for p in candidates["lightgbm"]: assert p["num_leaves"] in {15,31} and p["learning_rate"] in {.05,.1} and p["n_estimators"] in {100,300} and p["class_weight"] is None and p["scale_pos_weight"]==1 and p["is_unbalance"] is False

def test_default_threshold_models_are_deterministic()->None:
    x=np.asarray([[0.],[1.],[2.],[3.],[4.],[5.]]); y=np.asarray([0,0,0,1,1,1]); later=np.asarray([[.5],[2.5],[4.5]]); truth=np.asarray([0,1,1]); p={"C":1.,"penalty":"l2","solver":"liblinear","max_iter":2000,"class_weight":None,"fit_intercept":True}; first,_=evaluate("logistic_regression",p,42,x,y,later,truth); second,_=evaluate("logistic_regression",p,42,x,y,later,truth); assert first==second
    model=build_alternative_model("lightgbm",{"objective":"binary","class_weight":None,"scale_pos_weight":1.,"is_unbalance":False,"deterministic":True,"force_col_wise":True,"n_jobs":1,"verbosity":-1,"num_leaves":15,"learning_rate":.1,"n_estimators":10},42); assert model.get_params()["class_weight"] is None

def test_all_prior_manifests_unchanged()->None:
    config=load_config(DEFAULT_CONFIG_PATH); expected={"frozen_phase55_manifest":12,"frozen_phase6a_manifest":7,"frozen_phase6b_manifest":9,"frozen_phase6c_manifest":8,"frozen_phase6d1_manifest":10,"frozen_phase6d2_manifest":10}
    for key,count in expected.items(): assert len(verify_frozen_manifest(config[key]))==count
