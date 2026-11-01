"""Protocol tests for Phase 6D2."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,issparse

from src.experiments.phase6a import verify_frozen_manifest
from src.experiments.phase6b import baseline_parameters
from src.experiments.phase6d2 import DEFAULT_CONFIG_PATH,build_representation_pair,load_config,load_development_with_operands
from src.experiments.temporal_cv import make_positive_occurrence_folds

def test_temporal_folds_reused_and_final_test_excluded()->None:
    config=load_config(DEFAULT_CONFIG_PATH); partitions=load_development_with_operands(config); assert set(partitions)=={"train","validation"}
    folds=make_positive_occurrence_folds(partitions["train"],config["temporal_cv"]["train_positive_ends"],config["temporal_cv"]["validation_positive_ends"]); prior=json.loads((config["output_dir"].parent/"phase6d1/temporal_cv_folds.json").read_text()); indexed=partitions["train"].set_index("row_index")
    for fold,expected in zip(folds,prior,strict=True):
        train=indexed.loc[list(fold.train_indices)]; later=indexed.loc[list(fold.validation_indices)]; assert train.createdBlockNumber.max()<later.createdBlockNumber.min(); assert len(train)==expected["train_rows"] and len(later)==expected["validation_rows"]

def test_scaler_is_fold_training_only_and_sparse()->None:
    train=pd.DataFrame({"creator":["a","b"],"createdBlockNumber":[1,2],"label":[0,1]}); later=pd.DataFrame({"creator":["c"],"createdBlockNumber":[3]}); baseline=(csr_matrix([[1],[2]]),csr_matrix([[3]])); operand=(csr_matrix(np.ones((2,37))),csr_matrix(np.ones((1,37))*100)); creator=(csr_matrix([[0],[0]]),csr_matrix([[0]])); train_x,later_x,audit=build_representation_pair({"baseline":True,"groups":["A"]},"ridge",train,later,baseline,operand,creator)
    assert issparse(train_x) and issparse(later_x) and audit["scaler_fit_rows"]==2

def test_frozen_settings_and_all_manifests()->None:
    config=load_config(DEFAULT_CONFIG_PATH); assert baseline_parameters("ridge",{"class_weight":None})["class_weight"] is None; assert baseline_parameters("svm",{"class_weight":None})["class_weight"] is None; assert baseline_parameters("xgboost",{})["scale_pos_weight"]==1.0
    expected={"frozen_phase55_manifest":12,"frozen_phase6a_manifest":7,"frozen_phase6b_manifest":9,"frozen_phase6c_manifest":8,"frozen_phase6d1_manifest":10}
    for key,count in expected.items(): assert len(verify_frozen_manifest(config[key]))==count
