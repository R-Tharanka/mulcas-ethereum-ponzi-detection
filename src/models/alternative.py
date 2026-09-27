"""Alternative classifier constructors for the controlled Phase 6E study."""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression


def build_alternative_model(family:str,parameters:dict[str,Any],seed:int)->Any:
    values=dict(parameters)
    if family=="logistic_regression":
        if values.get("class_weight") is not None: raise ValueError("Class weighting is prohibited")
        return LogisticRegression(random_state=seed,**values)
    if family=="lightgbm":
        if values.get("class_weight") is not None or values.get("is_unbalance") is not False or float(values.get("scale_pos_weight",1))!=1: raise ValueError("LightGBM imbalance controls must remain neutral")
        return LGBMClassifier(random_state=seed,**values)
    raise ValueError(f"Unsupported alternative classifier: {family}")
