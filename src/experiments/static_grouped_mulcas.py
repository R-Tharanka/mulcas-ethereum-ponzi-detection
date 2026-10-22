"""Duplicate-group-safe successor to the exploratory random-split ensemble.

The historical ``scripts/static_random_mulcas.py`` result is intentionally not
modified. This module groups rows connected by normalized address or exact
normalized opcode sequence before creating fit, validation, test, and CV folds.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer

from scripts.static_random_mulcas import (
    MeanOpcodeWord2Vec,
    best_f1_threshold,
    ensemble_probabilities,
    evaluate,
    make_vectorizer,
    make_xgb_pipeline,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "raw" / "PonziDataset_20221114.csv"
HISTORICAL_RESULT = ROOT / "results" / "experiments" / "static_random_mulcas_results.json"
RESULT_DIR = ROOT / "results" / "experiments" / "static_grouped_mulcas"
EXPECTED_DATASET_SHA256 = "48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf"
EXPECTED_HISTORICAL_SHA256 = "4f8bd7d3c028317495fa7b8a3fa0efed1247f2b8f5c12287cf93192372204916"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_address(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def normalize_opcode(value: object) -> str:
    return " ".join(str(value).split())


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            smaller, larger = sorted((left_root, right_root))
            self.parent[larger] = smaller


def add_duplicate_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign connected groups using address OR exact opcode equality."""
    output = frame.copy()
    output["normalized_address"] = output["address"].map(normalize_address)
    output["normalized_opcode"] = output["opcode"].map(normalize_opcode)
    output["opcode_sha256"] = output["normalized_opcode"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )

    union_find = _UnionFind(len(output))
    first_address: dict[str, int] = {}
    first_opcode: dict[str, int] = {}
    for position, row in enumerate(output.itertuples(index=False)):
        if row.normalized_address:
            previous = first_address.setdefault(row.normalized_address, position)
            union_find.union(position, previous)
        previous = first_opcode.setdefault(row.opcode_sha256, position)
        union_find.union(position, previous)

    members: dict[int, list[int]] = {}
    for position in range(len(output)):
        members.setdefault(union_find.find(position), []).append(position)
    canonical = {
        position: min(group_members)
        for group_members in members.values()
        for position in group_members
    }
    output["duplicate_group"] = [f"G{canonical[i]:06d}" for i in range(len(output))]
    return output.drop(columns=["normalized_opcode"])


def load_grouped_data(path: Path = DATASET) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["address", "opcode", "label"], low_memory=False)
    frame.insert(0, "row_index", np.arange(len(frame), dtype=int))
    frame["opcode"] = frame["opcode"].fillna("").astype(str)
    if frame["opcode"].map(normalize_opcode).eq("").any():
        raise ValueError("Empty opcode sequences were found.")
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Labels must be 0 and 1.")
    return add_duplicate_groups(frame)


def _select_group_fold(frame: pd.DataFrame, n_splits: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Choose the most size/prevalence-balanced fold without model outcomes."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    target_rows = len(frame) / n_splits
    prevalence = float(frame.label.mean())
    candidates = []
    for fold, (train_idx, holdout_idx) in enumerate(
        splitter.split(frame.opcode, frame.label, groups=frame.duplicate_group)
    ):
        holdout = frame.iloc[holdout_idx]
        objective = (
            abs(len(holdout) - target_rows) / len(frame)
            + abs(float(holdout.label.mean()) - prevalence)
        )
        candidates.append((objective, fold, train_idx, holdout_idx))
    _, _, train_idx, holdout_idx = min(candidates, key=lambda item: (item[0], item[1]))
    return np.asarray(train_idx), np.asarray(holdout_idx)


def grouped_split(frame: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    development_idx, test_idx = _select_group_fold(frame, 5, seed)
    development = frame.iloc[development_idx].copy()
    fit_local, validation_local = _select_group_fold(development, 5, seed + 1)
    fit = development.iloc[fit_local].copy()
    validation = development.iloc[validation_local].copy()
    partitions = {
        "development": development,
        "fit": fit,
        "validation": validation,
        "test": frame.iloc[test_idx].copy(),
    }
    assert_group_isolation(partitions["fit"], partitions["validation"])
    assert_group_isolation(partitions["development"], partitions["test"])
    return {name: part.reset_index(drop=True) for name, part in partitions.items()}


def assert_group_isolation(left: pd.DataFrame, right: pd.DataFrame) -> None:
    for column in ("duplicate_group", "normalized_address", "opcode_sha256"):
        left_values = set(left[column])
        right_values = set(right[column])
        if column == "normalized_address":
            left_values.discard("")
            right_values.discard("")
        overlap = left_values & right_values
        if overlap:
            raise AssertionError(f"{column} crosses split boundary: {len(overlap)} values")


def grouped_cv_splits(frame: pd.DataFrame, seed: int = 42, n_splits: int = 3):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(frame.opcode, frame.label, groups=frame.duplicate_group))
    for train_idx, test_idx in splits:
        assert not (
            set(frame.iloc[train_idx].duplicate_group)
            & set(frame.iloc[test_idx].duplicate_group)
        )
    return splits


def tune_grouped_xgb(kind: str, fit: pd.DataFrame, seed: int, max_features: int, workers: int):
    cv = grouped_cv_splits(fit, seed)
    search = GridSearchCV(
        make_xgb_pipeline(kind, seed, max_features, workers),
        {
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.1],
            "model__n_estimators": [100, 200],
        },
        scoring="f1",
        cv=cv,
        n_jobs=1,
        refit=True,
        verbose=1,
        error_score="raise",
    )
    search.fit(fit.opcode, fit.label.to_numpy())
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def make_grouped_ridge(fit: pd.DataFrame, seed: int, max_features: int):
    features = FeatureUnion(
        [
            ("tfidf", make_vectorizer("tfidf", max_features)),
            (
                "word2vec",
                Pipeline(
                    [
                        ("embeddings", MeanOpcodeWord2Vec(seed=seed)),
                        ("normalize", Normalizer()),
                    ]
                ),
            ),
        ]
    )
    base = ImbPipeline(
        [
            ("features", features),
            ("smote", SMOTE(random_state=seed, k_neighbors=5)),
            ("model", RidgeClassifier(alpha=1.0)),
        ]
    )
    return CalibratedClassifierCV(
        estimator=base,
        method="sigmoid",
        cv=grouped_cv_splits(fit, seed),
    )


def grouping_summary(frame: pd.DataFrame) -> dict:
    sizes = frame.groupby("duplicate_group").size()
    label_counts = frame.groupby("duplicate_group").label.nunique()
    return {
        "groups": int(len(sizes)),
        "multirow_groups": int((sizes > 1).sum()),
        "rows_in_multirow_groups": int(sizes[sizes > 1].sum()),
        "largest_group_rows": int(sizes.max()),
        "mixed_label_groups": int((label_counts > 1).sum()),
    }


def partition_counts(partitions: dict[str, pd.DataFrame]) -> dict:
    return {
        name: {
            "rows": int(len(part)),
            "ponzi": int(part.label.sum()),
            "benign": int(len(part) - part.label.sum()),
            "groups": int(part.duplicate_group.nunique()),
        }
        for name, part in partitions.items()
    }


def write_membership(partitions: dict[str, pd.DataFrame], path: Path) -> None:
    memberships = []
    for partition in ("fit", "validation", "test"):
        part = partitions[partition].copy()
        part["partition"] = partition
        memberships.append(part[
            ["row_index", "address", "label", "duplicate_group", "opcode_sha256", "partition"]
        ])
    pd.concat(memberships, ignore_index=True).sort_values("row_index").to_csv(path, index=False)


def run(
    dataset: Path = DATASET,
    output_dir: Path = RESULT_DIR,
    seed: int = 42,
    workers: int = 4,
    max_features: int = 20_000,
    split_audit_only: bool = False,
) -> dict:
    dataset_hash = sha256_file(dataset)
    historical_hash_before = sha256_file(HISTORICAL_RESULT)
    if dataset_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Track A dataset hash mismatch; refusing to continue.")
    if historical_hash_before != EXPECTED_HISTORICAL_SHA256:
        raise RuntimeError("Historical 5ed8509 result hash mismatch; refusing to continue.")

    started = time.perf_counter()
    frame = load_grouped_data(dataset)
    partitions = grouped_split(frame, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_membership(partitions, output_dir / "split_membership.csv")

    base = {
        "experiment": "duplicate_group_safe_random_static_opcode_three_viewer",
        "status": "split_audit_complete" if split_audit_only else "complete",
        "successor_to_commit": "5ed8509613838f8e43268e075b58753f77c60e80",
        "historical_result": {
            "path": str(HISTORICAL_RESULT.relative_to(ROOT)),
            "sha256_before": historical_hash_before,
            "preserved": True,
        },
        "dataset_sha256": dataset_hash,
        "seed": seed,
        "group_policy": "connected components over normalized address OR exact normalized opcode",
        "split_policy": "5-fold stratified-group candidates; deterministic size/prevalence objective",
        "cv_policy": "3-fold StratifiedGroupKFold over fit duplicate groups",
        "grouping": grouping_summary(frame),
        "split_counts": partition_counts(partitions),
        "boundary_overlap": {
            "fit_validation_groups": 0,
            "fit_validation_addresses": 0,
            "fit_validation_opcodes": 0,
            "development_test_groups": 0,
            "development_test_addresses": 0,
            "development_test_opcodes": 0,
        },
        "limitations": [
            "Random grouped evaluation is not forward temporal evaluation.",
            "Exact address/opcode grouping does not establish fuzzy semantic family isolation.",
            "The three-view probability ensemble is MulCas-style, not the published MulCas cascade.",
        ],
    }
    if split_audit_only:
        base["runtime_seconds"] = time.perf_counter() - started
        (output_dir / "split_audit.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
        return base

    fit, validation, test = (
        partitions["fit"], partitions["validation"], partitions["test"]
    )
    count_model, count_params, count_cv = tune_grouped_xgb(
        "count", fit, seed, max_features, workers
    )
    tfidf_model, tfidf_params, tfidf_cv = tune_grouped_xgb(
        "tfidf", fit, seed, max_features, workers
    )
    ridge_model = make_grouped_ridge(fit, seed, max_features)
    ridge_model.fit(fit.opcode, fit.label.to_numpy())
    models = [count_model, tfidf_model, ridge_model]

    validation_scores = ensemble_probabilities(models, validation.opcode)
    threshold, validation_f1 = best_f1_threshold(validation.label.to_numpy(), validation_scores)
    test_scores = ensemble_probabilities(models, test.opcode)
    metrics = evaluate(test.label.to_numpy(), test_scores, threshold)
    predictions = (test_scores >= threshold).astype(int)
    prediction_table = test[
        ["row_index", "address", "label", "duplicate_group", "opcode_sha256"]
    ].copy()
    prediction_table["score"] = test_scores
    prediction_table["prediction"] = predictions
    prediction_table.to_csv(output_dir / "test_predictions.csv", index=False)

    historical = json.loads(HISTORICAL_RESULT.read_text(encoding="utf-8"))
    base.update(
        {
            "models": {
                "xgboost_count": {"best_params": count_params, "cv_f1": count_cv},
                "xgboost_tfidf": {"best_params": tfidf_params, "cv_f1": tfidf_cv},
                "ridge": {"alpha": 1.0, "calibration": "sigmoid_grouped_3fold"},
            },
            "validation_selected_threshold": threshold,
            "validation_f1": validation_f1,
            "test": metrics,
            "historical_row_random_test": historical["test"],
            "group_safe_minus_historical": {
                metric: float(metrics[source] - historical["test"][source])
                for metric, source in (
                    ("precision", "precision"),
                    ("recall", "recall"),
                    ("f1", "f1"),
                    ("average_precision", "average_precision"),
                )
            },
            "runtime_seconds": time.perf_counter() - started,
            "environment": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
    )
    base["historical_result"]["sha256_after"] = sha256_file(HISTORICAL_RESULT)
    base["historical_result"]["preserved"] = (
        base["historical_result"]["sha256_after"] == historical_hash_before
    )
    if not base["historical_result"]["preserved"]:
        raise RuntimeError("Historical result changed during successor experiment.")
    (output_dir / "static_grouped_mulcas_results.json").write_text(
        json.dumps(base, indent=2), encoding="utf-8"
    )
    return base

