"""Train and evaluate a random-split static MulCas-style ensemble.

Examples:
    python scripts/static_random_mulcas.py
    python scripts/static_random_mulcas.py --show-comparison-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "raw" / "PonziDataset_20221114.csv"
DEFAULT_OUTPUT = ROOT / "results" / "experiments" / "static_random_mulcas_results.json"
DEFAULT_BASELINE = ROOT / "results" / "baseline" / "paper_aligned" / "mulcas_results.json"


class OpcodeSentences:
    def __init__(self, documents):
        self.documents = documents

    def __iter__(self):
        for document in self.documents:
            yield str(document).split()


class MeanOpcodeWord2Vec(BaseEstimator, TransformerMixin):
    """Train Word2Vec on fit rows and mean-pool opcode vectors."""

    def __init__(self, vector_size=64, window=5, epochs=5, seed=42):
        self.vector_size = vector_size
        self.window = window
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y=None):
        documents = np.asarray(X, dtype=object).ravel()
        self.model_ = Word2Vec(
            sentences=OpcodeSentences(documents),
            vector_size=self.vector_size,
            window=self.window,
            min_count=1,
            workers=1,
            seed=self.seed,
            epochs=self.epochs,
        )
        return self

    def transform(self, X):
        documents = np.asarray(X, dtype=object).ravel()
        output = np.zeros((len(documents), self.vector_size), dtype=np.float32)
        vocabulary = self.model_.wv.key_to_index
        for index, document in enumerate(documents):
            tokens = [token for token in str(document).split() if token in vocabulary]
            if tokens:
                output[index] = np.mean(self.model_.wv[tokens], axis=0)
        return output


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=lambda column: column in {"opcode", "label", "address"},
        low_memory=False,
    )
    if not {"opcode", "label"}.issubset(frame.columns):
        raise ValueError("CSV must contain opcode and label columns.")
    frame["opcode"] = frame["opcode"].fillna("").astype(str)
    if frame["opcode"].str.strip().eq("").any():
        raise ValueError("Empty opcode sequences were found.")
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Labels must be 0 (Benign) and 1 (Ponzi).")
    return frame


def split_data(frame: pd.DataFrame, seed: int):
    outer = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_indices, test_indices = next(outer.split(frame.opcode, frame.label))
    train = frame.iloc[train_indices].reset_index(drop=True)
    test = frame.iloc[test_indices].reset_index(drop=True)

    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + 1)
    fit_indices, validation_indices = next(inner.split(train.opcode, train.label))
    fit = train.iloc[fit_indices].reset_index(drop=True)
    validation = train.iloc[validation_indices].reset_index(drop=True)
    return train, fit, validation, test


def make_vectorizer(kind: str, max_features: int):
    options = dict(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        min_df=2,
        max_features=max_features,
        dtype=np.float32,
    )
    if kind == "count":
        return CountVectorizer(ngram_range=(1, 1), **options)
    return TfidfVectorizer(ngram_range=(1, 3), **options)


def make_xgb_pipeline(kind: str, seed: int, max_features: int, workers: int):
    return ImbPipeline(
        [
            ("features", make_vectorizer(kind, max_features)),
            ("smote", SMOTE(random_state=seed, k_neighbors=5)),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=seed,
                    n_jobs=workers,
                    scale_pos_weight=1.0,
                    verbosity=0,
                ),
            ),
        ]
    )


def tune_xgb(kind: str, X, y, seed: int, max_features: int, workers: int):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
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
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def make_ridge_viewer(seed: int, max_features: int):
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
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=cv)


def ensemble_probabilities(models, documents):
    values = [model.predict_proba(documents)[:, 1] for model in models]
    return np.mean(np.column_stack(values), axis=1)


def best_f1_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    best = int(np.flatnonzero(np.isclose(scores, scores.max()))[-1])
    return float(thresholds[best]), float(scores[best])


def evaluate(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=[1], average=None, zero_division=0
    )
    curve_precision, curve_recall, _ = precision_recall_curve(labels, probabilities)
    return {
        "threshold": threshold,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
        "pr_auc_trapezoidal": float(auc(curve_recall, curve_precision)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Ponzi"],
            zero_division=0,
            output_dict=True,
        ),
    }


def build_comparison(baseline_path: Path, new_test: dict) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    published = baseline["published_mulcas"]
    old_test = baseline["experimental_reconstruction"]["test"]
    balanced = new_test.get("balanced_accuracy")
    if balanced is None:
        tpr = new_test["tp"] / (new_test["tp"] + new_test["fn"])
        tnr = new_test["tn"] / (new_test["tn"] + new_test["fp"])
        balanced = (tpr + tnr) / 2
    new = {
        **new_test,
        "balanced_accuracy": balanced,
        "pr_auc": new_test["average_precision"],
    }
    metric_names = (
        "accuracy", "balanced_accuracy", "precision", "recall", "f1", "pr_auc"
    )
    return {
        "note": "Published values and the two local tests use different evaluation protocols.",
        "rows": {
            "Published MulCas": {name: published.get(name) for name in metric_names},
            "Old paper-aligned test": {name: old_test.get(name) for name in metric_names},
            "New random-split test": {name: new.get(name) for name in metric_names},
        },
        "confusion_matrices": {
            "Old paper-aligned test": {
                name: int(old_test[name]) for name in ("tn", "fp", "fn", "tp")
            },
            "New random-split test": {
                name: int(new[name]) for name in ("tn", "fp", "fn", "tp")
            },
        },
        "new_minus_old_percentage_points": {
            name: 100 * (new[name] - old_test[name])
            for name in metric_names
            if new.get(name) is not None and old_test.get(name) is not None
        },
    }


def print_comparison(comparison: dict):
    metrics = (
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Bal Acc"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("pr_auc", "PR-AUC"),
    )

    def percent(value):
        return "N/R" if value is None else f"{100 * value:.2f}%"

    print("\nOLD AND NEW TEST COMPARISON")
    print(f"{'Result':<27}" + "".join(f"{title:>12}" for _, title in metrics))
    print("-" * (27 + 12 * len(metrics)))
    for label, values in comparison["rows"].items():
        print(f"{label:<27}" + "".join(
            f"{percent(values[name]):>12}" for name, _ in metrics
        ))
    print("\nConfusion matrices (positive class = Ponzi)")
    for label, values in comparison["confusion_matrices"].items():
        print(
            f"{label}: TP={values['tp']}, FP={values['fp']}, "
            f"TN={values['tn']}, FN={values['fn']}"
        )
    deltas = comparison["new_minus_old_percentage_points"]
    print("\nNew minus old (percentage points)")
    print(" | ".join(
        f"{title}: {deltas[name]:+.2f}" for name, title in metrics if name in deltas
    ))
    print(f"\nNote: {comparison['note']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--show-comparison-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-features", type=int, default=20_000)
    args = parser.parse_args()

    if args.show_comparison_only:
        if not args.output.is_file() or args.output.stat().st_size == 0:
            raise SystemExit(
                f"No saved model results found in {args.output}. "
                "Run without --show-comparison-only to train the model first."
            )
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        print_comparison(build_comparison(args.baseline_results, saved["test"]))
        return

    frame = load_data(args.dataset)
    train, fit, validation, test = split_data(frame, args.seed)
    print(
        f"Rows: {len(frame)}; Ponzi: {int(frame.label.sum())}; "
        f"benign: {len(frame) - int(frame.label.sum())}"
    )
    for name, part in (
        ("train", train), ("fit", fit), ("validation", validation), ("test", test)
    ):
        print(f"{name}: {len(part)} rows, {int(part.label.sum())} Ponzi")

    overlap = None
    if "address" in frame.columns:
        overlap = len(set(train.address.dropna()) & set(test.address.dropna()))
        print(f"Addresses shared by train and test: {overlap}")
        if overlap:
            print("WARNING: duplicate addresses across the split may make scores optimistic.")

    X_fit, y_fit = fit.opcode, fit.label.to_numpy()
    print("Tuning XGBoost term-count viewer...")
    count_model, count_params, count_cv_f1 = tune_xgb(
        "count", X_fit, y_fit, args.seed, args.max_features, args.workers
    )
    print("Tuning XGBoost n-gram TF-IDF viewer...")
    tfidf_model, tfidf_params, tfidf_cv_f1 = tune_xgb(
        "tfidf", X_fit, y_fit, args.seed, args.max_features, args.workers
    )
    print("Fitting calibrated Ridge TF-IDF + Word2Vec viewer...")
    ridge_model = make_ridge_viewer(args.seed, args.max_features)
    ridge_model.fit(X_fit, y_fit)
    models = [count_model, tfidf_model, ridge_model]

    validation_probabilities = ensemble_probabilities(models, validation.opcode)
    threshold, validation_f1 = best_f1_threshold(
        validation.label.to_numpy(), validation_probabilities
    )
    print(f"Validation-selected F1 threshold: {threshold:.6f} (F1={validation_f1:.4f})")

    test_probabilities = ensemble_probabilities(models, test.opcode)
    results = evaluate(test.label.to_numpy(), test_probabilities, threshold)
    print(
        f"Confusion matrix: TP={results['tp']}, FP={results['fp']}, "
        f"TN={results['tn']}, FN={results['fn']}"
    )
    print(f"Accuracy: {100 * results['accuracy']:.2f}%")
    print(f"Balanced accuracy: {100 * results['balanced_accuracy']:.2f}%")
    print(f"Precision: {100 * results['precision']:.2f}%")
    print(f"Recall: {100 * results['recall']:.2f}%")
    print(f"F1: {100 * results['f1']:.2f}%")
    print(f"PR-AUC (average precision): {100 * results['average_precision']:.2f}%")

    comparison = build_comparison(args.baseline_results, results)
    payload = {
        "experiment": "random_stratified_static_opcode_three_viewer",
        "dataset": str(args.dataset),
        "seed": args.seed,
        "split_counts": {
            name: {"rows": len(part), "ponzi": int(part.label.sum())}
            for name, part in (
                ("train", train), ("fit", fit),
                ("validation", validation), ("test", test)
            )
        },
        "train_test_shared_addresses": overlap,
        "smote": "Applied only inside training and cross-validation folds",
        "scale_pos_weight": 1.0,
        "word2vec": "64-dimensional training-only mean pooling",
        "xgboost_count": {"best_params": count_params, "cv_f1": count_cv_f1},
        "xgboost_tfidf": {"best_params": tfidf_params, "cv_f1": tfidf_cv_f1},
        "validation_optimal_f1": validation_f1,
        "test": results,
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results: {args.output}")
    print_comparison(comparison)


if __name__ == "__main__":
    main()
