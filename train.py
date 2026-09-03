from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import classification_report

from .features import SQLFeatureTransformer
from .models import make_classical_models, train_ann, train_hybrid_ann_svm, ann_features
from .metrics import evaluate, save_confusion_matrix, save_roc, save_comparison, dump_json

SEED = 42


def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Train six SQL injection detection models.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--query-column", default="query")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output", default="results")
    parser.add_argument("--models", default="models")
    parser.add_argument("--no-smote", action="store_true")
    args = parser.parse_args()
    set_seed()

    out = Path(args.output); model_dir = Path(args.models)
    out.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    if args.query_column not in df.columns or args.label_column not in df.columns:
        raise ValueError(f"CSV must contain '{args.query_column}' and '{args.label_column}' columns. Found: {list(df.columns)}")
    df = df[[args.query_column, args.label_column]].dropna()
    df[args.label_column] = df[args.label_column].astype(int)
    if not set(df[args.label_column].unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary: 0=benign, 1=malicious.")

    X_text = df[args.query_column].astype(str).values
    y = df[args.label_column].values
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.20, stratify=y, random_state=SEED
    )

    transformer = SQLFeatureTransformer(max_tfidf_features=5000)
    X_train = transformer.fit_transform(X_train_text)
    X_test = transformer.transform(X_test_text)
    joblib.dump(transformer, model_dir / "feature_transformer.joblib")

    # SMOTE is applied only after the train/test split, avoiding test leakage.
    X_fit, y_fit = X_train, y_train
    if not args.no_smote:
        dense = X_train.toarray()
        # Keep k_neighbors valid for small datasets.
        minority = min(np.bincount(y_train))
        if minority >= 2:
            k = min(5, minority - 1)
            X_fit, y_fit = SMOTE(random_state=SEED, k_neighbors=k).fit_resample(dense, y_train)
        else:
            X_fit, y_fit = dense, y_train
    else:
        X_fit = X_train.toarray()

    X_test_dense = X_test.toarray().astype(np.float32)
    X_fit = np.asarray(X_fit, dtype=np.float32)

    metrics = {}
    score_map = {}
    timing = {}
    trained = {}

    # Classical models.
    for name, model in make_classical_models().items():
        t0 = time.perf_counter(); model.fit(X_fit, y_fit); train_time = time.perf_counter() - t0
        t1 = time.perf_counter(); pred = model.predict(X_test_dense); infer_time = time.perf_counter() - t1
        try: score = model.predict_proba(X_test_dense)[:, 1]
        except Exception: score = model.decision_function(X_test_dense)
        m = evaluate(y_test, pred, score)
        metrics[name] = m
        score_map[name] = np.asarray(score)
        timing[name] = {"training_seconds": train_time, "test_inference_seconds": infer_time, "avg_inference_ms": infer_time / len(y_test) * 1000}
        joblib.dump(model, model_dir / f"{name}.joblib")
        save_confusion_matrix(y_test, pred, name, out)
        (out / f"{name}_classification_report.txt").write_text(classification_report(y_test, pred, digits=4), encoding="utf-8")

    # ANN uses a validation split from training data.
    X_ann_train, X_ann_val, y_ann_train, y_ann_val = train_test_split(
        X_fit, y_fit, test_size=0.15, stratify=y_fit, random_state=SEED
    )
    t0 = time.perf_counter()
    ann, history = train_ann(X_ann_train, y_ann_train, X_ann_val, y_ann_val)
    ann_train_time = time.perf_counter() - t0
    t1 = time.perf_counter(); ann_score = ann.predict(X_test_dense, verbose=0).ravel(); ann_infer = time.perf_counter() - t1
    ann_pred = (ann_score >= 0.5).astype(int)
    metrics["ANN"] = evaluate(y_test, ann_pred, ann_score)
    score_map["ANN"] = ann_score
    timing["ANN"] = {"training_seconds": ann_train_time, "test_inference_seconds": ann_infer, "avg_inference_ms": ann_infer / len(y_test) * 1000}
    ann.save(model_dir / "ANN.keras")
    pd.DataFrame(history.history).to_csv(out / "ANN_training_history.csv", index=False)
    save_confusion_matrix(y_test, ann_pred, "ANN", out)
    (out / "ANN_classification_report.txt").write_text(classification_report(y_test, ann_pred, digits=4), encoding="utf-8")

    # Hybrid: ANN penultimate layer -> SVM.
    t0 = time.perf_counter()
    hybrid_svm, z_test = train_hybrid_ann_svm(ann, X_fit, y_fit, X_test_dense)
    hybrid_train_time = time.perf_counter() - t0
    t1 = time.perf_counter(); hybrid_pred = hybrid_svm.predict(z_test); hybrid_infer = time.perf_counter() - t1
    hybrid_score = hybrid_svm.predict_proba(z_test)[:, 1]
    metrics["Hybrid_ANN_SVM"] = evaluate(y_test, hybrid_pred, hybrid_score)
    score_map["Hybrid_ANN_SVM"] = hybrid_score
    timing["Hybrid_ANN_SVM"] = {"svm_training_seconds": hybrid_train_time, "test_inference_seconds": hybrid_infer, "avg_inference_ms": hybrid_infer / len(y_test) * 1000}
    joblib.dump(hybrid_svm, model_dir / "Hybrid_ANN_SVM.joblib")
    save_confusion_matrix(y_test, hybrid_pred, "Hybrid_ANN_SVM", out)
    (out / "Hybrid_ANN_SVM_classification_report.txt").write_text(classification_report(y_test, hybrid_pred, digits=4), encoding="utf-8")

    # 10-fold CV on the training data for the classical models.
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    cv_results = {}
    for name, model in make_classical_models().items():
        try:
            scores = cross_val_score(model, X_fit, y_fit, cv=cv, scoring="f1", n_jobs=None)
            cv_results[name] = {"mean_f1": float(scores.mean()), "std_f1": float(scores.std()), "folds": scores.tolist()}
        except Exception as exc:
            cv_results[name] = {"error": str(exc)}

    save_comparison(metrics, out)
    save_roc(y_test, score_map, out)
    dump_json(metrics, out / "metrics.json")
    dump_json(timing, out / "timing.json")
    dump_json(cv_results, out / "10fold_cv_f1.json")
    dump_json({"rows": len(df), "class_counts": df[args.label_column].value_counts().sort_index().to_dict(), "test_rows": len(y_test), "seed": SEED}, out / "dataset_summary.json")
    print(pd.DataFrame(metrics).T.round(4))
    print(f"\nSaved results to {out.resolve()} and models to {model_dir.resolve()}")


if __name__ == "__main__":
    main()
