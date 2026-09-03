from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
from tensorflow import keras


def main():
    p = argparse.ArgumentParser(description="Predict whether a SQL query is benign or malicious.")
    p.add_argument("--models", default="models")
    p.add_argument("--query", required=True)
    args = p.parse_args()
    d = Path(args.models)
    transformer = joblib.load(d / "feature_transformer.joblib")
    X = transformer.transform_dense([args.query])

    models = [
        ("Naive_Bayes", joblib.load(d / "Naive_Bayes.joblib")),
        ("Decision_Tree", joblib.load(d / "Decision_Tree.joblib")),
        ("Random_Forest", joblib.load(d / "Random_Forest.joblib")),
        ("SVM", joblib.load(d / "SVM.joblib")),
    ]
    for name, model in models:
        pred = int(model.predict(X)[0])
        try: prob = float(model.predict_proba(X)[0, 1])
        except Exception: prob = None
        print(f"{name:18s}: {'MALICIOUS' if pred else 'BENIGN':9s}" + (f"  probability={prob:.4f}" if prob is not None else ""))

    ann = keras.models.load_model(d / "ANN.keras")
    score = float(ann.predict(X, verbose=0)[0, 0])
    print(f"{'ANN':18s}: {'MALICIOUS' if score >= .5 else 'BENIGN':9s}  probability={score:.4f}")
    extractor = keras.Model(ann.input, ann.get_layer("deep_features").output)
    z = extractor.predict(X, verbose=0)
    svm = joblib.load(d / "Hybrid_ANN_SVM.joblib")
    pred = int(svm.predict(z)[0]); prob = float(svm.predict_proba(z)[0, 1])
    print(f"{'Hybrid_ANN_SVM':18s}: {'MALICIOUS' if pred else 'BENIGN':9s}  probability={prob:.4f}")


if __name__ == "__main__":
    main()
