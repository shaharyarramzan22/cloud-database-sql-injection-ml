from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
)


def evaluate(y_true, y_pred, y_score=None) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if y_score is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["roc_auc"] = None
    return out


def save_confusion_matrix(y_true, y_pred, name: str, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_confusion_matrix.png", dpi=200)
    plt.close()


def save_roc(y_true, scores: dict[str, np.ndarray], out_dir: Path):
    plt.figure(figsize=(7, 5))
    for name, score in scores.items():
        try:
            fpr, tpr, _ = roc_curve(y_true, score)
            auc = roc_auc_score(y_true, score)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        except ValueError:
            pass
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves.png", dpi=200)
    plt.close()


def save_comparison(metrics: dict[str, dict], out_dir: Path):
    df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"})
    df.to_csv(out_dir / "model_comparison.csv", index=False)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        if metric in df:
            plt.figure(figsize=(9, 5))
            plt.bar(df["model"], df[metric])
            plt.ylim(0, 1.05)
            plt.ylabel(metric.upper())
            plt.title(f"Model Comparison — {metric.upper()}")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(out_dir / f"comparison_{metric}.png", dpi=200)
            plt.close()


def dump_json(data, path: Path):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
