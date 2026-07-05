"""Evaluation helpers for classifiers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_probability) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
    }


def save_evaluation_plots(model, X_test, y_test, output_dir: str | Path, prefix: str) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    confusion_path = output_dir / f"{prefix}_confusion_matrix.png"
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
    plt.title(f"Confusion Matrix - {prefix.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(confusion_path, dpi=160)
    plt.close()
    saved.append(confusion_path)

    roc_path = output_dir / f"{prefix}_roc_curve.png"
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"ROC Curve - {prefix.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=160)
    plt.close()
    saved.append(roc_path)
    return saved
