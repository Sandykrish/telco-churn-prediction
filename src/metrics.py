"""Metrics and diagnostic plots for the churn classifiers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def classification_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """The five headline numbers I report for every model."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Turn ``{model_name: metrics_dict}`` into a tidy table."""
    return pd.DataFrame(results).T.round(4)


def plot_roc_curves(probas: dict[str, np.ndarray], y_true, path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def plot_precision_recall(probas: dict[str, np.ndarray], y_true, path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in probas.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ax.plot(recall, precision, label=f"{name} (AP={average_precision_score(y_true, proba):.3f})")
    ax.axhline(np.mean(y_true), color="grey", ls="--", lw=1, label="churn base rate")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves")
    ax.legend(loc="upper right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def plot_confusion(y_true, y_pred, title: str, path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def plot_threshold_sweep(y_true, y_proba, path: Path | None = None) -> pd.DataFrame:
    """Precision / recall / F1 as the decision threshold moves - useful for picking a
    cut-off that matches the retention team's capacity."""
    thresholds = np.arange(0.1, 0.91, 0.05)
    rows = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        rows.append(
            {
                "threshold": round(t, 2),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred),
                "f1": f1_score(y_true, pred),
            }
        )
    sweep = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    for col in ("precision", "recall", "f1"):
        ax.plot(sweep["threshold"], sweep[col], marker="o", ms=3, label=col)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sweep")
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return sweep


def plot_feature_importance(importances: pd.Series, title: str, path: Path | None = None, top: int = 20) -> plt.Figure:
    top_feats = importances.sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(x=top_feats.values, y=top_feats.index, ax=ax, color="#3b6ea5")
    ax.set_title(title)
    ax.set_xlabel("importance")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig
