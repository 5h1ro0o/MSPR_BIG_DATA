"""
Évaluation — calcul centralisé des métriques et génération des graphiques.
Utilisé par tous les modèles pour produire des outputs comparables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Calcule l'ensemble des métriques de classification."""
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_weighted": round(
            f1_score(y_true, y_pred, average="weighted", zero_division=0), 4
        ),
        "precision_weighted": round(
            precision_score(y_true, y_pred, average="weighted", zero_division=0), 4
        ),
        "recall_weighted": round(
            recall_score(y_true, y_pred, average="weighted", zero_division=0), 4
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            metrics["roc_auc"] = round(roc_auc_score(y_true, proba_pos), 4)
        except Exception:
            pass
    return metrics


def regression_metrics(y_true, y_pred) -> dict:
    """Calcule les métriques de régression."""
    return {
        "r2": round(r2_score(y_true, y_pred), 4),
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
    }


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: list[str],
    model_name: str,
    output_dir: Path,
) -> Path:
    """Sauvegarde la matrice de confusion normalisée."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title in zip(axes, [cm, cm_norm], ["Comptages", "Normalisée"]):
        im = ax.imshow(data, interpolation="nearest", cmap="Blues")
        ax.set(
            title=f"{model_name} — Matrice de confusion ({title})",
            xlabel="Prédit",
            ylabel="Réel",
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.colorbar(im, ax=ax)
        thresh = data.max() / 2
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = (
                    f"{data[i, j]:.2f}"
                    if title == "Normalisée"
                    else str(int(data[i, j]))
                )
                ax.text(
                    j,
                    i,
                    val,
                    ha="center",
                    va="center",
                    color="white" if data[i, j] > thresh else "black",
                    fontsize=11,
                )

    plt.tight_layout()
    path = output_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_roc_curve(
    y_true,
    y_proba,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Sauvegarde la courbe ROC."""
    proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
    fpr, tpr, _ = roc_curve(y_true, proba_pos)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set(
        xlabel="Taux de Faux Positifs (1 - Spécificité)",
        ylabel="Taux de Vrais Positifs (Sensibilité)",
        title=f"{model_name} — Courbe ROC",
        xlim=[0, 1],
        ylim=[0, 1.02],
    )
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = output_dir / f"{model_name}_roc_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_feature_importance(
    df_importance: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    top_n: int = 25,
) -> Path:
    """Sauvegarde le graphique d'importance des features (horizontal bar)."""
    df = df_importance.head(top_n).sort_values("importance")

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    bars = ax.barh(
        df["feature"], df["importance"], xerr=df.get("std"), color="#4472C4", alpha=0.85
    )

    for bar, val in zip(bars, df["importance"]):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=8,
        )

    ax.set(
        xlabel="Importance (Mean Decrease Impurity)",
        title=f"{model_name} — Top {top_n} Features Importantes",
    )
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    plt.tight_layout()

    path = output_dir / f"{model_name}_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_predictions_vs_actual(
    y_true,
    y_pred,
    model_name: str,
    output_dir: Path,
    target_name: str = "Cible",
) -> Path:
    """Scatter plot prédictions vs réalité (régression)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="#E15759")
    lim = [min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1]
    ax.plot(lim, lim, "k--", lw=1, label="Parfait")
    ax.set(
        xlabel=f"{target_name} réelle",
        ylabel=f"{target_name} prédite",
        title=f"{model_name} — Prédictions vs Réalité",
    )
    ax.legend()

    ax = axes[1]
    residuals = np.array(y_pred) - np.array(y_true)
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, color="#59A14F")
    ax.axhline(0, color="k", linestyle="--", lw=1)
    ax.set(
        xlabel=f"{target_name} prédite",
        ylabel="Résidus",
        title=f"{model_name} — Analyse des résidus",
    )

    plt.tight_layout()
    path = output_dir / f"{model_name}_predictions_vs_actual.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_model_comparison(
    results: dict[str, dict],
    metric: str,
    output_dir: Path,
    title: str = "Comparaison des modèles",
) -> Path:
    """
    Graphique comparant plusieurs modèles sur une métrique.

    Args:
        results: {model_name: metrics_dict}
        metric:  clé de la métrique (ex. "test_accuracy", "cv_f1")
    """
    names = list(results.keys())
    values = [results[n].get(metric, 0) for n in names]
    colors = ["#4472C4", "#ED7D31", "#A9D18E", "#FF0000", "#7030A0"][: len(names)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        names, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5
    )
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set(ylabel=metric, title=title, ylim=[0, min(1.1, max(values) * 1.15)])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = output_dir / f"model_comparison_{metric}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
