"""
Évaluation — métriques robustes, intervalles de confiance, baseline, calibration.
Toutes les fonctions sont indépendantes du modèle et réutilisables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ── Intervalles de confiance ───────────────────────────────────────────────────

def wilson_ci(n_correct: int, n_total: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Intervalle de confiance de Wilson pour une proportion.
    Plus fiable que l'intervalle normal pour les petits échantillons ou les extrêmes.
    Retourne (borne_basse, borne_haute). Utilise uniquement math (stdlib).
    """
    from scipy.stats import norm as _norm

    z = float(_norm.ppf(1 - (1 - confidence) / 2))
    p = n_correct / n_total
    n = n_total
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / (1 + z**2 / n)
    return round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap pour l'IC d'une métrique quelconque.
    metric_fn(y_true, y_score) → float.
    """
    rng = np.random.RandomState(random_state)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        try:
            scores.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            pass
    scores = np.array(scores)
    alpha = (1 - confidence) / 2
    return round(float(np.percentile(scores, alpha * 100)), 4), round(float(np.percentile(scores, (1 - alpha) * 100)), 4)


# ── Baseline ──────────────────────────────────────────────────────────────────

def baseline_metrics(y_true: np.ndarray) -> dict:
    """
    Métriques d'un classifieur naïf (classe majoritaire).
    Référence minimale que tout modèle doit dépasser.
    """
    classes, counts = np.unique(y_true, return_counts=True)
    majority_class = classes[np.argmax(counts)]
    y_pred_baseline = np.full_like(y_true, majority_class)
    n = len(y_true)
    n_majority = int(counts.max())
    return {
        "baseline_majority_class": int(majority_class),
        "baseline_accuracy": round(n_majority / n, 4),
        "baseline_balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred_baseline), 4),
        "baseline_f1_weighted": round(f1_score(y_true, y_pred_baseline, average="weighted", zero_division=0), 4),
        "class_distribution": {int(c): round(int(cnt) / n, 4) for c, cnt in zip(classes, counts)},
    }


# ── Évaluation robuste complète ───────────────────────────────────────────────

def robust_classification_eval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    split: str = "test",
) -> dict:
    """
    Évaluation complète avec IC, métriques robustes et baseline.

    Retourne un dictionnaire préfixé par `split` (ex. 'test_accuracy', 'test_roc_auc').
    Inclut les IC de Wilson (accuracy) et bootstrap (AUC).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    n_correct = int((y_true == y_pred).sum())

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class  = recall_score(y_true, y_pred, average=None, zero_division=0)
    classes = np.unique(y_true)

    ci_low, ci_high = wilson_ci(n_correct, n, confidence)

    metrics = {
        f"{split}_n_samples":          n,
        f"{split}_n_correct":          n_correct,
        f"{split}_accuracy":           round(acc, 4),
        f"{split}_accuracy_ci_lo":     ci_low,
        f"{split}_accuracy_ci_hi":     ci_high,
        f"{split}_balanced_accuracy":  round(bal_acc, 4),
        f"{split}_f1_weighted":        round(f1_w, 4),
        f"{split}_f1_macro":           round(f1_m, 4),
        f"{split}_cohen_kappa":        round(kappa, 4),
    }

    # Métriques par classe
    label_names = {0: "macron", 1: "lepen"}
    for i, cls in enumerate(classes):
        suffix = label_names.get(int(cls), str(cls))
        if i < len(prec_per_class):
            metrics[f"{split}_precision_{suffix}"] = round(float(prec_per_class[i]), 4)
        if i < len(rec_per_class):
            metrics[f"{split}_recall_{suffix}"] = round(float(rec_per_class[i]), 4)

    # Métriques basées sur les probabilités
    if y_proba is not None:
        proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        try:
            auc_score = roc_auc_score(y_true, proba_pos)
            metrics[f"{split}_roc_auc"] = round(auc_score, 4)
            # IC bootstrap sur AUC
            auc_lo, auc_hi = bootstrap_ci(
                y_true, proba_pos,
                lambda yt, yp: roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else 0.5,
                n_bootstrap=n_bootstrap,
            )
            metrics[f"{split}_roc_auc_ci_lo"] = auc_lo
            metrics[f"{split}_roc_auc_ci_hi"] = auc_hi
        except Exception:
            pass
        try:
            brier = brier_score_loss(y_true, proba_pos)
            metrics[f"{split}_brier_score"] = round(brier, 4)
        except Exception:
            pass

    return metrics


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Alias léger pour compatibilité — préfère robust_classification_eval."""
    base = {
        "accuracy":            round(accuracy_score(y_true, y_pred), 4),
        "f1_weighted":         round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "precision_weighted":  round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall_weighted":     round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "confusion_matrix":    confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            base["roc_auc"] = round(roc_auc_score(y_true, proba_pos), 4)
        except Exception:
            pass
    return base


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "r2":   round(r2_score(y_true, y_pred), 4),
        "mae":  round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
    }


# ── Graphiques ────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, model_name, output_dir) -> Path:
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title in zip(axes, [cm, cm_norm], ["Comptages", "Normalisée"]):
        im = ax.imshow(data, interpolation="nearest", cmap="Blues")
        ax.set(
            title=f"{model_name} — Matrice de confusion ({title})",
            xlabel="Prédit", ylabel="Réel",
            xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
            xticklabels=class_names, yticklabels=class_names,
        )
        plt.colorbar(im, ax=ax)
        thresh = data.max() / 2
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = f"{data[i,j]:.2f}" if title == "Normalisée" else str(int(data[i,j]))
                ax.text(j, i, val, ha="center", va="center",
                        color="white" if data[i,j] > thresh else "black", fontsize=11)
    plt.tight_layout()
    path = output_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_roc_curve(y_true, y_proba, model_name, output_dir) -> Path:
    proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
    fpr, tpr, _ = roc_curve(y_true, proba_pos)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Hasard")
    ax.set(xlabel="Faux Positifs (1 − Spécificité)", ylabel="Vrais Positifs (Sensibilité)",
           title=f"{model_name} — Courbe ROC", xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = output_dir / f"{model_name}_roc_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_calibration(y_true, y_proba, model_name: str, output_dir: Path) -> Path:
    """
    Courbe de calibration (reliability diagram).
    Un modèle bien calibré a sa courbe proche de la diagonale.
    """
    proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
    fraction_pos, mean_pred = calibration_curve(y_true, proba_pos, n_bins=10, strategy="uniform")
    brier = brier_score_loss(y_true, proba_pos)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Calibration parfaite")
    ax.plot(mean_pred, fraction_pos, "o-", lw=2, color="#4472C4",
            label=f"Modèle (Brier={brier:.4f})")
    ax.set(xlabel="Probabilité prédite (Le Pen)",
           ylabel="Fraction réelle (Le Pen)",
           title=f"{model_name} — Courbe de calibration",
           xlim=[0, 1], ylim=[0, 1])
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(proba_pos[np.asarray(y_true) == 0], bins=20, alpha=0.6, color="#4472C4", label="Macron (0)")
    ax.hist(proba_pos[np.asarray(y_true) == 1], bins=20, alpha=0.6, color="#E15759", label="Le Pen (1)")
    ax.set(xlabel="Probabilité prédite (Le Pen)", ylabel="Nombre de communes",
           title=f"{model_name} — Distribution des probabilités")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"{model_name}_calibration.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_feature_importance(df_importance, model_name, output_dir, top_n=25) -> Path:
    df = df_importance.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    bars = ax.barh(df["feature"], df["importance"],
                   xerr=df["std"] if "std" in df.columns else None,
                   color="#4472C4", alpha=0.85)
    for bar, val in zip(bars, df["importance"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    ax.set(xlabel="Importance (Mean Decrease Impurity)",
           title=f"{model_name} — Top {top_n} variables")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    plt.tight_layout()
    path = output_dir / f"{model_name}_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_predictions_vs_actual(y_true, y_pred, model_name, output_dir, target_name="Cible") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="#E15759")
    lim = [min(float(min(y_true)), float(min(y_pred))) - 1,
           max(float(max(y_true)), float(max(y_pred))) + 1]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlabel=f"{target_name} réelle", ylabel=f"{target_name} prédite",
           title=f"{model_name} — Prédictions vs Réalité")
    ax = axes[1]
    residuals = np.array(y_pred) - np.array(y_true)
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, color="#59A14F")
    ax.axhline(0, color="k", linestyle="--", lw=1)
    ax.set(xlabel=f"{target_name} prédite", ylabel="Résidus",
           title=f"{model_name} — Analyse des résidus")
    plt.tight_layout()
    path = output_dir / f"{model_name}_predictions_vs_actual.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_model_comparison(results, metric, output_dir, title="Comparaison des modèles") -> Path:
    names  = list(results.keys())
    values = [results[n].get(metric, 0) for n in names]
    colors = ["#4472C4", "#ED7D31", "#A9D18E", "#FF0000", "#7030A0"][: len(names)]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set(ylabel=metric, title=title, ylim=[0, min(1.1, max(values) * 1.15)])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = output_dir / f"model_comparison_{metric}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
