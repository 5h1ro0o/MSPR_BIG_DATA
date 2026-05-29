"""
Helpers Plotly pour le dashboard Streamlit.
Toutes les fonctions retournent une figure Plotly (go.Figure).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

COLOR_MACRON  = "#2196F3"
COLOR_LEPEN   = "#F44336"
COLOR_NEUTRAL = "#9E9E9E"
TEMPLATE      = "plotly_dark"

def plotly_feature_importance(
    df_imp: pd.DataFrame,
    title: str = "Feature Importance",
    top_n: int = 20,
    color: str = COLOR_MACRON,
) -> go.Figure:
    """
    Barres horizontales triées par importance.

    Parameters
    ----------
    df_imp : DataFrame avec colonnes ['feature', 'importance']
    """
    df = df_imp.head(top_n).sort_values("importance")
    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker_color=color,
        text=[f"{v:.4f}" for v in df["importance"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Feature",
        template=TEMPLATE,
        height=max(400, top_n * 24),
        margin=dict(l=200, r=60, t=60, b=40),
    )
    return fig

def plotly_confusion_matrix(
    cm: np.ndarray | list,
    labels: list[str] | None = None,
    title: str = "Matrice de confusion",
) -> go.Figure:
    """
    Heatmap annotée pour matrice de confusion binaire ou multiclasse.
    """
    cm = np.array(cm)
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    text = [[f"{cm[i, j]}<br>({cm_norm[i, j]*100:.1f}%)"
             for j in range(cm.shape[1])]
            for i in range(cm.shape[0])]

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
        colorbar=dict(title="Proportion"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Prédit",
        yaxis_title="Réel",
        template=TEMPLATE,
        height=400,
    )
    return fig

def plotly_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    model_name: str = "",
    color: str = COLOR_MACRON,
) -> go.Figure:
    """
    Courbe ROC avec ligne diagonale de référence.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"{model_name} (AUC = {auc:.4f})",
        line=dict(color=color, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Aléatoire",
        line=dict(color=COLOR_NEUTRAL, width=1, dash="dash"),
    ))
    fig.update_layout(
        title=f"Courbe ROC — {model_name}",
        xaxis_title="Taux faux positifs (FPR)",
        yaxis_title="Taux vrais positifs (TPR)",
        template=TEMPLATE,
        height=450,
        legend=dict(x=0.6, y=0.1),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
    )
    return fig

def plotly_radar_comparison(
    results: dict[str, dict],
    metrics: list[str] | None = None,
    title: str = "Comparaison des modèles",
) -> go.Figure:
    """
    Radar chart comparant plusieurs modèles sur plusieurs métriques.
    Normalise chaque métrique dans [0, 1].
    """
    if metrics is None:
        metrics = ["val_accuracy", "val_auc", "val_f1"]

    present = [m for m in metrics
               if any(m in v for v in results.values())]
    if not present:
        fig = go.Figure()
        fig.update_layout(title="Aucune métrique disponible", template=TEMPLATE)
        return fig

    colors = [COLOR_MACRON, COLOR_LEPEN, "#4CAF50", "#FF9800", "#9C27B0"]
    fig = go.Figure()

    for i, (name, m) in enumerate(results.items()):
        vals = [m.get(metric, 0) or 0 for metric in present]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=present + [present[0]],
            fill="toself",
            name=name,
            line_color=colors[i % len(colors)],
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=title,
        template=TEMPLATE,
        height=500,
    )
    return fig

def plotly_predictions_map(
    pred_df: pd.DataFrame,
    title: str = "Prédictions par commune",
) -> go.Figure:
    """
    Bar chart du nombre de prédictions correctes/incorrectes par département.
    pred_df doit contenir : 'dept', 'correct' (0/1).
    """
    if "dept" not in pred_df.columns or "correct" not in pred_df.columns:
        fig = go.Figure()
        fig.update_layout(title="Colonnes manquantes", template=TEMPLATE)
        return fig

    summary = (pred_df
               .groupby(["dept", "correct"])
               .size()
               .reset_index(name="count"))
    summary["label"] = summary["correct"].map({1: "Correct", 0: "Incorrect"})
    summary["dept"] = summary["dept"].astype(str)

    fig = px.bar(
        summary, x="dept", y="count", color="label",
        color_discrete_map={"Correct": COLOR_MACRON, "Incorrect": COLOR_LEPEN},
        title=title,
        labels={"count": "Communes", "dept": "Département"},
        template=TEMPLATE,
        barmode="stack",
    )
    fig.update_layout(height=400)
    return fig

def plotly_proba_histogram(
    pred_df: pd.DataFrame,
    title: str = "Distribution des probabilités",
    proba_col: str = "proba_lepen",
) -> go.Figure:
    """
    Histogramme de la probabilité prédite, coloré par vainqueur prédit.
    """
    if proba_col not in pred_df.columns:
        fig = go.Figure()
        fig.update_layout(title="Colonne proba absente", template=TEMPLATE)
        return fig

    fig = go.Figure()
    for val, color, label in [(0, COLOR_MACRON, "Macron prédit"),
                               (1, COLOR_LEPEN, "Le Pen prédit")]:
        subset = pred_df[pred_df.get("prediction", pred_df[proba_col].round()) == val]
        if len(subset):
            fig.add_trace(go.Histogram(
                x=subset[proba_col],
                name=label,
                marker_color=color,
                opacity=0.7,
                nbinsx=30,
            ))
    fig.update_layout(
        title=title,
        xaxis_title="Probabilité Le Pen",
        yaxis_title="Nombre de communes",
        barmode="overlay",
        template=TEMPLATE,
        height=400,
    )
    return fig

def plotly_metric_bars(
    results: dict[str, dict],
    metric: str,
    title: str | None = None,
) -> go.Figure:
    """
    Barres verticales comparant une métrique entre modèles.
    """
    names = list(results.keys())
    vals  = [results[n].get(metric, 0) or 0 for n in names]
    colors = [COLOR_MACRON, COLOR_LEPEN, "#4CAF50", "#FF9800", "#9C27B0"]

    fig = go.Figure(go.Bar(
        x=names,
        y=vals,
        marker_color=colors[:len(names)],
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        title=title or metric,
        yaxis_title=metric,
        template=TEMPLATE,
        height=400,
        yaxis=dict(range=[0, max(vals) * 1.15 if vals else 1]),
    )
    return fig
