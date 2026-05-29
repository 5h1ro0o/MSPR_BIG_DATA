"""
Dashboard — Elections IDF 2022 | ML Pipeline

Lancement :
    cd etl_elections
    streamlit run ml/visualization/dashboard.py
"""

import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
except ImportError:
    print("pip install streamlit plotly")
    sys.exit(1)

from ml.config import ARTIFACTS

st.set_page_config(
    page_title="Elections IDF 2022",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* ── Reset & Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: #111318;
}
[data-testid="stSidebar"] {
    background: #16181f;
    border-right: 1px solid #22252e;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1280px;
}

/* ── Typography ── */
h1 {
    font-size: 1.65rem !important;
    font-weight: 600 !important;
    color: #e8eaf0 !important;
    letter-spacing: -0.02em;
    margin-bottom: 0.15rem !important;
}
h2 {
    font-size: 1.1rem !important;
    font-weight: 500 !important;
    color: #c9cdd8 !important;
    letter-spacing: -0.01em;
}
h3 {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #9095a8 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
p, li, label {
    color: #9095a8;
    font-size: 0.88rem;
}

/* ── Sidebar nav ── */
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.85rem !important;
    color: #9095a8 !important;
    padding: 0.1rem 0;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #e8eaf0 !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #1c1f29;
    border: 1px solid #22252e;
    border-radius: 6px;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: #6b7280 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 600 !important;
    color: #e8eaf0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Dividers ── */
hr {
    border: none;
    border-top: 1px solid #22252e !important;
    margin: 1.5rem 0 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid #22252e;
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Code blocks ── */
.stCode, code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    background: #1c1f29 !important;
    border: 1px solid #22252e !important;
    border-radius: 4px !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 4px;
    border-left-width: 3px;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #6b7280 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #e8eaf0 !important;
    border-bottom-color: #3b5bdb !important;
}

/* ── Status indicators ── */
.status-trained {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #40c057;
    margin-right: 6px;
    vertical-align: middle;
}
.status-pending {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #495057;
    margin-right: 6px;
    vertical-align: middle;
}
.model-row {
    padding: 0.4rem 0;
    font-size: 0.83rem;
    color: #9095a8;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Page header ── */
.page-header {
    border-bottom: 1px solid #22252e;
    padding-bottom: 0.9rem;
    margin-bottom: 1.6rem;
}
.page-subtitle {
    color: #6b7280;
    font-size: 0.85rem;
    margin-top: 0.2rem;
}

/* ── Stat card ── */
.stat-card {
    background: #1c1f29;
    border: 1px solid #22252e;
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
}
.stat-label {
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6b7280;
    margin-bottom: 0.4rem;
}
.stat-value {
    font-size: 1.6rem;
    font-weight: 600;
    color: #e8eaf0;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Sidebar title ── */
.sidebar-title {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin-bottom: 0.5rem;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] label {
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #6b7280 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #22252e !important;
    border-radius: 6px !important;
    background: #1c1f29 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

C_MACRON = "#3b5bdb"
C_LEPEN = "#c92a2a"
C_NEUTRAL = "#495057"
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#9095a8", family="Inter, system-ui"),
    margin=dict(t=40, b=30, l=10, r=10),
    xaxis=dict(gridcolor="#22252e", zerolinecolor="#22252e"),
    yaxis=dict(gridcolor="#22252e", zerolinecolor="#22252e"),
)


@st.cache_data(ttl=300)
def load_metrics() -> dict[str, dict]:
    results = {}
    for p in ARTIFACTS.glob("*_metrics_*.json"):
        try:
            data = json.loads(p.read_text())
            mname = p.stem.split("_metrics_")[0]
            results[mname] = data
        except Exception:
            pass
    return results


@st.cache_data(ttl=300)
def load_predictions_df(prefix: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(ARTIFACTS / f"{prefix}_predictions_*.csv")))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1])


@st.cache_data(ttl=300)
def load_dataset_cached() -> pd.DataFrame:
    try:
        from ml.preprocessing import load_dataset

        return load_dataset()
    except Exception:
        return pd.DataFrame()


def dataset_info() -> dict:
    """Retourne les infos du dataset local (chemin, taille, date de modification)."""
    from ml.config import DATA_PATH

    p = Path(DATA_PATH)
    if not p.exists():
        return {"exists": False, "path": str(p)}
    import datetime

    mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime)
    return {
        "exists": True,
        "path": str(p),
        "name": p.name,
        "size_mb": round(p.stat().st_size / 1_048_576, 1),
        "modified": mtime.strftime("%Y-%m-%d %H:%M"),
    }


def img_or_info(path: Path, caption: str = ""):
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Image non générée : {path.name} — lancez l'entraînement d'abord.")


def metric4(label, val, fmt=".4f"):
    if isinstance(val, float):
        st.metric(label, f"{val:{fmt}}")
    elif val is not None:
        st.metric(label, str(val))
    else:
        st.metric(label, "—")


def fmt_val(v, fmt=".4f"):
    return f"{v:{fmt}}" if isinstance(v, float) else ("—" if v is None else str(v))


def best_metric(m_dict, keys):
    for k in keys:
        if k in m_dict and isinstance(m_dict[k], float):
            return m_dict[k]
    return None


import os as _os


def _build_db_url() -> str | None:
    pwd = _os.environ.get("POSTGRES_PASSWORD", "")
    if not pwd:
        return None
    h = _os.environ.get("POSTGRES_HOST", "localhost")
    p = _os.environ.get("POSTGRES_PORT", "5432")
    d = _os.environ.get("POSTGRES_DB", "elections_idf")
    u = _os.environ.get("POSTGRES_USER", "etl_admin")
    return f"postgresql+psycopg2://{u}:{pwd}@{h}:{p}/{d}"


DB_URL = _build_db_url()


@st.cache_data(ttl=60, show_spinner=False)
def load_db_predictions(model_name: str = None) -> pd.DataFrame:
    if not DB_URL:
        return pd.DataFrame()
    try:
        from ml.training.db_store import load_predictions

        return load_predictions(DB_URL, model_name)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def load_db_metrics() -> pd.DataFrame:
    if not DB_URL:
        return pd.DataFrame()
    try:
        from ml.training.db_store import load_metrics as _lm

        return _lm(DB_URL)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def check_db_connection() -> bool:
    if not DB_URL:
        return False
    try:
        from sqlalchemy import create_engine, text

        eng = create_engine(DB_URL, pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


DB_CONNECTED = check_db_connection()

all_metrics = load_metrics()
df_main = load_dataset_cached()
ds_info = dataset_info()

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "lstm": "LSTM",
    "decision_tree": "Decision Tree",
    "mlp": "MLP",
}

with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">Elections IDF 2022</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<p style="font-size:0.8rem;color:#6b7280;margin-bottom:1.5rem;">'
        "Prédiction T2 — Macron vs Le Pen</p>",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Vue d'ensemble",
            "Random Forest",
            "Gradient Boosting",
            "LSTM",
            "Predictions",
            "Comparaison",
            "Stats",
            "Résultats DB",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<hr style="margin:0.8rem 0">', unsafe_allow_html=True)
    db_dot = "status-trained" if DB_CONNECTED else "status-pending"
    db_label = "PostgreSQL connecté" if DB_CONNECTED else "PostgreSQL non connecté"
    st.markdown(
        f'<div class="model-row">'
        f'<span class="{db_dot}"></span>'
        f'<span style="font-size:0.78rem;color:#6b7280">{db_label}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<hr style="margin:1.2rem 0">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Modeles</div>', unsafe_allow_html=True)

    for key, label in MODEL_LABELS.items():
        trained = key in all_metrics
        dot = "status-trained" if trained else "status-pending"
        suffix = (
            fmt_val(
                best_metric(
                    all_metrics.get(key, {}), ["val_accuracy", "test_accuracy"]
                ),
                ".3f",
            )
            if trained
            else "non entraîné"
        )
        st.markdown(
            f'<div class="model-row">'
            f'<span class="{dot}"></span>{label}'
            f'<span style="margin-left:auto;font-family:JetBrains Mono,monospace;'
            f'font-size:0.78rem;color:#6b7280">{suffix}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<hr style="margin:1.2rem 0">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Dataset</div>', unsafe_allow_html=True)
    if ds_info["exists"]:
        st.markdown(
            f'<div style="font-size:0.75rem;color:#6b7280;line-height:1.6">'
            f'<span style="color:#40c057;font-size:0.65rem">●</span> '
            f'{ds_info["name"]}<br>'
            f'<span style="font-family:JetBrains Mono,monospace">'
            f'{ds_info["size_mb"]} Mo · {ds_info["modified"]}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:0.75rem;color:#c92a2a">'
            '<span style="font-size:0.65rem">●</span> Dataset absent — lancez l\'ETL'
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<hr style="margin:1.2rem 0">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Entrainement</div>', unsafe_allow_html=True)
    st.code("python ml/run_training.py", language="bash")

if page == "Vue d'ensemble":
    st.markdown(
        """
    <div class="page-header">
        <h1>Elections Presidentielles 2022 — Ile-de-France</h1>
        <div class="page-subtitle">
            Modeles de prediction du vainqueur au second tour par commune (Macron / Le Pen)
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    n_communes = len(df_main) if len(df_main) else 0
    n_features = (
        len(
            [
                c
                for c in df_main.columns
                if not c.startswith("cible_")
                and c
                not in {
                    "code_commune",
                    "code_departement",
                    "libelle_departement",
                    "libelle_commune",
                }
            ]
        )
        if n_communes
        else 0
    )
    pct_macron = (
        (df_main["cible_t2_vainqueur"] == 0).mean() * 100
        if (n_communes and "cible_t2_vainqueur" in df_main.columns)
        else None
    )
    n_trained = len(all_metrics)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Communes IDF", f"{n_communes:,}" if n_communes else "—")
    c2.metric("Features ML", f"{n_features:,}" if n_features else "—")
    c3.metric("Communes Macron", f"{pct_macron:.1f}%" if pct_macron else "—")
    c4.metric("Communes Le Pen", f"{100-pct_macron:.1f}%" if pct_macron else "—")
    c5.metric("Modeles entraines", f"{n_trained}/5")

    st.markdown("<hr>", unsafe_allow_html=True)

    if all_metrics:
        st.markdown("#### Synthese des modeles")
        rows = []
        for name, m in all_metrics.items():
            rows.append(
                {
                    "Modele": MODEL_LABELS.get(name, name),
                    "Accuracy": fmt_val(
                        best_metric(m, ["val_accuracy", "test_accuracy"])
                    ),
                    "AUC-ROC": fmt_val(
                        best_metric(
                            m, ["val_auc", "test_auc", "test_roc_auc", "cv_roc_auc"]
                        )
                    ),
                    "F1": fmt_val(best_metric(m, ["val_f1", "test_f1"])),
                    "CV AUC": fmt_val(m.get("cv_roc_auc")),
                    "Temps (s)": fmt_val(m.get("training_time_s"), ".1f"),
                }
            )
        st.dataframe(
            pd.DataFrame(rows).set_index("Modele"),
            width="stretch",
        )

    if n_communes and "cible_t2_vainqueur" in df_main.columns:
        st.markdown("<hr>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Distribution T2")
            counts = (
                df_main["cible_t2_vainqueur"]
                .value_counts()
                .rename({0: "Macron", 1: "Le Pen"})
            )
            fig = px.pie(
                values=counts.values,
                names=counts.index,
                color_discrete_map={"Macron": C_MACRON, "Le Pen": C_LEPEN},
                hole=0.45,
            )
            fig.update_traces(
                textposition="outside", textinfo="percent+label", textfont_size=12
            )
            fig.update_layout(**PLOT_LAYOUT, showlegend=False, height=300)
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.markdown("#### Par departement")
            if "code_departement" in df_main.columns:
                dept_df = (
                    df_main.groupby(["code_departement", "libelle_departement"])
                    .apply(
                        lambda g: pd.Series(
                            {
                                "Macron": (g["cible_t2_vainqueur"] == 0).sum(),
                                "Le Pen": (g["cible_t2_vainqueur"] == 1).sum(),
                            }
                        )
                    )
                    .reset_index()
                    .sort_values("code_departement")
                )
                fig2 = px.bar(
                    dept_df,
                    x="libelle_departement",
                    y=["Macron", "Le Pen"],
                    barmode="stack",
                    color_discrete_map={"Macron": C_MACRON, "Le Pen": C_LEPEN},
                    labels={"value": "Communes", "libelle_departement": ""},
                )
                fig2.update_layout(
                    **PLOT_LAYOUT,
                    xaxis_tickangle=-30,
                    height=300,
                    legend=dict(orientation="h", y=1.08),
                )
                st.plotly_chart(fig2, width="stretch")

elif page == "Random Forest":
    st.markdown(
        """
    <div class="page-header">
        <h1>Random Forest</h1>
        <div class="page-subtitle">Classification binaire — vainqueur T2 par commune</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    m = all_metrics.get("random_forest", {})
    if not m:
        st.warning("Modele non entraine.")
        st.code("python ml/run_training.py --model rf", language="bash")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric4("Accuracy", best_metric(m, ["val_accuracy", "test_accuracy"]))
    with c2:
        metric4("F1", best_metric(m, ["val_f1", "test_f1"]))
    with c3:
        metric4("CV AUC", m.get("cv_roc_auc"))
    with c4:
        metric4("CV AUC ±", m.get("cv_roc_auc_std"))
    with c5:
        metric4("Temps (s)", m.get("training_time_s"), fmt=".1f")

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Matrice de confusion")
        img_or_info(ARTIFACTS / "random_forest_confusion_matrix.png")
    with col2:
        st.markdown("#### Courbe ROC")
        img_or_info(ARTIFACTS / "random_forest_roc_curve.png")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Importance des features")

    df_imp = pd.DataFrame()
    try:
        from ml.models.random_forest import RandomForestModel

        files = list(ARTIFACTS.glob("random_forest*.joblib"))
        if files:
            tag = files[0].stem.replace("random_forest", "").lstrip("_")
            mod = RandomForestModel(artifact_dir=ARTIFACTS)
            mod.load(tag=tag)
            df_imp = mod.get_feature_importance(top_n=30)
    except Exception:
        pass

    if not df_imp.empty:
        df_plot = df_imp.sort_values("importance")
        fig = go.Figure(
            go.Bar(
                x=df_plot["importance"],
                y=df_plot["feature"],
                orientation="h",
                marker=dict(
                    color=df_plot["importance"],
                    colorscale=[[0, "#1c3a6e"], [1, C_MACRON]],
                    showscale=False,
                ),
                error_x=dict(
                    array=df_plot.get("std", [0] * len(df_plot)), color="#495057"
                ),
                hovertemplate="%{y}<br>%{x:.5f}<extra></extra>",
            )
        )
        fig.update_layout(
            **PLOT_LAYOUT,
            height=max(480, len(df_plot) * 22),
            margin=dict(t=10, b=20, l=200, r=40),
            yaxis=dict(tickfont=dict(size=10), gridcolor="#22252e"),
            xaxis_title="Importance",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        img_or_info(ARTIFACTS / "random_forest_feature_importance.png")

    bp = m.get("best_params", {})
    if bp:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### Hyperparametres")
        cols = st.columns(min(len(bp), 6))
        for col, (k, v) in zip(cols, bp.items()):
            col.metric(k, str(v))

elif page == "Gradient Boosting":
    st.markdown(
        """
    <div class="page-header">
        <h1>Gradient Boosting</h1>
        <div class="page-subtitle">Classification binaire avec filtrage anti-fuite T2</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    m = all_metrics.get("gradient_boosting", {})
    if not m:
        st.warning("Modele non entraine.")
        st.code("python ml/run_training.py --model gb", language="bash")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric4("Accuracy", best_metric(m, ["val_accuracy", "test_accuracy"]))
    with c2:
        metric4("AUC-ROC", m.get("test_roc_auc"))
    with c3:
        metric4("CV AUC", m.get("cv_roc_auc"))
    with c4:
        metric4("CV AUC ±", m.get("cv_roc_auc_std"))
    with c5:
        metric4("Temps (s)", m.get("training_time_s"), fmt=".1f")

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Matrice de confusion")
        img_or_info(ARTIFACTS / "gradient_boosting_confusion_matrix.png")
    with col2:
        st.markdown("#### Courbe ROC")
        img_or_info(ARTIFACTS / "gradient_boosting_roc_curve.png")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Analyse complete")
    img_or_info(
        ARTIFACTS / "gradient_boosting_results.png",
        "Feature importance — Confusion — ROC — Distribution des probabilites",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Importance des features")

    df_imp = pd.DataFrame()
    try:
        from ml.models.gradient_boosting import GradientBoostingModel

        files = list(ARTIFACTS.glob("gradient_boosting*.joblib"))
        if files:
            tag = files[0].stem.replace("gradient_boosting", "").lstrip("_")
            mod = GradientBoostingModel(artifact_dir=ARTIFACTS)
            mod.load(tag=tag)
            df_imp = mod.get_feature_importance(top_n=30)
    except Exception:
        pass

    if not df_imp.empty:
        df_plot = df_imp.sort_values("importance")
        fig = go.Figure(
            go.Bar(
                x=df_plot["importance"],
                y=df_plot["feature"],
                orientation="h",
                marker=dict(
                    color=df_plot["importance"],
                    colorscale=[[0, "#2d1f1f"], [1, C_LEPEN]],
                    showscale=False,
                ),
                hovertemplate="%{y}<br>%{x:.5f}<extra></extra>",
            )
        )
        fig.update_layout(
            **PLOT_LAYOUT,
            height=max(480, len(df_plot) * 22),
            margin=dict(t=10, b=20, l=200, r=40),
            yaxis=dict(tickfont=dict(size=10), gridcolor="#22252e"),
            xaxis_title="Importance",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        img_or_info(ARTIFACTS / "gradient_boosting_feature_importance.png")

    top_csv = ARTIFACTS / "gb_top_features.csv"
    if top_csv.exists():
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### Top 20 features")
        st.dataframe(pd.read_csv(top_csv).head(20), width="stretch")

    bp = m.get("best_params", {})
    if bp:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### Parametres du modele")
        cols = st.columns(min(len(bp), 7))
        for col, (k, v) in zip(cols, bp.items()):
            col.metric(k, str(v))

elif page == "LSTM":
    st.markdown(
        """
    <div class="page-header">
        <h1>LSTM</h1>
        <div class="page-subtitle">
            Architecture duale — sequences temporelles (2012 / 2017 / 2022-T1) + features socio-economiques
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    m = all_metrics.get("lstm", {})
    if not m:
        st.warning("Modele non entraine.")
        st.code("python ml/run_training.py --model lstm", language="bash")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric4("Val Accuracy", m.get("val_accuracy"))
    with c2:
        metric4("Val AUC", m.get("val_auc"))
    with c3:
        metric4("Best Val AUC", m.get("best_val_auc"))
    with c4:
        metric4("Epoques", m.get("epochs_trained"))
    with c5:
        metric4("Temps (s)", m.get("training_time_s"), fmt=".1f")

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Courbes d'entrainement")
        img_or_info(ARTIFACTS / "lstm_training_curves.png")
    with col2:
        st.markdown("#### Matrice de confusion")
        img_or_info(ARTIFACTS / "lstm_confusion_matrix.png")

    st.markdown("<hr>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Courbe ROC")
        img_or_info(ARTIFACTS / "lstm_roc_curve.png")

    with col4:
        st.markdown("#### Architecture")
        st.code(
            "Input A  sequences (batch, 3, 8)\n"
            "  LSTM(64, return_sequences=True)\n"
            "  BatchNorm + Dropout(0.3)\n"
            "  LSTM(32)\n"
            "  BatchNorm + Dropout(0.3)\n"
            "  Dense(16, relu)\n"
            "\n"
            "Input B  socio-eco (batch, 17)\n"
            "  Dense(32, relu) + BatchNorm + Dropout(0.2)\n"
            "  Dense(16, relu)\n"
            "\n"
            "Fusion -> Dense(16) -> Dropout(0.2)\n"
            "      -> Dense(1, sigmoid)\n"
            "\n"
            "Loss      : BinaryCrossentropy\n"
            "Optimizer : Adam(lr=0.001)\n"
            "Callbacks : EarlyStopping(patience=15)\n"
            "            ReduceLROnPlateau",
            language="text",
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Metriques completes")
    metric_pairs = [(k, v) for k, v in m.items() if isinstance(v, (int, float))]
    cols = st.columns(3)
    for i, (k, v) in enumerate(metric_pairs):
        with cols[i % 3]:
            metric4(k, v)

elif page == "Predictions":
    st.markdown(
        """
    <div class="page-header">
        <h1>Predictions par commune</h1>
        <div class="page-subtitle">Resultats des modeles sur l'ensemble des communes IDF</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    available_models = {
        "random_forest": "rf",
        "gradient_boosting": "gb",
        "lstm": "lstm",
    }
    trained_models = {k: v for k, v in available_models.items() if k in all_metrics}

    if not trained_models:
        st.warning("Aucun modele entraine.")
        st.code("python ml/run_training.py", language="bash")
        st.stop()

    col_sel, col_dept, col_res, col_kpi = st.columns([2, 2, 2, 1])

    with col_sel:
        sel_model = st.selectbox(
            "Modele",
            list(trained_models.keys()),
            format_func=lambda k: MODEL_LABELS.get(k, k),
        )
    prefix = trained_models[sel_model]
    df_pred = load_predictions_df(prefix)

    if df_pred.empty:
        st.warning(f"Pas de fichier de predictions pour {sel_model}.")
        st.stop()

    with col_dept:
        depts = ["Tous"] + (
            sorted(df_pred["code_departement"].astype(str).unique().tolist())
            if "code_departement" in df_pred.columns
            else []
        )
        dept_f = st.selectbox("Departement", depts)

    with col_res:
        corr_f = st.selectbox("Afficher", ["Tous", "Correctes", "Incorrectes"])

    df_show = df_pred.copy()
    if dept_f != "Tous" and "code_departement" in df_show.columns:
        df_show = df_show[df_show["code_departement"].astype(str) == dept_f]
    if corr_f == "Correctes" and "correct" in df_show.columns:
        df_show = df_show[df_show["correct"] == 1]
    elif corr_f == "Incorrectes" and "correct" in df_show.columns:
        df_show = df_show[df_show["correct"] == 0]

    with col_kpi:
        st.metric("Communes", len(df_show))

    if "correct" in df_show.columns and len(df_show) > 0:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Accuracy", f"{df_show['correct'].mean():.1%}")
        col_b.metric("Erreurs", int((df_show["correct"] == 0).sum()))
        col_c.metric("Correctes", int(df_show["correct"].sum()))

    display_cols = [
        c
        for c in [
            "libelle_commune",
            "code_departement",
            "vainqueur_predit",
            "ground_truth",
            "correct",
            "proba_macron",
            "proba_lepen",
        ]
        if c in df_show.columns
    ]

    rename_map = {
        "vainqueur_predit": "Predit",
        "ground_truth": "Reel",
        "correct": "OK",
        "proba_macron": "P(Macron)",
        "proba_lepen": "P(Le Pen)",
        "libelle_commune": "Commune",
        "code_departement": "Dept",
    }

    def _color_row(row):
        ok = row.get("OK", 1)
        bg = "#1a2e1a" if ok == 1 else "#2e1a1a"
        return [f"background-color:{bg}"] * len(row)

    st.dataframe(
        df_show[display_cols]
        .rename(columns=rename_map)
        .style.apply(_color_row, axis=1),
        width="stretch",
        height=400,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        if "code_departement" in df_pred.columns and "correct" in df_pred.columns:
            st.markdown("#### Precision par departement")
            dept_acc = (
                df_pred.groupby("code_departement")["correct"]
                .agg(["sum", "count"])
                .reset_index()
            )
            dept_acc["accuracy"] = dept_acc["sum"] / dept_acc["count"]
            dept_acc = dept_acc.sort_values("code_departement")
            fig = go.Figure(
                go.Bar(
                    x=dept_acc["code_departement"].astype(str),
                    y=dept_acc["accuracy"],
                    marker_color=C_MACRON,
                    text=[f"{v:.1%}" for v in dept_acc["accuracy"]],
                    textposition="outside",
                )
            )
            fig.update_layout(
                **PLOT_LAYOUT,
                yaxis=dict(tickformat=".0%", range=[0, 1.1], gridcolor="#22252e"),
                xaxis_title="Departement",
                height=320,
            )
            st.plotly_chart(fig, width="stretch")

    with col_g2:
        if "proba_lepen" in df_pred.columns:
            st.markdown("#### Distribution P(Le Pen)")
            fig = px.histogram(
                df_pred,
                x="proba_lepen",
                nbins=50,
                color=(
                    "vainqueur_predit"
                    if "vainqueur_predit" in df_pred.columns
                    else None
                ),
                color_discrete_map={"Macron": C_MACRON, "Le Pen": C_LEPEN},
                labels={"proba_lepen": "P(Le Pen)", "count": "Communes"},
            )
            fig.add_vline(
                x=0.5,
                line_dash="dot",
                line_color="#6b7280",
                annotation_text="0.5",
                annotation_font_color="#6b7280",
            )
            fig.update_layout(
                **PLOT_LAYOUT, height=320, legend=dict(orientation="h", y=1.05)
            )
            st.plotly_chart(fig, width="stretch")

elif page == "Comparaison":
    st.markdown(
        """
    <div class="page-header">
        <h1>Comparaison des modeles</h1>
        <div class="page-subtitle">Performance comparee sur les metriques principales</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    for col, (key, label) in zip(cols, MODEL_LABELS.items()):
        with col:
            trained = key in all_metrics
            border = "#3b5bdb" if trained else "#22252e"
            status = "Entraine" if trained else "—"
            acc_val = ""
            if trained:
                acc = best_metric(all_metrics[key], ["val_accuracy", "test_accuracy"])
                auc = best_metric(
                    all_metrics[key], ["val_auc", "test_auc", "test_roc_auc"]
                )
                acc_val = f"{acc:.3f}" if acc else "—"
                auc_val = f"{auc:.3f}" if auc else "—"
            st.markdown(
                f"""
            <div style="border:1px solid {border};border-radius:6px;padding:1rem;
                        background:#1c1f29;min-height:90px">
                <div style="font-size:0.75rem;font-weight:500;color:#9095a8;
                            margin-bottom:0.4rem">{label}</div>
                <div style="font-size:0.78rem;color:{'#40c057' if trained else '#495057'};
                            margin-bottom:0.5rem">{status}</div>
                {"<div style='font-size:0.72rem;color:#6b7280;font-family:JetBrains Mono,monospace'>Acc " + acc_val + " / AUC " + auc_val + "</div>" if trained else ""}
            </div>
            """,
                unsafe_allow_html=True,
            )

    if len(all_metrics) < 2:
        st.info(
            f"Entrainez au moins 2 modeles pour la comparaison. ({len(all_metrics)}/5)"
        )
        st.stop()

    st.markdown("<hr>", unsafe_allow_html=True)

    METRIC_GROUPS = {
        "Accuracy": ["val_accuracy", "test_accuracy"],
        "AUC-ROC": ["val_auc", "test_auc", "test_roc_auc"],
        "F1": ["val_f1", "test_f1"],
    }
    COLORS = [C_MACRON, C_LEPEN, "#40c057", "#d9770a", "#9775fa"]

    col_a, col_b = st.columns(2)
    cols_plot = [col_a, col_b, col_a, col_b]

    for idx, (group_name, keys) in enumerate(METRIC_GROUPS.items()):
        data = {
            MODEL_LABELS.get(n, n): best_metric(m, keys) for n, m in all_metrics.items()
        }
        data = {k: v for k, v in data.items() if v is not None}
        if not data:
            continue

        fig = go.Figure(
            go.Bar(
                x=list(data.keys()),
                y=list(data.values()),
                marker_color=COLORS[: len(data)],
                text=[f"{v:.4f}" for v in data.values()],
                textposition="outside",
                textfont=dict(size=11),
            )
        )
        ymax = max(data.values()) * 1.18 if data else 1
        fig.update_layout(
            **PLOT_LAYOUT,
            title=dict(text=group_name, font=dict(size=13, color="#c9cdd8")),
            yaxis=dict(range=[0, min(1.05, ymax)], gridcolor="#22252e"),
            height=300,
        )
        with cols_plot[idx % 2]:
            st.plotly_chart(fig, width="stretch")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Profil global")

    radar_metrics = ["val_accuracy", "val_auc", "val_f1"]
    eligible = {
        n: m for n, m in all_metrics.items() if any(k in m for k in radar_metrics)
    }

    if eligible:
        fig = go.Figure()
        categories = ["Accuracy", "AUC-ROC", "F1"]
        for (name, m), color in zip(eligible.items(), COLORS):
            values = [
                best_metric(m, ["val_accuracy", "test_accuracy"]) or 0,
                best_metric(m, ["val_auc", "test_auc", "test_roc_auc"]) or 0,
                best_metric(m, ["val_f1", "test_f1"]) or 0,
            ]
            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=MODEL_LABELS.get(name, name),
                    line_color=color,
                    fillcolor=color,
                    opacity=0.15,
                )
            )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor="#22252e",
                    tickfont=dict(size=9, color="#6b7280"),
                ),
                angularaxis=dict(gridcolor="#22252e"),
                bgcolor="#16181f",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9095a8", family="Inter"),
            legend=dict(orientation="h", y=-0.1),
            height=420,
            margin=dict(t=20, b=60),
        )
        st.plotly_chart(fig, width="stretch")

elif page == "Stats":
    st.markdown(
        """
    <div class="page-header">
        <h1>Stats</h1>
        <div class="page-subtitle">Metriques brutes, distribution des features, configuration ML</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Artefacts",
            "Metriques JSON",
            "Dataset",
            "Features",
            "Config",
        ]
    )

    with tab1:
        st.markdown("#### Fichiers generes")
        if ARTIFACTS.exists():
            files = sorted(ARTIFACTS.iterdir())
            data = [
                {
                    "Fichier": f.name,
                    "Taille": f"{f.stat().st_size / 1024:.1f} KB",
                    "Type": f.suffix,
                }
                for f in files
                if f.is_file()
            ]
            if data:
                st.dataframe(pd.DataFrame(data), width="stretch")
            else:
                st.info("Aucun artefact genere.")
        else:
            st.info("Repertoire artifacts introuvable.")

    with tab2:
        st.markdown("#### Metriques completes (JSON)")
        if all_metrics:
            for name, metrics in all_metrics.items():
                with st.expander(MODEL_LABELS.get(name, name), expanded=True):
                    st.json(
                        {
                            k: v
                            for k, v in metrics.items()
                            if not isinstance(v, (list, np.ndarray))
                        }
                    )
        else:
            st.info("Aucune metrique disponible. Lancez l'entrainement.")

    with tab3:
        st.markdown("#### Dataset")
        if len(df_main):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Communes", len(df_main))
            c2.metric("Colonnes", len(df_main.columns))
            c3.metric("Nulls", int(df_main.isnull().sum().sum()))
            c4.metric("Doublons", int(df_main.duplicated().sum()))

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("#### Variables cibles 2022")
            cible_cols = [
                c for c in df_main.columns if c.startswith("cible_") and "pct" in c
            ]
            if cible_cols:
                st.dataframe(df_main[cible_cols].describe().round(3), width="stretch")

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("#### Communes par departement")
            dept_df = (
                df_main.groupby(["code_departement", "libelle_departement"])
                .agg(n_communes=("code_commune", "count"))
                .reset_index()
            )
            st.dataframe(dept_df, width="stretch")
        else:
            st.warning("Dataset non disponible.")

    with tab4:
        st.markdown("#### Distribution des features")
        if len(df_main):
            num_cols = sorted(
                [
                    c
                    for c in df_main.select_dtypes("number").columns
                    if not c.startswith("cible_")
                ]
            )
            feat = st.selectbox("Feature", num_cols, label_visibility="collapsed")

            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(
                    df_main,
                    x=feat,
                    nbins=50,
                    labels={feat: feat, "count": "Communes"},
                    color_discrete_sequence=[C_MACRON],
                )
                fig.update_layout(
                    **PLOT_LAYOUT, title=dict(text="Distribution", font=dict(size=12))
                )
                st.plotly_chart(fig, width="stretch")

            with col2:
                if "cible_t2_vainqueur" in df_main.columns:
                    fig2 = px.box(
                        df_main,
                        x=df_main["cible_t2_vainqueur"].map({0: "Macron", 1: "Le Pen"}),
                        y=feat,
                        color=df_main["cible_t2_vainqueur"].map(
                            {0: "Macron", 1: "Le Pen"}
                        ),
                        color_discrete_map={"Macron": C_MACRON, "Le Pen": C_LEPEN},
                    )
                    fig2.update_layout(
                        **PLOT_LAYOUT,
                        showlegend=False,
                        title=dict(text="Par vainqueur T2", font=dict(size=12)),
                    )
                    st.plotly_chart(fig2, width="stretch")

            if "cible_t2_pct_lepen" in df_main.columns:
                corr = (
                    df_main[num_cols].corrwith(df_main["cible_t2_pct_lepen"]).dropna()
                )
                top_corr = corr.abs().nlargest(20).index
                vals = corr[top_corr]

                fig3 = go.Figure(
                    go.Bar(
                        x=vals.values,
                        y=vals.index,
                        orientation="h",
                        marker=dict(
                            color=vals.values,
                            colorscale=[[0, C_MACRON], [0.5, "#22252e"], [1, C_LEPEN]],
                            cmid=0,
                            showscale=True,
                        ),
                    )
                )
                fig3.update_layout(
                    **PLOT_LAYOUT,
                    title=dict(
                        text="Correlation avec % Le Pen T2 (top 20)", font=dict(size=12)
                    ),
                    margin=dict(t=30, b=20, l=200, r=80),
                    yaxis=dict(tickfont=dict(size=10)),
                    height=460,
                )
                st.plotly_chart(fig3, width="stretch")
        else:
            st.warning("Dataset non disponible.")

    with tab5:
        st.markdown("#### Configuration ML")
        from ml.config import (
            CV_FOLDS,
            FEATURE_SETS,
            GB_BEST_PARAMS,
            LSTM_CONFIG,
            TARGETS,
            TEST_SIZE,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ensembles de features**")
            for name, feats in FEATURE_SETS.items():
                st.markdown(f"- `{name}` — {len(feats)} features")
            st.markdown(f"**Test size** `{TEST_SIZE}` — **CV folds** `{CV_FOLDS}`")

            st.markdown("<br>**Variables cibles**", unsafe_allow_html=True)
            for key, col in TARGETS.items():
                st.markdown(f"- `{key}` → `{col}`")

        with col2:
            st.markdown("**Parametres GB**")
            st.json(GB_BEST_PARAMS)
            st.markdown("**Config LSTM**")
            st.json(LSTM_CONFIG)

if page == "Résultats DB":
    st.markdown(
        """
    <div class="page-header">
        <h1>Résultats ML — Base de données</h1>
        <div class="page-subtitle">
            Prédictions et métriques stockées dans PostgreSQL après entraînement
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if not DB_CONNECTED:
        st.error(
            "PostgreSQL non accessible. Vérifiez POSTGRES_PASSWORD et POSTGRES_HOST "
            "dans les variables d'environnement."
        )
        st.code(
            "POSTGRES_HOST=localhost\nPOSTGRES_PASSWORD=elections2022\n"
            "POSTGRES_DB=elections_idf\nPOSTGRES_USER=etl_admin",
            language="bash",
        )
    else:
        df_metrics = load_db_metrics()
        df_preds_all = load_db_predictions()

        models_in_db = df_metrics["model_name"].nunique() if not df_metrics.empty else 0
        preds_count = len(df_preds_all)
        communes_count = df_preds_all["code_commune"].nunique() if preds_count else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Modèles en DB", models_in_db)
        c2.metric("Prédictions totales", f"{preds_count:,}")
        c3.metric("Communes couvertes", f"{communes_count:,}")
        c4.metric("Connexion", "✓ OK")

        st.markdown("<hr>", unsafe_allow_html=True)

        if df_metrics.empty:
            st.warning(
                "Aucune métrique en base. Lancez le pipeline complet : "
                "`docker-compose up pipeline` ou `python run_all.py`"
            )
        else:
            st.markdown("#### Métriques par modèle")
            pivot = df_metrics.pivot_table(
                index=["model_name", "feature_set", "target"],
                columns="metric_name",
                values="metric_value",
                aggfunc="last",
            ).reset_index()
            st.dataframe(pivot, use_container_width=True)

            st.markdown("#### Comparaison des précisions")
            acc_col = next(
                (
                    c
                    for c in df_metrics["metric_name"].unique()
                    if "accuracy" in c.lower()
                ),
                None,
            )
            if acc_col:
                df_acc = df_metrics[df_metrics["metric_name"] == acc_col].copy()
                fig_acc = px.bar(
                    df_acc.sort_values("metric_value", ascending=True),
                    x="metric_value",
                    y="model_name",
                    color="feature_set",
                    barmode="group",
                    orientation="h",
                    labels={
                        "metric_value": "Accuracy",
                        "model_name": "Modèle",
                        "feature_set": "Feature set",
                    },
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_acc.update_layout(
                    paper_bgcolor="#16181f",
                    plot_bgcolor="#1a1d27",
                    font=dict(color="#9095a8"),
                    xaxis=dict(range=[0.5, 1.0], tickformat=".1%"),
                    height=320,
                    margin=dict(t=20, b=20),
                )
                st.plotly_chart(fig_acc, use_container_width=True)

        if not df_preds_all.empty:
            st.markdown("#### Prédictions vs Réalité")

            model_choices = sorted(df_preds_all["model_name"].unique())
            sel = st.selectbox("Modèle", model_choices, key="db_model_sel")
            df_sel = df_preds_all[df_preds_all["model_name"] == sel].copy()

            if not df_sel.empty:
                has_correct = (
                    "correct" in df_sel.columns and df_sel["correct"].notna().any()
                )
                if has_correct:
                    acc_val = df_sel["correct"].astype(float).mean()
                    n_ok = df_sel["correct"].astype(bool).sum()
                    n_ko = (~df_sel["correct"].astype(bool)).sum()
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Accuracy globale", f"{acc_val:.1%}")
                    k2.metric("Communes correctes", f"{n_ok:,}")
                    k3.metric("Communes incorrectes", f"{n_ko:,}")

                col_l, col_r = st.columns(2)

                with col_l:
                    if "prediction" in df_sel.columns:
                        pred_counts = (
                            df_sel["prediction"]
                            .map(
                                {"0": "Macron", "1": "Le Pen", 0: "Macron", 1: "Le Pen"}
                            )
                            .value_counts()
                            .reset_index()
                        )
                        fig_pred = px.bar(
                            pred_counts,
                            x="prediction",
                            y="count",
                            color="prediction",
                            color_discrete_map={"Macron": C_MACRON, "Le Pen": C_LEPEN},
                            labels={"prediction": "Prédit", "count": "Communes"},
                            title="Distribution des prédictions",
                        )
                        fig_pred.update_layout(
                            paper_bgcolor="#16181f",
                            plot_bgcolor="#1a1d27",
                            font=dict(color="#9095a8"),
                            showlegend=False,
                            height=260,
                            margin=dict(t=30, b=20),
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)

                with col_r:
                    if "ground_truth" in df_sel.columns:
                        gt_counts = (
                            df_sel["ground_truth"]
                            .map(
                                {"0": "Macron", "1": "Le Pen", 0: "Macron", 1: "Le Pen"}
                            )
                            .value_counts()
                            .reset_index()
                        )
                        fig_gt = px.bar(
                            gt_counts,
                            x="ground_truth",
                            y="count",
                            color="ground_truth",
                            color_discrete_map={"Macron": C_MACRON, "Le Pen": C_LEPEN},
                            labels={"ground_truth": "Réel", "count": "Communes"},
                            title="Distribution réelle (vérité terrain)",
                        )
                        fig_gt.update_layout(
                            paper_bgcolor="#16181f",
                            plot_bgcolor="#1a1d27",
                            font=dict(color="#9095a8"),
                            showlegend=False,
                            height=260,
                            margin=dict(t=30, b=20),
                        )
                        st.plotly_chart(fig_gt, use_container_width=True)

                if (
                    has_correct
                    and "prediction" in df_sel.columns
                    and "ground_truth" in df_sel.columns
                ):
                    st.markdown("##### Matrice de confusion (données DB)")
                    labels_map = {
                        "0": "Macron",
                        "1": "Le Pen",
                        0: "Macron",
                        1: "Le Pen",
                    }
                    df_cm = df_sel.copy()
                    df_cm["pred_lbl"] = (
                        df_cm["prediction"].map(labels_map).fillna(df_cm["prediction"])
                    )
                    df_cm["true_lbl"] = (
                        df_cm["ground_truth"]
                        .map(labels_map)
                        .fillna(df_cm["ground_truth"])
                    )

                    cm_df = (
                        df_cm.groupby(["true_lbl", "pred_lbl"])
                        .size()
                        .reset_index(name="count")
                    )
                    cm_pivot = cm_df.pivot(
                        index="true_lbl", columns="pred_lbl", values="count"
                    ).fillna(0)

                    fig_cm = px.imshow(
                        cm_pivot,
                        text_auto=True,
                        color_continuous_scale=[[0, "#1a1d27"], [1, C_MACRON]],
                        labels={"x": "Prédit", "y": "Réel", "color": "Communes"},
                        title=f"Matrice de confusion — {sel}",
                    )
                    fig_cm.update_layout(
                        paper_bgcolor="#16181f",
                        plot_bgcolor="#1a1d27",
                        font=dict(color="#9095a8"),
                        height=320,
                        margin=dict(t=40, b=20),
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

                if (
                    "probability" in df_sel.columns
                    and df_sel["probability"].notna().any()
                ):
                    st.markdown("##### Distribution des probabilités de prédiction")
                    df_prob = df_sel[df_sel["probability"].notna()].copy()
                    df_prob["correct_lbl"] = (
                        df_prob["correct"]
                        .map(
                            {
                                True: "Correct",
                                False: "Incorrect",
                                1: "Correct",
                                0: "Incorrect",
                            }
                        )
                        .fillna("Inconnu")
                    )
                    fig_prob = px.histogram(
                        df_prob,
                        x="probability",
                        color="correct_lbl",
                        nbins=40,
                        barmode="overlay",
                        opacity=0.7,
                        color_discrete_map={"Correct": C_MACRON, "Incorrect": C_LEPEN},
                        labels={
                            "probability": "Probabilité prédite",
                            "count": "Communes",
                        },
                        title="Confiance du modèle vs correction",
                    )
                    fig_prob.update_layout(
                        paper_bgcolor="#16181f",
                        plot_bgcolor="#1a1d27",
                        font=dict(color="#9095a8"),
                        height=300,
                        margin=dict(t=40, b=20),
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)

                if "code_departement" in df_sel.columns and has_correct:
                    st.markdown("##### Précision par département")
                    df_dept = (
                        df_sel.groupby("code_departement")["correct"]
                        .agg(accuracy="mean", count="count")
                        .reset_index()
                    )
                    DEPT_NAMES = {
                        "75": "Paris (75)",
                        "77": "S&M (77)",
                        "78": "Yvelines (78)",
                        "91": "Essonne (91)",
                        "92": "Hts-de-Seine (92)",
                        "93": "Seine-St-Denis (93)",
                        "94": "Val-de-Marne (94)",
                        "95": "Val-d'Oise (95)",
                    }
                    df_dept["dept_label"] = (
                        df_dept["code_departement"]
                        .map(DEPT_NAMES)
                        .fillna(df_dept["code_departement"])
                    )
                    fig_dept = px.bar(
                        df_dept.sort_values("accuracy"),
                        x="accuracy",
                        y="dept_label",
                        orientation="h",
                        text=df_dept["accuracy"].map("{:.1%}".format),
                        labels={"accuracy": "Accuracy", "dept_label": "Département"},
                        color="accuracy",
                        color_continuous_scale=[
                            [0, C_LEPEN],
                            [0.5, "#8b9ab0"],
                            [1, C_MACRON],
                        ],
                        range_color=[0.7, 1.0],
                        title=f"Accuracy par département — {sel}",
                    )
                    fig_dept.update_traces(textposition="outside")
                    fig_dept.update_layout(
                        paper_bgcolor="#16181f",
                        plot_bgcolor="#1a1d27",
                        font=dict(color="#9095a8"),
                        coloraxis_showscale=False,
                        height=340,
                        margin=dict(t=40, b=20, l=140, r=60),
                        xaxis=dict(tickformat=".0%"),
                    )
                    st.plotly_chart(fig_dept, use_container_width=True)

                if has_correct:
                    wrong_df = df_sel[~df_sel["correct"].astype(bool)].copy()
                    if not wrong_df.empty:
                        st.markdown(f"##### Communes mal prédites ({len(wrong_df)})")
                        disp_cols = [
                            c
                            for c in [
                                "libelle_commune",
                                "code_departement",
                                "prediction",
                                "ground_truth",
                                "probability",
                            ]
                            if c in wrong_df.columns
                        ]
                        st.dataframe(
                            wrong_df[disp_cols].rename(
                                columns={
                                    "libelle_commune": "Commune",
                                    "code_departement": "Dept",
                                    "prediction": "Prédit",
                                    "ground_truth": "Réel",
                                    "probability": "Prob.",
                                }
                            ),
                            use_container_width=True,
                            height=280,
                        )

st.markdown("<hr>", unsafe_allow_html=True)
status_parts = []
for k, label in MODEL_LABELS.items():
    trained = k in all_metrics
    dot = "status-trained" if trained else "status-pending"
    status_parts.append(
        f'<span class="{dot}"></span>'
        f'<span style="color:#6b7280;font-size:0.75rem">{label}</span>'
    )

st.markdown(
    '<div style="display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap">'
    + "".join(status_parts)
    + f'<span style="color:#495057;font-size:0.75rem;margin-left:auto">'
    f"{len(all_metrics)}/5 modeles entraines</span>" + "</div>",
    unsafe_allow_html=True,
)
