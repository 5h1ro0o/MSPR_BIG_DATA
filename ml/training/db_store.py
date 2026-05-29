"""
ml/training/db_store.py
=======================
Stockage des résultats ML (prédictions + métriques) dans PostgreSQL.

Tables cibles :
  gold.model_predictions  — une ligne par (commune × modèle × run)
  gold.model_metrics      — une ligne par (métrique × modèle × run)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

import logging
log = logging.getLogger(__name__)

def _get_engine(database_url: str):
    return create_engine(database_url, pool_pre_ping=True)

def store_predictions(
    predictions_df: pd.DataFrame,
    model_name:    str,
    feature_set:   str,
    target:        str,
    run_id:        str,
    database_url:  str,
) -> bool:
    """
    Insère les prédictions d'un modèle dans gold.model_predictions.

    Le DataFrame doit avoir au minimum :
      - prediction, ground_truth, correct
      - optionnel : code_commune, libelle_commune, code_departement, probability/confidence
    """
    if predictions_df is None or predictions_df.empty:
        log.warning(f"  [db_store] predictions vides pour {model_name}")
        return False

    df = predictions_df.copy()
    df["run_id"]      = run_id
    df["model_name"]  = model_name
    df["feature_set"] = feature_set
    df["target"]      = target

    for col in ["code_commune", "libelle_commune", "code_departement"]:
        if col not in df.columns:
            df[col] = None
    for col in ["probability", "confidence"]:
        if col in df.columns:
            df = df.rename(columns={col: "probability"})
            break
    if "probability" not in df.columns:
        df["probability"] = None

    keep = ["run_id", "model_name", "feature_set", "target",
            "code_commune", "libelle_commune", "code_departement",
            "prediction", "probability", "ground_truth", "correct"]
    df = df[[c for c in keep if c in df.columns]]

    if "correct" in df.columns:
        df["correct"] = df["correct"].astype(bool)

    try:
        engine = _get_engine(database_url)
        with engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS gold"))
            conn.execute(
                text("DELETE FROM gold.model_predictions WHERE run_id=:rid AND model_name=:m"),
                {"rid": run_id, "m": model_name},
            )
        df.to_sql(
            "model_predictions",
            schema="gold",
            con=engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
        )
        log.info(f"  [db_store] {len(df):,} prédictions → gold.model_predictions ({model_name})")
        return True
    except Exception as e:
        log.warning(f"  [db_store] Erreur stockage prédictions ({model_name}): {e}")
        return False

def store_metrics(
    metrics:      dict[str, Any],
    model_name:   str,
    feature_set:  str,
    target:       str,
    run_id:       str,
    database_url: str,
) -> bool:
    """
    Insère les métriques d'un modèle dans gold.model_metrics.
    Seules les valeurs scalaires (float/int) sont stockées.
    """
    if not metrics:
        return False

    rows = []
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            rows.append({
                "run_id":       run_id,
                "model_name":   model_name,
                "feature_set":  feature_set,
                "target":       target,
                "metric_name":  metric_name,
                "metric_value": float(value),
            })

    if not rows:
        return False

    df = pd.DataFrame(rows)

    try:
        engine = _get_engine(database_url)
        with engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS gold"))
            conn.execute(
                text(
                    "DELETE FROM gold.model_metrics "
                    "WHERE run_id=:rid AND model_name=:m"
                ),
                {"rid": run_id, "m": model_name},
            )
        df.to_sql(
            "model_metrics",
            schema="gold",
            con=engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
        )
        log.info(f"  [db_store] {len(rows)} métriques → gold.model_metrics ({model_name})")
        return True
    except Exception as e:
        log.warning(f"  [db_store] Erreur stockage métriques ({model_name}): {e}")
        return False

def load_predictions(database_url: str, model_name: str = None) -> pd.DataFrame:
    """Charge les prédictions depuis la DB (dernier run par modèle)."""
    try:
        engine = _get_engine(database_url)
        where  = f"WHERE model_name = '{model_name}'" if model_name else ""
        query  = f"""
            SELECT DISTINCT ON (model_name, feature_set, target, code_commune)
                *
            FROM gold.model_predictions
            {where}
            ORDER BY model_name, feature_set, target, code_commune, created_at DESC
        """
        return pd.read_sql(query, engine)
    except Exception as e:
        log.warning(f"  [db_store] Erreur lecture prédictions: {e}")
        return pd.DataFrame()

def load_metrics(database_url: str) -> pd.DataFrame:
    """Charge les métriques depuis la vue gold.latest_model_metrics."""
    try:
        engine = _get_engine(database_url)
        return pd.read_sql("SELECT * FROM gold.latest_model_metrics", engine)
    except Exception as e:
        log.warning(f"  [db_store] Erreur lecture métriques: {e}")
        return pd.DataFrame()
