"""
Populate DB depuis les artefacts locaux (CSV + JSON).
Usage : python db/populate_from_artifacts.py
Conçu pour tourner DANS Docker (POSTGRES_HOST=postgres) OU en local.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from sqlalchemy import create_engine, text

ARTIFACTS = ROOT / "ml" / "artifacts"


# ── Connexion ─────────────────────────────────────────────────────────────────


def get_db_url() -> str:
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "elections_idf")
    user = os.environ.get("POSTGRES_USER", "etl_admin")
    return (
        f"postgresql+psycopg2://{user}:"
        + os.environ.get("POSTGRES_PASSWORD", "elections2022")  # NOSONAR
        + f"@{host}:{port}/{db}"
    )


# ── Map des CSVs de prédictions ───────────────────────────────────────────────
PRED_FILES = [
    (
        "gradient_boosting",
        "post_t1",
        "regression_macron",
        "gb_predictions_post_t1_regression_macron.csv",
    ),
    (
        "gradient_boosting",
        "pre_vote",
        "regression_macron",
        "gb_predictions_pre_vote_regression_macron.csv",
    ),
    (
        "random_forest",
        "post_t1",
        "regression_macron",
        "rf_predictions_post_t1_regression_macron.csv",
    ),
    (
        "random_forest",
        "pre_vote",
        "regression_macron",
        "rf_predictions_pre_vote_regression_macron.csv",
    ),
    (
        "decision_tree",
        "pre_vote",
        "regression_macron",
        "dt_predictions_pre_vote_regression_macron.csv",
    ),
    (
        "mlp",
        "pre_vote",
        "regression_macron",
        "mlp_predictions_pre_vote_regression_macron.csv",
    ),
    ("lstm", "lstm", "regression_macron", "lstm_predictions_regression_macron.csv"),
]

METRICS_FILES = [
    (
        "gradient_boosting",
        "post_t1",
        "regression_macron",
        "gradient_boosting_metrics_post_t1_regression_macron.json",
    ),
    (
        "gradient_boosting",
        "pre_vote",
        "regression_macron",
        "gradient_boosting_metrics_pre_vote_regression_macron.json",
    ),
    (
        "random_forest",
        "post_t1",
        "regression_macron",
        "random_forest_metrics_post_t1_regression_macron.json",
    ),
    (
        "random_forest",
        "pre_vote",
        "regression_macron",
        "random_forest_metrics_pre_vote_regression_macron.json",
    ),
    (
        "decision_tree",
        "pre_vote",
        "regression_macron",
        "decision_tree_metrics_pre_vote_regression_macron.json",
    ),
    (
        "mlp",
        "pre_vote",
        "regression_macron",
        "mlp_metrics_pre_vote_regression_macron.json",
    ),
    ("lstm", "lstm", "regression_macron", "lstm_metrics_lstm_regression_macron.json"),
]

_CREATE_GOLD_SCHEMA = "CREATE SCHEMA IF NOT EXISTS gold"


def populate_model_predictions(engine, run_id: str) -> None:
    ok, skipped = [], []
    for model_name, feature_set, target, fname in PRED_FILES:
        path = ARTIFACTS / fname
        if not path.exists():
            skipped.append(fname)
            continue
        df = pd.read_csv(path, dtype={"code_commune": str, "code_departement": str})
        df["run_id"] = run_id
        df["model_name"] = model_name
        df["feature_set"] = feature_set
        df["target"] = target

        if "probability" not in df.columns:
            df["probability"] = df.get("proba_lepen")
        if "correct" in df.columns:
            df["correct"] = df["correct"].astype(bool)

        keep = [
            "run_id",
            "model_name",
            "feature_set",
            "target",
            "code_commune",
            "libelle_commune",
            "code_departement",
            "prediction",
            "probability",
            "ground_truth",
            "correct",
        ]
        df_out = df[[c for c in keep if c in df.columns]]

        with engine.begin() as conn:
            conn.execute(text(_CREATE_GOLD_SCHEMA))
            conn.execute(
                text(
                    "DELETE FROM gold.model_predictions WHERE model_name=:m AND feature_set=:f AND target=:t"
                ),
                {"m": model_name, "f": feature_set, "t": target},
            )
        df_out.to_sql(
            "model_predictions",
            schema="gold",
            con=engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
        )
        ok.append(f"{model_name}/{feature_set}")
    log.info(
        f"  model_predictions : {len(ok)} modèles insérés"
        + (f" · {len(skipped)} SKIP" if skipped else "")
    )
    if skipped:
        log.warning(f"  Fichiers introuvables : {skipped}")


def populate_model_metrics(engine, run_id: str) -> None:
    ok, skipped = [], []
    for model_name, feature_set, target, fname in METRICS_FILES:
        path = ARTIFACTS / fname
        if not path.exists():
            skipped.append(fname)
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for k, v in m.items():
            if isinstance(v, (int, float)):
                rows.append(
                    {
                        "run_id": run_id,
                        "model_name": model_name,
                        "feature_set": feature_set,
                        "target": target,
                        "metric_name": k,
                        "metric_value": float(v),
                    }
                )
        if not rows:
            continue
        df = pd.DataFrame(rows)
        with engine.begin() as conn:
            conn.execute(text(_CREATE_GOLD_SCHEMA))
            conn.execute(
                text(
                    "DELETE FROM gold.model_metrics WHERE model_name=:m AND feature_set=:f AND target=:t"
                ),
                {"m": model_name, "f": feature_set, "t": target},
            )
        df.to_sql(
            "model_metrics",
            schema="gold",
            con=engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=200,
        )
        ok.append(f"{model_name}/{feature_set}")
    log.info(
        f"  model_metrics : {len(ok)} modèles insérés"
        + (f" · {len(skipped)} SKIP" if skipped else "")
    )
    if skipped:
        log.warning(f"  Fichiers introuvables : {skipped}")


def populate_ml_features(engine, csv_path: Path) -> None:
    if not csv_path.exists():
        log.warning(f"  ml_features SKIP — {csv_path} introuvable")
        return
    df = pd.read_csv(csv_path, dtype={"code_commune": str, "code_departement": str})
    df["code_commune"] = df["code_commune"].str.zfill(5)
    with engine.begin() as conn:
        conn.execute(text(_CREATE_GOLD_SCHEMA))
    df.to_sql(
        "ml_features",
        schema="gold",
        con=engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=500,
    )
    log.info(f"  ml_features : {len(df)} communes × {len(df.columns)} colonnes")


def populate_datamart(csv_path: Path, db_url: str) -> None:
    sys.path.insert(0, str(ROOT))
    try:
        from db.populate_datamart import run as _run

        ok = _run(csv_path=csv_path, db_url=db_url.replace("+psycopg2", ""))
        if not ok:
            log.warning("  datamart non alimenté (CSV ou DB indisponible)")
    except Exception as e:
        log.exception(f"  datamart ERREUR : {e}")


def _log_counts(engine) -> None:
    counts = []
    for q, label in [
        ("SELECT COUNT(*) FROM gold.model_predictions", "model_predictions"),
        ("SELECT COUNT(*) FROM gold.model_metrics", "model_metrics"),
        ("SELECT COUNT(*) FROM gold.ml_features", "ml_features"),
        ("SELECT COUNT(*) FROM datamart.dim_commune", "dim_commune"),
        ("SELECT COUNT(*) FROM datamart.fait_resultats_electoraux", "fait_resultats"),
    ]:
        try:
            with engine.connect() as conn:
                n = conn.execute(text(q)).scalar()
            counts.append(f"{label}={n:,}")
        except Exception:
            counts.append(f"{label}=ERR")
    log.info("  Lignes DB : " + " · ".join(counts))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    db_url = get_db_url()
    log.info(f"DB: {db_url.split('@')[-1]}")
    engine = create_engine(db_url, pool_pre_ping=True)

    run_id = (
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_artifacts"
    )

    gold_csv = Path(os.environ.get("DATA_PATH", "/data/dataset_elections_2022_idf.csv"))
    if not gold_csv.exists():
        gold_csv = (
            Path(os.environ.get("DATA_ROOT", "/data"))
            / "dataset_elections_2022_idf.csv"
        )

    populate_model_predictions(engine, run_id)
    populate_model_metrics(engine, run_id)
    populate_ml_features(engine, gold_csv)
    populate_datamart(gold_csv, db_url)
    _log_counts(engine)
    log.info("Done.")


if __name__ == "__main__":
    main()
