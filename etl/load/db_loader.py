"""
Chargement Gold → PostgreSQL.
Upsert idempotent via ON CONFLICT DO UPDATE (SQLAlchemy Core).
Utilisé optionnellement si DATABASE_URL est configuré.
"""

import pandas as pd
from sqlalchemy import create_engine, text

from monitoring.logger import get_logger
from security.encryption import SENSITIVE_DB_COLUMNS

log = get_logger(__name__)

GOLD_TABLE = "gold.ml_features"


def load_to_db(df: pd.DataFrame, database_url: str) -> None:
    """
    Charge le dataset gold dans PostgreSQL.
    Chiffre les colonnes sensibles (libelle_commune, libelle_departement) si
    ENCRYPTION_ENABLED=true dans la configuration.

    Args:
        df:           DataFrame gold
        database_url: URL SQLAlchemy (ex. postgresql+psycopg2://user:pass@host/db)
    """
    from config.settings import settings

    engine = create_engine(database_url, pool_pre_ping=True)

    df_out = df.copy()
    for col in df_out.select_dtypes(include="object").columns:
        df_out[col] = df_out[col].astype(str)

    encryptor = settings.encryptor
    if encryptor is not None:
        df_out = encryptor.encrypt_dataframe(df_out, SENSITIVE_DB_COLUMNS)
        log.info(f"Chiffrement activé — colonnes : {SENSITIVE_DB_COLUMNS}")

    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS gold"))

        df_out.to_sql(
            name="ml_features",
            schema="gold",
            con=conn,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=500,
        )

    log.info(f"Chargé {len(df_out):,} lignes dans {GOLD_TABLE}")


def check_db_connection(database_url: str) -> bool:
    """Vérifie la connexion à la base de données."""
    try:
        engine = create_engine(database_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("Connexion PostgreSQL OK")
        return True
    except Exception as e:
        log.warning(f"PostgreSQL non disponible : {e}")
        return False
