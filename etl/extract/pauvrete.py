"""
Extraction — base-cc-filosofi-2017.xlsx (revenus, pauvreté Filosofi 2017).
"""

from pathlib import Path
import pandas as pd

from etl.helpers import norm_code
from monitoring.logger import get_logger

log = get_logger(__name__)

IDF_DEPTS = frozenset({"75", "77", "78", "91", "92", "93", "94", "95"})


def extract_pauvrete(pauvrete_file: Path) -> pd.DataFrame:
    """
    Charge le fichier Filosofi 2017 (feuille COM, en-tête à la ligne 6).

    Returns:
        DataFrame brut filtré IDF.
    """
    if not pauvrete_file.exists():
        raise FileNotFoundError(f"Fichier introuvable : {pauvrete_file}")

    df = pd.read_excel(pauvrete_file, sheet_name="COM", header=5)
    df["CODGEO"] = norm_code(df["CODGEO"].astype(str))
    df = df[df["CODGEO"].str[:2].isin(IDF_DEPTS)].copy()

    log.info(f"Pauvrete brut : {len(df):,} lignes")
    return df
