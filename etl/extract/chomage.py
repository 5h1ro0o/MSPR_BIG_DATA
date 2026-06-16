"""
Extraction — données chômage :
  1. Historique IDF par commune (2007-2020)
  2. DS_RP_EMPLOI 2022 (niveau commune, toutes CSP)
"""

from pathlib import Path

import pandas as pd

from etl.helpers import norm_code
from monitoring.logger import get_logger

log = get_logger(__name__)

IDF_DEPTS = frozenset({"75", "77", "78", "91", "92", "93", "94", "95"})


def extract_chomage_historique(chomage_hist_file: Path) -> pd.DataFrame:
    """
    Charge le fichier de chômage historique par commune IDF.

    Returns:
        DataFrame brut avec toutes les colonnes du fichier source.
    """
    if not chomage_hist_file.exists():
        log.warning(f"Fichier chômage absent : {chomage_hist_file} — chômage historique = NaN")
        return pd.DataFrame(columns=["code_commune"])

    df = pd.read_csv(chomage_hist_file, sep=";", encoding="utf-8")
    df["code_commune"] = norm_code(df["insee"])

    log.info(f"Chomage historique brut : {len(df):,} lignes")
    return df


def extract_emploi_2022(emploi_file: Path) -> pd.DataFrame:
    """
    Charge le fichier emploi 2022 par commune.
    Supporte base-cc (CODGEO) et DS (GEO_OBJECT) — détection auto du séparateur.

    Returns:
        DataFrame brut — filtrage IDF fait en transform.
    """
    if not emploi_file.exists():
        log.warning(f"Fichier emploi absent : {emploi_file} — taux chômage 2022 = NaN")
        return pd.DataFrame(columns=["code_commune"])

    for sep in (";", ",", "\t"):
        try:
            df = pd.read_csv(emploi_file, sep=sep, low_memory=False)
            if len(df.columns) > 3:
                log.info(f"Emploi 2022 brut : {len(df):,} lignes")
                return df
        except Exception:
            continue

    log.warning(f"Impossible de parser : {emploi_file}")
    return pd.DataFrame(columns=["code_commune"])
