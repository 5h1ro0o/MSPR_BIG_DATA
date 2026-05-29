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
        raise FileNotFoundError(f"Fichier introuvable : {chomage_hist_file}")

    df = pd.read_csv(chomage_hist_file, sep=";", encoding="utf-8")
    df["code_commune"] = norm_code(df["insee"])

    log.info(f"Chomage historique brut : {len(df):,} lignes")
    return df


def extract_emploi_2022(emploi_file: Path) -> pd.DataFrame:
    """
    Charge DS_RP_EMPLOI_LR_COMP_2022_data.csv (complet, sans filtrage).

    Returns:
        DataFrame brut — le filtrage COM/CSP/_T/2022/IDF est fait en transform.
    """
    if not emploi_file.exists():
        raise FileNotFoundError(f"Fichier introuvable : {emploi_file}")

    df = pd.read_csv(emploi_file, sep=";", low_memory=False)
    log.info(f"Emploi 2022 brut : {len(df):,} lignes")
    return df
