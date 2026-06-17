"""
Extraction — Filosofi 2017 (revenus, pauvreté par commune).
Supporte CSV (source INSEE directe) et XLSX (data.gouv.fr, feuille COM en-tête ligne 6).
"""

from pathlib import Path

import pandas as pd

from etl.helpers import norm_code
from monitoring.logger import get_logger

log = get_logger(__name__)

IDF_DEPTS = frozenset({"75", "77", "78", "91", "92", "93", "94", "95"})


def extract_pauvrete(pauvrete_file: Path) -> pd.DataFrame:
    """
    Charge le fichier Filosofi 2017.

    Formats supportés :
    - CSV (insee.fr direct) : séparateur ';', header ligne 0, toutes géographies
    - XLSX (data.gouv.fr)   : feuille 'COM', header ligne 5 ou 0

    Returns:
        DataFrame brut filtré IDF communes (CODGEO 5 chars).
    """
    # Chercher aussi l'extension alternative si le fichier principal est absent
    if not pauvrete_file.exists():
        alt_ext = ".xlsx" if pauvrete_file.suffix.lower() == ".csv" else ".csv"
        alt = pauvrete_file.with_suffix(alt_ext)
        if alt.exists():
            pauvrete_file = alt
        else:
            log.warning(
                f"Fichier Filosofi absent : {pauvrete_file} — pauvrete ignorée (features revenue = NaN)"
            )
            return pd.DataFrame(columns=["code_commune"])

    ext = pauvrete_file.suffix.lower()
    df = None

    try:
        if ext in (".xlsx", ".xls"):
            df = _read_filosofi_xlsx(pauvrete_file)
        else:
            df = _read_filosofi_csv(pauvrete_file)
    except Exception as e:
        log.warning(
            f"Fichier Filosofi invalide ({e}) — pauvrete ignorée (revenus = NaN)"
        )
        return pd.DataFrame(columns=["code_commune"])

    if df is None or "CODGEO" not in df.columns:
        log.warning("Filosofi : colonne CODGEO introuvable — pauvrete ignorée")
        return pd.DataFrame(columns=["code_commune"])

    df["CODGEO"] = norm_code(df["CODGEO"].astype(str))
    # Garder uniquement les communes IDF (code 5 chars, dept en IDF)
    df = df[
        (df["CODGEO"].str.len() == 5) & (df["CODGEO"].str[:2].isin(IDF_DEPTS))
    ].copy()

    log.info(f"Pauvrete brut : {len(df):,} lignes")
    return df


def _read_filosofi_xlsx(path: Path) -> pd.DataFrame:
    """Lit le fichier XLSX Filosofi, essaie plusieurs combinaisons feuille/header."""
    for sheet in ("COM", 0):
        for hdr in (5, 0, 1, 4, 6):
            try:
                df = pd.read_excel(path, sheet_name=sheet, header=hdr)
                if "CODGEO" in df.columns and len(df.columns) > 5:
                    return df
            except Exception:
                continue
    raise ValueError(
        f"Aucune combinaison feuille/header n'a trouvé la colonne CODGEO dans {path.name}"
    )


def _read_filosofi_csv(path: Path) -> pd.DataFrame:
    """Lit le fichier CSV Filosofi (INSEE direct, séparateur ';')."""
    for sep in (";", ",", "\t"):
        try:
            df = pd.read_csv(
                path, sep=sep, dtype=str, low_memory=False, encoding="utf-8"
            )
            if "CODGEO" in df.columns and len(df.columns) > 5:
                return df
        except Exception:
            continue
    for sep in (";", ",", "\t"):
        try:
            df = pd.read_csv(
                path, sep=sep, dtype=str, low_memory=False, encoding="latin-1"
            )
            if "CODGEO" in df.columns and len(df.columns) > 5:
                return df
        except Exception:
            continue
    raise ValueError(f"Impossible de parser le CSV Filosofi : {path.name}")
