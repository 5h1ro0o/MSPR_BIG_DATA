"""
Fonctions utilitaires partagées par tous les modules ETL.
"""

import unicodedata

import numpy as np
import pandas as pd


def norm_code(series: pd.Series) -> pd.Series:
    """Normalise les codes communes en strings de 5 caractères (zfill)."""
    return series.astype(str).str.strip().str.zfill(5)


def pct(numerator: pd.Series, denominator: pd.Series, decimals: int = 2) -> pd.Series:
    """Calcule un pourcentage en gérant les divisions par zéro."""
    return (numerator / denominator.replace(0, np.nan) * 100).round(decimals)


def norm_nom(s) -> str:
    """Supprime les accents et met en majuscules (MÉLENCHON → MELENCHON)."""
    if pd.isna(s):
        return ""
    nfkd = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).upper().strip()


def cap_outliers(df: pd.DataFrame, cols: list, factor: float = 3.0) -> pd.DataFrame:
    """
    Écrête les outliers par la méthode IQR.
    Les valeurs < Q1 − factor*IQR sont ramenées à Q1 − factor*IQR,
    idem pour Q3 + factor*IQR.
    """
    for col in cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - factor * iqr, upper=q3 + factor * iqr)
    return df


def impute_missing(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Impute les valeurs manquantes :
      - médiane si |skewness| > 1  (distribution asymétrique)
      - moyenne sinon
      - 0.0 si la colonne est entièrement nulle (source absente)
    """
    for col in cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].isna().sum() == 0:
            continue
        skew = df[col].skew()
        filler = df[col].median() if abs(skew) > 1 else df[col].mean()
        if pd.isna(filler):
            filler = 0.0  # colonne entièrement absente — source non disponible
        df[col] = df[col].fillna(filler)
    return df


def recalc_winner_marge(
    df: pd.DataFrame,
    prefix: str,
    cand_a: str,
    cand_b: str,
    label_a: int = 0,
    label_b: int = 1,
) -> pd.DataFrame:
    """
    Recalcule {prefix}_vainqueur et {prefix}_marge APRES tout le nettoyage.
    Utilise .values pour éviter les désalignements d'index pandas.
    """
    col_a = f"{prefix}_pct_{cand_a}"
    col_b = f"{prefix}_pct_{cand_b}"
    if col_a in df.columns and col_b in df.columns:
        df[f"{prefix}_vainqueur"] = np.where(
            df[col_a].values >= df[col_b].values, label_a, label_b
        ).astype(int)
        df[f"{prefix}_marge"] = np.abs(df[col_a].values - df[col_b].values).round(2)
    return df
