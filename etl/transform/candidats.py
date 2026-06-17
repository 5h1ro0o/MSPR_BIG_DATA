"""
Transformation — résultats par candidat.
Produit les features historiques (2012/2017) et les variables cibles (2022).
"""

import pandas as pd

from etl.helpers import pct
from monitoring.logger import get_logger

log = get_logger(__name__)

_LE_PEN = "LE PEN"

CANDS_2012_T1 = {
    "HOLLANDE": "hollande",
    "SARKOZY": "sarkozy",
    _LE_PEN: "lepen",
    "MELENCHON": "melenchon",
    "BAYROU": "bayrou",
}
CANDS_2012_T2 = {"HOLLANDE": "hollande", "SARKOZY": "sarkozy"}

CANDS_2017_T1 = {
    "MACRON": "macron",
    "FILLON": "fillon",
    "MELENCHON": "melenchon",
    _LE_PEN: "lepen",
    "HAMON": "hamon",
}
CANDS_2017_T2 = {"MACRON": "macron", _LE_PEN: "lepen"}

CANDS_2022_T1 = {
    "MELENCHON": "melenchon",
    "MACRON": "macron",
    _LE_PEN: "lepen",
    "ZEMMOUR": "zemmour",
    "PECRESSE": "pecresse",
    "JADOT": "jadot",
}
CANDS_2022_T2 = {"MACRON": "macron", _LE_PEN: "lepen"}


def pivot_candidats(
    df: pd.DataFrame,
    election: str,
    candidates: dict[str, str],
    prefix: str,
) -> pd.DataFrame:
    """
    Pour une élection donnée, agrège les voix par commune × candidat,
    calcule les pourcentages sur les exprimés, et pivote en colonnes larges.

    Args:
        df:         DataFrame issu de extract_candidats_raw (filtré par election)
        election:   id_election (ex. "2017_pres_t2")
        candidates: {nom_normalise: label_court} (ex. {"MACRON": "macron"})
        prefix:     préfixe des colonnes (ex. "h17_t2")

    Returns:
        DataFrame avec une ligne par commune et une colonne par candidat + autres.
    """
    sub = df[df["id_election"] == election].copy()
    if sub.empty:
        log.warning(f"Aucune donnée pour {election}")
        return pd.DataFrame(columns=["code_commune"])

    agg = sub.groupby(["code_commune", "nom_norm"], as_index=False)["voix"].sum()

    exprimes = agg.groupby("code_commune")["voix"].sum().rename("total_exprimes")
    agg = agg.merge(exprimes, on="code_commune")
    agg["pct_val"] = pct(agg["voix"], agg["total_exprimes"])

    communes = agg["code_commune"].unique()
    result = pd.DataFrame({"code_commune": communes})

    for nom_norm, label in candidates.items():
        sub_cand = agg[agg["nom_norm"] == nom_norm][["code_commune", "pct_val"]].rename(
            columns={"pct_val": f"{prefix}_pct_{label}"}
        )
        result = result.merge(sub_cand, on="code_commune", how="left")

    named_cols = [f"{prefix}_pct_{lbl}" for lbl in candidates.values()]
    existing = [c for c in named_cols if c in result.columns]
    result[f"{prefix}_pct_autres"] = (
        (100 - result[existing].fillna(0).sum(axis=1)).clip(lower=0).round(2)
    )

    for col in named_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0.0)

    log.debug(f"pivot_candidats [{election}] : {len(result):,} communes")
    return result


def transform_historique(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Produit les features historiques préfixées h12_ et h17_.
    Les colonnes vainqueur/marge sont recalculées après tout le nettoyage
    dans assembler.py, pas ici.

    Args:
        df_raw: données brutes de extract_candidats_raw (filtré 2012/2017)

    Returns:
        DataFrame silver — une ligne par commune, 20 colonnes features.
    """
    h12_t1 = pivot_candidats(df_raw, "2012_pres_t1", CANDS_2012_T1, "h12_t1")
    h12_t2 = pivot_candidats(df_raw, "2012_pres_t2", CANDS_2012_T2, "h12_t2")
    h17_t1 = pivot_candidats(df_raw, "2017_pres_t1", CANDS_2017_T1, "h17_t1")
    h17_t2 = pivot_candidats(df_raw, "2017_pres_t2", CANDS_2017_T2, "h17_t2")

    hist = h12_t1
    for part in [h12_t2, h17_t1, h17_t2]:
        hist = hist.merge(part, on="code_commune", how="outer")

    hist.drop(
        columns=["h12_t2_pct_autres", "h17_t2_pct_autres"],
        errors="ignore",
        inplace=True,
    )

    log.info(
        f"Historique silver : {len(hist):,} communes, {len(hist.columns)} colonnes"
    )
    return hist


def transform_cibles(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Produit les variables cibles préfixées cible_.
    Les colonnes vainqueur/marge sont recalculées dans assembler.py.

    Args:
        df_raw: données brutes de extract_candidats_raw (filtré 2022)

    Returns:
        DataFrame silver — une ligne par commune, cibles T1 + T2.
    """
    c22_t1 = pivot_candidats(df_raw, "2022_pres_t1", CANDS_2022_T1, "cible_t1")

    pct_cols = [f"cible_t1_pct_{lbl}" for lbl in CANDS_2022_T1.values()]
    existing = [c for c in pct_cols if c in c22_t1.columns]
    c22_t1["cible_t1_premier"] = (
        c22_t1[existing].idxmax(axis=1).str.replace("cible_t1_pct_", "", regex=False)
    )

    c22_t2 = pivot_candidats(df_raw, "2022_pres_t2", CANDS_2022_T2, "cible_t2")
    c22_t2.drop(columns=["cible_t2_pct_autres"], errors="ignore", inplace=True)

    cibles = c22_t1.merge(c22_t2, on="code_commune", how="outer")
    log.info(
        f"Cibles silver : {len(cibles):,} communes, {len(cibles.columns)} colonnes"
    )
    return cibles
