"""
Extraction — dossier_complet.csv (INSEE RP 2022, ~642 MB, 1976 colonnes).
Sélectionne uniquement les colonnes nécessaires pour économiser la mémoire.
"""

from pathlib import Path

import pandas as pd

from etl.helpers import norm_code
from monitoring.logger import get_logger

log = get_logger(__name__)

IDF_DEPTS = frozenset({"75", "77", "78", "91", "92", "93", "94", "95"})

COL_MAP = {
    "CODGEO": "code_commune",
    "P22_POP": "pop_totale",
    "P22_POP0014": "pop_0014",
    "P22_POP1529": "pop_1529",
    "P22_POP3044": "pop_3044",
    "P22_POP4559": "pop_4559",
    "P22_POP6074": "pop_6074",
    "P22_POP7589": "pop_7589",
    "P22_POP90P": "pop_90p",
    "P22_POPH": "pop_hommes",
    "P22_POPF": "pop_femmes",
    "C22_POP15P": "pop_15p",
    "C22_POP15P_STAT_GSEC11_21": "csp_agriculteurs",
    "C22_POP15P_STAT_GSEC12_22": "csp_artisans_commercants",
    "C22_POP15P_STAT_GSEC13_23": "csp_cadres",
    "C22_POP15P_STAT_GSEC14_24": "csp_prof_intermediaires",
    "C22_POP15P_STAT_GSEC15_25": "csp_employes",
    "C22_POP15P_STAT_GSEC16_26": "csp_ouvriers",
    "C22_POP15P_STAT_GSEC32": "csp_retraites",
    "C22_POP15P_STAT_GSEC40": "csp_autres_inactifs",
    "P22_NSCOL15P": "pop_nscol15p",
    "P22_NSCOL15P_DIPLMIN": "dipl_aucun",
    "P22_NSCOL15P_BEPC": "dipl_bepc",
    "P22_NSCOL15P_CAPBEP": "dipl_capbep",
    "P22_NSCOL15P_BAC": "dipl_bac",
    "P22_NSCOL15P_SUP2": "dipl_sup_bac2",
    "P22_NSCOL15P_SUP34": "dipl_sup_bac34",
    "P22_NSCOL15P_SUP5": "dipl_sup_bac5",
    "P22_ACT1564": "actifs_1564",
    "P22_ACTOCC1564": "actifs_occupes_1564",
    "P22_CHOM1564": "chomeurs_1564",
    "P22_LOG": "nb_logements",
    "P22_LOGVAC": "nb_logements_vacants",
    "P22_RP_PROP": "log_proprietaires",
    "P22_RP_LOC": "log_locataires",
    "P22_RP_LOCHLMV": "log_hlm",
    "C22_MEN": "nb_menages",
    "C22_MENPSEUL": "men_seuls",
    "C22_MENFAM": "men_familiaux",
    "C22_MENCOUPAENF": "men_couple_enfants",
    "C22_MENFAMMONO": "men_monoparentaux",
    "MED21": "revenu_median_2021",
}


def extract_demographique(demographique_file: Path) -> pd.DataFrame:
    """
    Charge dossier_complet.csv en ne lisant que les colonnes du COL_MAP.

    Args:
        demographique_file: chemin vers dossier_complet.csv

    Returns:
        DataFrame brut (bronze) filtré IDF, colonnes renommées.
    """
    if not demographique_file.exists():
        log.warning(f"Fichier démographique absent : {demographique_file} — CSP/diplômes = NaN")
        return pd.DataFrame(columns=["code_commune"])

    df = pd.read_csv(
        demographique_file,
        sep=";",
        low_memory=False,
        encoding="utf-8",
        usecols=lambda c: c in COL_MAP,
    )

    df["CODGEO"] = norm_code(df["CODGEO"])
    df = df[df["CODGEO"].str[:2].isin(IDF_DEPTS)].copy()

    available = {k: v for k, v in COL_MAP.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available).copy()

    for col in df.select_dtypes(include="object").columns:
        if col == "code_commune":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(f"Demographique brut : {len(df):,} lignes, {len(df.columns)} colonnes")
    return df
