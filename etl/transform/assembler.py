"""
Assembleur — fusion de toutes les couches silver en dataset gold.
Applique le nettoyage final (outliers, imputation, recalcul vainqueurs).
"""

import numpy as np
import pandas as pd

from etl.helpers import cap_outliers, impute_missing, recalc_winner_marge
from monitoring.logger import get_logger

log = get_logger(__name__)

WINNER_SPECS = [
    ("h12_t2", "hollande", "sarkozy", 0, 1),
    ("h17_t2", "macron", "lepen", 0, 1),
    ("cible_t2", "macron", "lepen", 0, 1),
]


def assemble_dataset(
    participation: pd.DataFrame,
    demographique: pd.DataFrame,
    pauvrete: pd.DataFrame,
    chomage: pd.DataFrame,
    emploi: pd.DataFrame,
    historique: pd.DataFrame,
    cibles: pd.DataFrame,
    iqr_factor: float = 3.0,
) -> pd.DataFrame:
    """
    Fusionne les 7 couches silver sur code_commune.
    Nettoie (outliers IQR, imputation médiane/moyenne).
    Recalcule les vainqueurs/marges T2 APRÈS nettoyage (important !).

    Returns:
        DataFrame gold — une ligne par commune, prêt pour ML.
    """
    log.info("Fusion des couches silver...")

    dataset = participation.copy()
    dataset = dataset.merge(demographique, on="code_commune", how="left")
    dataset = dataset.merge(pauvrete, on="code_commune", how="left")
    dataset = dataset.merge(chomage, on="code_commune", how="left")
    dataset = dataset.merge(emploi, on="code_commune", how="left")
    dataset = dataset.merge(historique, on="code_commune", how="left")
    dataset = dataset.merge(cibles, on="code_commune", how="left")

    log.info(f"Après fusion : {len(dataset):,} lignes, {len(dataset.columns)} colonnes")

    n_before = len(dataset)
    dataset.drop_duplicates(subset="code_commune", keep="first", inplace=True)
    log.info(f"Doublons supprimés : {n_before - len(dataset)}")

    key_targets = [
        c
        for c in ["cible_t2_vainqueur", "cible_t2_pct_macron", "cible_t2_pct_lepen"]
        if c in dataset.columns
    ]
    n_before = len(dataset)
    dataset.dropna(subset=key_targets, inplace=True)
    log.info(f"Lignes sans cibles 2022 supprimées : {n_before - len(dataset)}")

    # Ensure code_departement is stored as a 2-char zero-padded string, not integer
    if "code_departement" in dataset.columns:
        dataset["code_departement"] = (
            dataset["code_departement"]
            .astype(str)
            .str.replace(r"\.0+$", "", regex=True)
            .str.strip()
            .str.zfill(2)
        )

    numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
    cible_cols = [c for c in dataset.columns if c.startswith("cible_")]
    cible_num = [c for c in cible_cols if pd.api.types.is_numeric_dtype(dataset[c])]
    derived_cols = [
        c for c in numeric_cols if c.endswith("_vainqueur") or c.endswith("_marge")
    ]
    feature_num = [
        c for c in numeric_cols if c not in cible_num and c not in derived_cols
    ]

    log.info(f"Ecrêtage outliers IQR × {iqr_factor} sur {len(feature_num)} features...")
    dataset = cap_outliers(dataset, feature_num, factor=iqr_factor)

    n_miss = dataset[feature_num].isna().sum().sum()
    log.info(f"Imputation de {n_miss:,} valeurs manquantes...")
    dataset = impute_missing(dataset, feature_num)

    # Passe finale : colonnes encore NaN (source entièrement absente) → 0
    for col in feature_num:
        if dataset[col].isna().any():
            fallback = dataset[col].median()
            dataset[col] = dataset[col].fillna(fallback if not pd.isna(fallback) else 0.0)

    # Colonnes texte avec NaN (libelle_commune, etc.) → chaîne vide
    str_cols = dataset.select_dtypes(include=["object"]).columns
    for col in str_cols:
        if dataset[col].isna().any():
            dataset[col] = dataset[col].fillna("")

    id_cols = [
        "code_commune",
        "code_departement",
        "libelle_departement",
        "libelle_commune",
    ]
    part_prefixes = (
        "inscrits",
        "abstentions",
        "votants",
        "blancs",
        "nuls",
        "exprimes",
        "taux_participation",
        "taux_abstention",
        "taux_blancs",
        "taux_nuls",
        "taux_exprimes",
        "delta_participation",
        "ratio_blancs_nuls",
    )
    part_cols = [
        c for c in dataset.columns if any(c.startswith(p) for p in part_prefixes)
    ]
    hist_cols = sorted(
        c for c in dataset.columns if c.startswith("h12_") or c.startswith("h17_")
    )
    cible_cols_sorted = sorted(cible_cols)
    socio_cols = [
        c
        for c in dataset.columns
        if c not in id_cols
        and c not in part_cols
        and c not in hist_cols
        and c not in cible_cols_sorted
    ]

    final_order = (
        id_cols + sorted(part_cols) + sorted(socio_cols) + hist_cols + cible_cols_sorted
    )
    dataset = dataset[[c for c in final_order if c in dataset.columns]]

    for prefix, cand_a, cand_b, la, lb in WINNER_SPECS:
        dataset = recalc_winner_marge(dataset, prefix, cand_a, cand_b, la, lb)

    n_nulls = dataset.isnull().sum().sum()
    log.info(
        f"Dataset gold : {len(dataset):,} lignes × {len(dataset.columns)} colonnes | "
        f"{n_nulls} nulls | "
        f"depts={sorted(dataset['code_departement'].unique().tolist())}"
    )

    return dataset
