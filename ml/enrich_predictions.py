"""
Enrichit les CSV de prédictions avec les vraies colonnes T1/T2 du dataset gold.
À relancer après chaque entraînement si trainer.py n'est pas encore mis à jour.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ML_DIR   = Path(__file__).parent
ARTIFACTS = ML_DIR / "artifacts"
DATA_PATH = Path(
    os.environ.get("DATA_PATH", str(ML_DIR.parent.parent / "dataset_elections_2022_idf.csv"))
)

T1_COLS = [
    "cible_t1_pct_macron",
    "cible_t1_pct_melenchon",
    "cible_t1_pct_lepen",
    "cible_t1_pct_zemmour",
    "cible_t1_pct_pecresse",
    "cible_t1_pct_jadot",
    "cible_t1_pct_autres",
    "cible_t1_premier",
    "taux_participation_t1",
]
T2_COLS = [
    "cible_t2_pct_macron",
    "cible_t2_pct_lepen",
    "cible_t2_marge",
]

PRED_FILES = [
    "gb_predictions_post_t1_classification_t2.csv",
    "gb_predictions_pre_vote_classification_t2.csv",
    "rf_predictions_post_t1_classification_t2.csv",
    "lstm_predictions_classification_t2.csv",
]


def enrich():
    if not DATA_PATH.exists():
        # Cherche dans le répertoire parent
        alt = ML_DIR.parent / "dataset_elections_2022_idf.csv"
        if alt.exists():
            data_path = alt
        else:
            print(f"Dataset introuvable : {DATA_PATH}")
            return
    else:
        data_path = DATA_PATH

    print(f"Chargement du dataset : {data_path}")
    gold = pd.read_csv(data_path, low_memory=False, dtype={"code_commune": str})

    # Colonnes disponibles dans gold
    available_t1 = [c for c in T1_COLS if c in gold.columns]
    available_t2 = [c for c in T2_COLS if c in gold.columns]
    extra_cols   = ["code_commune"] + available_t1 + available_t2

    gold_sub = gold[extra_cols].copy()
    gold_sub["code_commune"] = gold_sub["code_commune"].astype(str).str.zfill(5)

    enriched = 0
    for fname in PRED_FILES:
        path = ARTIFACTS / fname
        if not path.exists():
            print(f"  [SKIP] {fname} — fichier absent")
            continue

        pred = pd.read_csv(path, dtype={"code_commune": str})
        pred["code_commune"] = pred["code_commune"].astype(str).str.zfill(5)

        # Vérifie si déjà enrichi
        already = [c for c in available_t1 + available_t2 if c in pred.columns]
        if len(already) == len(available_t1) + len(available_t2):
            print(f"  [OK] {fname} — déjà enrichi ({len(already)} colonnes présentes)")
            continue

        merged = pred.merge(gold_sub, on="code_commune", how="left")

        # Arrondi 2 décimales pour les pourcentages
        for col in available_t1 + available_t2:
            if col in merged.columns and merged[col].dtype == float:
                merged[col] = merged[col].round(2)

        merged.to_csv(path, index=False)
        print(f"  [ENRICHI] {fname} — +{len(available_t1) + len(available_t2)} colonnes")
        enriched += 1

    print(f"\nTerminé. {enriched} fichier(s) enrichi(s).")


if __name__ == "__main__":
    enrich()
