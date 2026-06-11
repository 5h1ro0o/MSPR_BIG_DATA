"""
ETL Extract — Données de sécurité (criminalité et délinquance)
==============================================================
Source : data.gouv.fr — Base nationale des crimes et délits enregistrés
par la Police et la Gendarmerie nationales.

Indicateurs extraits (agrégés au niveau commune / département IDF) :
  - taux_crimes_personnes  : crimes et délits contre les personnes (/ 1 000 hab.)
  - taux_crimes_biens      : atteintes aux biens (vols, dégradations) (/ 1 000 hab.)
  - taux_crimes_total      : total général (/ 1 000 hab.)
  - taux_crimes_drogues    : infractions liées aux stupéfiants (/ 1 000 hab.)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from monitoring.logger import get_logger

log = get_logger(__name__)

DEPTS_IDF = {"75", "77", "78", "91", "92", "93", "94", "95"}

# Correspondance type de crime → catégorie agrégée
CATEGORIES = {
    "personnes": [
        "Coups et blessures volontaires",
        "Violences sexuelles",
        "Menaces ou chantages",
    ],
    "biens": [
        "Cambriolages de logement",
        "Vols avec violence",
        "Vols sans violence contre des personnes",
        "Vols de véhicules",
        "Destructions et dégradations volontaires",
    ],
    "drogues": [
        "Usage de stupéfiants",
        "Trafic de stupéfiants",
    ],
}


def extract_securite(data_root: Path) -> pd.DataFrame | None:
    """
    Charge et agrège les données de criminalité pour les départements IDF.

    Retourne un DataFrame avec une ligne par commune (code_departement)
    et les taux / 1 000 habitants calculés.
    Retourne None si le fichier source est introuvable.
    """
    # Cherche le fichier récursivement dans data_root et ses sous-dossiers
    candidates = list(data_root.glob("**/*crime*delinquance*departement*.csv")) + \
                 list(data_root.glob("**/*delin*departement*.csv")) + \
                 list(data_root.glob("**/*crimes_delits*.csv"))

    if not candidates:
        log.warning(
            f"Fichier criminalité introuvable dans {data_root}. "
            "Téléchargez depuis : https://www.data.gouv.fr/fr/pages/donnees-securite/"
        )
        return None

    path = candidates[0]
    log.info(f"Chargement sécurité : {path.name}")

    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=";", encoding="latin-1", low_memory=False)

    # Normalisation colonnes
    df.columns = [c.strip().lower() for c in df.columns]

    # Identifier les colonnes clés (nommage variable selon les millésimes)
    col_dept  = next((c for c in df.columns if "dep" in c and len(c) < 10), None)
    col_type  = next((c for c in df.columns if "classe" in c or "type" in c or "nature" in c), None)
    col_count = next((c for c in df.columns if "faits" in c or "nombre" in c or "valeur" in c), None)
    col_pop   = next((c for c in df.columns if "pop" in c), None)
    col_year  = next((c for c in df.columns if "annee" in c or "année" in c or "an" == c), None)

    if not all([col_dept, col_type, col_count]):
        log.error(f"Colonnes attendues introuvables — skip sécurité. Colonnes dispo: {list(df.columns)[:10]}")
        return None

    # Filtrer IDF
    df[col_dept] = df[col_dept].astype(str).str.zfill(2)
    df = df[df[col_dept].isin(DEPTS_IDF)].copy()

    # Dernière année disponible
    if col_year:
        last_year = df[col_year].max()
        df = df[df[col_year] == last_year].copy()
        log.info(f"Année retenue : {last_year}")

    # Agrégation par département
    rows = []
    for dept_code, grp in df.groupby(col_dept):
        total_faits = grp[col_count].sum()
        pop = grp[col_pop].iloc[0] if col_pop else 1

        faits_per_cat = {}
        for cat, labels in CATEGORIES.items():
            mask = grp[col_type].isin(labels)
            faits_per_cat[f"taux_crimes_{cat}"] = (
                grp.loc[mask, col_count].sum() / max(pop, 1) * 1000
            )

        rows.append({
            "code_departement":      dept_code,
            "taux_crimes_total":     round(total_faits / max(pop, 1) * 1000, 2),
            "taux_crimes_personnes": round(faits_per_cat.get("taux_crimes_personnes", 0), 2),
            "taux_crimes_biens":     round(faits_per_cat.get("taux_crimes_biens", 0), 2),
            "taux_crimes_drogues":   round(faits_per_cat.get("taux_crimes_drogues", 0), 2),
        })

    result = pd.DataFrame(rows)
    log.info(f"Sécurité : {len(result)} départements IDF extraits")
    return result
