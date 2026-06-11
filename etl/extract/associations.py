"""
ETL Extract — Données de vie associative (RNA)
===============================================
Source : data.gouv.fr — Répertoire National des Associations (RNA)

Indicateurs extraits :
  - nb_associations_actives    : associations actives par commune
  - densite_associative        : associations / 1 000 habitants
  - pct_asso_sportives         : % d'associations sportives
  - pct_asso_culturelles       : % d'associations culturelles
  - pct_asso_sociales          : % d'associations d'action sociale

URL source : https://www.data.gouv.fr/fr/datasets/repertoire-national-des-associations/
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from monitoring.logger import get_logger

log = get_logger(__name__)

DEPTS_IDF = {"75", "77", "78", "91", "92", "93", "94", "95"}

# Mots-clés dans l'objet/titre de l'association pour la classification
KEYWORDS = {
    "sportives":   ["sport", "tennis", "football", "basket", "natation", "gym", "athlét", "cyclisme", "rugby"],
    "culturelles": ["cultur", "musique", "théâtre", "art", "danse", "cinéma", "patrimoin", "littérat"],
    "sociales":    ["social", "solidar", "humanitaire", "aide", "handicap", "retraite", "sénior", "jeunesse"],
}


def _classify(objet: str) -> str:
    """Classifie une association par mots-clés dans son objet social."""
    if not isinstance(objet, str):
        return "autre"
    o = objet.lower()
    for cat, keywords in KEYWORDS.items():
        if any(k in o for k in keywords):
            return cat
    return "autre"


def extract_associations(data_root: Path) -> pd.DataFrame | None:
    """
    Agrège le nombre et le type d'associations actives par commune IDF.

    Retourne un DataFrame avec une ligne par commune (code_commune)
    et les indicateurs de vie associative calculés.
    """
    candidates = (
        list(data_root.glob("**/*rna*commune*.csv")) +
        list(data_root.glob("**/*associations*idf*.csv")) +
        list(data_root.glob("**/*rna*.csv"))
    )

    if not candidates:
        log.warning(
            f"Fichier RNA introuvable dans {data_root}. "
            "Téléchargez depuis : "
            "https://www.data.gouv.fr/fr/datasets/repertoire-national-des-associations/"
        )
        return None

    # Prendre le plus petit fichier d'abord (évite le fichier complet de 2 Go)
    path = sorted(candidates, key=lambda p: p.stat().st_size)[0]
    log.info(f"Chargement RNA : {path.name} ({path.stat().st_size / 1_048_576:.1f} Mo)")

    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8", dtype=str, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=";", encoding="latin-1", dtype=str, low_memory=False)

    df.columns = [c.strip().lower() for c in df.columns]

    col_commune = next((c for c in df.columns if "commune" in c or "siret_commune" in c or "adrs_codepostal" in c), None)
    col_objet   = next((c for c in df.columns if "objet" in c or "titre" in c or "libelle" in c), None)
    col_etat    = next((c for c in df.columns if "etat" in c or "dissolut" in c or "valid" in c), None)

    if not col_commune:
        log.error(f"Colonne commune RNA introuvable. Colonnes: {list(df.columns)[:10]}")
        return None

    # Extraire code commune INSEE depuis code postal si nécessaire
    df["_cp"] = df[col_commune].astype(str).str.strip().str[:2]
    df = df[df["_cp"].isin(DEPTS_IDF)].copy()

    if df.empty:
        log.warning(
            f"Associations : 0 lignes IDF après filtrage — colonne={col_commune} — "
            "vérifiez le format du fichier RNA (attendu: code postal 5 chiffres)"
        )
        return None

    # Filtrer actives si disponible
    if col_etat:
        actives_mask = ~df[col_etat].astype(str).str.upper().isin(["D", "DISSOUTE", "0", "INVALIDE"])
        df = df[actives_mask].copy()

    # Classification
    if col_objet:
        df["_categorie"] = df[col_objet].apply(_classify)

    # Agrégation par code commune (code postal → approximation)
    rows = []
    for commune_cp, grp in df.groupby(col_commune):
        total = len(grp)
        row = {
            "code_commune_cp":        str(commune_cp).zfill(5),
            "nb_associations_actives": total,
        }
        if col_objet:
            for cat in ["sportives", "culturelles", "sociales"]:
                row[f"pct_asso_{cat}"] = round(
                    (grp["_categorie"] == cat).sum() / max(total, 1) * 100, 2
                )
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        log.warning("Associations : aucun résultat après agrégation — source ignorée")
        return None

    log.info(
        f"Associations : {len(result)} communes, "
        f"{int(result['nb_associations_actives'].sum())} associations actives"
    )
    return result
