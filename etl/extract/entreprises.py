"""
ETL Extract — Données d'activité économique (nombre d'entreprises)
==================================================================
Source : INSEE — Démographie des entreprises, Fichier FLORES/CLAP
ou Base Sirene (stock établissements actifs par commune).

Indicateurs extraits :
  - nb_entreprises_actives     : nombre d'établissements actifs
  - densite_entreprises        : établissements actifs / 1 000 habitants
  - pct_entreprises_commerce   : % dans le commerce/réparation
  - pct_entreprises_services   : % dans les services aux entreprises
  - taux_creation_entreprises  : taux de création (si disponible)

Format source accepté :
  - Fichier CSV Sirene (stock établissements) avec colonne CODECOMMUNE / DEPET
  - Fichier INSEE CLAP (nb établissements par NAF et commune)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from monitoring.logger import get_logger

log = get_logger(__name__)

DEPTS_IDF = {"75", "77", "78", "91", "92", "93", "94", "95"}

# Sections NAF pour les catégories agrégées
NAF_COMMERCE  = {"G"}   # Commerce ; réparation d'automobiles
NAF_SERVICES  = {"J", "K", "L", "M", "N"}  # Services aux entreprises
NAF_INDUSTRIE = {"B", "C", "D", "E"}       # Industries


def extract_entreprises(data_root: Path) -> pd.DataFrame | None:
    """
    Agrège le nombre d'entreprises / établissements actifs par commune IDF.

    Retourne un DataFrame avec une ligne par commune (code_commune)
    et les indicateurs d'activité économique calculés.
    """
    # Noms de fichiers possibles, recherche récursive dans les sous-dossiers
    candidates = (
        list(data_root.glob("**/*sirene*commune*.csv")) +
        list(data_root.glob("**/*clap*commune*.csv")) +
        list(data_root.glob("**/*etab*commune*.csv")) +
        list(data_root.glob("**/*entreprises*idf*.csv")) +
        list(data_root.glob("**/*etablissements*actifs*.csv"))
    )

    if not candidates:
        log.warning(
            f"Fichier entreprises introuvable dans {data_root}. "
            "Téléchargez le stock Sirene depuis : "
            "https://www.data.gouv.fr/datasets/search?q=sirene+commune"
        )
        return None

    path = candidates[0]
    log.info(f"Chargement entreprises : {path.name}")

    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8", dtype=str, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=";", encoding="latin-1", dtype=str, low_memory=False)

    df.columns = [c.strip().lower() for c in df.columns]

    # Colonnes clés (nommage variable)
    col_commune = next((c for c in df.columns if "commune" in c or "codecom" in c), None)
    col_dept    = next((c for c in df.columns if ("dep" in c and "code" in c) or c == "dep"), None)
    col_naf     = next((c for c in df.columns if "naf" in c or "activite" in c or "section" in c), None)
    col_etat    = next((c for c in df.columns if "etat" in c or "statut" in c), None)

    if not col_commune:
        log.error(f"Colonne commune introuvable — skip entreprises. Colonnes: {list(df.columns)[:10]}")
        return None

    # Filtrer IDF via le code département (2 premiers chiffres du code commune)
    df["_dept"] = df[col_commune].astype(str).str[:2]
    df = df[df["_dept"].isin(DEPTS_IDF)].copy()

    # Filtrer établissements actifs si colonne état disponible
    if col_etat:
        df = df[df[col_etat].astype(str).str.upper().isin(["A", "1", "ACTIF"])].copy()

    # Agrégation par commune
    rows = []
    for commune_code, grp in df.groupby(col_commune):
        total = len(grp)
        naf_section = grp[col_naf].str[:1] if col_naf else pd.Series(dtype=str)

        rows.append({
            "code_commune":             str(commune_code).zfill(5),
            "nb_entreprises_actives":   total,
            "pct_entreprises_commerce": round((naf_section.isin(NAF_COMMERCE).sum() / max(total, 1)) * 100, 2) if col_naf else None,
            "pct_entreprises_services": round((naf_section.isin(NAF_SERVICES).sum() / max(total, 1)) * 100, 2) if col_naf else None,
            "pct_entreprises_industrie": round((naf_section.isin(NAF_INDUSTRIE).sum() / max(total, 1)) * 100, 2) if col_naf else None,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        log.warning("Entreprises : aucun résultat après agrégation — source ignorée")
        return None
    log.info(
        f"Entreprises : {len(result)} communes IDF extraites, "
        f"{int(result['nb_entreprises_actives'].sum())} établissements total"
    )
    return result
