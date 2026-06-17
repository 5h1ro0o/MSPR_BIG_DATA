"""
etl/extract/datagouv.py
=======================
Téléchargement automatique de tous les datasets nécessaires depuis data.gouv.fr.

Datasets récupérés :
  1. Résultats présidentielles 2012 / 2017 / 2022 (T1 + T2) par commune
     → elections/elections_{dept}.csv  (participation par BV)
     → candidats_results.csv          (votes par candidat × commune × élection)
  2. INSEE RP 2022 — Dossier complet  → demographique +csp/dossier_complet.csv
  3. Filosofi 2017 — revenus locaux   → pauvreté/base-cc-filosofi-2017.xlsx
  4. DS_RP_EMPLOI 2022                → chomage/DS_RP_EMPLOI_LR_COMP_2022_data.csv
  5. Chômage IDF historique           → chomage/chomage-...csv

API data.gouv.fr : https://www.data.gouv.fr/api/1/
"""

from __future__ import annotations

import io
import logging
import re
import time
import unicodedata
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://www.data.gouv.fr/api/1"
IDF_DEPTS = frozenset({"75", "77", "78", "91", "92", "93", "94", "95"})

ELECTION_SLUGS: dict[str, list[str]] = {
    "2022_pres_t1": [
        # Slugs testés sur data.gouv.fr — "subcom" dans les noms de fichiers = niveau commune
        "election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour",
        "resultats-de-l-election-presidentielle-2022-1er-tour",
        "resultats-presidentielle-2022-1er-tour",
        "presidentielle-2022-resultats-communes-t1",
        "election-presidentielle-2022-tour-1",
        "election-presidentielle-2022-1er-tour",
    ],
    "2022_pres_t2": [
        # "2eme" est le bon suffixe data.gouv.fr (pas "2nd" ni "second")
        "election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-2eme-tour",
        "election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-2nd-tour",
        "resultats-de-l-election-presidentielle-2022-2eme-tour",
        "resultats-presidentielle-2022-2eme-tour",
        "presidentielle-2022-resultats-communes-t2",
        "election-presidentielle-2022-tour-2",
        "election-presidentielle-2022-2eme-tour",
    ],
    "2017_pres_t1": [
        # Slug correct vérifié sur data.gouv.fr
        "election-presidentielle-des-23-avril-et-7-mai-2017-resultats-definitifs-du-1er-tour-par-communes",
        "election-presidentielle-des-23-avril-et-7-mai-2017-resultats-definitifs-du-1er-tour",
        "election-presidentielle-des-23-avril-et-7-mai-2017-resultats-du-1er-tour-1",
        "resultats-de-l-election-presidentielle-2017-1er-tour",
        "election-presidentielle-2017-resultats-communes-t1",
    ],
    "2017_pres_t2": [
        # Slug correct vérifié sur data.gouv.fr
        "election-presidentielle-des-23-avril-et-7-mai-2017-resultats-definitifs-du-2nd-tour-par-communes",
        "election-presidentielle-des-23-avril-et-7-mai-2017-resultats-definitifs-du-2eme-tour",
        "election-presidentielle-des-23-avril-et-7-mai-2017-resultats-du-2eme-tour-2",
        "resultats-de-l-election-presidentielle-2017-2eme-tour",
        "election-presidentielle-2017-resultats-communes-t2",
    ],
    "2012_pres_t1": [
        # Slug correct vérifié sur data.gouv.fr
        "election-presidentielle-2012-1er-tour-par-communes",
        "election-presidentielle-2012-resultats-572126",
        "elections-presidentielles-1965-2012-1",
        "resultats-de-l-election-presidentielle-2012-1er-tour",
        "election-presidentielle-2012-resultats-communes-t1",
    ],
    "2012_pres_t2": [
        "election-presidentielle-2012-resultats-572126",
        "elections-presidentielles-1965-2012-1",
        "resultats-de-l-election-presidentielle-2012-2eme-tour",
        "election-presidentielle-2012-resultats-communes-t2",
    ],
}
ELECTION_SEARCH_QUERIES = {
    "2022_pres_t1": "election presidentielle 2022 premier tour commune resultats definitifs",
    "2022_pres_t2": "election presidentielle 2022 deuxieme tour commune resultats definitifs",
    "2017_pres_t1": "election presidentielle 2017 premier tour commune resultats definitifs",
    "2017_pres_t2": "election presidentielle 2017 deuxieme tour commune resultats definitifs",
    "2012_pres_t1": "election presidentielle 2012 premier tour commune resultats definitifs",
    "2012_pres_t2": "election presidentielle 2012 deuxieme tour commune resultats definitifs",
}
ELECTION_TOUR_HINT = {
    "2012_pres_t1": ["tour 1", "premier", "t1", "1er", "1"],
    "2012_pres_t2": ["tour 2", "second", "t2", "2ème", "2eme", "deuxième", "2nd", "2"],
    "2017_pres_t1": ["tour 1", "premier", "t1", "1er"],
    "2017_pres_t2": ["tour 2", "second", "t2", "2ème", "2eme", "deuxième", "2nd"],
    "2022_pres_t1": ["t1", "1er", "tour 1", "premier"],
    "2022_pres_t2": ["t2", "2nd", "tour 2", "second", "2eme", "2ème"],
}

OTHER_DATASETS = [
    {
        "key": "demographics",
        # Source directe INSEE — dossier complet RP 2022 par commune (ZIP → CSV)
        "direct_url": "https://www.insee.fr/fr/statistiques/fichier/5359146/dossier_complet.zip",
        "zip_file_hint": "dossier_complet",
        "slugs": [
            "bases-de-donnees-de-synthese-du-recensement-de-la-population-2022",
            "base-de-donnees-de-synthese-du-recensement-de-la-population-2022",
            "dossier-complet-du-recensement-de-la-population-2022",
            "recensement-de-la-population-2022",
            "bases-de-donnees-de-synthese-au-niveau-de-la-commune-2022",
            "bases-de-donnees-de-synthese-au-niveau-de-la-commune",
            "bases-de-donnees-et-fichiers-details-du-recensement-de-la-population",
            "recensement-de-la-population-2022-base-de-donnees-commune",
            "bases-de-donnees-communales-du-recensement-de-la-population-2022",
            "dossier-complet-recensement-2022",
        ],
        "resource_hints": [
            "dossier_complet",
            "dossier complet",
            "synthese",
            "commune",
            "2022",
            "csv",
        ],
        "formats": ["csv", "zip"],
        "target": "demographique +csp/dossier_complet.csv",
        "required": True,
        "search_query": "dossier complet commune recensement population 2022 csv insee",
    },
    {
        "key": "filosofi",
        # Source directe INSEE — Filosofi 2017 tous niveaux géo (ZIP → CSV plat)
        "direct_url": "https://www.insee.fr/fr/statistiques/fichier/4507225/base-filosofi-2017_CSV.zip",
        "zip_file_hint": "filosofi",
        "slugs": [
            "base-cc-filosofi-2017",
            "cc-filosofi-2017-donnees-de-synthese",
            "revenus-et-pauvrete-des-menages-en-2017-resultats-communaux",
            "revenus-et-pauvrete-des-menages-en-2017",
            "statistiques-filosofi-2017",
            "filosofi-2017-statistiques-locales",
            "revenus-et-pauvrete-des-menages-aux-niveaux-national-et-local-revenus-localises-sociaux-et-fiscaux",
            "revenus-localises-sociaux-et-fiscaux-filosofi-2017",
            "filosofi-2017-commune",
        ],
        "resource_hints": [
            "filosofi",
            "base-cc",
            "cc-filosofi",
            "2017",
            "commune",
            "revenus",
        ],
        "formats": ["csv", "xlsx", "xls", "zip"],
        "target": "pauvreté/base-cc-filosofi-2017.csv",
        "required": True,
        "search_query": "base cc filosofi 2017 commune revenus pauvrete menages insee",
    },
    {
        "key": "emploi_2022",
        # Source directe INSEE — emploi-population active 2022 par commune (ZIP → CSV)
        "direct_url": "https://www.insee.fr/fr/statistiques/fichier/8581444/base-cc-emploi-pop-active-2022_csv.zip",
        "zip_file_hint": "emploi",
        "slugs": [
            "emploi-par-commune-recensement-2022",
            "statistiques-de-lemploi-au-recensement-2022",
            "bases-de-donnees-communales-emploi-2022",
            "structure-emplois-commune-2022",
            "ds-rp-emploi-lr-comp-2022",
        ],
        "resource_hints": ["emploi", "commune", "2022", "pop_active", "actif"],
        "formats": ["csv", "zip"],
        "target": "chomage/base-cc-emploi-pop-active-2022.csv",
        "required": False,
        "search_query": "emploi population active commune 2022 recensement population insee",
    },
    {
        "key": "chomage_idf",
        # Source principale : data.iledefrance.fr (OpenDataSoft) — pas sur data.gouv.fr
        "direct_url": (
            "https://data.iledefrance.fr/api/explore/v2.1/catalog/datasets/"
            "chomage-nombre-de-chomeurs-exhaustif-des-communes-dile-de-france-donnee-insee/"
            "exports/csv?lang=fr&timezone=Europe%2FParis&use_labels=false&delimiter=%3B"
        ),
        "slugs": [
            "taux-de-chomage-locaux-communes-idf",
            "chomage-communes-ile-de-france",
            "taux-de-chomage-localises-annuels-france",
        ],
        "resource_hints": ["chomage", "commune", "ile-de-france", "idf"],
        "formats": ["csv", "zip"],
        "target": "chomage/chomage-nombre-de-chomeurs-exhaustif-des-communes-dile-de-france-donnee-insee.csv",
        "required": True,
        "search_query": "chomage commune ile de france insee exhaustif 2020",
    },
    # ── Sources optionnelles (enrichissement — non bloquantes) ────────────────
    {
        "key": "securite",
        "slugs": [
            "bases-statistiques-communales-de-la-delinquance-enregistree-par-la-police-et-la-gendarmerie-nationales",
            "stat-crimes-delits",
            "crimes-et-delits-enregistres-par-les-services-de-gendarmerie-et-de-police",
        ],
        "resource_hints": ["departement", "crimes", "delits", "delinquance"],
        "formats": ["csv", "zip"],
        "target": "securite/crimes_delits_departements.csv",
        "required": False,
    },
    {
        "key": "associations",
        "slugs": [
            "repertoire-national-des-associations",
            "rna-repertoire-national-associations",
        ],
        # Préférer l'export complet (rna_import_*.zip large) vs imports mensuels (3-5 Mo)
        # Le fichier complet RNA contient toutes les asso actives (100-300 Mo compressé)
        "resource_hints": ["rna", "associations", "import"],
        "formats": ["csv", "zip"],
        "target": "associations/rna_associations_idf.csv",
        "required": False,
        "prefer_largest": True,  # évite de prendre l'import mensuel (3.9 Mo)
    },
    {
        "key": "entreprises",
        "slugs": [
            "nombre-detablissements-actifs-par-tranche-deffectif-salariale-et-par-commune",
            "etablissements-actifs-par-commune",
            "base-sirene-par-unite-legale",
        ],
        "resource_hints": ["commune", "etablissements", "entreprises", "sirene"],
        "formats": ["csv", "zip"],
        "target": "entreprises/etablissements_actifs_commune.csv",
        "required": False,
    },
]


_ELECTION_LABELS = {
    "2022_pres_t1": "Élections 2022 T1",
    "2022_pres_t2": "Élections 2022 T2",
    "2017_pres_t1": "Élections 2017 T1",
    "2017_pres_t2": "Élections 2017 T2",
    "2012_pres_t1": "Élections 2012 T1",
    "2012_pres_t2": "Élections 2012 T2",
}
_DATASET_LABELS = {
    "demographics": "Démographie RP 2022",
    "filosofi": "Filosofi 2017",
    "emploi_2022": "Emploi 2022",
    "chomage_idf": "Chômage IDF",
    "securite": "Sécurité",
    "associations": "Associations RNA",
    "entreprises": "Entreprises",
}


def fetch_all_datasets(data_root: Path) -> dict[str, bool]:
    """
    Télécharge tous les datasets manquants dans data_root.

    Returns:
        Dictionnaire {dataset_key: success_bool}
    """
    data_root = Path(data_root)
    results: dict[str, bool] = {}

    log.info("━━━ FETCH DATASETS ━━━")

    ok = _fetch_elections(data_root)
    results["elections"] = ok

    for cfg in OTHER_DATASETS:
        ok = _fetch_other_dataset(cfg, data_root)
        results[cfg["key"]] = ok

    _log_summary(results)
    return results


def _fetch_elections(data_root: Path) -> bool:
    """
    Télécharge les 6 fichiers d'élections (2012 T1/T2, 2017 T1/T2, 2022 T1/T2),
    produit elections/elections_{dept}.csv et candidats_results.csv.
    """
    elections_dir = data_root / "elections"
    elections_dir.mkdir(parents=True, exist_ok=True)

    candidats_file = data_root / "candidats_results.csv"

    all_depts_present = all(
        (elections_dir / f"elections_{dept}.csv").exists() for dept in IDF_DEPTS
    )
    if candidats_file.exists() and all_depts_present:
        # A previous run may have cached only 2017/2012 data when the 2022 resource
        # selection pointed to a dpt-level file (0 communes). Validate 2022 presence.
        has_2022 = False
        for dept in sorted(IDF_DEPTS):
            try:
                df_ck = pd.read_csv(
                    elections_dir / f"elections_{dept}.csv", dtype=str, low_memory=False
                )
                if (
                    "id_election" in df_ck.columns
                    and df_ck["id_election"]
                    .isin({"2022_pres_t1", "2022_pres_t2"})
                    .any()
                ):
                    has_2022 = True
                    break
            except Exception:
                pass
        if has_2022:
            log.info(f"  ✓  {'Élections (× 6)':<22} déjà présentes — skip")
            return True
        log.info(
            "  Élections cache invalide (données 2022 absentes) — re-téléchargement…"
        )

    all_participation: list[pd.DataFrame] = []
    all_candidats: list[pd.DataFrame] = []

    for id_election, slugs in ELECTION_SLUGS.items():
        label = _ELECTION_LABELS.get(id_election, id_election)
        raw = _download_election_csv(id_election, slugs)
        if raw is None:
            log.warning(f"  ✗  {label:<22} introuvable sur data.gouv.fr")
            continue

        participation, candidats = _parse_election_csv(raw, id_election)
        if participation is None:
            log.warning(f"  ✗  {label:<22} fichier téléchargé mais colonnes manquantes")
            continue

        n_com = len(participation)
        n_cand = len(candidats) if candidats is not None else 0
        cand_str = f" · {n_cand:,} candidats-lignes" if n_cand else ""
        log.info(f"  ✓  {label:<22} {n_com:,} communes{cand_str}")
        all_participation.append(participation)
        if candidats is not None:
            all_candidats.append(candidats)

    if not all_participation:
        log.error("  Aucune donnée d'élection téléchargée — ETL bloqué")
        return False

    participation_full = pd.concat(all_participation, ignore_index=True)
    _write_elections_by_dept(participation_full, elections_dir)

    if all_candidats:
        candidats_full = pd.concat(all_candidats, ignore_index=True)
        candidats_full.to_csv(candidats_file, sep=";", index=False)
        log.info(f"     {'Candidats':<22} {len(candidats_full):,} lignes total")

    return True


def _download_election_csv(
    id_election: str, slugs: list[str]
) -> Optional[pd.DataFrame]:
    """Cherche et télécharge le CSV de résultats par commune pour une élection."""
    dataset = None
    for slug in slugs:
        dataset = _get_dataset(slug)
        if dataset:
            log.debug(f"    [{id_election}] slug trouvé : {slug}")
            break

    if dataset is None:
        query = ELECTION_SEARCH_QUERIES.get(
            id_election, f"election presidentielle {id_election[:4]} commune resultats"
        )
        log.debug(f"    [{id_election}] fallback recherche : {query!r}")
        # Essayer les 5 premiers résultats de recherche (pas seulement le top-1)
        candidates_from_search = _search_datasets(query, limit=5)
        for candidate in candidates_from_search:
            dataset = candidate
            resources_c = _get_all_dataset_resources(dataset)
            tour_hints_c = ELECTION_TOUR_HINT.get(id_election, [])
            ranked_c = _pick_election_resources_ranked(resources_c, tour_hints_c)
            if ranked_c:
                log.debug(
                    f"    [{id_election}] dataset trouvé via recherche : {dataset.get('slug')}"
                )
                break
            dataset = None
        else:
            dataset = None

    if dataset is None:
        log.debug(f"    [{id_election}] dataset introuvable")
        return None

    resources = _get_all_dataset_resources(dataset)
    tour_hints = ELECTION_TOUR_HINT.get(id_election, [])

    ranked = _pick_election_resources_ranked(resources, tour_hints)
    if not ranked:
        log.warning(f"    [{id_election}] aucune ressource CSV/XLSX dans le dataset")
        return None

    # Try each resource in score order until one parses with a commune column
    _COMMUNE_NORM = {
        _normalize_col(a) for a in _COL_ALIASES.get("code_commune", [])
    } | {"code_commune"}

    for resource in ranked:
        url = resource.get("latest") or resource.get("url")
        fmt = (resource.get("format") or "").lower()
        title = resource.get("title", "")
        log.debug(f"    [{id_election}] essai ressource : {title!r} ({fmt})")

        df = _download_and_parse(url, fmt)
        if df is None or df.empty:
            continue

        # Reject immediately if no commune column present
        norm_cols = {_normalize_col(c) for c in df.columns}
        if not (norm_cols & _COMMUNE_NORM):
            log.debug(
                f"    [{id_election}] pas de colonne commune dans {title!r} — essai suivant…"
            )
            continue

        # Reject per-candidate files (one candidate only).
        # Wide-format XLSX files have either:
        #   - Numbered Nom columns: "Nom.1", "Nom 1" etc.  (dot/space-numbered)
        #   - "Unnamed: N" columns alongside a "Nom" column (NaN headers in XLSX, 2022 format)
        # Both are exempt. Only apply the candidate-count check for plain long-format files.
        _nom_numbered = [
            c for c in df.columns if re.match(r"^nom[.\s]\d+$", c, re.IGNORECASE)
        ]
        _has_unnamed = any(str(c).startswith("Unnamed:") for c in df.columns)
        _nom_col_raw = next(
            (c for c in df.columns if re.match(r"^nom$", c, re.IGNORECASE)), None
        )
        _is_wide = bool(_nom_numbered) or (_has_unnamed and _nom_col_raw is not None)
        if not _is_wide:
            _min_cands = 5 if id_election.endswith("_t1") else 2
            if _nom_col_raw is not None:
                _n_cands = df[_nom_col_raw].dropna().nunique()
                if _n_cands < _min_cands:
                    log.debug(
                        f"    [{id_election}] {_n_cands} candidat(s) dans {title!r} "
                        f"(≥{_min_cands} attendus) — fichier par candidat, essai suivant…"
                    )
                    continue

        return df

    log.warning(f"    [{id_election}] aucune ressource avec colonne commune trouvée")
    return None


def _pick_election_resources_ranked(
    resources: list[dict],
    tour_hints: list[str],
) -> list[dict]:
    """
    Retourne TOUTES les ressources valides triées par score décroissant.
    Préférence : commune > CSV > XLSX > bureau de vote / national / dpt.

    Vocabulaire data.gouv.fr pour les élections 2022 :
      niveau-subcom  = commune (agrégation des bureaux de vote par commune)
      niveau-burvot  = bureau de vote (granularité infra-communale)
      france-entiere = périmètre géographique France, PAS un agrégat national —
                       un fichier "niveau-subcom-france-entiere" contient UNE LIGNE
                       PAR COMMUNE pour toute la France. Ne pas pénaliser ce terme.
    """
    COMMUNE_LEVEL = [
        "-com-",
        "niveau-com",
        "par-commune",
        "par commune",
        "communes",
        "commune",
        "niveau com",
        "par com",
        # data.gouv.fr 2022 : "subcom" = niveau commune (sous-division du département)
        "subcom",
        "niveau-subcom",
        "sous-com",
    ]
    BV_LEVEL = [
        "burvot",
        "bureau de vote",
        "bureaux de vote",
        "bv-t",
        # "subcom" RETIRÉ — c'est le niveau commune sur data.gouv.fr 2022
        "niveau-burvot",
        "niveau-bv",
        "scrutin",
        "cirlg",
        "circo",
        "circonscription",
        # Département level
        "dpt",
        "niveau-dpt",
        "niveau dpt",
        "niveau-dep",
        "niveau dep",
        "dep-t",
        # Région level
        "-reg-",
        "niveau-reg",
        "niveau reg",
        "reg-t",
        # Agrégat national (1 seule ligne) — PAS "france-entiere" qui est le périmètre
        "-fra-",
        "niveau-fra",
        "niveau fra",
        "fra-t",
        "-nat-",
        "national",
        # NOTE : "france-entiere" RETIRÉ de BV_LEVEL — c'est le périmètre géo (scope),
        # pas le niveau d'agrégation. "resultats-par-niveau-subcom-t1-france-entiere"
        # contient bien des données COMMUNES pour toute la France.
    ]

    scored: list[tuple[int, dict]] = []
    for r in resources:
        title = _normalize_col(r.get("title") or "")
        url = r.get("latest") or r.get("url") or ""
        url_name = _normalize_col(url.rsplit("/", 1)[-1].split("?")[0])
        text = title + " " + url_name

        fmt = (r.get("format") or "").lower().strip()
        # Normalize MIME types
        if fmt and "/" in fmt:
            if "csv" in fmt or "dsv" in fmt:
                fmt = "csv"
            elif "openxmlformat" in fmt or "spreadsheetml" in fmt:
                fmt = "xlsx"
            elif "ms-excel" in fmt or "vnd.ms-excel" in fmt:
                fmt = "xls"
            elif "zip" in fmt or "gzip" in fmt or "x-compressed" in fmt:
                fmt = "zip"
            else:
                fmt = ""
        if not fmt:
            for ext in ("csv", "xlsx", "xls", "zip"):
                if url.lower().endswith(f".{ext}"):
                    fmt = ext
                    break

        if fmt not in ("csv", "xlsx", "xls", "zip"):
            continue

        score = 0
        if any(_normalize_col(k) in text for k in COMMUNE_LEVEL):
            score += 20
        if fmt == "csv":
            score += 6
        elif fmt in ("xlsx", "xls"):
            score += 2
        if tour_hints and any(_normalize_col(h) in text for h in tour_hints):
            score += 8
        if any(_normalize_col(k) in text for k in BV_LEVEL):
            score -= 25
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored]


def _pick_election_resource(
    resources: list[dict],
    tour_hints: list[str],
) -> Optional[dict]:
    """Retourne la meilleure ressource (top-1 du ranking)."""
    ranked = _pick_election_resources_ranked(resources, tour_hints)
    return ranked[0] if ranked else None


def _download_and_parse(url: str, fmt: str) -> Optional[pd.DataFrame]:
    """Télécharge une URL et retourne un DataFrame."""
    try:
        r = _get_with_retry(url)
        content = r.content

        if fmt == "zip" or url.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                names = zf.namelist()
                for name in names:
                    low = name.lower()
                    if low.endswith(".csv"):
                        content = zf.read(name)
                        fmt = "csv"
                        break
                    if low.endswith((".xlsx", ".xls")):
                        content = zf.read(name)
                        fmt = "xlsx"
                        break

        if fmt in ("xlsx", "xls"):
            df_xl = pd.read_excel(io.BytesIO(content), dtype=str)
            # Some election files (e.g. 2017 xls) have a merged title row on row 0,
            # making most columns "Unnamed:". Try increasing header rows, but only
            # accept a row that actually looks like a header (contains known column
            # name fragments rather than numeric data values).
            HEADER_FRAGMENTS = {
                "code",
                "lib",
                "inscrit",
                "votant",
                "exprim",
                "nom",
                "prenom",
                "voix",
                "departement",
                "commune",
                "tour",
                "resultat",
            }
            unnamed_count = sum(
                1 for c in df_xl.columns if str(c).startswith("Unnamed:")
            )
            if unnamed_count > len(df_xl.columns) // 2:
                for skip in range(1, 6):
                    try:
                        df2 = pd.read_excel(io.BytesIO(content), header=skip, dtype=str)
                        uc2 = sum(
                            1 for c in df2.columns if str(c).startswith("Unnamed:")
                        )
                        # Accept only if fewer unnamed AND at least one recognizable column header
                        col_names_lower = {str(c).lower() for c in df2.columns}
                        has_real_header = any(
                            any(frag in col for frag in HEADER_FRAGMENTS)
                            for col in col_names_lower
                        )
                        if uc2 < unnamed_count and has_real_header:
                            df_xl = df2
                            break
                    except Exception:
                        continue
            return df_xl

        for enc in ("utf-8", "latin-1", "cp1252"):
            for sep in (";", ",", "\t"):
                try:
                    df = pd.read_csv(
                        io.BytesIO(content),
                        sep=sep,
                        dtype=str,
                        encoding=enc,
                        low_memory=False,
                    )
                    if len(df.columns) > 5:
                        return df
                except Exception:
                    continue

        log.warning(f"    Impossible de parser le fichier téléchargé depuis {url}")
        return None

    except Exception as e:
        log.warning(f"    Erreur téléchargement {url}: {e}")
        return None


_COL_ALIASES = {
    "code_departement": [
        "code du département",
        "code département",
        "code departement",
        "num_dpt",
        "coddpt",
        "dept",
    ],
    "libelle_departement": [
        "libellé du département",
        "libelle département",
        "libelle departement",
        "lib_dpt",
    ],
    "code_commune": [
        "code de la commune",
        "code commune",
        "codecomm",
        "com",
        "codgeo",
        "code_commune",
        "codeinsee",
        "code_insee",
        "insee",
    ],
    "libelle_commune": [
        "libellé de la commune",
        "libelle commune",
        "lib_commune",
        "libellé de la commune ",
    ],
    "inscrits": ["inscrits", "nb_inscrits", "inscrits "],
    "abstentions": ["abstentions"],
    "votants": ["votants"],
    "blancs": ["blancs"],
    "nuls": ["nuls"],
    "exprimes": ["exprimés", "exprimes", "nb_exp"],
    "nom": ["nom", "nom_candidat", "candidat"],
    "prenom": ["prénom", "prenom"],
    "voix": ["voix", "nb_voix"],
}


def _normalize_col(s: str) -> str:
    """Normalise un nom de colonne : minuscules, sans accents, sans espaces."""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s


def _map_columns(df: pd.DataFrame) -> dict[str, str]:
    """Retourne un mapping {nom_interne: nom_dans_df} pour les colonnes disponibles."""
    norm_cols = {_normalize_col(c): c for c in df.columns}
    mapping: dict[str, str] = {}
    for internal, aliases in _COL_ALIASES.items():
        for alias in aliases:
            norm_alias = _normalize_col(alias)
            if norm_alias in norm_cols:
                mapping[internal] = norm_cols[norm_alias]
                break
    return mapping


def _melt_wide_candidats(
    df_idf: pd.DataFrame, id_election: str
) -> Optional[pd.DataFrame]:
    """
    Extrait tous les candidats d'un xlsx large.

    Deux formats supportés :
    - "Nom.N" / "Nom N" : pandas a dédupliqué les colonnes répétées (2017 xlsx / CSV)
    - "Unnamed: N" : les headers des candidats 2+ étaient vides dans l'xlsx (2022 format)
      Structure 2022 : 7 colonnes par candidat = N°Panneau, Sexe, Nom, Prénom, Voix, %Ins, %Exp
    """
    cols = list(df_idf.columns)
    id_cols = [c for c in ["code_departement", "code_commune"] if c in df_idf.columns]

    # — Format "Unnamed: N" (2022 data.gouv.fr XLSX) —
    has_unnamed = any(str(c).startswith("Unnamed:") for c in cols)
    if has_unnamed and "nom" in cols:
        nom_idx = cols.index("nom")
        voix_idx = cols.index("voix") if "voix" in cols else None
        prenom_idx = cols.index("prenom") if "prenom" in cols else None
        if voix_idx is None:
            return None
        # Offsets within each per-candidate block (relative to Nom column)
        voix_off = voix_idx - nom_idx  # typically 2 (Nom, Prénom, Voix)
        prenom_off = (prenom_idx - nom_idx) if prenom_idx is not None else None

        # Detect stride: find the first Unnamed column with a candidate name
        # (surname = uppercase string ≥ 2 chars; filters out Sexe = "M"/"F")
        cand2_nom_idx = None
        for i, c in enumerate(cols):
            if i <= nom_idx or not str(c).startswith("Unnamed:"):
                continue
            sample = df_idf.iloc[:5, i].dropna().astype(str)
            if sample.str.match(r"^[A-ZÁÉÀÙÂÊÎÔÛÄËÏÖÜ\-\' ]{2,}$").any():
                cand2_nom_idx = i
                break

        frames = []

        def _append_candidate(nom_c, voix_c, prenom_c):
            fr = df_idf[id_cols + [nom_c]].copy()
            fr.rename(columns={nom_c: "nom"}, inplace=True)
            fr = fr[
                fr["nom"].notna() & (fr["nom"].astype(str).str.strip() != "")
            ].copy()
            if fr.empty:
                return
            fr["voix"] = (
                (
                    pd.to_numeric(df_idf.loc[fr.index, voix_c], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                if voix_c
                else 0
            )
            fr["prenom"] = (
                (df_idf.loc[fr.index, prenom_c].fillna("").astype(str).str.strip())
                if prenom_c
                else ""
            )
            frames.append(fr)

        # Candidate 1 uses named columns
        voix_c1 = cols[voix_idx] if voix_idx is not None else None
        prenom_c1 = cols[prenom_idx] if prenom_idx is not None else None
        _append_candidate("nom", voix_c1, prenom_c1)

        if cand2_nom_idx is not None:
            stride = cand2_nom_idx - nom_idx
            cur = cand2_nom_idx
            while cur < len(cols):
                v_idx = cur + voix_off
                p_idx = (cur + prenom_off) if prenom_off is not None else None
                voix_c = cols[v_idx] if v_idx < len(cols) else None
                prenom_c = (
                    cols[p_idx] if p_idx is not None and p_idx < len(cols) else None
                )
                _append_candidate(cols[cur], voix_c, prenom_c)
                cur += stride

        if not frames:
            return None
        result = pd.concat(frames, ignore_index=True)
        result["id_election"] = id_election
        out_cols = [
            "id_election",
            "code_departement",
            "code_commune",
            "voix",
            "nom",
            "prenom",
        ]
        return result[[c for c in out_cols if c in result.columns]]

    # — Format "Nom.N" / "Nom N" (dot- or space-numbered columns) —
    def _sorted_cols(pattern: str) -> list[str]:
        matched = [c for c in cols if re.match(pattern, c, re.IGNORECASE)]

        def _key(c):
            m = re.search(r"[.\s](\d+)$", c)
            return int(m.group(1)) if m else -1

        return sorted(matched, key=_key)

    nom_cols = _sorted_cols(r"^nom(?:[.\s]\d+)?$")
    voix_cols = _sorted_cols(r"^voix(?:[.\s]\d+)?$")
    prenom_cols = _sorted_cols(r"^pr[eé]nom(?:[.\s]\d+)?$")

    if not nom_cols or not voix_cols:
        return None

    frames = []
    for i, nom_col in enumerate(nom_cols):
        frame = df_idf[id_cols + [nom_col]].copy()
        frame.rename(columns={nom_col: "nom"}, inplace=True)
        mask = frame["nom"].notna() & (frame["nom"].astype(str).str.strip() != "")
        frame = frame[mask].copy()
        if frame.empty:
            continue

        if i < len(voix_cols):
            frame["voix"] = (
                df_idf.loc[frame.index, voix_cols[i]]
                .astype(str)
                .str.replace(r"\s+", "", regex=True)
                .str.replace(",", ".")
                .pipe(pd.to_numeric, errors="coerce")
                .fillna(0)
                .astype(int)
            )
        else:
            frame["voix"] = 0

        if i < len(prenom_cols):
            frame["prenom"] = (
                df_idf.loc[frame.index, prenom_cols[i]]
                .fillna("")
                .astype(str)
                .str.strip()
            )
        else:
            frame["prenom"] = ""

        frames.append(frame)

    if not frames:
        return None

    result = pd.concat(frames, ignore_index=True)
    result["id_election"] = id_election
    out_cols = [
        "id_election",
        "code_departement",
        "code_commune",
        "voix",
        "nom",
        "prenom",
    ]
    return result[[c for c in out_cols if c in result.columns]]


def _parse_election_csv(
    df: pd.DataFrame,
    id_election: str,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Parse un DataFrame brut d'élection (data.gouv.fr format) et retourne :
      - participation : une ligne par bureau de vote (ou commune) IDF
      - candidats     : une ligne par (commune × candidat) IDF
    """
    col_map = _map_columns(df)
    # code_departement can be derived from a full 5-digit code_commune (e.g. 2012 T1 CodeInsee)
    required = {"code_commune", "inscrits", "votants", "exprimes"}
    if not required.issubset(col_map):
        missing = required - set(col_map)
        log.warning(
            f"  [{id_election}] colonnes manquantes : {missing} "
            f"— disponibles : {list(df.columns)[:15]}"
        )
        return None, None

    df = df.rename(columns={v: k for k, v in col_map.items()})

    if "code_departement" in df.columns:
        df["code_departement"] = (
            df["code_departement"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0+$", "", regex=True)
            .str.lstrip("0")
            .str.zfill(2)
        )

    # Construct 5-digit INSEE commune code.
    # Election files from data.gouv.fr typically store the within-dept commune number
    # (3 digits, e.g. "056" for Paris) NOT the full 5-digit INSEE code ("75056").
    # We detect this by checking whether all commune codes are ≤ 3 chars after stripping.
    raw_commune = (
        df["code_commune"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
        .str.lstrip("0")
        .fillna("")
    )
    max_commune_len = raw_commune.str.len().max()
    if max_commune_len is not None and int(max_commune_len) <= 3:
        # Within-dept code → needs code_departement to build full INSEE code
        if "code_departement" not in df.columns:
            log.warning(
                f"    {id_election}: commune 3 chiffres mais dept absent — skip"
            )
            return None, None
        df["code_commune"] = df["code_departement"] + raw_commune.str.zfill(3)
    else:
        # Full 5-digit code (e.g. CodeInsee from 2012 T1 CSV)
        df["code_commune"] = raw_commune.str.zfill(5)
        # Derive dept from first 2 chars if not already present
        if "code_departement" not in df.columns:
            df["code_departement"] = df["code_commune"].str[:2]
        df["code_departement"] = (
            df["code_departement"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0+$", "", regex=True)
            .str.lstrip("0")
            .str.zfill(2)
        )

    df_idf = df[df["code_departement"].isin(IDF_DEPTS)].copy()
    if df_idf.empty:
        log.warning(f"  [{id_election}] aucune commune IDF après filtrage dept")
        return None, None

    for col in [
        "inscrits",
        "abstentions",
        "votants",
        "blancs",
        "nuls",
        "exprimes",
        "voix",
    ]:
        if col in df_idf.columns:
            df_idf[col] = (
                df_idf[col]
                .astype(str)
                .str.replace(r"\s", "", regex=True)
                .str.replace(",", ".")
                .replace("", "0")
                .pipe(pd.to_numeric, errors="coerce")
                .fillna(0)
                .astype(int)
            )

    part_cols = [
        "code_departement",
        "libelle_departement",
        "code_commune",
        "libelle_commune",
        "inscrits",
        "abstentions",
        "votants",
        "blancs",
        "nuls",
        "exprimes",
    ]
    avail_part = [c for c in part_cols if c in df_idf.columns]
    num_cols = [
        c
        for c in ["inscrits", "abstentions", "votants", "blancs", "nuls", "exprimes"]
        if c in avail_part
    ]
    str_cols = [
        c
        for c in [
            "code_departement",
            "libelle_departement",
            "code_commune",
            "libelle_commune",
        ]
        if c in avail_part
    ]
    agg_dict = {c: "first" for c in str_cols if c != "code_commune"}
    agg_dict.update({c: "sum" for c in num_cols})
    participation = (
        df_idf[avail_part].groupby("code_commune", as_index=False).agg(agg_dict)
    )
    participation["id_election"] = id_election
    participation["code_bv"] = "00"
    if "abstentions" not in participation.columns:
        participation["abstentions"] = (
            participation["inscrits"] - participation["votants"]
        )
    if "blancs" not in participation.columns:
        participation["blancs"] = 0
    if "nuls" not in participation.columns:
        participation["nuls"] = 0

    def safe_pct(a, b):
        return (a / b.replace(0, float("nan")) * 100).round(2)

    part = participation
    part["ratio_abstentions_inscrits"] = safe_pct(
        part.get("abstentions", 0), part["inscrits"]
    )
    part["ratio_votants_inscrits"] = safe_pct(part.get("votants", 0), part["inscrits"])
    part["ratio_exprimes_inscrits"] = safe_pct(
        part.get("exprimes", 0), part["inscrits"]
    )
    part["ratio_exprimes_votants"] = safe_pct(
        part.get("exprimes", 0), part.get("votants", part["inscrits"])
    )

    candidats = None
    # Wide format detection:
    # (a) "Nom.N" or "Nom N" numbered columns (dot/space — pandas dedup or pre-named)
    # (b) "Unnamed: N" columns alongside a "nom" column (2022 XLSX: NaN headers for cands 2+)
    nom_extra = [
        c for c in df_idf.columns if re.match(r"^nom(?:[.\s]\d+)?$", c, re.IGNORECASE)
    ]
    _nom_numbered = [c for c in nom_extra if re.search(r"[.\s]\d+$", c)]
    _unnamed_wide = "nom" in df_idf.columns and any(
        str(c).startswith("Unnamed:") for c in df_idf.columns
    )
    if len(nom_extra) > 1 or _nom_numbered or _unnamed_wide:
        # Wide format with duplicated Nom/Voix columns (2022, 2017 xlsx)
        candidats = _melt_wide_candidats(df_idf, id_election)
    elif "nom" in df_idf.columns and "voix" in df_idf.columns:
        cand_cols = [
            "code_departement",
            "code_commune",
            "nom",
            "prenom" if "prenom" in df_idf.columns else "nom",
            "voix",
        ]
        cand = df_idf[[c for c in cand_cols if c in df_idf.columns]].copy()
        cand["id_election"] = id_election
        if "prenom" not in cand.columns:
            cand["prenom"] = ""
        keep = [
            "id_election",
            "code_departement",
            "code_commune",
            "voix",
            "nom",
            "prenom",
        ]
        candidats = cand[[c for c in keep if c in cand.columns]]

    return participation, candidats


def _write_elections_by_dept(df: pd.DataFrame, elections_dir: Path) -> None:
    """Sauvegarde les données de participation par département dans elections_{dept}.csv."""
    if "code_departement" not in df.columns:
        log.error("Colonne code_departement manquante — impossible d'écrire par dept")
        return

    for dept in IDF_DEPTS:
        dept_df = df[df["code_departement"] == dept].copy()
        if dept_df.empty:
            continue
        path = elections_dir / f"elections_{dept}.csv"
        dept_df.to_csv(path, index=False)
        log.debug(f"  Écrit {path.name} : {len(dept_df):,} lignes")


def _is_valid_data_content(content: bytes, fmt: str) -> bool:
    """Vérifie que le contenu téléchargé n'est pas une page HTML ou un fichier vide."""
    if not content or len(content) < 1024:
        return False
    head = content[:512].lower()
    if b"<!doctype" in head or b"<html" in head or b"<head>" in head:
        return False
    # Excel/ZIP : accept both PK magic (xlsx/zip) and OLE2 magic (old binary xls).
    # Some datasets report "xls" but serve xlsx, and vice-versa.
    if fmt in ("xlsx", "xls", "zip"):
        is_pk = content[:2] == b"PK"
        is_ole2 = content[:4] == b"\xd0\xcf\x11\xe0"
        if not (is_pk or is_ole2):
            return False
    return True


def _try_download_resource(
    dataset: dict, cfg: dict, expected_fmt: str
) -> Optional[bytes]:
    """
    Tente de télécharger une ressource valide du dataset.
    Essaie toutes les ressources classées par score jusqu'à en trouver une valide.
    Si prefer_largest=True, trie par taille décroissante (évite imports mensuels minuscules).
    """
    resources = _get_all_dataset_resources(dataset)
    ranked = _pick_resources_ranked(
        resources, cfg.get("resource_hints", []), cfg.get("formats", ["csv"])
    )
    if not ranked:
        return None

    if cfg.get("prefer_largest"):
        # Pour RNA et autres sources : préférer le plus grand fichier (export complet)
        ranked = sorted(ranked, key=lambda r: r.get("filesize") or 0, reverse=True)

    for resource in ranked:
        url = resource.get("latest") or resource.get("url")
        if not url:
            continue

        fmt = (resource.get("format") or "").lower().strip()
        if "," in fmt or (fmt and " " in fmt):
            fmt = ""
        if fmt and "/" in fmt:
            if "csv" in fmt or "dsv" in fmt:
                fmt = "csv"
            elif "openxmlformat" in fmt or "spreadsheetml" in fmt:
                fmt = "xlsx"
            elif "ms-excel" in fmt or "vnd.ms-excel" in fmt:
                fmt = "xls"
            elif "zip" in fmt or "gzip" in fmt or "x-compressed" in fmt:
                fmt = "zip"
            else:
                fmt = ""
        if not fmt:
            for ext in ("csv", "xlsx", "xls", "zip"):
                if url.lower().endswith(f".{ext}"):
                    fmt = ext
                    break
        if not fmt:
            fmt = expected_fmt

        title = resource.get("title", "?")
        log.debug(f"    Téléchargement : {title!r} ({fmt}) …")

        try:
            r = _get_with_retry(url, timeout=300)
            content = r.content

            if url.endswith(".zip") or fmt == "zip":
                target_ext = "." + expected_fmt
                ext_map = {".csv": "csv", ".xlsx": "xlsx", ".xls": "xls"}
                try:
                    with zipfile.ZipFile(io.BytesIO(content)) as zf:
                        candidates = [
                            n
                            for n in zf.namelist()
                            if Path(n).suffix.lower() == target_ext
                            and not n.startswith("__")
                        ]
                        if not candidates:
                            candidates = [
                                n
                                for n in zf.namelist()
                                if Path(n).suffix.lower() in ext_map
                                and not n.startswith("__")
                            ]
                        if candidates:
                            name = candidates[0]
                            content = zf.read(name)
                            fmt = ext_map.get(Path(name).suffix.lower(), expected_fmt)
                except zipfile.BadZipFile:
                    pass

            if _is_valid_data_content(content, fmt):
                return content
            log.debug(
                f"    Contenu invalide ({len(content):,} bytes, fmt={fmt}) — essai suivant…"
            )

        except Exception as e:
            log.debug(f"    Erreur téléchargement ({title!r}) : {e} — essai suivant…")

    return None


def _fetch_other_dataset(cfg: dict, data_root: Path) -> bool:
    """Télécharge un dataset socio-éco (data.gouv.fr ou URL directe)."""
    key = cfg["key"]
    label = _DATASET_LABELS.get(key, key)
    target = data_root / cfg["target"]
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_fmt = Path(cfg["target"]).suffix.lstrip(".").lower()
    required = cfg.get("required", True)

    if target.exists():
        size_mo = target.stat().st_size / 1_048_576
        log.info(f"  ✓  {label:<22} déjà présent ({size_mo:.1f} Mo) — skip")
        return True

    # 1. URL directe (data.iledefrance.fr, insee.fr, etc.)
    if "direct_url" in cfg:
        log.debug(f"  [{key}] téléchargement direct : {cfg['direct_url'][:60]}…")
        try:
            r = _get_with_retry(cfg["direct_url"], timeout=600)
            content = r.content
            dl_fmt = expected_fmt

            # Extraire le fichier de données depuis une archive ZIP
            if cfg["direct_url"].lower().endswith(".zip") or content[:2] == b"PK":
                try:
                    with zipfile.ZipFile(io.BytesIO(content)) as zf:
                        names = zf.namelist()
                        ext_set = {".csv", ".xlsx", ".xls"}
                        target_ext = f".{expected_fmt}"
                        # Priorité 1 : même extension que la cible
                        candidates = [
                            n
                            for n in names
                            if Path(n).suffix.lower() == target_ext
                            and not n.startswith("__")
                        ]
                        # Priorité 2 : n'importe quel fichier de données
                        if not candidates:
                            candidates = [
                                n
                                for n in names
                                if Path(n).suffix.lower() in ext_set
                                and not n.startswith("__")
                            ]
                        if candidates:
                            hint = cfg.get("zip_file_hint", "")
                            if hint:
                                hinted = [
                                    n
                                    for n in candidates
                                    if hint.lower() in Path(n).name.lower()
                                ]
                                if hinted:
                                    candidates = hinted
                            # Sélectionner le plus gros fichier non-meta (fichier de données)
                            candidates.sort(
                                key=lambda n: (
                                    "meta" in Path(n).name.lower(),
                                    -zf.getinfo(n).file_size,  # plus gros en premier
                                )
                            )
                            best = candidates[0]
                            content = zf.read(best)
                            dl_fmt = Path(best).suffix.lower().lstrip(".")
                            log.debug(
                                f"  [{key}] extrait du ZIP : {best} "
                                f"({dl_fmt}, {zf.getinfo(best).file_size:,} octets)"
                            )
                except zipfile.BadZipFile:
                    pass

            if _is_valid_data_content(content, dl_fmt):
                target.write_bytes(content)
                size_mo = len(content) / 1_048_576
                if "iledefrance" in cfg["direct_url"]:
                    source = "data.iledefrance.fr"
                elif "insee.fr" in cfg["direct_url"]:
                    source = "insee.fr"
                else:
                    source = "URL directe"
                log.info(f"  ✓  {label:<22} {size_mo:.1f} Mo [{source}]")
                return True
            log.debug(
                f"  [{key}] contenu direct invalide ({len(content):,} bytes) — essai slugs…"
            )
        except Exception as e:
            log.debug(f"  [{key}] URL directe échouée : {e} — essai slugs…")

    # 2. Slugs data.gouv.fr (du plus spécifique au plus générique)
    for slug in cfg.get("slugs", []):
        log.debug(f"  [{key}] essai slug : {slug}")
        dataset = _get_dataset(slug)
        if not dataset:
            continue
        content = _try_download_resource(dataset, cfg, expected_fmt)
        if content is not None:
            target.write_bytes(content)
            size_mo = len(content) / 1_048_576
            log.info(f"  ✓  {label:<22} {size_mo:.1f} Mo [data.gouv.fr / {slug[:40]}]")
            return True

    # 3. Recherche textuelle en dernier recours
    query = cfg.get("search_query") or " ".join(cfg.get("resource_hints", [])[:3])
    log.debug(f"  [{key}] fallback recherche : {query!r}")
    dataset = _search_dataset(query)
    if dataset:
        content = _try_download_resource(dataset, cfg, expected_fmt)
        if content is not None:
            target.write_bytes(content)
            size_mo = len(content) / 1_048_576
            log.info(f"  ✓  {label:<22} {size_mo:.1f} Mo [data.gouv.fr / recherche]")
            return True

    if required:
        log.warning(f"  ✗  {label:<22} introuvable — features associées = NaN")
    else:
        log.info(f"  –  {label:<22} non disponible (optionnel)")
    return False


def _pick_resources_ranked(
    resources: list[dict],
    hints: list[str],
    formats: list[str],
) -> list[dict]:
    """
    Retourne TOUTES les ressources valides triées par score décroissant.
    Rejette les descriptions multi-format ("csv, xlsx, xls..." ou "xls csv") :
    ce sont des pages de documentation, pas des fichiers de données.
    Infère le format depuis l'URL si le champ format est absent ou invalide.
    """
    accepted = set(formats) | {"zip"}
    scored: list[tuple[int, dict]] = []
    for r in resources:
        title = (r.get("title") or "").lower()
        url = r.get("latest") or r.get("url") or ""
        url_name = url.rsplit("/", 1)[-1].split("?")[0].lower()

        fmt = (r.get("format") or "").lower().strip()

        # Multi-format description → documentation resource, not a data file.
        if "," in fmt or (fmt and " " in fmt):
            fmt = ""

        # Normalize MIME types → simple extension (data.gouv.fr sometimes returns
        # "text/csv", "application/vnd.ms-excel", "application/zip", etc.)
        if fmt and "/" in fmt:
            if "csv" in fmt or "dsv" in fmt or "tab" in fmt:
                fmt = "csv"
            elif "openxmlformat" in fmt or "spreadsheetml" in fmt:
                fmt = "xlsx"
            elif "ms-excel" in fmt or "vnd.ms-excel" in fmt:
                fmt = "xls"
            elif "zip" in fmt or "gzip" in fmt or "x-compressed" in fmt:
                fmt = "zip"
            else:
                fmt = ""

        if not fmt:
            for ext in ("csv", "xlsx", "xls", "zip"):
                if url.lower().endswith(f".{ext}"):
                    fmt = ext
                    break

        if fmt not in accepted:
            continue

        # Score: title match = 3 pts, URL basename match = 4 pts (URL is more reliable)
        score = 0
        for h in hints:
            h_low = h.lower()
            if h_low in title:
                score += 3
            if h_low in url_name:
                score += 4
        if fmt != "zip":
            score += 2
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored]


def _pick_resource(
    resources: list[dict],
    hints: list[str],
    formats: list[str],
) -> Optional[dict]:
    """Retourne la meilleure ressource (top-1 du ranking)."""
    ranked = _pick_resources_ranked(resources, hints, formats)
    return ranked[0] if ranked else None


def _get_all_dataset_resources(dataset: dict) -> list[dict]:
    """
    Returns ALL resources for a dataset.
    data.gouv.fr may truncate the embedded resources list for large datasets;
    the paginated /resources/ endpoint always returns the full set.
    """
    embedded = dataset.get("resources", [])
    dataset_id = dataset.get("id")
    if not dataset_id:
        return embedded

    try:
        all_res: list[dict] = []
        for page in range(1, 21):  # safety cap: 20 pages × 50 = 1,000 resources max
            r = requests.get(
                f"{BASE_URL}/datasets/{dataset_id}/resources/",
                params={"page": page, "page_size": 50},
                timeout=20,
            )
            if r.status_code != 200:
                break
            body = r.json()
            page_data = body.get("data", body) if isinstance(body, dict) else body
            if not isinstance(page_data, list) or not page_data:
                break
            all_res.extend(page_data)
            if len(page_data) < 50:
                break
        if all_res:
            log.debug(
                f"    Ressources paginées : {len(all_res)} (embedded={len(embedded)})"
            )
            return all_res
    except Exception as exc:
        log.debug(f"    Pagination ressources échouée : {exc}")

    return embedded


def _get_dataset(slug: str) -> Optional[dict]:
    """Récupère un dataset data.gouv.fr par son slug exact."""
    try:
        r = requests.get(f"{BASE_URL}/datasets/{slug}/", timeout=20)
        if r.status_code == 200:
            return r.json()
        r2 = requests.get(f"{BASE_URL}/datasets/", params={"slug": slug}, timeout=20)
        if r2.status_code == 200:
            data = r2.json().get("data", [])
            for d in data:
                if d.get("slug") == slug:
                    return d
    except Exception as e:
        log.debug(f"    _get_dataset({slug}): {e}")
    return None


def _search_dataset(query: str) -> Optional[dict]:
    """Recherche un dataset data.gouv.fr par mots-clés (retourne le top résultat)."""
    try:
        r = requests.get(
            f"{BASE_URL}/datasets/",
            params={"q": query, "page_size": 5, "sort": "relevance"},
            timeout=20,
        )
        if r.status_code == 200:
            data = r.json().get("data", [])
            return data[0] if data else None
    except Exception as e:
        log.debug(f"    _search_dataset({query!r}): {e}")
    return None


def _search_datasets(query: str, limit: int = 5) -> list[dict]:
    """Recherche plusieurs datasets data.gouv.fr — pour le fallback élections."""
    try:
        r = requests.get(
            f"{BASE_URL}/datasets/",
            params={"q": query, "page_size": limit, "sort": "relevance"},
            timeout=20,
        )
        if r.status_code == 200:
            return r.json().get("data", [])[:limit]
    except Exception as e:
        log.debug(f"    _search_datasets({query!r}): {e}")
    return []


def _get_with_retry(
    url: str, max_retries: int = 3, timeout: int = 120
) -> requests.Response:
    """GET avec retry exponentiel."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout, stream=False)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait = 2**attempt
                log.debug(f"    Retry {attempt+1}/{max_retries} dans {wait}s : {e}")
                time.sleep(wait)
            else:
                raise


def _log_summary(results: dict[str, bool]) -> None:
    all_labels = {"elections": "Élections"}
    all_labels.update(_DATASET_LABELS)
    required_keys = {cfg["key"] for cfg in OTHER_DATASETS if cfg.get("required", True)}
    required_keys.add("elections")

    ok = [k for k, v in results.items() if v]
    nok_required = [k for k, v in results.items() if not v and k in required_keys]
    nok_optional = [k for k, v in results.items() if not v and k not in required_keys]

    n_total = len(results)
    n_ok = len(ok)
    parts = [f"{n_ok}/{n_total} OK"]
    if nok_required:
        parts.append(f"{len(nok_required)} requis manquant(s)")
    if nok_optional:
        parts.append(f"{len(nok_optional)} optionnel(s) ignoré(s)")
    level = log.warning if nok_required else log.info
    level(f"━━━ FETCH terminé | {' · '.join(parts)} ━━━")

    for k in nok_required:
        log.warning(
            f"  ✗  {all_labels.get(k, k):<22} requis — placez le fichier dans DATA_ROOT si disponible"
        )
    for k in nok_optional:
        log.debug(f"  –  {all_labels.get(k, k):<22} optionnel non chargé")
