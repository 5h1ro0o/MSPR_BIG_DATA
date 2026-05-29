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

ELECTION_SLUGS = {
    "2022_pres_t1": "election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour",
    "2022_pres_t2": "election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-2nd-tour",
    "2017_pres_t1": "election-presidentielle-des-23-avril-et-7-mai-2017-resultats-definitifs-du-1er-tour",
    "2017_pres_t2": "election-presidentielle-des-23-avril-et-7-mai-2017-resultats-definitifs-du-2eme-tour",
    "2012_pres_t1": "election-presidentielle-des-22-avril-et-6-mai-2012-resultats-definitifs-par-departement-et-commune",
    "2012_pres_t2": "election-presidentielle-des-22-avril-et-6-mai-2012-resultats-definitifs-par-departement-et-commune",
}
ELECTION_TOUR_HINT = {
    "2012_pres_t1": ["tour 1", "premier", "t1", "1er"],
    "2012_pres_t2": ["tour 2", "second", "t2", "2ème", "2eme", "deuxième"],
}

OTHER_DATASETS = [
    {
        "key": "demographics",
        "slugs": [
            "bases-de-donnees-de-synthese-au-niveau-de-la-commune-2022",
            "recensement-de-la-population-2022-base-de-donnees-commune",
            "bases-de-donnees-communales-du-recensement-de-la-population-2022",
        ],
        "resource_hints": ["dossier_complet", "dossier complet"],
        "formats": ["csv", "zip"],
        "target": "demographique +csp/dossier_complet.csv",
        "required": True,
    },
    {
        "key": "filosofi",
        "slugs": [
            "revenus-et-pauvrete-des-menages-en-2017",
            "statistiques-filosofi-2017",
            "filosofi-2017-statistiques-locales",
        ],
        "resource_hints": ["filosofi", "commune", "base-cc"],
        "formats": ["xlsx", "xls", "zip"],
        "target": "pauvreté/base-cc-filosofi-2017.xlsx",
        "required": True,
    },
    {
        "key": "emploi_2022",
        "slugs": [
            "emploi-par-commune-recensement-2022",
            "statistiques-de-lemploi-au-recensement-2022",
            "bases-de-donnees-communales-emploi-2022",
        ],
        "resource_hints": ["emploi", "DS_RP_EMPLOI", "commune", "2022"],
        "formats": ["csv", "zip"],
        "target": "chomage/DS_RP_EMPLOI_LR_COMP_2022_data.csv",
        "required": True,
    },
    {
        "key": "chomage_idf",
        "slugs": [
            "taux-de-chomage-locaux-communes-idf",
            "chomage-communes-ile-de-france",
            "chomage-localise-zone-emploi",
        ],
        "resource_hints": ["commune", "chomage", "ile-de-france", "idf"],
        "formats": ["csv", "zip"],
        "target": "chomage/chomage-nombre-de-chomeurs-exhaustif-des-communes-dile-de-france-donnee-insee.csv",
        "required": True,
    },
]


def fetch_all_datasets(data_root: Path) -> dict[str, bool]:
    """
    Télécharge tous les datasets manquants dans data_root.

    Returns:
        Dictionnaire {dataset_key: success_bool}
    """
    data_root = Path(data_root)
    results: dict[str, bool] = {}

    log.info("=== FETCH data.gouv.fr ===")

    ok = _fetch_elections(data_root)
    results["elections"] = ok

    for cfg in OTHER_DATASETS:
        target = data_root / cfg["target"]
        if target.exists() and target.stat().st_size > 1_000:
            log.info(f"  [SKIP] {cfg['key']} déjà présent : {target.name}")
            results[cfg["key"]] = True
            continue
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

    existing_elections = list(elections_dir.glob("elections_??.csv"))
    if len(existing_elections) >= 8 and candidats_file.exists():
        log.info("  [SKIP] Fichiers élections déjà présents")
        return True

    all_participation: list[pd.DataFrame] = []
    all_candidats: list[pd.DataFrame] = []

    for id_election, slug in ELECTION_SLUGS.items():
        log.info(f"  Téléchargement {id_election} …")
        raw = _download_election_csv(id_election, slug)
        if raw is None:
            log.warning(f"  [WARN] {id_election} non téléchargé")
            continue

        participation, candidats = _parse_election_csv(raw, id_election)
        if participation is not None:
            all_participation.append(participation)
        if candidats is not None:
            all_candidats.append(candidats)

    if not all_participation:
        log.error("  Aucune donnée d'élection téléchargée")
        return False

    participation_full = pd.concat(all_participation, ignore_index=True)
    _write_elections_by_dept(participation_full, elections_dir)

    if all_candidats:
        candidats_full = pd.concat(all_candidats, ignore_index=True)
        candidats_full.to_csv(candidats_file, sep=";", index=False)
        log.info(
            f"  Candidats : {len(candidats_full):,} lignes → {candidats_file.name}"
        )

    return True


def _download_election_csv(id_election: str, slug: str) -> Optional[pd.DataFrame]:
    """Cherche et télécharge le CSV de résultats par commune pour une élection."""
    dataset = _get_dataset(slug)
    if dataset is None:
        log.warning(f"    Dataset introuvable : {slug}")
        return None

    resources = dataset.get("resources", [])
    tour_hints = ELECTION_TOUR_HINT.get(id_election, [])

    resource = _pick_election_resource(resources, tour_hints)
    if resource is None:
        log.warning(f"    Aucune ressource CSV pour {id_election}")
        return None

    url = resource.get("url") or resource.get("latest")
    fmt = (resource.get("format") or "").lower()
    title = resource.get("title", "")
    log.info(f"    Ressource sélectionnée : {title!r} ({fmt})")

    return _download_and_parse(url, fmt)


def _pick_election_resource(
    resources: list[dict],
    tour_hints: list[str],
) -> Optional[dict]:
    """
    Choisit la ressource la plus pertinente dans la liste.
    Priorité : résultats par commune > CSV > XLSX/ZIP > bureau de vote (BV).
    """
    COMMUNE_KEYWORDS = ["commune", "résultats", "resultats", "détaillé", "detail"]
    BV_KEYWORDS = ["burvot", "bureau de vote", "bureaux de vote", "bv-t"]

    scored: list[tuple[int, dict]] = []
    for r in resources:
        title = _normalize_col(r.get("title") or "")
        fmt = (r.get("format") or "").lower()
        if fmt not in ("csv", "xlsx", "xls", "zip"):
            continue
        score = 0
        if any(_normalize_col(k) in title for k in COMMUNE_KEYWORDS):
            score += 10
        if fmt == "csv":
            score += 6
        elif fmt in ("xlsx", "xls"):
            score += 2
        if tour_hints and any(_normalize_col(h) in title for h in tour_hints):
            score += 8
        if any(_normalize_col(k) in title for k in BV_KEYWORDS):
            score -= 15
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else None


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
            return pd.read_excel(io.BytesIO(content), dtype=str)

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
    required = {"code_departement", "code_commune", "inscrits", "votants", "exprimes"}
    if not required.issubset(col_map):
        missing = required - set(col_map)
        log.warning(f"    Colonnes manquantes pour {id_election}: {missing}")
        log.warning(f"    Colonnes disponibles: {list(df.columns)[:20]}")
        return None, None

    df = df.rename(columns={v: k for k, v in col_map.items()})

    df["code_departement"] = (
        df["code_departement"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
        .str.zfill(2)
    )
    df["code_commune"] = (
        df["code_commune"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
        .str.zfill(5)
    )

    df_idf = df[df["code_departement"].isin(IDF_DEPTS)].copy()
    if df_idf.empty:
        log.warning(f"    Aucune commune IDF dans {id_election}")
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
    participation = df_idf[avail_part].drop_duplicates(subset=["code_commune"]).copy()
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
    if "nom" in df_idf.columns and "voix" in df_idf.columns:
        cand_cols = [
            "code_departement",
            "code_commune",
            "nom",
            "prenom" if "prenom" in df_idf.columns else "nom",
            "voix",
        ]
        cand = df_idf[[c for c in cand_cols if c in df_idf.columns]].copy()
        cand["id_election"] = id_election
        cand = cand.rename(
            columns={
                "code_departement": "code_departement",
                "code_commune": "code_commune",
            }
        )
        if "prenom" not in cand.columns:
            cand["prenom"] = ""
        candidats = cand[
            ["id_election", "code_departement", "code_commune", "voix", "nom", "prenom"]
        ]

    log.info(
        f"    {id_election} : {len(participation):,} communes IDF"
        + (f" | {len(candidats):,} candidat-lignes" if candidats is not None else "")
    )
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
        log.info(f"  Écrit {path.name} : {len(dept_df):,} lignes")


def _fetch_other_dataset(cfg: dict, data_root: Path) -> bool:
    """Télécharge un dataset non-élection depuis data.gouv.fr."""
    key = cfg["key"]
    target = data_root / cfg["target"]
    target.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"  Recherche dataset : {key}")

    dataset = None
    for slug in cfg["slugs"]:
        dataset = _get_dataset(slug)
        if dataset:
            log.info(f"    Trouvé via slug : {slug}")
            break

    if dataset is None:
        query = " ".join(cfg["resource_hints"][:3])
        log.info(f"    Fallback recherche : {query!r}")
        dataset = _search_dataset(query)

    if dataset is None:
        log.warning(f"  [WARN] Dataset {key} introuvable sur data.gouv.fr")
        return False

    resources = dataset.get("resources", [])
    resource = _pick_resource(resources, cfg["resource_hints"], cfg["formats"])
    if resource is None:
        log.warning(f"  [WARN] Aucune ressource {cfg['formats']} pour {key}")
        return False

    url = resource.get("url") or resource.get("latest")
    fmt = (resource.get("format") or "").lower() or target.suffix.lstrip(".")
    title = resource.get("title", "?")
    log.info(f"    Téléchargement : {title!r} ({fmt}) …")

    try:
        r = _get_with_retry(url)
        content = r.content

        if url.endswith(".zip") or fmt == "zip":
            target_ext = Path(cfg["target"]).suffix.lower()
            ext_map = {".csv": "csv", ".xlsx": "xlsx", ".xls": "xls"}
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                candidates = [
                    n
                    for n in zf.namelist()
                    if Path(n).suffix.lower() == target_ext and not n.startswith("__")
                ]
                if not candidates:
                    candidates = [
                        n
                        for n in zf.namelist()
                        if Path(n).suffix.lower() in ext_map and not n.startswith("__")
                    ]
                if candidates:
                    name = candidates[0]
                    content = zf.read(name)
                    fmt = ext_map.get(Path(name).suffix.lower(), "csv")

        target.write_bytes(content)
        log.info(f"    Sauvegardé : {target.name} ({len(content)/1_048_576:.1f} Mo)")
        return True

    except Exception as e:
        log.warning(f"  [WARN] Erreur téléchargement {key}: {e}")
        return False


def _pick_resource(
    resources: list[dict],
    hints: list[str],
    formats: list[str],
) -> Optional[dict]:
    """Sélectionne la ressource la plus pertinente.

    Accepte aussi les ressources ZIP (archives pouvant contenir le fichier cible).
    """
    accepted = set(formats) | {"zip"}
    scored: list[tuple[int, dict]] = []
    for r in resources:
        title = (r.get("title") or "").lower()
        fmt = (r.get("format") or "").lower()
        if not any(f in fmt for f in accepted):
            continue
        score = sum(3 if h.lower() in title else 0 for h in hints)
        if fmt != "zip":
            score += 2
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else None


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
    """Recherche un dataset data.gouv.fr par mots-clés."""
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
    ok = [k for k, v in results.items() if v]
    nok = [k for k, v in results.items() if not v]
    log.info(f"=== FETCH terminé | OK: {len(ok)} | KO: {len(nok)} ===")
    if nok:
        log.warning(f"  Datasets manquants : {nok}")
        log.warning(
            "  → Placez ces fichiers manuellement dans DATA_ROOT si disponibles localement."
        )
