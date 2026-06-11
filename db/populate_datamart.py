"""
Populate Datamart — Alimentation du schéma en étoile depuis le dataset gold.
=============================================================================
Ce script lit le CSV gold (dataset_elections_2022_idf.csv) et alimente les
5 tables du datamart analytique (schéma en étoile) dans PostgreSQL :

  datamart.dim_commune         — 1268 communes IDF + indicateurs socio-éco
  datamart.dim_contexte        — indicateurs par commune × millésime (2017, 2022)
  datamart.fait_resultats_electoraux — résultats T1/T2 pour 2012, 2017, 2022

dim_candidat et dim_temps sont pré-remplis par db/init.sql.

Usage :
    python db/populate_datamart.py [--csv PATH] [--dry-run]

Variables d'environnement :
    DATABASE_URL  ex. postgresql://etl_writer:pwd@localhost:5432/elections
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import psycopg2
    import psycopg2.extras

    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


# ─── Candidats T2 2022 dans le CSV gold ──────────────────────────────────────
# Mapping colonne CSV → nom_candidat dans dim_candidat
CANDIDATS_T2_2022 = {
    "cible_t2_pct_macron": "Emmanuel Macron",
    "cible_t2_pct_lepen": "Marine Le Pen",
}
CANDIDATS_T1_2022 = {
    "cible_t1_pct_macron": "Emmanuel Macron",
    "cible_t1_pct_lepen": "Marine Le Pen",
    "cible_t1_pct_melenchon": "Jean-Luc Mélenchon",
    "cible_t1_pct_zemmour": "Éric Zemmour",
    "cible_t1_pct_pecresse": "Valérie Pécresse",
    "cible_t1_pct_jadot": "Yannick Jadot",
}
CANDIDATS_T2_2017 = {
    "h17_t2_pct_macron": "Emmanuel Macron",
    "h17_t2_pct_lepen": "Marine Le Pen",
}
CANDIDATS_T1_2017 = {
    "h17_t1_pct_macron": "Emmanuel Macron",
    "h17_t1_pct_lepen": "Marine Le Pen",
    "h17_t1_pct_melenchon": "Jean-Luc Mélenchon",
    "h17_t1_pct_fillon": "François Fillon",
    "h17_t1_pct_hamon": "Benoît Hamon",
}
CANDIDATS_T2_2012 = {
    "h12_t2_pct_hollande": "François Hollande",
    "h12_t2_pct_sarkozy": "Nicolas Sarkozy",
}
CANDIDATS_T1_2012 = {
    "h12_t1_pct_hollande": "François Hollande",
    "h12_t1_pct_sarkozy": "Nicolas Sarkozy",
    "h12_t1_pct_lepen": "Marine Le Pen",
    "h12_t1_pct_melenchon": "Jean-Luc Mélenchon",
    "h12_t1_pct_bayrou": "Autres candidats",
}


def _col(df: pd.DataFrame, name: str, default=None):
    """Retourne df[name] si la colonne existe, sinon une Série remplie de default."""
    return (
        df[name]
        if name in df.columns
        else pd.Series([default] * len(df), index=df.index)
    )


def load_gold(csv_path: Path) -> pd.DataFrame:
    print(f"[load] Lecture du CSV gold : {csv_path}")
    df = pd.read_csv(csv_path, dtype={"code_commune": str, "code_departement": str})
    df["code_commune"] = df["code_commune"].str.zfill(5)
    df["code_departement"] = df["code_departement"].str.zfill(2)
    print(f"[load] {len(df)} communes × {len(df.columns)} colonnes")
    return df


# ─── Populate dim_commune ─────────────────────────────────────────────────────


def populate_dim_commune(cur, df: pd.DataFrame) -> dict[str, int]:
    """Insère les communes dans dim_commune. Retourne {code_commune: sk_commune}."""
    rows = []
    for _, r in df.iterrows():
        rows.append(
            (
                str(r["code_commune"]),
                str(r["code_departement"]),
                r.get("libelle_departement"),
                r.get("libelle_commune"),
                r.get("revenu_median_2021") or r.get("revenu_median"),
                r.get("taux_chomage_rp2022"),
                r.get("taux_pauvrete_2017"),
                r.get("pct_dipl_superieur"),
                r.get("pct_csp_cadres"),
                r.get("pct_log_hlm"),
                r.get("pop_totale"),
                r.get("taux_crimes_total"),
                r.get("nb_entreprises_actives"),
                r.get("nb_associations_actives"),
            )
        )

    psycopg2.extras.execute_batch(
        cur,
        """
        INSERT INTO datamart.dim_commune
            (code_commune, code_departement, libelle_departement, libelle_commune,
             revenu_median, taux_chomage, taux_pauvrete, pct_dipl_superieur,
             pct_csp_cadres, pct_log_hlm, pop_totale,
             taux_crimes_total, nb_entreprises, nb_associations)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (code_commune) DO UPDATE SET
            revenu_median       = EXCLUDED.revenu_median,
            taux_chomage        = EXCLUDED.taux_chomage,
            taux_pauvrete       = EXCLUDED.taux_pauvrete,
            pct_dipl_superieur  = EXCLUDED.pct_dipl_superieur,
            pct_csp_cadres      = EXCLUDED.pct_csp_cadres,
            pct_log_hlm         = EXCLUDED.pct_log_hlm,
            pop_totale          = EXCLUDED.pop_totale,
            taux_crimes_total   = EXCLUDED.taux_crimes_total,
            nb_entreprises      = EXCLUDED.nb_entreprises,
            nb_associations     = EXCLUDED.nb_associations,
            updated_at          = NOW()
        """,
        rows,
        page_size=500,
    )

    cur.execute("SELECT code_commune, sk_commune FROM datamart.dim_commune")
    return {row[0]: row[1] for row in cur.fetchall()}


# ─── Populate dim_contexte ────────────────────────────────────────────────────


def populate_dim_contexte(cur, df: pd.DataFrame) -> dict[tuple, int]:
    """Insère le contexte socio-éco par commune × millésime (2017, 2022)."""
    rows = []
    for _, r in df.iterrows():
        commune = str(r["code_commune"])
        # Millésime 2017
        rows.append(
            (
                commune,
                2017,
                r.get("revenu_median_2017"),
                r.get("taux_chomage_2009"),  # proxy si pas de 2017
                r.get("taux_pauvrete_2017"),
                r.get("pct_dipl_superieur"),
                r.get("pct_csp_cadres"),
                r.get("pct_pop_senior_60p"),
                r.get("pct_log_hlm"),
            )
        )
        # Millésime 2022
        rows.append(
            (
                commune,
                2022,
                r.get("revenu_median_2021"),
                r.get("taux_chomage_rp2022"),
                r.get("taux_pauvrete_2017"),  # dernière valeur disponible
                r.get("pct_dipl_superieur"),
                r.get("pct_csp_cadres"),
                r.get("pct_pop_senior_60p"),
                r.get("pct_log_hlm"),
            )
        )

    psycopg2.extras.execute_batch(
        cur,
        """
        INSERT INTO datamart.dim_contexte
            (code_commune, millesime, revenu_median, taux_chomage, taux_pauvrete,
             pct_dipl_superieur, pct_csp_cadres, pct_pop_senior_60p, pct_log_hlm)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (code_commune, millesime) DO NOTHING
        """,
        rows,
        page_size=500,
    )

    cur.execute(
        "SELECT code_commune, millesime, sk_contexte FROM datamart.dim_contexte"
    )
    return {(row[0], row[1]): row[2] for row in cur.fetchall()}


# ─── Populate fait_resultats_electoraux ───────────────────────────────────────


def _insert_faits(cur, faits: list[tuple]) -> None:
    psycopg2.extras.execute_batch(
        cur,
        """
        INSERT INTO datamart.fait_resultats_electoraux
            (sk_commune, sk_candidat, sk_temps, sk_contexte,
             nb_voix, pct_voix, taux_participation, taux_abstention,
             vainqueur_t2, marge_vainqueur_t2)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT DO NOTHING
        """,
        faits,
        page_size=1000,
    )


def populate_faits(
    cur,
    df: pd.DataFrame,
    sk_communes: dict[str, int],
    sk_contexte: dict[tuple, int],
) -> None:
    # Récupère les SK candidats et temps depuis la DB
    cur.execute("SELECT nom_candidat, sk_candidat FROM datamart.dim_candidat")
    sk_cands = {row[0]: row[1] for row in cur.fetchall()}

    cur.execute("SELECT annee, tour, sk_temps FROM datamart.dim_temps")
    sk_temps = {(row[0], row[1]): row[2] for row in cur.fetchall()}

    elections = [
        # (annee, tour, millesime_contexte, candidats_map, col_particip, col_abstention, col_vainqueur, col_marge)
        (
            2022,
            2,
            2022,
            CANDIDATS_T2_2022,
            "taux_participation_t2",
            "taux_abstention_t2",
            "cible_t2_vainqueur",
            "cible_t2_marge",
        ),
        (
            2022,
            1,
            2022,
            CANDIDATS_T1_2022,
            "taux_participation_t1",
            "taux_abstention_t1",
            None,
            None,
        ),
        (
            2017,
            2,
            2017,
            CANDIDATS_T2_2017,
            None,
            None,
            "h17_t2_vainqueur",
            "h17_t2_marge",
        ),
        (2017, 1, 2017, CANDIDATS_T1_2017, None, None, None, None),
        (
            2012,
            2,
            2017,
            CANDIDATS_T2_2012,
            None,
            None,
            "h12_t2_vainqueur",
            "h12_t2_marge",
        ),
        (2012, 1, 2017, CANDIDATS_T1_2012, None, None, None, None),
    ]

    total_faits = 0
    for (
        annee,
        tour,
        millesime,
        cands_map,
        col_part,
        col_abs,
        col_vainq,
        col_marge,
    ) in elections:
        sk_t = sk_temps.get((annee, tour))
        if sk_t is None:
            print(f"[faits] WARN: sk_temps introuvable pour ({annee}, {tour})")
            continue

        faits = []
        for _, r in df.iterrows():
            commune = str(r["code_commune"])
            sk_c = sk_communes.get(commune)
            if sk_c is None:
                continue
            sk_ctx = sk_contexte.get((commune, millesime))

            particip = r.get(col_part) if col_part and col_part in df.columns else None
            abstention = r.get(col_abs) if col_abs and col_abs in df.columns else None
            vainq = r.get(col_vainq) if col_vainq and col_vainq in df.columns else None
            marge = r.get(col_marge) if col_marge and col_marge in df.columns else None

            for col_pct, nom_cand in cands_map.items():
                if col_pct not in df.columns:
                    continue
                sk_cand = sk_cands.get(nom_cand)
                if sk_cand is None:
                    continue
                pct = r.get(col_pct)
                faits.append(
                    (
                        sk_c,
                        sk_cand,
                        sk_t,
                        sk_ctx,
                        None,  # nb_voix non disponible dans le gold agrégé
                        float(pct) if pd.notna(pct) else None,
                        float(particip) if pd.notna(particip) else None,
                        float(abstention) if pd.notna(abstention) else None,
                        int(vainq) if pd.notna(vainq) else None,
                        float(marge) if pd.notna(marge) else None,
                    )
                )

        _insert_faits(cur, faits)
        total_faits += len(faits)
        print(f"[faits] {annee} T{tour} : {len(faits)} lignes insérées")

    print(f"[faits] Total : {total_faits} faits insérés")


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _build_db_url() -> str:
    """Construit l'URL PostgreSQL depuis DATABASE_URL ou les var. d'env. Docker."""
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "elections_idf")
    user = os.environ.get("POSTGRES_USER", "etl_admin")
    pwd = os.environ.get("POSTGRES_PASSWORD", "elections2022")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"


def _default_csv() -> Path:
    """Résout le chemin CSV depuis DATA_PATH (Docker) ou le répertoire par défaut."""
    env = os.environ.get("DATA_PATH")
    if env:
        return Path(env)
    return ROOT.parent / "dataset_elections_2022_idf.csv"


# ─── API programmatique (appelée depuis run_all.py) ───────────────────────────


def run(csv_path: Path | None = None, db_url: str | None = None) -> bool:
    """
    Alimente le schéma datamart depuis le CSV gold.
    Retourne True si succès, False sinon (ne lève pas d'exception).
    """
    if not HAS_PSYCOPG2:
        print("[WARN] psycopg2 absent — datamart non alimenté.")
        return False

    csv_path = Path(csv_path) if csv_path else _default_csv()
    if not csv_path.exists():
        print(f"[WARN] CSV introuvable ({csv_path}) — datamart non alimenté.")
        return False

    db_url = db_url or _build_db_url()

    try:
        df = load_gold(csv_path)
        conn = psycopg2.connect(db_url)
        try:
            with conn:
                with conn.cursor() as cur:
                    sk_communes = populate_dim_commune(cur, df)
                    sk_ctx = populate_dim_contexte(cur, df)
                    populate_faits(cur, df, sk_communes, sk_ctx)
            print(f"[OK] Datamart alimenté — {len(sk_communes)} communes.")
            return True
        finally:
            conn.close()
    except Exception as exc:
        print(f"[ERROR] Datamart : {exc}")
        return False


# ─── Main CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alimente le schéma datamart en étoile."
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Chemin vers le CSV gold (défaut: DATA_PATH ou ../dataset_elections_2022_idf.csv)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Valide la lecture sans écrire en base"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else _default_csv()
    if not csv_path.exists():
        print(f"[ERROR] CSV introuvable : {csv_path}")
        sys.exit(1)

    df = load_gold(csv_path)

    if args.dry_run:
        print("[dry-run] Lecture OK —", len(df), "communes. Pas d'écriture en base.")
        return

    if not HAS_PSYCOPG2:
        print("[ERROR] psycopg2 non installé. Lancez : pip install psycopg2-binary")
        sys.exit(1)

    db_url = _build_db_url()
    print(f"[db] Connexion à {db_url.split('@')[-1]}")

    ok = run(csv_path=csv_path, db_url=db_url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
