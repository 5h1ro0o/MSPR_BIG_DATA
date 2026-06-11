"""
ml/projection/generate_projections.py
======================================
Génère des projections électorales à T+1, T+2, T+3 ans à partir du modèle
Gradient Boosting pré-vote (sans données T1 2022 — seul modèle applicable
à une élection future dont on ne connaît pas encore le premier tour).

Méthode :
  1. Calculer les tendances socio-économiques par commune depuis les données historiques
  2. Extrapoler ces tendances pour T+1 (≈2025), T+2 (≈2026), T+3 (≈2027)
  3. Appliquer le modèle GB pré-vote sur les features projetées
  4. Sauvegarder 3 CSV de prédictions + 1 CSV de synthèse par département

Hypothèses :
  - Les features sans données multi-années restent constantes (hypothèse conservatrice)
  - Les features historiques (h12, h17) sont maintenues : la géographie électorale
    évolue lentement et n'est pas projetable sur 3 ans
  - Les tendances observées se poursuivent de façon linéaire
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ML_DIR    = Path(__file__).parent.parent
ARTIFACTS = ML_DIR / "artifacts"
DATA_PATH = Path(
    os.environ.get("DATA_PATH", str(ML_DIR.parent.parent / "dataset_elections_2022_idf.csv"))
)

# Horizons de projection (en années depuis la dernière donnée ≈ 2022)
HORIZONS = {1: "2025", 2: "2026", 3: "2027"}

# Features pour lesquelles une tendance est calculable à partir des données historiques
# Format : feature_actuelle → (feature_ancienne, année_ancienne, année_actuelle)
TRENDABLE = {
    "taux_chomage_rp2022": ("taux_chomage_2009", 2009, 2022),
    "revenu_median_2021":  ("revenu_median_2017", 2017, 2021),
}

# Tendances démographiques nationales IDF annuelles (source : projections INSEE 2020-2050)
# Valeurs en points de pourcentage par an
DEMO_TRENDS = {
    "pct_pop_0014":      -0.05,
    "pct_pop_1529":      -0.03,
    "pct_pop_3044":      -0.02,
    "pct_pop_4559":      -0.04,
    "pct_pop_6074":      +0.06,
    "pct_pop_7589":      +0.04,
    "pct_pop_90p":       +0.02,
    "pct_pop_senior_60p": +0.07,
}

# Plages valides pour clipper les valeurs projetées (évite les extrapolations absurdes)
FEATURE_BOUNDS = {
    "taux_chomage_rp2022":  (2.0, 35.0),
    "revenu_median_2021":   (10_000, 100_000),
    "pct_pop_0014":         (0.0, 35.0),
    "pct_pop_1529":         (0.0, 35.0),
    "pct_pop_3044":         (0.0, 35.0),
    "pct_pop_4559":         (0.0, 35.0),
    "pct_pop_6074":         (0.0, 30.0),
    "pct_pop_7589":         (0.0, 20.0),
    "pct_pop_90p":          (0.0, 10.0),
    "pct_pop_senior_60p":   (0.0, 50.0),
}


def compute_per_commune_trends(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Calcule le delta annuel par commune pour chaque feature trendable.
    Retourne un dict {feature_actuelle: pd.Series(delta_annuel par commune)}.
    """
    deltas = {}
    for feat_current, (feat_old, yr_old, yr_current) in TRENDABLE.items():
        if feat_current not in df.columns or feat_old not in df.columns:
            continue
        n_years = yr_current - yr_old
        delta = (df[feat_current] - df[feat_old]) / n_years
        deltas[feat_current] = delta
        print(f"  Tendance {feat_current}: delta/an = {delta.mean():.3f} (mediane {delta.median():.3f})")
    return deltas


def project_features(
    X: pd.DataFrame,
    deltas: dict[str, pd.Series],
    horizon_years: int,
) -> pd.DataFrame:
    """
    Applique les tendances socio-économiques et démographiques sur un horizon donné.
    Retourne une copie de X avec les features mises à jour.
    """
    X_proj = X.copy()

    # Tendances issues des données historiques par commune
    for feat, delta in deltas.items():
        if feat in X_proj.columns:
            X_proj[feat] = X_proj[feat] + delta * horizon_years
            if feat in FEATURE_BOUNDS:
                lo, hi = FEATURE_BOUNDS[feat]
                X_proj[feat] = X_proj[feat].clip(lo, hi)

    # Tendances démographiques nationales (constantes pour toutes les communes)
    for feat, annual_delta in DEMO_TRENDS.items():
        if feat in X_proj.columns:
            X_proj[feat] = X_proj[feat] + annual_delta * horizon_years
            if feat in FEATURE_BOUNDS:
                lo, hi = FEATURE_BOUNDS[feat]
                X_proj[feat] = X_proj[feat].clip(lo, hi)

    return X_proj


def load_model_and_data() -> tuple:
    """Charge le modèle GB pré-vote et le dataset gold."""
    model_path = ARTIFACTS / "gradient_boosting_pre_vote_classification_t2.joblib"
    meta_path  = ARTIFACTS / "gradient_boosting_pre_vote_classification_t2_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle pré-vote introuvable : {model_path}\n"
            "Lancez d'abord : python -X utf8 ml/run_training.py --model gb --feature-set pre_vote"
        )

    print(f"  Chargement modèle : {model_path.name}")
    pipeline = joblib.load(model_path)

    with open(meta_path) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    print(f"  Features : {len(feature_names)} colonnes")

    if not DATA_PATH.exists():
        alt = ML_DIR.parent / "dataset_elections_2022_idf.csv"
        if alt.exists():
            data_path = alt
        else:
            raise FileNotFoundError(f"Dataset gold introuvable : {DATA_PATH}")
    else:
        data_path = DATA_PATH

    print(f"  Dataset  : {data_path.name}")
    df = pd.read_csv(data_path, low_memory=False)

    return pipeline, feature_names, df


def generate():
    """Génère les 3 CSVs de projections (T+1, T+2, T+3) et le CSV de synthèse."""
    print("\n" + "="*60)
    print("  PROJECTIONS TEMPORELLES — T+1, T+2, T+3")
    print("="*60)

    pipeline, feature_names, df = load_model_and_data()

    # Colonnes identifiantes
    id_cols = ["code_commune", "code_departement", "libelle_departement", "libelle_commune"]
    id_df   = df[[c for c in id_cols if c in df.columns]].copy()

    # Vérifier que toutes les features du modèle sont disponibles
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        print(f"  [WARN] {len(missing)} features absentes du dataset : {missing[:5]}...")
    available = [f for f in feature_names if f in df.columns]

    # Construire la matrice de features de base (2022)
    mask = df[available].notna().all(axis=1)
    X_base = df.loc[mask, available].copy()
    id_base = id_df.loc[mask].reset_index(drop=True)
    X_base = X_base.reset_index(drop=True)

    print(f"  Communes avec données complètes : {len(X_base):,}")

    # Calculer les tendances socio-économiques par commune
    print("\n  Calcul des tendances socio-économiques...")
    deltas = compute_per_commune_trends(df.loc[mask].reset_index(drop=True))

    all_horizons = []

    for horizon, annee in HORIZONS.items():
        print(f"\n  Projection {annee} (T+{horizon})...")

        X_proj = project_features(X_base, deltas, horizon)

        # Imputer les NaN résiduels (colonne d'imputation du pipeline)
        proba = pipeline.predict_proba(X_proj)
        pred  = pipeline.predict(X_proj)

        result = id_base.copy()
        result["horizon_years"]  = horizon
        result["annee_cible"]    = annee
        result["prediction"]     = pred
        result["vainqueur_predit"] = pd.Series(pred).map({0: "Macron", 1: "Le Pen"})
        result["proba_macron"]   = proba[:, 0].round(4)
        result["proba_lepen"]    = proba[:, 1].round(4)

        # Indicateurs projetés (pour visualisation de l'évolution)
        result["chomage_projete"] = X_proj.get("taux_chomage_rp2022", pd.Series(dtype=float)).round(2)
        result["revenu_projete"]  = X_proj.get("revenu_median_2021",  pd.Series(dtype=float)).round(0)

        # Résultat 2022 comme baseline
        result["baseline_2022_macron"] = X_base.get(
            "taux_chomage_rp2022", pd.Series(dtype=float)  # placeholder
        )
        # On re-prédit sur la base 2022 pour avoir la comparaison
        proba_base = pipeline.predict_proba(X_base)
        result["proba_macron_2022"] = proba_base[:, 0].round(4)
        result["proba_lepen_2022"]  = proba_base[:, 1].round(4)
        result["delta_macron"]      = (result["proba_macron"] - result["proba_macron_2022"]).round(4)

        out_path = ARTIFACTS / f"projections_{annee}_t{horizon}.csv"
        result.to_csv(out_path, index=False)
        print(f"  Sauvegardé : {out_path.name}")

        # Stats rapides
        n_macron = (pred == 0).sum()
        n_lepen  = (pred == 1).sum()
        print(f"    Macron prédit : {n_macron} communes ({n_macron/len(pred)*100:.1f}%)")
        print(f"    Le Pen prédit : {n_lepen} communes ({n_lepen/len(pred)*100:.1f}%)")

        all_horizons.append(result)

    # CSV de synthèse par département (pour les courbes temporelles)
    print("\n  Génération synthèse par département...")
    dept_rows = []
    for horizon_df in all_horizons:
        horizon = int(horizon_df["horizon_years"].iloc[0])
        annee   = str(horizon_df["annee_cible"].iloc[0])
        for dept_code, group in horizon_df.groupby("code_departement"):
            dept_rows.append({
                "code_departement":    dept_code,
                "libelle_departement": group["libelle_departement"].iloc[0],
                "annee_cible":         annee,
                "horizon_years":       horizon,
                "n_communes":          len(group),
                "proba_macron_mean":   group["proba_macron"].mean().round(4),
                "proba_lepen_mean":    group["proba_lepen"].mean().round(4),
                "proba_macron_2022":   group["proba_macron_2022"].mean().round(4),
                "n_macron_predit":     (group["prediction"] == 0).sum(),
                "n_lepen_predit":      (group["prediction"] == 1).sum(),
            })

    # Ajouter la ligne 2022 (baseline)
    proba_base_all = pipeline.predict_proba(X_base)
    pred_base_all  = pipeline.predict(X_base)
    base_df = id_base.copy()
    base_df["proba_macron"] = proba_base_all[:, 0]
    base_df["proba_lepen"]  = proba_base_all[:, 1]
    base_df["prediction"]   = pred_base_all
    for dept_code, group in base_df.groupby("code_departement"):
        dept_rows.append({
            "code_departement":    dept_code,
            "libelle_departement": group["libelle_departement"].iloc[0],
            "annee_cible":         "2022",
            "horizon_years":       0,
            "n_communes":          len(group),
            "proba_macron_mean":   group["proba_macron"].mean().round(4),
            "proba_lepen_mean":    group["proba_lepen"].mean().round(4),
            "proba_macron_2022":   group["proba_macron"].mean().round(4),
            "n_macron_predit":     (group["prediction"] == 0).sum(),
            "n_lepen_predit":      (group["prediction"] == 1).sum(),
        })

    summary_df = pd.DataFrame(dept_rows).sort_values(["code_departement", "annee_cible"])
    summary_path = ARTIFACTS / "projections_synthese_depts.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Synthèse : {summary_path.name} ({len(summary_df)} lignes)")

    # Métadonnées
    meta_out = {
        "model_used": "gradient_boosting_pre_vote_classification_t2",
        "horizons": HORIZONS,
        "n_communes": int(len(X_base)),
        "trendable_features": list(TRENDABLE.keys()),
        "demo_trends_applied": list(DEMO_TRENDS.keys()),
        "methodology": (
            "Extrapolation linéaire des tendances observées 2009-2022 (chômage) "
            "et 2017-2021 (revenu). Tendances démographiques nationales IDF appliquées. "
            "Features historiques (h12, h17) maintenues constantes."
        ),
        "limitations": (
            "Projection linéaire — ne capture pas les chocs économiques ni "
            "les évolutions politiques. Incertitude croissante avec l'horizon."
        ),
    }
    meta_path_out = ARTIFACTS / "projections_meta.json"
    meta_path_out.write_text(json.dumps(meta_out, indent=2, ensure_ascii=False))
    print(f"  Méta     : {meta_path_out.name}")
    print("\n  Projections terminées.")


if __name__ == "__main__":
    generate()
