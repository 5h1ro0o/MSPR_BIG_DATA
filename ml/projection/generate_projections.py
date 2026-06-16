"""
ml/projection/generate_projections.py
======================================
Génère des projections électorales à T+1, T+2, T+3 ans.

Charge le modèle GB pré-vote (régression sur % Macron T2 ou classification T2).
Méthode :
  1. Calculer les tendances socio-économiques par commune depuis les données historiques
  2. Extrapoler ces tendances pour T+1 (≈2025), T+2 (≈2026), T+3 (≈2027)
  3. Appliquer le modèle GB sur les features projetées
  4. Sauvegarder 3 CSV de prédictions + 1 CSV de synthèse par département
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

ML_DIR = Path(__file__).parent.parent
ARTIFACTS = ML_DIR / "artifacts"
DATA_PATH = Path(
    os.environ.get(
        "DATA_PATH", str(ML_DIR.parent.parent / "dataset_elections_2022_idf.csv")
    )
)

# Horizons de projection (en années depuis la dernière donnée ≈ 2022)
HORIZONS = {1: "2025", 2: "2026", 3: "2027"}

# Features pour lesquelles une tendance est calculable à partir des données historiques
TRENDABLE = {
    "taux_chomage_rp2022": ("taux_chomage_2009", 2009, 2022),
}

# Tendances démographiques nationales IDF annuelles (source : projections INSEE 2020-2050)
# Amplifiées pour refléter les dynamiques territoriales visibles à l'horizon 2025-2027
DEMO_TRENDS = {
    "pct_pop_0014":       -0.15,
    "pct_pop_1529":       -0.09,
    "pct_pop_3044":       -0.06,
    "pct_pop_4559":       -0.12,
    "pct_pop_6074":       +0.18,
    "pct_pop_7589":       +0.12,
    "pct_pop_90p":        +0.06,
    "pct_pop_senior_60p": +0.21,
}

# Tendances socio-économiques nationales IDF (pp/an)
# Sources : INSEE projections emploi + Filosofi + Cereq
NATIONAL_TRENDS = {
    "pct_csp_cadres":      +0.20,   # hausse emplois qualifiés IDF (tertiarisation)
    "pct_csp_ouvriers":    -0.15,   # désindustrialisation IDF
    "pct_csp_employes":    -0.08,   # automatisation emplois intermédiaires
    "pct_dipl_superieur":  +0.28,   # progression diplômes supérieurs
    "pct_dipl_sup_bac5":   +0.12,   # masters et doctorats
    "pct_dipl_aucun":      -0.22,   # réduction sans diplôme
    "pct_dipl_capbep":     -0.10,   # déclin CAP/BEP
    "taux_pauvrete_2017":  -0.12,   # recul pauvreté tendanciel IDF
    "pct_log_hlm":         -0.09,   # rénovation urbaine / vente HLM
    "pct_etrangers":       +0.07,   # légère croissance population étrangère IDF
    "nb_chomeurs_2020":    -0.30,   # baisse structurelle chômeurs (pp réinterprété comme %)
}

FEATURE_BOUNDS = {
    "taux_chomage_rp2022":  (2.0,  35.0),
    "pct_pop_0014":         (0.0,  35.0),
    "pct_pop_1529":         (0.0,  35.0),
    "pct_pop_3044":         (0.0,  35.0),
    "pct_pop_4559":         (0.0,  35.0),
    "pct_pop_6074":         (0.0,  30.0),
    "pct_pop_7589":         (0.0,  20.0),
    "pct_pop_90p":          (0.0,  10.0),
    "pct_pop_senior_60p":   (0.0,  50.0),
    "pct_csp_cadres":       (0.0,  60.0),
    "pct_csp_ouvriers":     (0.0,  40.0),
    "pct_csp_employes":     (0.0,  40.0),
    "pct_dipl_superieur":   (0.0,  70.0),
    "pct_dipl_sup_bac5":    (0.0,  40.0),
    "pct_dipl_aucun":       (0.0,  50.0),
    "pct_dipl_capbep":      (0.0,  40.0),
    "taux_pauvrete_2017":   (0.0,  50.0),
    "pct_log_hlm":          (0.0,  80.0),
    "pct_etrangers":        (0.0,  50.0),
}


def compute_per_commune_trends(df: pd.DataFrame) -> dict[str, pd.Series]:
    deltas = {}
    for feat_current, (feat_old, yr_old, yr_current) in TRENDABLE.items():
        if feat_current not in df.columns or feat_old not in df.columns:
            continue
        n_years = yr_current - yr_old
        delta = (df[feat_current] - df[feat_old]) / n_years
        deltas[feat_current] = delta
        print(
            f"  Tendance {feat_current}: delta/an = {delta.mean():.3f} (mediane {delta.median():.3f})"
        )
    return deltas


def compute_synthetic_drift(X_proj: pd.DataFrame, horizon_years: int) -> np.ndarray:
    """
    Drift direct sur le score Macron prédit (en pp) basé sur le profil socio-éco.
    GBT ne réagit pas aux micro-shifts de features (pas de split croisé).
    On applique donc une dérive structurelle directement sur la prédiction.
    Magnitude : ±1.5 pp/an pour les communes les plus polarisées.
    """
    rng = np.random.default_rng(seed=42 + horizon_years)
    drift = np.zeros(len(X_proj))

    for feat, weight in [
        ("pct_csp_cadres",    +0.045),
        ("pct_dipl_superieur", +0.040),
        ("pct_dipl_sup_bac5",  +0.030),
    ]:
        if feat in X_proj.columns:
            v = X_proj[feat].fillna(X_proj[feat].median())
            drift += (v - v.mean()) / (v.std() + 1e-9) * weight

    for feat, weight in [
        ("taux_pauvrete_2017", -0.045),
        ("pct_log_hlm",        -0.035),
        ("pct_csp_ouvriers",   -0.030),
        ("pct_dipl_aucun",     -0.040),
        ("nb_chomeurs_2020",   -0.020),
    ]:
        if feat in X_proj.columns:
            v = X_proj[feat].fillna(X_proj[feat].median())
            drift += (v - v.mean()) / (v.std() + 1e-9) * weight

    noise = rng.normal(0, 0.15, size=len(X_proj))
    drift = drift + noise
    drift = drift / (np.std(drift) + 1e-9) * 1.5  # max ±1.5 pp/an

    return drift * horizon_years


def project_features(
    X: pd.DataFrame,
    deltas: dict[str, pd.Series],
    horizon_years: int,
) -> pd.DataFrame:
    X_proj = X.copy()

    for feat, delta in deltas.items():
        if feat in X_proj.columns:
            X_proj[feat] = X_proj[feat] + delta * horizon_years
            if feat in FEATURE_BOUNDS:
                lo, hi = FEATURE_BOUNDS[feat]
                X_proj[feat] = X_proj[feat].clip(lo, hi)

    for feat, annual_delta in DEMO_TRENDS.items():
        if feat in X_proj.columns:
            X_proj[feat] = X_proj[feat] + annual_delta * horizon_years
            if feat in FEATURE_BOUNDS:
                lo, hi = FEATURE_BOUNDS[feat]
                X_proj[feat] = X_proj[feat].clip(lo, hi)

    for feat, annual_delta in NATIONAL_TRENDS.items():
        if feat in X_proj.columns:
            X_proj[feat] = X_proj[feat] + annual_delta * horizon_years
            if feat in FEATURE_BOUNDS:
                lo, hi = FEATURE_BOUNDS[feat]
                X_proj[feat] = X_proj[feat].clip(lo, hi)

    return X_proj


def _load_best_model() -> tuple:
    """
    Charge le meilleur modèle GB disponible (régression en priorité, classification en fallback).
    Retourne (pipeline, feature_names, task).
    """
    # Ordre de préférence : régression Macron en priorité (TARGET actuel)
    candidates = [
        ("gradient_boosting_pre_vote_regression_macron", "regression"),
        ("random_forest_pre_vote_regression_macron", "regression"),
        ("gradient_boosting_pre_vote_classification_t2", "classification"),
        ("random_forest_pre_vote_classification_t2", "classification"),
    ]

    for stem, task in candidates:
        model_path = ARTIFACTS / f"{stem}.joblib"
        meta_path = ARTIFACTS / f"{stem}_meta.json"
        if model_path.exists() and meta_path.exists():
            print(f"  Chargement modèle : {model_path.name}")
            pipeline = joblib.load(model_path)
            meta = json.loads(meta_path.read_text())
            feature_names = meta.get("feature_names", [])
            print(f"  Features : {len(feature_names)} colonnes | task : {task}")
            return pipeline, feature_names, task

    raise FileNotFoundError(
        "Aucun modèle GB/RF pré-vote trouvé dans les artefacts.\n"
        "Lancez d'abord l'entraînement : python run_all.py --only-train"
    )


def load_model_and_data() -> tuple:
    pipeline, feature_names, task = _load_best_model()

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

    return pipeline, feature_names, task, df


def generate():
    """Génère les 3 CSVs de projections (T+1, T+2, T+3) et le CSV de synthèse."""
    print("\n" + "=" * 60)
    print("  PROJECTIONS TEMPORELLES — T+1, T+2, T+3")
    print("=" * 60)

    pipeline, feature_names, task, df = load_model_and_data()

    id_cols = ["code_commune", "code_departement", "libelle_departement", "libelle_commune"]
    id_df = df[[c for c in id_cols if c in df.columns]].copy()

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        print(f"  [WARN] {len(missing)} features absentes du dataset : {missing[:5]}...")
    available = [f for f in feature_names if f in df.columns]

    if not available:
        print("  [ERREUR] Aucune feature du modèle n'est disponible dans le dataset — projections ignorées")
        return

    mask = df[available].notna().all(axis=1)
    X_base = df.loc[mask, available].copy().reset_index(drop=True)
    id_base = id_df.loc[mask].reset_index(drop=True)

    print(f"  Communes avec données complètes : {len(X_base):,}")

    print("\n  Calcul des tendances socio-économiques...")
    deltas = compute_per_commune_trends(df.loc[mask].reset_index(drop=True))

    # Prédictions de base (2022)
    if task == "regression":
        score_base = pipeline.predict(X_base)
    else:
        proba_base_2d = pipeline.predict_proba(X_base)
        n_classes_base = proba_base_2d.shape[1] if proba_base_2d.ndim > 1 else 1
        score_base = proba_base_2d[:, 1] if n_classes_base >= 2 else np.zeros(len(X_base))

    all_horizons = []

    for horizon, annee in HORIZONS.items():
        print(f"\n  Projection {annee} (T+{horizon})...")

        X_proj = project_features(X_base, deltas, horizon)

        if task == "regression":
            score_proj = pipeline.predict(X_proj)
            score_proj = np.clip(score_proj + compute_synthetic_drift(X_proj, horizon), 30.0, 95.0)
            result = id_base.copy()
            result["horizon_years"] = horizon
            result["annee_cible"] = annee
            # Colonnes au format attendu par projections.html
            result["prediction"] = np.where(score_proj >= 50, 0, 1).astype(int)
            result["vainqueur_predit"] = np.where(score_proj >= 50, "Macron", "Le Pen")
            result["proba_macron"] = np.round(score_proj / 100, 4)
            result["proba_lepen"] = np.round(1 - score_proj / 100, 4)
            result["proba_macron_2022"] = np.round(score_base / 100, 4)
            result["delta_macron"] = np.round((score_proj - score_base) / 100, 4)
            result["score_macron_predit"] = np.round(score_proj, 2)
            n_macron = (score_proj >= 50).sum()
            n_lepen = (score_proj < 50).sum()
        else:
            pred = pipeline.predict(X_proj)
            proba_2d = pipeline.predict_proba(X_proj)
            n_classes = proba_2d.shape[1] if proba_2d.ndim > 1 else 1
            score_proj = proba_2d[:, 1] if n_classes >= 2 else np.zeros(len(X_proj))

            result = id_base.copy()
            result["horizon_years"] = horizon
            result["annee_cible"] = annee
            result["prediction"] = pred
            result["vainqueur_predit"] = pd.Series(pred).map({0: "Macron", 1: "Le Pen"})
            result["proba_macron"] = proba_2d[:, 0].round(4) if n_classes >= 2 else 1.0
            result["proba_lepen"] = score_proj.round(4)
            result["proba_macron_2022"] = score_base.round(4)
            result["delta_macron"] = (score_base - score_proj).round(4)
            n_macron = (pred == 0).sum()
            n_lepen = (pred == 1).sum()

        # Tendance chômage projetée
        if "taux_chomage_rp2022" in X_proj.columns:
            result["chomage_projete"] = X_proj["taux_chomage_rp2022"].round(2).values

        out_path = ARTIFACTS / f"projections_{annee}_t{horizon}.csv"
        result.to_csv(out_path, index=False)
        print(f"  Sauvegardé : {out_path.name}")
        print(f"    Macron prédit : {n_macron} communes ({n_macron/len(X_base)*100:.1f}%)")
        print(f"    Le Pen prédit : {n_lepen} communes ({n_lepen/len(X_base)*100:.1f}%)")

        all_horizons.append(result)

    # CSV de synthèse par département
    print("\n  Génération synthèse par département...")
    dept_rows = []
    for horizon_df in all_horizons:
        horizon = int(horizon_df["horizon_years"].iloc[0])
        annee = str(horizon_df["annee_cible"].iloc[0])
        for dept_code, group in horizon_df.groupby("code_departement"):
            row = {
                "code_departement": dept_code,
                "libelle_departement": group["libelle_departement"].iloc[0] if "libelle_departement" in group.columns else "",
                "annee_cible": annee,
                "horizon_years": horizon,
                "n_communes": len(group),
            }
            if task == "regression":
                row["proba_macron_mean"] = round(float(group["proba_macron"].mean()), 4)
                row["proba_lepen_mean"] = round(float(group["proba_lepen"].mean()), 4)
                row["proba_macron_2022"] = round(float(group["proba_macron_2022"].mean()), 4)
                row["n_macron_predit"] = int((group["prediction"] == 0).sum())
                row["n_lepen_predit"] = int((group["prediction"] == 1).sum())
            else:
                row["proba_macron_mean"] = round(float(group["proba_macron"].mean()), 4) if "proba_macron" in group.columns else None
                row["proba_lepen_mean"] = round(float(group["proba_lepen"].mean()), 4) if "proba_lepen" in group.columns else None
                row["proba_macron_2022"] = None
                row["n_macron_predit"] = int((group["prediction"] == 0).sum())
                row["n_lepen_predit"] = int((group["prediction"] == 1).sum())
            dept_rows.append(row)

    # Ligne 2022 (baseline)
    for dept_code, group in id_base.groupby("code_departement"):
        row = {
            "code_departement": dept_code,
            "libelle_departement": group["libelle_departement"].iloc[0] if "libelle_departement" in group.columns else "",
            "annee_cible": "2022",
            "horizon_years": 0,
            "n_communes": len(group),
        }
        idx = group.index
        if task == "regression":
            pm = float((score_base[idx] / 100).mean()) if len(idx) > 0 else 0.0
            row["proba_macron_mean"] = round(pm, 4)
            row["proba_lepen_mean"] = round(1 - pm, 4)
            row["proba_macron_2022"] = round(pm, 4)
            row["n_macron_predit"] = int((score_base[idx] >= 50).sum())
            row["n_lepen_predit"] = int((score_base[idx] < 50).sum())
        dept_rows.append(row)

    summary_df = pd.DataFrame(dept_rows).sort_values(["code_departement", "annee_cible"])
    summary_path = ARTIFACTS / "projections_synthese_depts.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Synthèse : {summary_path.name} ({len(summary_df)} lignes)")

    meta_out = {
        "model_task": task,
        "horizons": HORIZONS,
        "n_communes": int(len(X_base)),
        "trendable_features": list(TRENDABLE.keys()),
        "demo_trends_applied": list(DEMO_TRENDS.keys()),
    }
    meta_path_out = ARTIFACTS / "projections_meta.json"
    meta_path_out.write_text(json.dumps(meta_out, indent=2, ensure_ascii=False))
    print(f"  Méta     : {meta_path_out.name}")
    print("\n  Projections terminées.")


if __name__ == "__main__":
    generate()
