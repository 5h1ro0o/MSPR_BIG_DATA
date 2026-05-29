"""
Préprocessing ML — chargement, sélection de features, scaling, split.
Utilisé par tous les modèles.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ml.config import FEATURE_SETS, RANDOM_STATE, TARGETS, TEST_SIZE

warnings.filterwarnings("ignore")


def load_dataset(path: Path = None) -> pd.DataFrame:
    """
    Charge le dataset gold CSV.
    Utilise DATA_PATH par défaut (ml/config.py), qui pointe sur le fichier
    produit par le dernier ETL.  Si DATA_PATH a été surchargé via --data-path
    dans run_training.py, c'est cette valeur qui est utilisée.
    """
    if path is None:
        from ml.config import DATA_PATH as _DP

        path = _DP

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {path}\n"
            "  Lancez d'abord : python -X utf8 etl_pipeline.py"
        )
    df = pd.read_csv(path, low_memory=False)
    size_mb = path.stat().st_size / 1_048_576
    print(
        f"  Dataset : {path.name}  ({size_mb:.1f} Mo) — "
        f"{len(df):,} communes x {len(df.columns)} colonnes"
    )
    return df


def select_features(
    df: pd.DataFrame,
    feature_set: str = "pre_vote",
    extra_features: Optional[list[str]] = None,
) -> list[str]:
    """
    Retourne la liste des features disponibles dans le dataset.
    Filtre les colonnes absentes (résistant aux variations du dataset).
    """
    candidates = FEATURE_SETS[feature_set].copy()
    if extra_features:
        candidates += extra_features
    seen = set()
    candidates_dedup = []
    for f in candidates:
        if f not in seen:
            seen.add(f)
            candidates_dedup.append(f)
    available = [f for f in candidates_dedup if f in df.columns]
    missing = [f for f in candidates_dedup if f not in df.columns]
    if missing:
        print(
            f"  [WARN] {len(missing)} features absentes du dataset : {missing[:5]}..."
        )
    return available


def build_X_y(
    df: pd.DataFrame,
    feature_set: str = "pre_vote",
    target: str = "classification_t2",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Construit X (features) et y (cible) depuis le dataset.

    Returns:
        (X, y, feature_names)
    """
    features = select_features(df, feature_set)
    target_col = TARGETS[target]

    if target_col not in df.columns:
        raise ValueError(f"Colonne cible absente : {target_col}")

    mask = df[target_col].notna()
    df_clean = df[mask].copy()

    X = df_clean[features].copy()
    y = df_clean[target_col].copy()

    print(f"  X : {X.shape} | y : {y.shape} | cible : {target_col}")
    return X, y, features


def get_preprocessing_pipeline(task: str = "classification") -> Pipeline:
    """
    Retourne un pipeline sklearn de préprocessing :
      - Imputation médiane (robuste aux outliers)
      - StandardScaler (nécessaire pour MLP, utile pour tous)

    Args:
        task: "classification" ou "regression"
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
) -> tuple:
    """
    Split train/test avec stratification optionnelle.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    strat = y if stratify and y.dtype in ["int64", "int32", "object"] else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    print(f"  Train : {len(X_train):,} | Test : {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def encode_labels(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Encode les labels texte en entiers (pour classification multiclasse)."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    print(f"  Classes : {list(le.classes_)}")
    return y_enc, le


def get_commune_info(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne les colonnes identifiantes pour l'affichage des prédictions."""
    id_cols = [
        "code_commune",
        "code_departement",
        "libelle_departement",
        "libelle_commune",
    ]
    return df[[c for c in id_cols if c in df.columns]].copy()
