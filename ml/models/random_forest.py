"""
Random Forest — implémentation complète.

Supporte :
  - Classification binaire  (cible_t2_vainqueur : 0=Macron, 1=Le Pen)
  - Classification multiclasse (cible_t1_premier)
  - Régression              (cible_t2_pct_macron, cible_t2_marge, …)

Pipeline :
  1. Préprocessing   — imputation médiane + StandardScaler
  2. GridSearchCV    — recherche d'hyperparamètres avec CV stratifiée
  3. Évaluation      — métriques complètes + matrice de confusion + feature importance
  4. Persistance     — joblib pour le pipeline complet
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import (
    ARTIFACTS,
    CV_FOLDS,
    RANDOM_STATE,
    RF_PARAM_GRID,
    RF_PARAM_GRID_FAST,
)
from ml.models.base import BaseModel


class RandomForestModel(BaseModel):
    """
    Random Forest avec recherche d'hyperparamètres et évaluation complète.
    Utilise un Pipeline sklearn (imputation → scaling → RF) pour garantir
    l'absence de data leakage lors de la CV.
    """

    name = "random_forest"

    def __init__(
        self,
        task: str = "classification",
        artifact_dir: Path = ARTIFACTS,
        n_jobs: int = -1,
    ):
        """
        Args:
            task:         "classification" ou "regression"
            artifact_dir: répertoire de sauvegarde des artefacts
            n_jobs:       nombre de cœurs (-1 = tous)
        """
        super().__init__(artifact_dir)
        self.task = task
        self.n_jobs = n_jobs
        self.pipeline: Optional[Pipeline] = None
        self.best_params_: dict = {}
        self.cv_results_: dict = {}
        self.label_classes_: Optional[np.ndarray] = None

    def build(self, **kwargs) -> Pipeline:
        """Construit le pipeline sklearn complet."""
        if self.task == "classification":
            estimator = RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=self.n_jobs,
                **kwargs,
            )
        else:
            estimator = RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=self.n_jobs,
                **kwargs,
            )

        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("rf", estimator),
            ]
        )
        return pipeline

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        use_grid_search: bool = True,
        fast_search: bool = True,
        n_iter_random: int = 30,
    ) -> dict:
        """
        Entraîne le Random Forest.

        Args:
            X_train:         features d'entraînement
            y_train:         cible d'entraînement
            X_val:           features de validation (optionnel, pour métriques finales)
            y_val:           cible de validation
            use_grid_search: True = GridSearch / RandomizedSearch, False = paramètres par défaut
            fast_search:     True = grille réduite (RF_PARAM_GRID_FAST), False = grille complète
            n_iter_random:   nombre d'itérations pour RandomizedSearchCV

        Returns:
            Dictionnaire de métriques d'entraînement.
        """
        self.feature_names = list(X_train.columns)

        print(f"\n{'='*60}")
        print(f"  Random Forest — {self.task.upper()}")
        print(f"  Features : {len(self.feature_names)} | Train : {len(X_train):,}")
        print(f"{'='*60}")

        t0 = time.time()

        if use_grid_search:
            self.pipeline = self._run_search(
                X_train, y_train, fast_search, n_iter_random
            )
        else:
            self.pipeline = self.build()
            self.pipeline.fit(X_train, y_train)

        elapsed = time.time() - t0
        print(f"  Entraînement terminé en {elapsed:.1f}s")

        y_pred_train = self.pipeline.predict(X_train)
        y_proba_train = (
            self.pipeline.predict_proba(X_train)
            if self.task == "classification"
            else None
        )
        train_metrics = self._compute_metrics(
            y_train, y_pred_train, split="train", y_proba=y_proba_train
        )

        val_metrics = {}
        if X_val is not None and y_val is not None:
            y_pred_val = self.pipeline.predict(X_val)
            y_proba_val = (
                self.pipeline.predict_proba(X_val)
                if self.task == "classification"
                else None
            )
            val_metrics = self._compute_metrics(
                y_val, y_pred_val, split="val", y_proba=y_proba_val
            )

        cv_metrics = self._run_cross_validation(X_train, y_train)

        self.metrics = {**train_metrics, **val_metrics, **cv_metrics}
        self.metrics["training_time_s"] = round(elapsed, 2)
        if self.best_params_:
            self.metrics["best_params"] = self.best_params_

        self.model = self.pipeline
        self.is_trained = True

        self._print_metrics_summary()
        return self.metrics

    def _run_search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        fast: bool,
        n_iter: int,
    ) -> Pipeline:
        """Lance RandomizedSearchCV sur le pipeline."""
        param_grid = RF_PARAM_GRID_FAST if fast else RF_PARAM_GRID

        pipeline_grid = {f"rf__{k}": v for k, v in param_grid.items()}

        if self.task == "regression":
            pipeline_grid.pop("rf__class_weight", None)

        base_pipeline = self.build()

        cv = (
            StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            if self.task == "classification"
            else KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        )
        scoring = "f1_weighted" if self.task == "classification" else "r2"

        print(f"  RandomizedSearchCV : {n_iter} itérations × {CV_FOLDS} folds...")
        search = RandomizedSearchCV(
            base_pipeline,
            param_distributions=pipeline_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=self.n_jobs,
            random_state=RANDOM_STATE,
            verbose=0,
            refit=True,
        )
        search.fit(X_train, y_train)

        self.best_params_ = {
            k.replace("rf__", ""): v for k, v in search.best_params_.items()
        }
        self.cv_results_ = {
            "best_score": round(search.best_score_, 4),
            "best_params": self.best_params_,
        }
        print(f"  Meilleur score CV ({scoring}) : {search.best_score_:.4f}")
        print(f"  Meilleurs params : {self.best_params_}")
        return search.best_estimator_

    def _run_cross_validation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> dict:
        """Cross-validation finale sur le pipeline optimal."""
        cv = (
            StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            if self.task == "classification"
            else KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        )

        if self.task == "classification":
            scoring = {
                "accuracy": "accuracy",
                "f1": "f1_weighted",
                "precision": "precision_weighted",
                "recall": "recall_weighted",
            }
            y_unique = np.unique(y_train)
            if len(y_unique) == 2:
                scoring["roc_auc"] = "roc_auc"
        else:
            scoring = {
                "r2": "r2",
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
            }

        print(f"  Cross-validation finale ({CV_FOLDS} folds)...")
        cv_results = cross_validate(
            self.pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=self.n_jobs,
        )

        result = {}
        for metric, values in cv_results.items():
            if metric.startswith("test_"):
                name = metric.replace("test_", "cv_")
                val = abs(np.mean(values))
                std = abs(np.std(values))
                result[name] = round(val, 4)
                result[f"{name}_std"] = round(std, 4)
                print(f"    {name:25s}: {val:.4f} ± {std:.4f}")

        return result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Retourne les prédictions (labels ou valeurs)."""
        if not self.is_trained:
            raise RuntimeError(
                "Modèle non entraîné. Appelez train() ou load() d'abord."
            )
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Retourne les probabilités par classe (classification uniquement)."""
        if self.task != "classification":
            return None
        return self.pipeline.predict_proba(X)

    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        split: str = "test",
        y_proba: Optional[np.ndarray] = None,
    ) -> dict:
        """Calcule les métriques adaptées à la tâche."""
        result = {}
        prefix = f"{split}_"

        if self.task == "classification":
            result[f"{prefix}accuracy"] = round(accuracy_score(y_true, y_pred), 4)
            result[f"{prefix}f1"] = round(
                f1_score(y_true, y_pred, average="weighted", zero_division=0), 4
            )
            result[f"{prefix}precision"] = round(
                precision_score(y_true, y_pred, average="weighted", zero_division=0), 4
            )
            result[f"{prefix}recall"] = round(
                recall_score(y_true, y_pred, average="weighted", zero_division=0), 4
            )

            if y_proba is not None and len(np.unique(y_true)) == 2:
                try:
                    proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                    result[f"{prefix}roc_auc"] = round(
                        roc_auc_score(y_true, proba_pos), 4
                    )
                except Exception:
                    pass

            result[f"{prefix}confusion_matrix"] = confusion_matrix(
                y_true, y_pred
            ).tolist()
        else:
            result[f"{prefix}r2"] = round(r2_score(y_true, y_pred), 4)
            result[f"{prefix}mae"] = round(mean_absolute_error(y_true, y_pred), 4)
            result[f"{prefix}rmse"] = round(
                np.sqrt(mean_squared_error(y_true, y_pred)), 4
            )

        return result

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        """Évaluation complète sur le jeu de test avec rapport détaillé."""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        metrics = self._compute_metrics(y_test, y_pred, split="test", y_proba=y_proba)
        self.metrics.update(metrics)

        print(f"\n{'='*60}")
        print(f"  Évaluation — {self.name.upper()} ({self.task})")
        print(f"{'='*60}")

        if self.task == "classification":
            print(f"  Accuracy  : {metrics.get('test_accuracy', 'N/A'):.4f}")
            print(f"  F1 (wtd)  : {metrics.get('test_f1', 'N/A'):.4f}")
            print(f"  Precision : {metrics.get('test_precision', 'N/A'):.4f}")
            print(f"  Recall    : {metrics.get('test_recall', 'N/A'):.4f}")
            if "test_roc_auc" in metrics:
                print(f"  ROC-AUC   : {metrics['test_roc_auc']:.4f}")
            print("\n  Classification Report :")
            print(classification_report(y_test, y_pred, zero_division=0))
        else:
            print(f"  R²   : {metrics.get('test_r2', 'N/A'):.4f}")
            print(f"  MAE  : {metrics.get('test_mae', 'N/A'):.4f}")
            print(f"  RMSE : {metrics.get('test_rmse', 'N/A'):.4f}")

        return metrics

    def get_feature_importance(self, top_n: int = 30) -> pd.DataFrame:
        """
        Retourne l'importance des features (Mean Decrease in Impurity).

        Returns:
            DataFrame trié par importance décroissante.
        """
        if not self.is_trained:
            raise RuntimeError("Modèle non entraîné.")

        rf_step = self.pipeline.named_steps["rf"]
        importances = rf_step.feature_importances_
        std = np.std(
            [tree.feature_importances_ for tree in rf_step.estimators_], axis=0
        )

        df_imp = (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": importances,
                    "std": std,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        return df_imp.head(top_n)

    def get_predictions_with_communes(
        self,
        X: pd.DataFrame,
        commune_info: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combine les prédictions avec les informations des communes.

        Returns:
            DataFrame avec code_commune, libelle, prédiction (et proba si classif).
        """
        y_pred = self.predict(X)
        result = commune_info.copy().reset_index(drop=True)
        result["prediction"] = y_pred

        if self.task == "classification":
            y_prob = self.predict_proba(X)
            if y_prob is not None:
                for i, cls in enumerate(self.pipeline.named_steps["rf"].classes_):
                    result[f"proba_{cls}"] = y_prob[:, i].round(3)

        return result

    def save(self, tag: str = "") -> Path:
        """Sauvegarde via joblib (plus efficace que pickle pour sklearn)."""
        suffix = f"_{tag}" if tag else ""
        model_path = self.artifact_dir / f"{self.name}{suffix}.joblib"
        meta_path = self.artifact_dir / f"{self.name}{suffix}_meta.json"

        joblib.dump(self.pipeline, model_path, compress=3)

        meta = {
            "name": self.name,
            "task": self.task,
            "feature_names": self.feature_names,
            "metrics": {
                k: v for k, v in self.metrics.items() if not isinstance(v, list)
            },
            "best_params": self.best_params_,
            "is_trained": self.is_trained,
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        print(
            f"  Sauvegardé : {model_path} ({model_path.stat().st_size / 1024:.0f} KB)"
        )
        return model_path

    def load(self, tag: str = "") -> "RandomForestModel":
        """Charge le modèle joblib + métadonnées."""
        suffix = f"_{tag}" if tag else ""
        model_path = self.artifact_dir / f"{self.name}{suffix}.joblib"
        meta_path = self.artifact_dir / f"{self.name}{suffix}_meta.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé : {model_path}")

        self.pipeline = joblib.load(model_path)
        self.model = self.pipeline

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.feature_names = meta.get("feature_names", [])
            self.metrics = meta.get("metrics", {})
            self.best_params_ = meta.get("best_params", {})
            self.is_trained = True

        print(f"  Chargé : {model_path}")
        return self

    def _print_metrics_summary(self) -> None:
        print("\n  Résumé des métriques :")
        for k, v in self.metrics.items():
            if isinstance(v, (int, float)):
                print(
                    f"    {k:35s}: {v:.4f}"
                    if isinstance(v, float)
                    else f"    {k:35s}: {v}"
                )
