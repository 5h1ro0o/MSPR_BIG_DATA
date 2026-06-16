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
from ml.training.evaluate import (
    baseline_metrics,
    plot_calibration,
    robust_classification_eval,
)


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
        """Construit le pipeline sklearn complet. OOB activé pour estimation gratuite."""
        if self.task == "classification":
            estimator = RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=self.n_jobs,
                oob_score=True,  # estimation non biaisée de la généralisation, gratuite
                **kwargs,
            )
        else:
            estimator = RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=self.n_jobs,
                oob_score=True,
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
        X_val: Optional[pd.DataFrame] = None,  # ignoré — RF n'a pas d'early stopping
        y_val: Optional[pd.Series] = None,
        use_grid_search: bool = True,
        fast_search: bool = True,
        n_iter_random: int = 30,
    ) -> dict:
        """
        Entraîne le Random Forest sur X_train uniquement.

        X_val/y_val ne sont PAS utilisés ici — réservés pour evaluate().
        La CV interne à RandomizedSearchCV remplace la double CV.
        """
        self.feature_names = list(X_train.columns)
        y_arr = np.asarray(y_train)

        print(f"\n{'='*60}")
        print(f"  Random Forest — {self.task.upper()}")
        print(f"  Features : {len(self.feature_names)} | Train : {len(X_train):,}")
        print(f"{'='*60}")

        t0 = time.time()

        if use_grid_search:
            # RandomizedSearchCV fait déjà la CV → pas besoin d'une 2ème CV après
            self.pipeline = self._run_search(
                X_train, y_train, fast_search, n_iter_random
            )
        else:
            self.pipeline = self.build()
            self.pipeline.fit(X_train, y_train)

        elapsed = time.time() - t0
        print(f"  Entraînement terminé en {elapsed:.1f}s")

        # Score OOB (gratuit avec RF, estimation non biaisée de la généralisation)
        rf_step = self.pipeline.named_steps["rf"]
        oob_score = getattr(rf_step, "oob_score_", None)
        if oob_score is not None:
            print(f"  OOB score (estimation généralisation) : {oob_score:.4f}")

        # Métriques train — diagnostique d'overfitting uniquement
        y_pred_tr = self.pipeline.predict(X_train)
        train_acc = round(
            float(
                accuracy_score(y_arr, y_pred_tr)
                if self.task == "classification"
                else r2_score(y_arr, y_pred_tr)
            ),
            4,
        )

        self.metrics = {
            "n_train": len(X_train),
            "train_accuracy": train_acc,
            "training_time_s": round(elapsed, 2),
        }
        if oob_score is not None:
            self.metrics["oob_score"] = round(float(oob_score), 4)
        if self.best_params_:
            self.metrics["best_params"] = self.best_params_
        if self.cv_results_:
            self.metrics["cv_r2"] = self.cv_results_.get("best_score")
            self.metrics["cv_r2_std"] = self.cv_results_.get("best_score_std")

        self.model = self.pipeline
        self.is_trained = True
        print(f"  train_accuracy = {train_acc:.4f}")
        if oob_score:
            print(f"  OOB (généralisation estimée) = {oob_score:.4f}")
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

        # n_jobs limité à 4 pour éviter OOM Docker avec RF × 15 iter × 5 folds
        safe_jobs = min(self.n_jobs if self.n_jobs > 0 else 4, 4)
        print(f"  RandomizedSearchCV : {n_iter} itérations × {CV_FOLDS} folds (n_jobs={safe_jobs})...")
        search = RandomizedSearchCV(
            base_pipeline,
            param_distributions=pipeline_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=safe_jobs,
            random_state=RANDOM_STATE,
            verbose=0,
            refit=True,
        )
        search.fit(X_train, y_train)

        self.best_params_ = {
            k.replace("rf__", ""): v for k, v in search.best_params_.items()
        }
        best_idx = search.best_index_
        self.cv_results_ = {
            "best_score": round(search.best_score_, 4),
            "best_score_std": round(float(search.cv_results_["std_test_score"][best_idx]), 4),
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

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Évaluation finale sur le set de test (jamais vu pendant l'entraînement).
        Utilise robust_classification_eval avec IC, baseline et calibration.
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        y_true = np.asarray(y_test)

        if self.task == "classification":
            metrics = robust_classification_eval(y_true, y_pred, y_proba, split="test")
            bl = baseline_metrics(y_true)
            metrics.update(bl)
            metrics["improvement_over_baseline"] = round(
                metrics["test_accuracy"] - bl["baseline_accuracy"], 4
            )
            try:
                plot_calibration(y_true, y_proba, self.name, self.artifact_dir)
            except Exception:
                pass
            print(f"\n{'─'*55}")
            print(f"  SET DE TEST — {metrics['test_n_samples']} communes")
            print(
                f"  Accuracy          : {metrics['test_accuracy']:.4f}  IC95% [{metrics['test_accuracy_ci_lo']:.4f}, {metrics['test_accuracy_ci_hi']:.4f}]"
            )
            print(f"  Balanced accuracy : {metrics['test_balanced_accuracy']:.4f}")
            print(
                f"  ROC-AUC           : {metrics.get('test_roc_auc','N/A')}  IC95% [{metrics.get('test_roc_auc_ci_lo','?')}, {metrics.get('test_roc_auc_ci_hi','?')}]"
            )
            print(f"  Cohen's κ         : {metrics['test_cohen_kappa']:.4f}")
            print(f"  Brier score       : {metrics.get('test_brier_score','N/A')}")
            print(f"  Baseline accuracy : {bl['baseline_accuracy']:.4f}")
            print(f"  Gain vs baseline  : +{metrics['improvement_over_baseline']:.4f}")
            print(f"{'─'*55}")
        else:
            metrics = {
                "train_r2": round(float(r2_score(y_true, y_pred)), 4),
                "train_mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
                "train_rmse": round(
                    float(np.sqrt(mean_squared_error(y_true, y_pred))), 4
                ),
            }
            print(
                f"  R²={metrics['train_r2']:.4f}  MAE={metrics['train_mae']:.4f}  RMSE={metrics['train_rmse']:.4f}"
            )

        self.metrics.update(metrics)
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
        raw_pred = self.predict(X)
        result = commune_info.copy().reset_index(drop=True)

        if self.task == "regression":
            result["prediction"] = np.where(raw_pred >= 50, 0, 1)
            result["vainqueur_predit"] = np.where(raw_pred >= 50, "Macron", "Le Pen")
            result["proba_macron"] = np.round(raw_pred / 100, 4)
            result["proba_lepen"] = np.round(1 - raw_pred / 100, 4)
            result["probability"] = result["proba_lepen"]
            result["score_macron_predit"] = np.round(raw_pred, 2)
        else:
            result["prediction"] = raw_pred
            y_prob = self.predict_proba(X)
            if y_prob is not None:
                for i, cls in enumerate(self.pipeline.named_steps["rf"].classes_):
                    result[f"proba_{cls}"] = y_prob[:, i].round(3)
                result["vainqueur_predit"] = pd.Series(raw_pred).map({0: "Macron", 1: "Le Pen"})
                if y_prob.shape[1] >= 2:
                    result["proba_macron"] = y_prob[:, 0].round(4)
                    result["proba_lepen"] = y_prob[:, 1].round(4)
                    result["probability"] = y_prob[:, 1].round(4)

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
