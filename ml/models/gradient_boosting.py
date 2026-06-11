"""
Gradient Boosting — implémentation complète.
Adapté du script de référence : a implementer/gradientboosting/gradient_boosting_elections.py

Spécificités :
  - Anti-leakage : exclut toutes les colonnes T2 de participation
  - sklearn GradientBoostingClassifier (sans dépendance externe)
  - RandomizedSearchCV + cross-validation 5-fold
  - Export : predictions CSV + feature importance CSV + graphique 4-panneaux
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import (
    ARTIFACTS,
    CV_FOLDS,
    GB_BEST_PARAMS,
    GB_PARAM_GRID,
    LEAKAGE_KEYWORDS,
    RANDOM_STATE,
)
from ml.models.base import BaseModel
from ml.training.evaluate import (
    baseline_metrics,
    plot_calibration,
    robust_classification_eval,
)


class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting pour la prédiction électorale IDF.

    Pipeline : SimpleImputer → StandardScaler → GradientBoosting
    Anti-leakage : colonnes T2 exclues automatiquement.
    """

    name = "gradient_boosting"

    def __init__(
        self,
        task: str = "classification",
        artifact_dir: Path = ARTIFACTS,
    ):
        super().__init__(artifact_dir)
        self.task = task
        self.pipeline: Optional[Pipeline] = None
        self.best_params_: dict = {}
        self.cv_scores_: np.ndarray = np.array([])

    def build(self, **kwargs) -> Pipeline:
        """Construit le pipeline sklearn avec les paramètres fournis."""
        params = {**GB_BEST_PARAMS, **kwargs}
        if self.task == "regression":
            params.pop("class_weight", None)

        if self.task == "classification":
            estimator = GradientBoostingClassifier(**params)
        else:
            estimator = GradientBoostingRegressor(**params)

        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("gb", estimator),
            ]
        )

    @staticmethod
    def filter_leakage(feature_names: list[str]) -> list[str]:
        """Supprime les colonnes qui fuiteraient le résultat T2."""
        return [f for f in feature_names if not any(kw in f for kw in LEAKAGE_KEYWORDS)]

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,  # ignoré — early stopping interne au GB
        y_val: Optional[pd.Series] = None,
        use_search: bool = False,
        n_iter: int = 20,
    ) -> dict:
        """
        Entraîne le Gradient Boosting sur X_train uniquement.

        L'early stopping est géré en interne via validation_fraction dans GB_BEST_PARAMS.
        X_val/y_val ne sont PAS utilisés ici — ils doivent être réservés
        pour l'évaluation finale (appel séparé à evaluate()).
        """
        clean_features = self.filter_leakage(list(X_train.columns))
        X_tr = X_train[clean_features].copy()
        self.feature_names = clean_features
        y_arr = np.asarray(y_train)

        print(f"\n{'='*60}")
        print(f"  Gradient Boosting — {self.task.upper()}")
        print(f"  Features : {len(clean_features)} (après anti-leakage)")
        print(
            f"  Train : {len(X_tr):,} communes  (early stopping interne sur {int(len(X_tr)*GB_BEST_PARAMS.get('validation_fraction',0.1)*100):.0f}%)"
        )
        print(f"{'='*60}")

        t0 = time.time()

        if use_search:
            self.pipeline = self._run_search(X_tr, y_train, n_iter)
        else:
            self.pipeline = self.build()
            self.pipeline.fit(X_tr, y_train)
            self.best_params_ = GB_BEST_PARAMS.copy()

        elapsed = round(time.time() - t0, 2)

        # Early stopping: afficher le nombre d'itérations réellement utilisées
        gb_step = self.pipeline.named_steps["gb"]
        n_estimators_used = getattr(gb_step, "n_estimators_", None) or getattr(
            gb_step, "n_estimators", "?"
        )
        print(
            f"  Entraînement terminé en {elapsed}s — {n_estimators_used} arbres utilisés"
        )

        # Cross-validation sur l'ensemble d'entraînement (estimation robuste de la généralisation)
        scoring = "roc_auc" if self.task == "classification" else "r2"
        cv = (
            StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            if self.task == "classification"
            else KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        )
        print(f"  Cross-validation {CV_FOLDS}-fold sur train ({scoring})...")
        self.cv_scores_ = cross_val_score(
            self.build(),  # pipeline vierge — pas le modèle déjà fitté
            X_tr,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        print(
            f"  CV {scoring} : {self.cv_scores_.mean():.4f} ± {self.cv_scores_.std():.4f}"
        )

        # Métriques train (diagnostique d'overfitting, non utilisées comme référence)
        y_pred_tr = self.pipeline.predict(X_tr)
        train_acc = round(float(accuracy_score(y_arr, y_pred_tr)), 4)

        self.metrics = {
            "n_train": len(X_tr),
            "train_accuracy": train_acc,  # indicateur d'overfitting, pas une performance cible
            "cv_roc_auc": round(float(self.cv_scores_.mean()), 4),
            "cv_roc_auc_std": round(float(self.cv_scores_.std()), 4),
            "training_time_s": elapsed,
            "n_features": len(clean_features),
            "n_estimators_used": n_estimators_used,
            "best_params": self.best_params_,
        }

        self.model = self.pipeline
        self.is_trained = True
        print(
            f"  train_accuracy = {train_acc:.4f} (un score de 1.0 signifie sur-apprentissage)"
        )
        print(
            f"  Référence robuste : CV AUC = {self.metrics['cv_roc_auc']:.4f} ± {self.metrics['cv_roc_auc_std']:.4f}"
        )
        return self.metrics

    def _run_search(self, X_tr, y_train, n_iter: int) -> Pipeline:
        """RandomizedSearchCV sur GB_PARAM_GRID."""
        base = self.build()
        grid = {f"gb__{k}": v for k, v in GB_PARAM_GRID.items()}
        cv = (
            StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            if self.task == "classification"
            else KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        )
        scoring = "roc_auc" if self.task == "classification" else "r2"

        print(f"  RandomizedSearchCV : {n_iter} itérations...")
        search = RandomizedSearchCV(
            base,
            grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
            verbose=0,
        )
        search.fit(X_tr, y_train)
        self.best_params_ = {
            k.replace("gb__", ""): v for k, v in search.best_params_.items()
        }
        print(f"  Meilleur score CV : {search.best_score_:.4f}")
        print(f"  Meilleurs params  : {self.best_params_}")
        return search.best_estimator_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Modèle non entraîné.")
        return self.pipeline.predict(X[self.feature_names])

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if self.task != "classification":
            return None
        return self.pipeline.predict_proba(X[self.feature_names])

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Évaluation finale sur le set de test (jamais touché pendant l'entraînement).
        Inclut : IC Wilson sur accuracy, IC bootstrap sur AUC, baseline, calibration.
        """
        X_clean = X_test[self.feature_names]
        y_pred = self.pipeline.predict(X_clean)
        y_true = np.asarray(y_test)

        if self.task == "classification":
            y_proba_2d = self.pipeline.predict_proba(X_clean)
            metrics = robust_classification_eval(
                y_true, y_pred, y_proba_2d, split="test"
            )

            # Baseline (classifieur naïf)
            bl = baseline_metrics(y_true)
            metrics.update(bl)
            metrics["improvement_over_baseline"] = round(
                metrics["test_accuracy"] - bl["baseline_accuracy"], 4
            )

            # Calibration — sauvegarde du graphique
            try:
                plot_calibration(y_true, y_proba_2d, self.name, self.artifact_dir)
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
            reg = {
                "test_r2": round(float(r2_score(y_true, y_pred)), 4),
                "test_mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
                "test_rmse": round(
                    float(np.sqrt(mean_squared_error(y_true, y_pred))), 4
                ),
            }
            metrics = reg
            print(
                f"  R²={metrics['test_r2']:.4f}  MAE={metrics['test_mae']:.4f}  RMSE={metrics['test_rmse']:.4f}"
            )

        self.metrics.update(metrics)
        return metrics

    def _compute_metrics(self, y_true, y_pred, split: str) -> dict:
        p = f"{split}_"
        if self.task == "classification":
            return {
                f"{p}accuracy": round(accuracy_score(y_true, y_pred), 4),
                f"{p}f1": round(
                    f1_score(y_true, y_pred, average="weighted", zero_division=0), 4
                ),
                f"{p}precision": round(
                    precision_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    ),
                    4,
                ),
                f"{p}recall": round(
                    recall_score(y_true, y_pred, average="weighted", zero_division=0), 4
                ),
                f"{p}confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            }
        else:
            return {
                f"{p}r2": round(r2_score(y_true, y_pred), 4),
                f"{p}mae": round(mean_absolute_error(y_true, y_pred), 4),
                f"{p}rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
            }

    def get_feature_importance(self, top_n: int = 30) -> pd.DataFrame:
        if not self.is_trained:
            raise RuntimeError("Modèle non entraîné.")
        gb_step = self.pipeline.named_steps["gb"]
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": gb_step.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
            .head(top_n)
        )

    def plot_results(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_all: pd.DataFrame,
        y_all: pd.Series,
    ) -> Path:
        """
        Graphique 4-panneaux identique au script de référence :
          (a) Feature Importance Top-20
          (b) Matrice de confusion
          (c) Courbe ROC
          (d) Distribution des probabilités
        """
        X_clean_test = X_test[self.feature_names]
        X_clean_all = X_all[self.feature_names]

        y_pred = self.pipeline.predict(X_clean_test)
        y_proba_all = self.pipeline.predict_proba(X_clean_all)[:, 1]
        y_proba = self.pipeline.predict_proba(X_clean_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        df_imp = self.get_feature_importance(top_n=20)

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle(
            "Gradient Boosting — Élections Présidentielles 2022 IDF",
            fontsize=14,
            fontweight="bold",
        )

        ax = axes[0, 0]
        colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(df_imp)))
        df_imp.sort_values("importance").plot(
            kind="barh",
            x="feature",
            y="importance",
            ax=ax,
            color=colors,
            legend=False,
        )
        ax.set_title("Top 20 Features les plus importantes")
        ax.set_xlabel("Importance")
        ax.tick_params(axis="y", labelsize=8)

        ax = axes[0, 1]
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Macron", "Le Pen"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Matrice de confusion (Test)\nAccuracy : {accuracy*100:.1f}%")

        ax = axes[1, 0]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.fill_between(fpr, tpr, alpha=0.1, color="steelblue")
        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Taux de vrais positifs")
        ax.set_title("Courbe ROC")
        ax.legend()

        ax = axes[1, 1]
        ax.hist(
            y_proba_all[y_all == 0],
            bins=30,
            alpha=0.6,
            color="steelblue",
            label="Macron (réel)",
            density=True,
        )
        ax.hist(
            y_proba_all[y_all == 1],
            bins=30,
            alpha=0.6,
            color="crimson",
            label="Le Pen (réel)",
            density=True,
        )
        ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Seuil 0.5")
        ax.set_xlabel("Probabilité prédite Le Pen")
        ax.set_ylabel("Densité")
        ax.set_title("Distribution des probabilités")
        ax.legend()

        plt.tight_layout()
        path = self.artifact_dir / "gradient_boosting_results.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Graphique sauvegardé : {path}")
        return path

    def save(self, tag: str = "") -> Path:
        suffix = f"_{tag}" if tag else ""
        model_path = self.artifact_dir / f"{self.name}{suffix}.joblib"
        meta_path = self.artifact_dir / f"{self.name}{suffix}_meta.json"
        joblib.dump(self.pipeline, model_path, compress=3)
        meta = {
            "name": self.name,
            "task": self.task,
            "feature_names": self.feature_names,
            "metrics": {
                k: v
                for k, v in self.metrics.items()
                if not isinstance(v, (list, np.ndarray))
            },
            "best_params": self.best_params_,
            "cv_scores": self.cv_scores_.tolist() if len(self.cv_scores_) else [],
            "is_trained": self.is_trained,
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        print(f"  Sauvegardé : {model_path} ({model_path.stat().st_size/1024:.0f} KB)")
        return model_path

    def load(self, tag: str = "") -> "GradientBoostingModel":
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
            self.cv_scores_ = np.array(meta.get("cv_scores", []))
            self.is_trained = True
        return self

    def get_predictions_with_communes(
        self,
        X: pd.DataFrame,
        commune_info: pd.DataFrame,
    ) -> pd.DataFrame:
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        result = commune_info.reset_index(drop=True).copy()
        result["prediction"] = y_pred
        result["vainqueur_predit"] = pd.Series(y_pred).map({0: "Macron", 1: "Le Pen"})
        if y_proba is not None:
            result["proba_macron"] = y_proba[:, 0].round(4)
            result["proba_lepen"] = y_proba[:, 1].round(4)
        return result

    def _print_summary(self):
        print("\n  Métriques finales :")
        for k, v in self.metrics.items():
            if isinstance(v, float):
                print(f"    {k:35s}: {v:.4f}")
            elif isinstance(v, (int, str)) and not isinstance(v, bool):
                print(f"    {k:35s}: {v}")
