"""
MLP (Multi-Layer Perceptron) — sklearn MLPClassifier.
Réseau de neurones fully-connected pour la classification électorale T2.

Architecture : Input → 128 → Dropout(ReLU) → 64 → 32 → Output (sigmoid)
Normalisation StandardScaler obligatoire (MLP sensible aux échelles).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import ARTIFACTS, RANDOM_STATE
from ml.models.base import BaseModel


class MLPModel(BaseModel):
    """
    Réseau de neurones MLP — sklearn MLPClassifier.
    StandardScaler intégré dans le pipeline (obligatoire pour MLP).
    Feature importance via permutation importance (sklearn.inspection).
    """

    name = "mlp"

    def __init__(self, task: str = "classification", artifact_dir: Path = ARTIFACTS):
        super().__init__(artifact_dir)
        self.task = task

    def build(self, **kwargs) -> Pipeline:
        common = dict(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=600,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=RANDOM_STATE,
        )
        common.update(kwargs)
        if self.task == "regression":
            mlp = MLPRegressor(**{k: v for k, v in common.items()})
        else:
            mlp = MLPClassifier(**{k: v for k, v in common.items()})
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("mlp", mlp),
            ]
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        use_grid_search: bool = False,
        **kwargs,
    ) -> dict:
        t0 = time.time()
        self.feature_names = list(X_train.columns)
        self.model = self.build()

        if self.task == "regression":
            from sklearn.model_selection import KFold

            cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            cv_scoring = "r2"
        else:
            from sklearn.model_selection import StratifiedKFold

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            cv_scoring = "roc_auc"

        if use_grid_search and self.task == "classification":
            from sklearn.model_selection import RandomizedSearchCV

            param_grid = {
                "mlp__hidden_layer_sizes": [(32, 16), (64, 32), (64, 32, 16)],
                "mlp__alpha": [0.001, 0.01, 0.1],
                "mlp__learning_rate_init": [0.001, 0.005],
            }
            search = RandomizedSearchCV(
                self.model,
                param_grid,
                n_iter=12,
                cv=cv,
                scoring="balanced_accuracy",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                refit=True,
            )
            search.fit(X_train, y_train)
            self.model = search.best_estimator_
        else:
            self.model.fit(X_train, y_train)

        train_score = self.model.score(X_train, y_train)

        cv_scores = cross_val_score(
            self.build(),
            X_train,
            y_train,
            cv=cv,
            scoring=cv_scoring,
            n_jobs=1,   # séquentiel — MLP parallèle = OOM Docker
            error_score=np.nan,
        )

        score_key = "train_r2" if self.task == "regression" else "train_accuracy"
        self.metrics = {
            "n_train": len(X_train),
            score_key: round(float(train_score), 4),
            "training_time_s": round(time.time() - t0, 2),
            f"cv_{cv_scoring}": round(float(np.nanmean(cv_scores)), 4),
            f"cv_{cv_scoring}_std": round(float(np.nanstd(cv_scores)), 4),
        }
        self.is_trained = True
        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict(X)
        return raw if self.task == "regression" else raw.astype(int)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if self.task == "regression":
            return None
        try:
            return self.model.predict_proba(X)
        except Exception:
            return None

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        y_pred = self.predict(X_test)

        if self.task == "regression":
            y_true = np.asarray(y_test, dtype=float)
            y_hat = np.asarray(y_pred, dtype=float)
            self.metrics.update({
                "train_n_samples": len(X_test),
                "train_r2": round(float(r2_score(y_true, y_hat)), 4),
                "train_mae": round(float(mean_absolute_error(y_true, y_hat)), 4),
                "train_rmse": round(float(np.sqrt(mean_squared_error(y_true, y_hat))), 4),
            })
            print(
                f"  MLP Régression — R²={self.metrics['train_r2']:.4f} "
                f"MAE={self.metrics['train_mae']:.4f} "
                f"RMSE={self.metrics['train_rmse']:.4f}"
            )
        else:
            y_proba = self.predict_proba(X_test)
            baseline = float(pd.Series(y_test).value_counts(normalize=True).max())
            test_acc = accuracy_score(y_test, y_pred)
            try:
                if y_proba is not None and y_proba.ndim > 1 and y_proba.shape[1] >= 2 and len(np.unique(y_test)) >= 2:
                    test_auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    test_auc = None
            except Exception:
                test_auc = None

            self.metrics.update({
                "test_n_samples": len(X_test),
                "test_n_correct": int((y_pred == y_test.values).sum()),
                "test_accuracy": round(test_acc, 4),
                "test_balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
                "test_f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
                "test_f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
                "test_cohen_kappa": round(float(cohen_kappa_score(y_test, y_pred)), 4),
                "test_roc_auc": round(test_auc, 4) if test_auc is not None else None,
                "baseline_accuracy": round(baseline, 4),
                "improvement_over_baseline": round(test_acc - baseline, 4),
                "class_distribution": {
                    str(k): round(v, 4)
                    for k, v in pd.Series(y_test).value_counts(normalize=True).items()
                },
            })
        return self.metrics

    def get_feature_importance(self, top_n: int = 25) -> pd.DataFrame:
        """Importance proxy via norme L1 des poids de la 1ère couche du MLP."""
        if not self.is_trained or not self.feature_names:
            return pd.DataFrame(columns=["feature", "importance"])
        try:
            mlp_step = self.model.named_steps["mlp"]
            weights = np.abs(mlp_step.coefs_[0]).mean(axis=1)
            total = weights.sum() or 1.0
            imp = (
                pd.DataFrame(
                    {
                        "feature": self.feature_names[: len(weights)],
                        "importance": weights / total,
                    }
                )
                .sort_values("importance", ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )
            return imp
        except Exception:
            return pd.DataFrame(
                {
                    "feature": self.feature_names[:top_n],
                    "importance": [0.0] * min(top_n, len(self.feature_names)),
                }
            )

    def get_predictions_with_communes(
        self, X_all: pd.DataFrame, comm: pd.DataFrame
    ) -> pd.DataFrame:
        raw_pred = self.predict(X_all)
        df = comm.copy().reset_index(drop=True)
        if self.task == "regression":
            df["prediction"] = np.where(raw_pred >= 50, 0, 1)
            df["vainqueur_predit"] = np.where(raw_pred >= 50, "Macron", "Le Pen")
            df["proba_macron"] = np.round(raw_pred / 100, 4)
            df["proba_lepen"] = np.round(1 - raw_pred / 100, 4)
            df["probability"] = df["proba_lepen"]
            df["score_macron_predit"] = np.round(raw_pred, 2)
        else:
            df["prediction"] = raw_pred
            df["vainqueur_predit"] = pd.Series(raw_pred).map({0: "Macron", 1: "Le Pen"})
            probas = self.predict_proba(X_all)
            if probas is not None:
                df["proba_macron"] = probas[:, 0].round(4)
                if probas.ndim > 1 and probas.shape[1] >= 2:
                    df["proba_lepen"] = probas[:, 1].round(4)
                    df["probability"] = probas[:, 1].round(4)
        return df

    def save(self, tag: str = "") -> Path:
        suffix = f"_{tag}" if tag else ""
        model_path = self.artifact_dir / f"{self.name}{suffix}.joblib"
        meta_path = self.artifact_dir / f"{self.name}{suffix}_meta.json"
        joblib.dump(self.model, model_path)
        meta = {
            "name": self.name,
            "task": self.task,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "is_trained": self.is_trained,
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        print(f"  Modèle MLP sauvegardé : {model_path}")
        return model_path
