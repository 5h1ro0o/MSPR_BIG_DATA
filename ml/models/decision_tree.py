"""
Decision Tree — sklearn DecisionTreeClassifier.
Modèle interprétable : règles de décision lisibles, arbre visualisable.
Intérêt : comprendre les seuils socio-économiques discriminants
          (ex. revenu_median > X → Macron, taux_chomage > Y → Le Pen).
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
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from ml.config import ARTIFACTS, RANDOM_STATE
from ml.models.base import BaseModel

DT_PARAM_GRID = {
    "dt__max_depth": [3, 5, 6, 8, 10],  # None supprimé — trop sujet au mémorisation
    "dt__min_samples_split": [5, 10, 20, 30],
    "dt__min_samples_leaf": [2, 5, 10, 15],
    "dt__class_weight": ["balanced", None],  # important pour Macron dominant en IDF
    "dt__ccp_alpha": [0.0, 0.001, 0.005],  # élagage post-entraînement
}


class DecisionTreeModel(BaseModel):
    """
    Arbre de décision — interprétable, utile pour l'analyse des règles de vote.
    GridSearchCV sur DT_PARAM_GRID (cv=5).
    Exporte les règles textuelles et le DOT graphviz.
    """

    name = "decision_tree"

    def __init__(self, task: str = "classification", artifact_dir: Path = ARTIFACTS):
        super().__init__(artifact_dir)
        self.task = task

    def build(self, **kwargs) -> Pipeline:
        params = {"class_weight": "balanced", **kwargs}
        dt = DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("dt", dt),
            ]
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        use_grid_search: bool = True,
        **kwargs,
    ) -> dict:
        t0 = time.time()
        self.feature_names = list(X_train.columns)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        if use_grid_search:
            base = self.build()
            search = GridSearchCV(
                base,
                DT_PARAM_GRID,
                cv=skf,
                scoring="balanced_accuracy",
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            self.model = search.best_estimator_
            best_params = search.best_params_
        else:
            self.model = self.build(
                max_depth=6, min_samples_split=10, min_samples_leaf=5
            )
            self.model.fit(X_train, y_train)
            best_params = {}

        train_acc = self.model.score(X_train, y_train)
        cv_aucs = cross_val_score(
            self.build(),
            X_train,
            y_train,
            cv=skf,
            scoring="roc_auc",
            n_jobs=-1,
        )

        self.metrics = {
            "n_train": len(X_train),
            "train_accuracy": round(float(train_acc), 4),
            "training_time_s": round(time.time() - t0, 2),
            "cv_roc_auc": round(float(cv_aucs.mean()), 4),
            "cv_roc_auc_std": round(float(cv_aucs.std()), 4),
            "best_params": {k.replace("dt__", ""): v for k, v in best_params.items()},
        }
        self.is_trained = True

        # Export des règles textuelles (interprétabilité)
        self._export_rules()
        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        try:
            return self.model.predict_proba(X)
        except Exception:
            return None

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        baseline = float(pd.Series(y_test).value_counts(normalize=True).max())

        test_acc = accuracy_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_proba[:, 1]) if y_proba is not None else None

        self.metrics.update(
            {
                "test_n_samples": len(X_test),
                "test_n_correct": int((y_pred == y_test.values).sum()),
                "test_accuracy": round(test_acc, 4),
                "test_balanced_accuracy": round(
                    float(balanced_accuracy_score(y_test, y_pred)), 4
                ),
                "test_f1_weighted": round(
                    float(f1_score(y_test, y_pred, average="weighted")), 4
                ),
                "test_f1_macro": round(
                    float(f1_score(y_test, y_pred, average="macro")), 4
                ),
                "test_cohen_kappa": round(float(cohen_kappa_score(y_test, y_pred)), 4),
                "test_roc_auc": round(test_auc, 4) if test_auc else None,
                "baseline_accuracy": round(baseline, 4),
                "improvement_over_baseline": round(test_acc - baseline, 4),
                "class_distribution": {
                    str(k): round(v, 4)
                    for k, v in pd.Series(y_test).value_counts(normalize=True).items()
                },
            }
        )
        return self.metrics

    def get_feature_importance(self, top_n: int = 25) -> pd.DataFrame:
        dt_step = self.model.named_steps["dt"]
        imp = (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": dt_step.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        return imp

    def get_predictions_with_communes(
        self, X_all: pd.DataFrame, comm: pd.DataFrame
    ) -> pd.DataFrame:
        preds = self.predict(X_all)
        probas = self.predict_proba(X_all)
        df = comm.copy().reset_index(drop=True)
        df["prediction"] = preds
        if probas is not None:
            df["proba_macron"] = probas[:, 0].round(4)
            df["proba_lepen"] = probas[:, 1].round(4)
            df["probability"] = probas[:, 1].round(4)
        return df

    def _export_rules(self) -> None:
        """Exporte les règles de décision en texte + DOT graphviz."""
        if not self.is_trained:
            return
        try:
            dt_step = self.model.named_steps["dt"]
            rules = export_text(dt_step, feature_names=self.feature_names, max_depth=5)
            rules_path = self.artifact_dir / "decision_tree_rules.txt"
            rules_path.write_text(rules, encoding="utf-8")
            print(f"  Règles DT exportées : {rules_path}")
        except Exception as e:
            print(f"  [WARN] Export règles DT : {e}")

        try:
            from sklearn.tree import export_graphviz

            dt_step = self.model.named_steps["dt"]
            dot_data = export_graphviz(
                dt_step,
                feature_names=self.feature_names,
                class_names=["Macron (0)", "Le Pen (1)"],
                filled=True,
                rounded=True,
                max_depth=4,
            )
            dot_path = self.artifact_dir / "decision_tree.dot"
            dot_path.write_text(dot_data, encoding="utf-8")
            print(f"  Arbre DT DOT : {dot_path}")
        except Exception:
            pass

    def export_tree_rules(self) -> str:
        dt_step = self.model.named_steps["dt"]
        return export_text(dt_step, feature_names=self.feature_names, max_depth=6)

    def export_tree_dot(self, output_path: Optional[Path] = None) -> str:
        from sklearn.tree import export_graphviz

        dt_step = self.model.named_steps["dt"]
        dot_data = export_graphviz(
            dt_step,
            feature_names=self.feature_names,
            class_names=["Macron (0)", "Le Pen (1)"],
            filled=True,
            rounded=True,
            max_depth=4,
        )
        if output_path:
            Path(output_path).write_text(dot_data, encoding="utf-8")
        return dot_data

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
        print(f"  Modèle DT sauvegardé : {model_path}")
        return model_path
