"""
Decision Tree — squelette prêt à compléter.

Intérêt dans ce projet :
  - Modèle interprétable : arbre visualisable (communes → règles de vote)
  - Baseline simple avant les modèles ensemblistes
  - Utile pour comprendre les seuils décisionnels (ex: revenu_median > X → Macron)

Modèles :
  - DecisionTreeClassifier  (cible_t2_vainqueur, cible_t1_premier)
  - DecisionTreeRegressor   (cible_t2_pct_macron, cible_t2_marge)

Usage prévu :
    from ml.models.decision_tree import DecisionTreeModel

    model = DecisionTreeModel(task="classification")
    model.train(X_train, y_train, X_val=X_test, y_val=y_test)
    model.evaluate(X_test, y_test)
    model.export_tree_dot()   # export pour graphviz
    model.save()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.models.base import BaseModel
from ml.config import ARTIFACTS

DT_PARAM_GRID = {
    "dt__max_depth": [3, 5, 7, 10, None],
    "dt__min_samples_split": [2, 5, 10, 20],
    "dt__min_samples_leaf": [1, 2, 5, 10],
    "dt__criterion": ["gini", "entropy"],
    "dt__max_features": [None, "sqrt", "log2"],
    "dt__ccp_alpha": [0.0, 0.001, 0.01],
}


class DecisionTreeModel(BaseModel):
    """
    Arbre de décision — interprétable, bon pour l'analyse des règles de vote.

    TODO :
      1. Implémenter build() — DecisionTreeClassifier ou DecisionTreeRegressor
      2. Implémenter train() — GridSearchCV sur DT_PARAM_GRID
      3. Implémenter evaluate() — métriques + visualisation arbre
      4. Implémenter export_tree_dot() — export graphviz pour visualisation
      5. Implémenter export_tree_rules() — extraction des règles textuelles
         (sklearn.tree.export_text)
    """

    name = "decision_tree"

    def __init__(
        self,
        task: str = "classification",
        artifact_dir: Path = ARTIFACTS,
    ):
        super().__init__(artifact_dir)
        self.task = task

    def build(self, **kwargs):
        """
        TODO :
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer

            estimator = DecisionTreeClassifier(random_state=RANDOM_STATE, **kwargs)
            # Pas de StandardScaler nécessaire pour les arbres (invariant aux échelles)
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("dt", estimator),
            ])
        """
        raise NotImplementedError("DecisionTreeModel.build() non implémentée.")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> dict:
        """
        TODO :
          1. build() → pipeline
          2. GridSearchCV sur DT_PARAM_GRID (cv=5)
          3. Métriques train + val
          4. self.metrics, self.is_trained = True
        """
        raise NotImplementedError("DecisionTreeModel.train() non implémentée.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("DecisionTreeModel.predict() non implémentée.")

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        raise NotImplementedError("DecisionTreeModel.predict_proba() non implémentée.")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        raise NotImplementedError("DecisionTreeModel.evaluate() non implémentée.")

    def get_feature_importance(self, top_n: int = 30) -> pd.DataFrame:
        """TODO : pipeline.named_steps["dt"].feature_importances_"""
        raise NotImplementedError(
            "DecisionTreeModel.get_feature_importance() non implémentée."
        )

    def export_tree_dot(self, output_path: Optional[Path] = None) -> str:
        """
        TODO : exporter l'arbre au format DOT pour graphviz.

            from sklearn.tree import export_graphviz
            dot_data = export_graphviz(
                pipeline.named_steps["dt"],
                feature_names=self.feature_names,
                class_names=["Macron", "Le Pen"],
                filled=True, rounded=True,
                max_depth=4,   # limiter la profondeur pour la lisibilité
            )
            if output_path:
                output_path.write_text(dot_data)
            return dot_data
        """
        raise NotImplementedError(
            "DecisionTreeModel.export_tree_dot() non implémentée."
        )

    def export_tree_rules(self) -> str:
        """
        TODO : exporter les règles de décision en texte.

            from sklearn.tree import export_text
            return export_text(
                pipeline.named_steps["dt"],
                feature_names=self.feature_names,
            )
        """
        raise NotImplementedError(
            "DecisionTreeModel.export_tree_rules() non implémentée."
        )
