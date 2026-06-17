"""
Predictor — interface unifiée pour charger un modèle entraîné et prédire.
Utilisé par le dashboard et l'API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from ml.config import ARTIFACTS
from ml.preprocessing import build_X_y, get_commune_info, load_dataset


class Predictor:
    """
    Charge un modèle sauvegardé et produit des prédictions enrichies
    avec les informations géographiques des communes.
    """

    def __init__(self, model_name: str, tag: str = "", artifact_dir: Path = ARTIFACTS):
        """
        Args:
            model_name: "random_forest" | "gradient_boosting" | "decision_tree" | "mlp" | "lstm"
            tag:        suffixe utilisé lors de la sauvegarde (ex. "pre_vote_classification_t2")
            artifact_dir: répertoire des artefacts
        """
        self.model_name = model_name
        self.tag = tag
        self.artifact_dir = Path(artifact_dir)
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Instancie et charge le modèle selon son nom."""
        if self.model_name == "random_forest":
            from ml.models.random_forest import RandomForestModel

            meta = self._load_meta()
            task = meta.get("task", "classification")
            self.model = RandomForestModel(task=task, artifact_dir=self.artifact_dir)
            self.model.load(tag=self.tag)

        elif self.model_name == "gradient_boosting":
            from ml.models.gradient_boosting import GradientBoostingModel

            meta = self._load_meta()
            self.model = GradientBoostingModel(
                task=meta.get("task", "classification"), artifact_dir=self.artifact_dir
            )
            self.model.load(tag=self.tag)

        elif self.model_name == "decision_tree":
            from ml.models.decision_tree import DecisionTreeModel

            meta = self._load_meta()
            self.model = DecisionTreeModel(
                task=meta.get("task", "classification"), artifact_dir=self.artifact_dir
            )
            self.model.load(tag=self.tag)

        elif self.model_name == "mlp":
            from ml.models.mlp import MLPModel

            meta = self._load_meta()
            self.model = MLPModel(
                task=meta.get("task", "classification"), artifact_dir=self.artifact_dir
            )
            self.model.load(tag=self.tag)

        elif self.model_name == "lstm":
            from ml.models.lstm import LSTMModel

            self.model = LSTMModel(artifact_dir=self.artifact_dir)
            self.model.load(tag=self.tag)

        else:
            raise ValueError(f"Modèle inconnu : {self.model_name}")

    def _load_meta(self) -> dict:
        """Charge les métadonnées JSON du modèle."""
        suffix = f"_{self.tag}" if self.tag else ""
        meta_path = self.artifact_dir / f"{self.model_name}{suffix}_meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        return {}

    def predict_all_communes(
        self,
        feature_set: str = "pre_vote",
        target: str = "classification_t2",
    ) -> pd.DataFrame:
        """
        Prédit sur toutes les communes du dataset et retourne un DataFrame enrichi.

        Returns:
            DataFrame avec code_commune, libelle, prédiction (et probas si classif).
        """
        df = load_dataset()
        X, y, _ = build_X_y(df, feature_set=feature_set, target=target)
        commune_info = get_commune_info(df)

        y_pred = self.model.predict(X)
        result = commune_info.reset_index(drop=True).copy()
        result["prediction"] = y_pred
        result["ground_truth"] = y.reset_index(drop=True)
        result["correct"] = (result["prediction"] == result["ground_truth"]).astype(int)

        if self.model.task == "classification":
            y_proba = self.model.predict_proba(X)
            if y_proba is not None:
                rf_step = getattr(self.model.pipeline, "named_steps", {}).get("rf")
                classes = getattr(rf_step, "classes_", [0, 1]) if rf_step else [0, 1]
                for i, cls in enumerate(classes):
                    result[f"proba_{cls}"] = y_proba[:, i].round(4)

        return result

    def predict_single_commune(
        self,
        code_commune: str,
        feature_set: str = "pre_vote",
        target: str = "classification_t2",
    ) -> dict:
        """
        Prédit pour une seule commune et retourne un dict avec tous les détails.
        """
        df = load_dataset()
        mask = df["code_commune"] == code_commune
        if mask.sum() == 0:
            raise ValueError(f"Commune introuvable : {code_commune}")

        X, y, _ = build_X_y(df, feature_set=feature_set, target=target)
        commune_row = df[mask].index.intersection(X.index)
        if len(commune_row) == 0:
            raise ValueError(f"Données insuffisantes pour {code_commune}")

        X_commune = X.loc[commune_row]
        y_commune = y.loc[commune_row]

        prediction = self.model.predict(X_commune)[0]
        result = {
            "code_commune": code_commune,
            "libelle_commune": df.loc[mask, "libelle_commune"].values[0],
            "prediction": prediction,
            "ground_truth": y_commune.values[0],
            "correct": prediction == y_commune.values[0],
        }

        y_proba = self.model.predict_proba(X_commune)
        if y_proba is not None:
            result["proba_macron"] = round(float(y_proba[0, 0]), 4)
            result["proba_lepen"] = round(float(y_proba[0, 1]), 4)

        return result

    @property
    def metrics(self) -> dict:
        """Retourne les métriques du modèle chargé."""
        return getattr(self.model, "metrics", {})

    @property
    def feature_importance(self) -> Optional[pd.DataFrame]:
        """Retourne l'importance des features si disponible."""
        try:
            return self.model.get_feature_importance()
        except (AttributeError, RuntimeError):
            return None
