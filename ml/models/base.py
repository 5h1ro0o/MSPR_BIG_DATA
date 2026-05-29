"""
Classe de base abstraite pour tous les modèles ML du projet.
Définit l'interface commune : fit, predict, evaluate, save, load.
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Interface commune pour tous les modèles du projet."""

    name: str = "base"
    task: str = "classification"

    def __init__(self, artifact_dir: Path):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.feature_names: list[str] = []
        self.metrics: dict = {}
        self.is_trained: bool = False

    @abstractmethod
    def build(self, **kwargs) -> Any:
        """Instancie et retourne l'estimateur sklearn (ou équivalent)."""

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> dict:
        """Entraîne le modèle. Retourne les métriques d'entraînement."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Retourne les prédictions (labels ou valeurs continues)."""

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Retourne les probabilités (classification uniquement). None si non dispo."""
        return None

    def save(self, tag: str = "") -> Path:
        """Sérialise le modèle entraîné + métadonnées."""
        suffix = f"_{tag}" if tag else ""
        model_path = self.artifact_dir / f"{self.name}{suffix}.pkl"
        meta_path = self.artifact_dir / f"{self.name}{suffix}_meta.json"

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        meta = {
            "name": self.name,
            "task": self.task,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "is_trained": self.is_trained,
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        print(f"  Modèle sauvegardé : {model_path}")
        return model_path

    def load(self, tag: str = "") -> "BaseModel":
        """Charge un modèle persisté."""
        suffix = f"_{tag}" if tag else ""
        model_path = self.artifact_dir / f"{self.name}{suffix}.pkl"
        meta_path = self.artifact_dir / f"{self.name}{suffix}_meta.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé : {model_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.feature_names = meta.get("feature_names", [])
            self.metrics = meta.get("metrics", {})
            self.is_trained = meta.get("is_trained", True)

        print(f"  Modèle chargé : {model_path}")
        return self

    def summary(self) -> str:
        lines = [
            f"Modèle      : {self.name}",
            f"Tâche       : {self.task}",
            f"Entraîné    : {self.is_trained}",
            f"Nb features : {len(self.feature_names)}",
            "Métriques   :",
        ]
        for k, v in self.metrics.items():
            lines.append(f"  {k:30s}: {v}")
        return "\n".join(lines)
