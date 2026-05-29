"""
MLP (Multi-Layer Perceptron) — squelette prêt à compléter.

Modèles :
  - Option A : sklearn MLPClassifier / MLPRegressor  (simple, pas de GPU)
  - Option B : PyTorch nn.Module                     (flexible, GPU possible)

Spécificités électorales :
  - StandardScaler OBLIGATOIRE (MLP sensible aux échelles)
  - Risque d'overfitting avec 1268 communes → dropout + early stopping
  - Batch normalization recommandée entre les couches

Architecture suggérée :
  Input (n_features) → 128 → Dropout(0.3) → 64 → Dropout(0.2) → 32 → Output
  Activation : ReLU (couches cachées) + Sigmoid/Softmax (sortie)

Usage prévu :
    from ml.models.mlp import MLPModel

    model = MLPModel(task="classification", backend="sklearn")
    model.train(X_train, y_train, X_val=X_test, y_val=y_test)
    model.evaluate(X_test, y_test)
    model.plot_training_curve()   # courbe loss
    model.save()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.models.base import BaseModel
from ml.config import ARTIFACTS

MLP_PARAM_GRID_SKLEARN = {
    "mlp__hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
    "mlp__activation": ["relu", "tanh"],
    "mlp__alpha": [0.0001, 0.001, 0.01],
    "mlp__learning_rate_init": [0.001, 0.01],
    "mlp__max_iter": [200, 500],
    "mlp__early_stopping": [True],
    "mlp__validation_fraction": [0.1],
}

PYTORCH_HIDDEN_SIZES = [128, 64, 32]
PYTORCH_DROPOUT = 0.3
PYTORCH_EPOCHS = 200
PYTORCH_LR = 1e-3
PYTORCH_BATCH_SIZE = 64


class MLPModel(BaseModel):
    """
    MLP — réseau de neurones fully connected.

    TODO :
      1. Choisir backend : "sklearn" (simple) ou "pytorch" (flexible)
      2. Implémenter build() pour le backend choisi
      3. Implémenter train() avec early stopping
      4. Implémenter plot_training_curve() — loss train vs val par époque
      5. Pour PyTorch : implémenter _train_epoch() et _validate()
      6. Gérer la sauvegarde torch.save() / torch.load() si PyTorch
    """

    name = "mlp"

    def __init__(
        self,
        task: str = "classification",
        backend: str = "sklearn",
        artifact_dir: Path = ARTIFACTS,
    ):
        super().__init__(artifact_dir)
        self.task = task
        self.backend = backend
        self.history: dict = {"train_loss": [], "val_loss": []}

    def build(self, **kwargs):
        """
        TODO Option A — sklearn :
            from sklearn.neural_network import MLPClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer

            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
                ("mlp",     MLPClassifier(random_state=RANDOM_STATE, **kwargs)),
            ])

        TODO Option B — PyTorch :
            class ElectionMLP(nn.Module):
                def __init__(self, input_dim, hidden_sizes, dropout, n_classes):
                    super().__init__()
                    layers = []
                    prev = input_dim
                    for h in hidden_sizes:
                        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                                   nn.ReLU(), nn.Dropout(dropout)]
                        prev = h
                    layers.append(nn.Linear(prev, n_classes))
                    self.net = nn.Sequential(*layers)

                def forward(self, x):
                    return self.net(x)

            return ElectionMLP(input_dim, PYTORCH_HIDDEN_SIZES, PYTORCH_DROPOUT, n_classes)
        """
        raise NotImplementedError("MLPModel.build() non implémentée.")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        epochs: int = PYTORCH_EPOCHS,
        batch_size: int = PYTORCH_BATCH_SIZE,
        lr: float = PYTORCH_LR,
    ) -> dict:
        """
        TODO :
          sklearn : RandomizedSearchCV sur MLP_PARAM_GRID_SKLEARN
          PyTorch :
            - DataLoader(TensorDataset(X_tensor, y_tensor), batch_size)
            - Boucle epochs : forward → loss → backward → optimizer.step()
            - Early stopping sur val_loss (patience=20)
            - Stocker self.history["train_loss"] et self.history["val_loss"]
        """
        raise NotImplementedError("MLPModel.train() non implémentée.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("MLPModel.predict() non implémentée.")

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        raise NotImplementedError("MLPModel.predict_proba() non implémentée.")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        raise NotImplementedError("MLPModel.evaluate() non implémentée.")

    def plot_training_curve(self) -> None:
        """
        TODO : courbe loss / accuracy par époque.

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(self.history["train_loss"], label="Train loss")
            ax.plot(self.history["val_loss"],   label="Val loss")
            ax.set_title("MLP — Courbe d'entraînement")
            ax.legend()
            plt.savefig(self.artifact_dir / "mlp_training_curve.png", dpi=150)
        """
        raise NotImplementedError("MLPModel.plot_training_curve() non implémentée.")
