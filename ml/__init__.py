"""
ML package — Élections présidentielles IDF 2022.

Modèles disponibles :
    rf   → Random Forest
    gb   → Gradient Boosting
    lstm → LSTM (TensorFlow requis)
    dt   → Decision Tree (à implémenter)
    mlp  → MLP (à implémenter)
"""
from ml.config import ARTIFACTS, TARGETS, FEATURE_SETS, RANDOM_STATE

__all__ = ["ARTIFACTS", "TARGETS", "FEATURE_SETS", "RANDOM_STATE"]
