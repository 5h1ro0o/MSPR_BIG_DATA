"""
ML package — Élections présidentielles IDF 2022.

Modèles disponibles :
    rf   → Random Forest
    gb   → Gradient Boosting
    lstm → LSTM (TensorFlow requis)
    dt   → Decision Tree (à implémenter)
    mlp  → MLP (à implémenter)
"""

from ml.config import ARTIFACTS, FEATURE_SETS, RANDOM_STATE, TARGETS

__all__ = ["ARTIFACTS", "TARGETS", "FEATURE_SETS", "RANDOM_STATE"]
