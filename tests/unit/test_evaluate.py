"""Tests unitaires — ml/training/evaluate.py (fonctions pures + plots mockés)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ml.training.evaluate import (
    baseline_metrics,
    bootstrap_ci,
    classification_metrics,
    regression_metrics,
    robust_classification_eval,
    wilson_ci,
)


# ── Intervalles de confiance ──────────────────────────────────────────────────


class TestWilsonCI:
    def test_basic_proportion(self):
        lo, hi = wilson_ci(90, 100)
        assert lo < 0.9 < hi
        assert 0 <= lo <= 1
        assert 0 <= hi <= 1

    def test_perfect_accuracy(self):
        lo, hi = wilson_ci(100, 100)
        assert lo <= 1.0
        assert hi == 1.0

    def test_zero_accuracy(self):
        lo, hi = wilson_ci(0, 100)
        assert lo == 0.0

    def test_confidence_wider_at_99(self):
        lo95, hi95 = wilson_ci(50, 100, confidence=0.95)
        lo99, hi99 = wilson_ci(50, 100, confidence=0.99)
        assert (hi99 - lo99) > (hi95 - lo95)

    def test_half_correct(self):
        lo, hi = wilson_ci(50, 100)
        assert lo < 0.5 < hi


class TestBootstrapCI:
    def test_returns_tuple_ordered(self):
        y_true = np.array([0, 1, 0, 1, 0])
        y_score = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        lo, hi = bootstrap_ci(
            y_true, y_score, lambda yt, ys: float(np.mean(ys)), n_bootstrap=50
        )
        assert lo <= hi

    def test_stable_with_fixed_seed(self):
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.2, 0.8, 0.3, 0.7])
        lo1, hi1 = bootstrap_ci(y_true, y_score, np.mean, n_bootstrap=100, random_state=42)
        lo2, hi2 = bootstrap_ci(y_true, y_score, np.mean, n_bootstrap=100, random_state=42)
        assert lo1 == lo2
        assert hi1 == hi2


# ── Baseline ─────────────────────────────────────────────────────────────────


class TestBaselineMetrics:
    def test_majority_class_identified(self):
        y = np.array([0, 0, 0, 1])
        m = baseline_metrics(y)
        assert m["baseline_majority_class"] == 0
        assert m["baseline_accuracy"] == pytest.approx(0.75)

    def test_balanced_classes(self):
        y = np.array([0, 1])
        m = baseline_metrics(y)
        assert m["baseline_accuracy"] == pytest.approx(0.5)

    def test_returns_class_distribution(self):
        y = np.array([0, 1, 0])
        m = baseline_metrics(y)
        assert "class_distribution" in m
        assert 0 in m["class_distribution"]

    def test_all_same_class(self):
        y = np.array([1, 1, 1, 1])
        m = baseline_metrics(y)
        assert m["baseline_accuracy"] == 1.0


# ── Métriques de régression ──────────────────────────────────────────────────


class TestRegressionMetrics:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        m = regression_metrics(y, y)
        assert m["r2"] == pytest.approx(1.0)
        assert m["mae"] == pytest.approx(0.0)
        assert m["rmse"] == pytest.approx(0.0)

    def test_constant_offset(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        m = regression_metrics(y_true, y_pred)
        assert m["mae"] == pytest.approx(0.5)

    def test_keys_present(self):
        y = np.array([1.0, 2.0])
        m = regression_metrics(y, y)
        assert set(m.keys()) == {"r2", "mae", "rmse"}


# ── Métriques de classification ───────────────────────────────────────────────


class TestClassificationMetrics:
    def test_perfect_classification(self):
        y = np.array([0, 1, 0, 1])
        m = classification_metrics(y, y)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_confusion_matrix_2x2(self):
        y = np.array([0, 0, 1, 1])
        pred = np.array([0, 1, 0, 1])
        m = classification_metrics(y, pred)
        cm = m["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_f1_weighted_present(self):
        y = np.array([0, 1, 0, 1])
        m = classification_metrics(y, y)
        assert "f1_weighted" in m

    def test_roc_auc_with_proba(self):
        y = np.array([0, 1, 0, 1])
        y_proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        m = classification_metrics(y, y, y_proba=y_proba)
        assert "roc_auc" in m
        assert m["roc_auc"] == pytest.approx(1.0)


# ── Évaluation robuste ────────────────────────────────────────────────────────


class TestRobustClassificationEval:
    def test_basic_eval_keys(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 0])
        m = robust_classification_eval(y_true, y_pred, split="test")
        assert "test_accuracy" in m
        assert "test_balanced_accuracy" in m
        assert "test_f1_weighted" in m
        assert "test_f1_macro" in m
        assert "test_cohen_kappa" in m

    def test_with_proba_adds_auc(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
        m = robust_classification_eval(y_true, y_pred, y_proba=y_proba, n_bootstrap=10)
        assert "test_roc_auc" in m

    def test_custom_split_prefix(self):
        y = np.array([0, 1, 0, 1])
        m = robust_classification_eval(y, y, split="val")
        assert all(k.startswith("val_") for k in m)

    def test_n_samples_correct(self):
        y = np.array([0, 1, 0, 1, 0])
        m = robust_classification_eval(y, y, split="test")
        assert m["test_n_samples"] == 5

    def test_ci_lo_le_hi(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        m = robust_classification_eval(y, y, split="test")
        assert m["test_accuracy_ci_lo"] <= m["test_accuracy_ci_hi"]


# ── Graphiques (matplotlib mocké) ────────────────────────────────────────────


class TestPlotFunctions:
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_confusion_matrix(self, mock_close, mock_savefig, tmp_path):
        from ml.training.evaluate import plot_confusion_matrix
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 0])
        path = plot_confusion_matrix(y_true, y_pred, ["macron", "lepen"], "test_model", tmp_path)
        assert mock_savefig.called
        assert "test_model" in str(path)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_roc_curve(self, mock_close, mock_savefig, tmp_path):
        from ml.training.evaluate import plot_roc_curve
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        path = plot_roc_curve(y_true, y_proba, "model", tmp_path)
        assert mock_savefig.called

    def test_plot_roc_curve_raises_on_single_class(self, tmp_path):
        from ml.training.evaluate import plot_roc_curve
        y_true = np.array([0, 0, 0])
        y_proba = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
        with pytest.raises(ValueError):
            plot_roc_curve(y_true, y_proba, "model", tmp_path)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_predictions_vs_actual(self, mock_close, mock_savefig, tmp_path):
        from ml.training.evaluate import plot_predictions_vs_actual
        y_true = np.array([60.0, 70.0, 80.0])
        y_pred = np.array([62.0, 68.0, 81.0])
        path = plot_predictions_vs_actual(y_true, y_pred, "gb", tmp_path)
        assert mock_savefig.called

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_feature_importance(self, mock_close, mock_savefig, tmp_path):
        from ml.training.evaluate import plot_feature_importance
        df = pd.DataFrame({"feature": ["a", "b", "c"], "importance": [0.5, 0.3, 0.2]})
        path = plot_feature_importance(df, "rf", tmp_path, top_n=3)
        assert mock_savefig.called

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_model_comparison(self, mock_close, mock_savefig, tmp_path):
        from ml.training.evaluate import plot_model_comparison
        results = {"rf": {"r2": 0.9}, "gb": {"r2": 0.85}}
        path = plot_model_comparison(results, "r2", tmp_path)
        assert mock_savefig.called

    def test_plot_model_comparison_all_zero_skips_plot(self, tmp_path):
        from ml.training.evaluate import plot_model_comparison
        results = {"rf": {"r2": None}, "gb": {"r2": None}}
        path = plot_model_comparison(results, "r2", tmp_path)
        assert path is not None
