"""Tests unitaires — ml/models/base.py (via sous-classe concrète)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.models.base import BaseModel


class ConcreteModel(BaseModel):
    """Implémentation minimale pour tester la classe abstraite BaseModel."""

    name = "test_model"
    task = "regression"

    def build(self, **kwargs):
        return None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model = {"coef": 1.0}
        self.feature_names = list(X_train.columns)
        self.is_trained = True
        self.metrics = {"r2": 0.9, "mae": 1.5}
        return self.metrics

    def predict(self, X):
        return np.zeros(len(X))


@pytest.fixture
def tmp_model(tmp_path):
    return ConcreteModel(artifact_dir=tmp_path)


@pytest.fixture
def trained_model(tmp_path):
    m = ConcreteModel(artifact_dir=tmp_path)
    X = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0], "feat_b": [4.0, 5.0, 6.0]})
    y = pd.Series([10.0, 20.0, 30.0])
    m.train(X, y)
    return m


class TestBaseModelInit:
    def test_artifact_dir_set(self, tmp_path):
        m = ConcreteModel(artifact_dir=tmp_path)
        assert m.artifact_dir == tmp_path

    def test_artifact_dir_created(self, tmp_path):
        new_dir = tmp_path / "model_artifacts"
        ConcreteModel(artifact_dir=new_dir)
        assert new_dir.exists()

    def test_initial_state(self, tmp_model):
        assert tmp_model.model is None
        assert tmp_model.feature_names == []
        assert tmp_model.metrics == {}
        assert not tmp_model.is_trained


class TestBaseModelTrain:
    def test_train_sets_is_trained(self, tmp_model):
        X = pd.DataFrame({"a": [1, 2]})
        y = pd.Series([10.0, 20.0])
        tmp_model.train(X, y)
        assert tmp_model.is_trained

    def test_train_stores_feature_names(self, tmp_model):
        X = pd.DataFrame({"feat_a": [1, 2], "feat_b": [3, 4]})
        y = pd.Series([10.0, 20.0])
        tmp_model.train(X, y)
        assert tmp_model.feature_names == ["feat_a", "feat_b"]

    def test_train_returns_metrics(self, tmp_model):
        X = pd.DataFrame({"x": [1, 2]})
        y = pd.Series([1.0, 2.0])
        result = tmp_model.train(X, y)
        assert isinstance(result, dict)
        assert "r2" in result


class TestBaseModelPredict:
    def test_predict_returns_array(self, trained_model):
        X = pd.DataFrame({"feat_a": [1.0], "feat_b": [4.0]})
        preds = trained_model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == 1

    def test_predict_proba_returns_none(self, trained_model):
        X = pd.DataFrame({"feat_a": [1.0]})
        assert trained_model.predict_proba(X) is None


class TestBaseModelSave:
    def test_save_creates_pkl(self, trained_model, tmp_path):
        trained_model.save()
        assert (tmp_path / "test_model.pkl").exists()

    def test_save_creates_meta_json(self, trained_model, tmp_path):
        trained_model.save()
        assert (tmp_path / "test_model_meta.json").exists()

    def test_save_with_tag(self, trained_model, tmp_path):
        trained_model.save(tag="v1")
        assert (tmp_path / "test_model_v1.pkl").exists()
        assert (tmp_path / "test_model_v1_meta.json").exists()

    def test_save_meta_content(self, trained_model, tmp_path):
        trained_model.save()
        meta = json.loads((tmp_path / "test_model_meta.json").read_text())
        assert meta["name"] == "test_model"
        assert meta["task"] == "regression"
        assert meta["is_trained"] is True
        assert meta["feature_names"] == ["feat_a", "feat_b"]


class TestBaseModelLoad:
    def test_load_restores_feature_names(self, trained_model, tmp_path):
        trained_model.save()
        m2 = ConcreteModel(artifact_dir=tmp_path)
        m2.load()
        assert m2.feature_names == ["feat_a", "feat_b"]

    def test_load_restores_metrics(self, trained_model, tmp_path):
        trained_model.save()
        m2 = ConcreteModel(artifact_dir=tmp_path)
        m2.load()
        assert m2.metrics["r2"] == pytest.approx(0.9)

    def test_load_sets_is_trained(self, trained_model, tmp_path):
        trained_model.save()
        m2 = ConcreteModel(artifact_dir=tmp_path)
        m2.load()
        assert m2.is_trained

    def test_load_with_tag(self, trained_model, tmp_path):
        trained_model.save(tag="prod")
        m2 = ConcreteModel(artifact_dir=tmp_path)
        m2.load(tag="prod")
        assert m2.is_trained

    def test_load_raises_if_not_found(self, tmp_path):
        m = ConcreteModel(artifact_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            m.load()

    def test_save_load_roundtrip(self, trained_model, tmp_path):
        trained_model.save(tag="rt")
        m2 = ConcreteModel(artifact_dir=tmp_path)
        m2.load(tag="rt")
        assert m2.feature_names == trained_model.feature_names
        assert m2.metrics == trained_model.metrics


class TestBaseModelSummary:
    def test_summary_contains_name(self, trained_model):
        s = trained_model.summary()
        assert "test_model" in s

    def test_summary_contains_task(self, trained_model):
        s = trained_model.summary()
        assert "regression" in s

    def test_summary_contains_metrics(self, trained_model):
        s = trained_model.summary()
        assert "r2" in s

    def test_summary_is_string(self, trained_model):
        assert isinstance(trained_model.summary(), str)
