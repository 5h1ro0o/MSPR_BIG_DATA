"""
Trainer — orchestration de l'entraînement de tous les modèles.
Chaque fonction `train_XXX()` est autonome et produit artefacts + métriques.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import os

from ml.config import ARTIFACTS, TARGETS, RANDOM_STATE, FEATURE_SETS
from ml.preprocessing import (
    load_dataset, build_X_y, split_data, get_commune_info,
)
from ml.training.evaluate import (
    plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, plot_predictions_vs_actual,
    plot_model_comparison,
)

def train_random_forest(
    feature_set: str = "pre_vote",
    target: str = "classification_t2",
    fast_search: bool = True,
    n_iter: int = 20,
    save: bool = True,
) -> dict:
    """Entraîne le Random Forest avec recherche d'hyperparamètres."""
    from ml.models.random_forest import RandomForestModel

    df = load_dataset()
    task = "regression" if "regression" in target else "classification"
    tag  = f"{feature_set}_{target}"

    X, y, features = build_X_y(df, feature_set=feature_set, target=target)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = RandomForestModel(task=task, artifact_dir=ARTIFACTS)
    model.train(X_train, y_train, X_val=X_test, y_val=y_test,
                use_grid_search=True, fast_search=fast_search, n_iter_random=n_iter)
    model.evaluate(X_test, y_test)

    y_pred = model.predict(X_test)
    if task == "classification":
        plot_confusion_matrix(y_test, y_pred, ["Macron (0)", "Le Pen (1)"],
                              "random_forest", ARTIFACTS)
        y_proba = model.predict_proba(X_test)
        if y_proba is not None:
            try:
                plot_roc_curve(y_test, y_proba, "random_forest", ARTIFACTS)
            except Exception:
                pass
    else:
        plot_predictions_vs_actual(y_test, y_pred, "random_forest", ARTIFACTS, target)

    df_imp = model.get_feature_importance(top_n=25)
    plot_feature_importance(df_imp, "random_forest", ARTIFACTS)

    X_all, y_all, _ = build_X_y(df, feature_set=feature_set, target=target)
    comm = get_commune_info(df).loc[y_all.index].reset_index(drop=True)
    pred_df = model.get_predictions_with_communes(X_all, comm)
    pred_df["ground_truth"] = y_all.reset_index(drop=True)
    pred_df["correct"] = (pred_df["prediction"] == pred_df["ground_truth"]).astype(int)
    pred_df.to_csv(ARTIFACTS / f"rf_predictions_{tag}.csv", index=False)

    if save:
        model.save(tag=tag)
    _save_metrics(model.metrics, "random_forest", tag)
    _store_to_db(pred_df, model.metrics, "random_forest", feature_set, target, tag)
    print(f"\n  Artefacts RF sauvegardés dans : {ARTIFACTS}")
    return model.metrics

def train_gradient_boosting(
    feature_set: str = "pre_vote",
    target: str = "classification_t2",
    use_search: bool = False,
    n_iter: int = 20,
    save: bool = True,
) -> dict:
    """
    Entraîne le Gradient Boosting.
    Anti-leakage automatique (colonnes T2 exclues dans le modèle).
    """
    from ml.models.gradient_boosting import GradientBoostingModel

    df = load_dataset()
    task = "regression" if "regression" in target else "classification"
    tag  = f"{feature_set}_{target}"

    X, y, features = build_X_y(df, feature_set=feature_set, target=target)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = GradientBoostingModel(task=task, artifact_dir=ARTIFACTS)
    model.train(X_train, y_train, X_val=X_test, y_val=y_test,
                use_search=use_search, n_iter=n_iter)
    model.evaluate(X_test, y_test)

    if task == "classification":
        X_all, y_all, _ = build_X_y(df, feature_set=feature_set, target=target)
        model.plot_results(X_test, y_test, X_all, y_all)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        plot_confusion_matrix(y_test, y_pred, ["Macron (0)", "Le Pen (1)"],
                              "gradient_boosting", ARTIFACTS)
        if y_proba is not None:
            try:
                plot_roc_curve(y_test, y_proba, "gradient_boosting", ARTIFACTS)
            except Exception:
                pass

    df_imp = model.get_feature_importance(top_n=25)
    plot_feature_importance(df_imp, "gradient_boosting", ARTIFACTS)
    df_imp.to_csv(ARTIFACTS / "gb_top_features.csv", index=False)

    X_all, y_all, _ = build_X_y(df, feature_set=feature_set, target=target)
    comm = get_commune_info(df).loc[y_all.index].reset_index(drop=True)
    pred_df = model.get_predictions_with_communes(X_all, comm)
    pred_df["ground_truth"] = y_all.reset_index(drop=True)
    pred_df["correct"] = (pred_df["prediction"] == pred_df["ground_truth"]).astype(int)
    pred_df.to_csv(ARTIFACTS / f"gb_predictions_{tag}.csv", index=False)

    if save:
        model.save(tag=tag)
    _save_metrics(model.metrics, "gradient_boosting", tag)
    _store_to_db(pred_df, model.metrics, "gradient_boosting", feature_set, target, tag)
    print(f"\n  Artefacts GB sauvegardés dans : {ARTIFACTS}")
    return model.metrics

def train_lstm(
    target: str = "classification_t2",
    save: bool = True,
) -> dict:
    """
    Entraîne le LSTM dual-input.
    feature_set forcé à "lstm" (contient toutes les colonnes nécessaires
    aux séquences temporelles + socio-économiques).
    """
    from ml.models.lstm import LSTMModel, TF_AVAILABLE

    if not TF_AVAILABLE:
        print("  [SKIP] TensorFlow non disponible. Installez : pip install tensorflow")
        return {}

    df  = load_dataset()
    tag = f"lstm_{target}"

    X, y, features = build_X_y(df, feature_set="lstm", target=target)

    idx = np.arange(len(X))
    y_arr = y.values
    from sklearn.model_selection import train_test_split
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE,
                                      stratify=y_arr)

    X_train = X.iloc[idx_tr].reset_index(drop=True)
    X_test  = X.iloc[idx_te].reset_index(drop=True)
    y_train = pd.Series(y_arr[idx_tr])
    y_test  = pd.Series(y_arr[idx_te])

    print(f"\n  Distribution cible :")
    print(f"  Macron gagne  : {(y_arr==0).sum()} communes ({(y_arr==0).mean()*100:.1f}%)")
    print(f"  Le Pen gagne  : {(y_arr==1).sum()} communes ({(y_arr==1).mean()*100:.1f}%)")

    model = LSTMModel(artifact_dir=ARTIFACTS)
    model.train(X_train, y_train, X_val=X_test, y_val=y_test)
    model.evaluate(X_test, y_test)

    model.plot_training_curves()
    model.plot_confusion_matrix(X_test, y_test)

    y_proba = model.predict_proba(X_test)
    if y_proba is not None:
        try:
            plot_roc_curve(y_test, y_proba, "lstm", ARTIFACTS)
        except Exception:
            pass

    X_all, y_all, _ = build_X_y(df, feature_set="lstm", target=target)
    comm = get_commune_info(df).loc[y_all.index].reset_index(drop=True)
    pred_df = model.get_predictions_with_communes(X_all.reset_index(drop=True), comm)
    pred_df["ground_truth"] = y_all.reset_index(drop=True)
    pred_df["correct"] = (pred_df["prediction"] == pred_df["ground_truth"]).astype(int)
    pred_df.to_csv(ARTIFACTS / f"lstm_predictions_{target}.csv", index=False)

    if save:
        model.save(tag=tag)
    _save_metrics(model.metrics, "lstm", tag)
    _store_to_db(pred_df, model.metrics, "lstm", "lstm", target, tag)
    print(f"\n  Artefacts LSTM sauvegardés dans : {ARTIFACTS}")
    return model.metrics

def compare_all_models(results: dict[str, dict]) -> None:
    """Génère les graphiques de comparaison entre modèles."""
    if len(results) < 2:
        return

    sample = next(iter(results.values()), {})
    is_classif = "test_accuracy" in sample or "val_accuracy" in sample

    if is_classif:
        for metric in ["test_accuracy", "val_accuracy", "test_f1", "val_f1",
                       "test_roc_auc", "test_auc", "val_auc", "best_val_auc"]:
            filtered = {k: v for k, v in results.items() if metric in v}
            if len(filtered) >= 2:
                plot_model_comparison(filtered, metric, ARTIFACTS,
                                      f"Comparaison — {metric}")
    else:
        for metric in ["test_r2", "test_mae", "test_rmse"]:
            filtered = {k: v for k, v in results.items() if metric in v}
            if len(filtered) >= 2:
                plot_model_comparison(filtered, metric, ARTIFACTS,
                                      f"Comparaison — {metric}")

def _save_metrics(metrics: dict, model_name: str, tag: str) -> None:
    """Sauvegarde les métriques en JSON."""
    path = ARTIFACTS / f"{model_name}_metrics_{tag}.json"
    clean = {k: v for k, v in metrics.items()
             if not isinstance(v, (list, np.ndarray))}
    path.write_text(json.dumps(clean, indent=2, default=str))

def _store_to_db(
    pred_df:     pd.DataFrame,
    metrics:     dict,
    model_name:  str,
    feature_set: str,
    target:      str,
    tag:         str,
) -> None:
    """Stocke prédictions + métriques dans PostgreSQL si DATABASE_URL est défini."""
    db_url = os.environ.get("DATABASE_URL") or os.environ.get("DB_URL")
    if not db_url:
        host = os.environ.get("POSTGRES_HOST", "localhost")
        port = os.environ.get("POSTGRES_PORT", "5432")
        db   = os.environ.get("POSTGRES_DB",   "elections_idf")
        user = os.environ.get("POSTGRES_USER",  "etl_admin")
        pwd  = os.environ.get("POSTGRES_PASSWORD", "")
        if not pwd:
            return
        db_url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"

    from ml.training.db_store import store_predictions, store_metrics
    import datetime
    run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S") + f"_{model_name}"

    store_predictions(pred_df, model_name, feature_set, target, run_id, db_url)
    store_metrics(metrics, model_name, feature_set, target, run_id, db_url)
