"""
LSTM — implémentation complète avec TensorFlow / Keras.
Adapté du script de référence : a implementer/lstm/train_lstm.py

Architecture duale :
  - Branch LSTM  : séquences temporelles 2012 → 2017 → 2022-T1
                   (chaque pas = 8 features électorales homogènes)
  - Branch Dense : features socio-économiques statiques (17 features)
  - Fusion → Dense → Sigmoid (classification binaire)

Séquence de formation des inputs :
  X doit être un DataFrame contenant toutes les colonnes LSTM
  (définies dans FEATURE_SETS["lstm"]).
  _prepare_inputs() sépare séquences et socio, normalise, reshape.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from ml.config import (
    ARTIFACTS,
    FEATURES_SOCIO_LSTM,
    LSTM_CONFIG,
    LSTM_PERIOD_FEATURES,
    RANDOM_STATE,
)
from ml.models.base import BaseModel

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    import tensorflow as tf

    try:
        from keras import Input, Model
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from keras.layers import LSTM as KerasLSTM
        from keras.layers import (
            BatchNormalization,
            Concatenate,
            Dense,
            Dropout,
        )
        from keras.metrics import AUC as KerasAUC
        from keras.optimizers import Adam
    except Exception:
        from tensorflow.keras import Input, Model
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.layers import LSTM as KerasLSTM
        from tensorflow.keras.layers import (
            BatchNormalization,
            Concatenate,
            Dense,
            Dropout,
        )
        from tensorflow.keras.optimizers import Adam

        KerasAUC = tf.keras.metrics.AUC

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARN] TensorFlow non disponible. Installez avec : pip install tensorflow")

_PERIOD_MAP = {
    "2012": {
        "pct_lepen": "h12_t1_pct_lepen",
        "pct_gauche": "h12_t1_pct_hollande",
        "pct_droite": "h12_t1_pct_sarkozy",
        "pct_centre": "h12_t1_pct_bayrou",
        "pct_melenchon": "h12_t1_pct_melenchon",
        "pct_autres": "h12_t1_pct_autres",
        "marge_t2": "h12_t2_marge",
        "vainqueur_t2": "h12_t2_vainqueur",
    },
    "2017": {
        "pct_lepen": "h17_t1_pct_lepen",
        "pct_gauche": "h17_t1_pct_hamon",
        "pct_droite": "h17_t1_pct_fillon",
        "pct_centre": "h17_t1_pct_macron",
        "pct_melenchon": "h17_t1_pct_melenchon",
        "pct_autres": "h17_t1_pct_autres",
        "marge_t2": "h17_t2_marge",
        "vainqueur_t2": "h17_t2_vainqueur",
    },
    "2022_t1": {
        "pct_lepen": "cible_t1_pct_lepen",
        "pct_gauche": "cible_t1_pct_jadot",
        "pct_droite": "cible_t1_pct_pecresse",
        "pct_centre": "cible_t1_pct_macron",
        "pct_melenchon": "cible_t1_pct_melenchon",
        "pct_autres": "cible_t1_pct_zemmour",
        "marge_t2": None,
        "vainqueur_t2": None,
    },
}
N_ELEC_FEATURES = len(LSTM_PERIOD_FEATURES)


class LSTMModel(BaseModel):
    """
    LSTM dual-input pour la prédiction électorale IDF 2022.

    Input A : séquences temporelles (batch, 3, 8)
    Input B : features socio-économiques statiques (batch, 17)
    Output  : probabilité 0-1 (classification) OU % Macron continu (régression)
    """

    name = "lstm"

    def __init__(
        self,
        artifact_dir: Path = ARTIFACTS,
        config: Optional[dict] = None,
        task: str = "classification",
    ):
        super().__init__(artifact_dir)
        self.config = config or LSTM_CONFIG.copy()
        self.task = task
        self.keras_model = None
        self.scaler_seq = StandardScaler()
        self.scaler_soc = StandardScaler()
        self.history_ = {}

    def _require_tf(self):
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow est requis pour LSTMModel.\n"
                "Installez avec : pip install tensorflow"
            )

    def _build_sequences(self, X: pd.DataFrame) -> np.ndarray:
        """
        Construit les séquences 3D (n, 3, 8) depuis le DataFrame.
        Chaque pas de temps représente une élection.
        """
        steps = []
        for period, col_map in _PERIOD_MAP.items():
            step_data = np.zeros((len(X), N_ELEC_FEATURES), dtype=np.float32)
            for i, feat_name in enumerate(LSTM_PERIOD_FEATURES):
                src_col = col_map.get(feat_name)
                if src_col and src_col in X.columns:
                    step_data[:, i] = X[src_col].fillna(0).values
            steps.append(step_data)

        return np.stack(steps, axis=1).astype(np.float32)

    def _get_socio(self, X: pd.DataFrame) -> np.ndarray:
        """Extrait les features socio-économiques disponibles."""
        available = [f for f in FEATURES_SOCIO_LSTM if f in X.columns]
        return X[available].fillna(0).values.astype(np.float32)

    def _prepare_inputs(
        self,
        X: pd.DataFrame,
        fit_scalers: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retourne (X_seq_scaled, X_socio_scaled) prêts pour Keras.
        Si fit_scalers=True, ajuste les scalers sur ces données.
        """
        X_seq = self._build_sequences(X)
        X_socio = self._get_socio(X)
        n, seq_len, n_feat = X_seq.shape

        X_seq_flat = X_seq.reshape(-1, n_feat)
        if fit_scalers:
            self.scaler_seq.fit(X_seq_flat)
            self.scaler_soc.fit(X_socio)

        X_seq_scaled = self.scaler_seq.transform(X_seq_flat).reshape(n, seq_len, n_feat)
        X_socio_scaled = self.scaler_soc.transform(X_socio)
        return X_seq_scaled.astype(np.float32), X_socio_scaled.astype(np.float32)

    def build(self, n_socio: int = None, **kwargs) -> "Model":
        """
        Construit le modèle Keras dual-input.
        Architecture identique au script de référence.
        """
        self._require_tf()
        cfg = self.config
        n_soc = n_socio or len([f for f in FEATURES_SOCIO_LSTM])

        tf.random.set_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        input_seq = Input(shape=(3, N_ELEC_FEATURES), name="input_sequences")
        x = KerasLSTM(cfg["lstm_units_1"], return_sequences=True, name="lstm_1")(
            input_seq
        )
        x = BatchNormalization()(x)
        x = Dropout(cfg["dropout"])(x)
        x = KerasLSTM(cfg["lstm_units_2"], return_sequences=False, name="lstm_2")(x)
        x = BatchNormalization()(x)
        x = Dropout(cfg["dropout"])(x)
        x = Dense(cfg["dense_units_2"], activation="relu", name="lstm_dense")(x)

        input_socio = Input(shape=(n_soc,), name="input_socio")
        s = Dense(cfg["dense_units_1"], activation="relu")(input_socio)
        s = BatchNormalization()(s)
        s = Dropout(cfg["dropout_socio"])(s)
        s = Dense(cfg["dense_units_2"], activation="relu")(s)

        merged = Concatenate()([x, s])
        merged = Dense(cfg["merged_units"], activation="relu")(merged)
        merged = Dropout(cfg["dropout_socio"])(merged)

        if self.task == "regression":
            output = Dense(1, activation="linear", name="output")(merged)
            keras_model = Model(inputs=[input_seq, input_socio], outputs=output)
            keras_model.compile(
                optimizer=Adam(learning_rate=cfg["learning_rate"]),
                loss="mse",
                metrics=["mae"],
            )
        else:
            output = Dense(1, activation="sigmoid", name="output")(merged)
            keras_model = Model(inputs=[input_seq, input_socio], outputs=output)
            keras_model.compile(
                optimizer=Adam(learning_rate=cfg["learning_rate"]),
                loss="binary_crossentropy",
                metrics=["accuracy", KerasAUC(name="auc")],
            )
        return keras_model

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> dict:
        """
        Entraîne le LSTM dual-input.

        Args:
            X_train: DataFrame contenant TOUTES les colonnes nécessaires
                     (features électorales + socio-éco, sans les cibles T2)
            y_train: cible binaire (0=Macron, 1=Le Pen)
        """
        self._require_tf()
        cfg = self.config

        print(f"\n{'='*60}")
        print(f"  LSTM — {self.task.upper()} (dual-input)")
        print(f"  Train : {len(X_train):,} communes | Séquences : 2012→2017→2022-T1")
        print(f"{'='*60}")

        t0 = time.time()

        X_seq_tr, X_soc_tr = self._prepare_inputs(X_train, fit_scalers=True)
        y_tr = y_train.values.astype(np.float32)

        validation_data = None
        if X_val is not None and y_val is not None:
            X_seq_v, X_soc_v = self._prepare_inputs(X_val, fit_scalers=False)
            validation_data = ([X_seq_v, X_soc_v], y_val.values.astype(np.float32))

        n_socio = X_soc_tr.shape[1]
        self.keras_model = self.build(n_socio=n_socio)
        print("\n  Architecture :")
        self.keras_model.summary(print_fn=lambda x: print(f"    {x}"))

        if self.task == "regression":
            monitor_metric = "val_mae" if validation_data else "mae"
            es_mode = "min"
            fit_kwargs: dict = {}
        else:
            n_total = len(y_tr)
            n_macron = int((y_tr == 0).sum())
            n_lepen  = int((y_tr == 1).sum())
            fit_kwargs = {
                "class_weight": {
                    0: n_total / (2 * max(n_macron, 1)),
                    1: n_total / (2 * max(n_lepen, 1)),
                }
            }
            print(f"  Macron : {n_macron} | Le Pen : {n_lepen} | Poids : {fit_kwargs['class_weight']}")
            monitor_metric = "val_auc" if validation_data else "auc"
            es_mode = "max"

        callbacks = [
            EarlyStopping(
                monitor=monitor_metric,
                patience=cfg["patience"],
                restore_best_weights=True,
                mode=es_mode,
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if validation_data else "loss",
                factor=cfg["lr_factor"],
                patience=cfg["lr_patience"],
                min_lr=cfg["min_lr"],
            ),
        ]

        print(f"\n  Entraînement (max {cfg['epochs']} époques, patience={cfg['patience']})...")
        hist = self.keras_model.fit(
            [X_seq_tr, X_soc_tr],
            y_tr,
            validation_data=validation_data,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            callbacks=callbacks,
            verbose=1,
            **fit_kwargs,
        )

        self.history_ = hist.history
        elapsed = round(time.time() - t0, 2)
        print(f"\n  Terminé en {elapsed}s | {len(hist.history['loss'])} époques")

        y_raw_tr = self.keras_model.predict([X_seq_tr, X_soc_tr], verbose=0).flatten()

        if self.task == "regression":
            from sklearn.metrics import mean_absolute_error as _mae, r2_score as _r2
            self.metrics = {
                "train_mae": round(float(_mae(y_tr, y_raw_tr)), 4),
                "train_r2":  round(float(_r2(y_tr, y_raw_tr)), 4),
                "epochs_trained": len(hist.history["loss"]),
                "training_time_s": elapsed,
            }
            if validation_data:
                y_raw_v = self.keras_model.predict([X_seq_v, X_soc_v], verbose=0).flatten()
                from sklearn.metrics import mean_squared_error as _mse
                self.metrics.update({
                    "val_r2":   round(float(_r2(y_val.values, y_raw_v)), 4),
                    "val_mae":  round(float(_mae(y_val.values, y_raw_v)), 4),
                    "val_rmse": round(float(np.sqrt(_mse(y_val.values, y_raw_v))), 4),
                })
                print(f"  Val R²  : {self.metrics['val_r2']:.4f}")
                print(f"  Val MAE : {self.metrics['val_mae']:.4f}")
        else:
            y_pred_tr = (y_raw_tr > 0.5).astype(int)
            best_val_auc = max(hist.history.get("val_auc", hist.history.get("auc", [0])))
            self.metrics = {
                "train_accuracy": round(accuracy_score(y_tr, y_pred_tr), 4),
                "train_auc":      round(roc_auc_score(y_tr, y_raw_tr), 4),
                "best_val_auc":   round(float(best_val_auc), 4),
                "epochs_trained": len(hist.history["loss"]),
                "training_time_s": elapsed,
            }
            if validation_data:
                y_raw_v = self.keras_model.predict([X_seq_v, X_soc_v], verbose=0).flatten()
                y_pred_v = (y_raw_v > 0.5).astype(int)
                self.metrics.update({
                    "val_accuracy": round(accuracy_score(y_val, y_pred_v), 4),
                    "val_auc":      round(roc_auc_score(y_val, y_raw_v), 4),
                    "val_f1":       round(f1_score(y_val, y_pred_v, zero_division=0), 4),
                })
                print(f"  Val Accuracy : {self.metrics['val_accuracy']:.4f}")
                print(f"  Val AUC      : {self.metrics['val_auc']:.4f}")
                print(classification_report(
                    y_val, y_pred_v,
                    target_names=["Macron (0)", "Le Pen (1)"],
                    zero_division=0,
                ))

        self.feature_names = list(X_train.columns)
        self.model = self.keras_model
        self.is_trained = True
        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError(
                "Modèle non entraîné. Appelez train() ou load() d'abord."
            )
        X_seq, X_soc = self._prepare_inputs(X, fit_scalers=False)
        raw = self.keras_model.predict([X_seq, X_soc], verbose=0).flatten()
        if self.task == "regression":
            return raw
        return (raw > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("Modèle non entraîné.")
        X_seq, X_soc = self._prepare_inputs(X, fit_scalers=False)
        raw = self.keras_model.predict([X_seq, X_soc], verbose=0).flatten()
        if self.task == "regression":
            return raw
        return np.column_stack([1 - raw, raw])

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        X_seq, X_soc = self._prepare_inputs(X_test, fit_scalers=False)
        raw = self.keras_model.predict([X_seq, X_soc], verbose=0).flatten()

        if self.task == "regression":
            from sklearn.metrics import (
                mean_absolute_error as _mae,
                mean_squared_error as _mse,
                r2_score as _r2,
                balanced_accuracy_score as _balacc,
            )
            y_true_bin = (y_test.values < 50).astype(int)
            y_pred_bin = (raw < 50).astype(int)
            metrics = {
                "test_r2":                round(float(_r2(y_test, raw)), 4),
                "test_mae":               round(float(_mae(y_test, raw)), 4),
                "test_rmse":              round(float(np.sqrt(_mse(y_test, raw))), 4),
                "test_balanced_accuracy": round(float(_balacc(y_true_bin, y_pred_bin)), 4),
            }
            self.metrics.update(metrics)
            print(f"\n  Test R²   : {metrics['test_r2']:.4f}")
            print(f"  Test MAE  : {metrics['test_mae']:.4f} pp")
            print(f"  Test RMSE : {metrics['test_rmse']:.4f} pp")
            return metrics

        y_pred = (raw > 0.5).astype(int)
        metrics = {
            "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
            "test_auc": round(roc_auc_score(y_test, raw), 4),
            "test_f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "test_confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        self.metrics.update(metrics)
        print(f"\n  Test Accuracy : {metrics['test_accuracy']:.4f}")
        print(f"  Test AUC      : {metrics['test_auc']:.4f}")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=["Macron (0)", "Le Pen (1)"],
                zero_division=0,
            )
        )
        return metrics

    def plot_training_curves(self) -> Path:
        """
        Courbes d'entraînement loss / accuracy / AUC par époque.
        Fidèle au script de référence.
        """
        if not self.history_:
            raise RuntimeError("Pas d'historique d'entraînement.")

        h = self.history_
        n_plots = sum(
            [
                "loss" in h,
                "accuracy" in h,
                "auc" in h,
            ]
        )

        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        fig.suptitle(
            "Entraînement LSTM — Élections 2022 IDF", fontsize=14, fontweight="bold"
        )

        plot_idx = 0
        for metric, title in [
            ("loss", "Loss (Binary Crossentropy)"),
            ("accuracy", "Accuracy"),
            ("auc", "AUC-ROC"),
        ]:
            if metric not in h:
                continue
            ax = axes[plot_idx]
            ax.plot(h[metric], label="Train", color="steelblue")
            if f"val_{metric}" in h:
                ax.plot(h[f"val_{metric}"], label="Validation", color="coral")
            ax.set_title(title)
            ax.set_xlabel("Époque")
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(alpha=0.3)
            plot_idx += 1

        plt.tight_layout()
        path = self.artifact_dir / "lstm_training_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Courbes sauvegardées : {path}")
        return path

    def plot_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series) -> Path:
        """Matrice de confusion sur le jeu de test."""
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Macron prédit", "Le Pen prédit"])
        ax.set_yticklabels(["Macron réel", "Le Pen réel"])
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=20,
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        ax.set_title("Matrice de Confusion — LSTM (Test Set)")
        plt.colorbar(im)
        plt.tight_layout()
        path = self.artifact_dir / "lstm_confusion_matrix.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def save(self, tag: str = "") -> Path:
        suffix = f"_{tag}" if tag else ""
        base_path = self.artifact_dir / f"{self.name}{suffix}"

        keras_path = Path(str(base_path) + "_model.keras")
        try:
            self.keras_model.save(str(keras_path))
        except Exception:
            keras_path = Path(str(base_path) + "_model.h5")
            self.keras_model.save(str(keras_path))

        with open(str(base_path) + "_scalers.pkl", "wb") as f:
            pickle.dump({"seq": self.scaler_seq, "soc": self.scaler_soc}, f)

        meta = {
            "name": self.name,
            "task": self.task,
            "feature_names": self.feature_names,
            "metrics": {
                k: v
                for k, v in self.metrics.items()
                if not isinstance(v, (list, np.ndarray))
            },
            "config": self.config,
            "is_trained": self.is_trained,
        }
        Path(str(base_path) + "_meta.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )
        print(f"  LSTM sauvegarde : {keras_path}")
        return keras_path

    def load(self, tag: str = "") -> "LSTMModel":
        self._require_tf()
        suffix = f"_{tag}" if tag else ""
        base_path = self.artifact_dir / f"{self.name}{suffix}"

        keras_path = None
        for candidate in [
            Path(str(base_path) + "_model.keras"),
            Path(str(base_path) + "_model.h5"),
            Path(str(base_path) + "_keras"),
        ]:
            if candidate.exists():
                keras_path = candidate
                break

        scaler_path = Path(str(base_path) + "_scalers.pkl")
        meta_path = Path(str(base_path) + "_meta.json")

        if keras_path is None:
            raise FileNotFoundError(
                f"Modèle Keras non trouvé pour tag='{tag}' dans {self.artifact_dir}"
            )

        self.keras_model = tf.keras.models.load_model(str(keras_path))
        self.model = self.keras_model

        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scalers = pickle.load(f)
            self.scaler_seq = scalers["seq"]
            self.scaler_soc = scalers["soc"]

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.feature_names = meta.get("feature_names", [])
            self.metrics = meta.get("metrics", {})
            self.config = meta.get("config", self.config)
            self.is_trained = True

        print(f"  LSTM chargé depuis : {keras_path}")
        return self

    def get_predictions_with_communes(
        self,
        X: pd.DataFrame,
        commune_info: pd.DataFrame,
    ) -> pd.DataFrame:
        raw = self.predict(X)
        result = commune_info.reset_index(drop=True).copy()

        if self.task == "regression":
            result["score_macron_predit"] = raw.round(4)
            result["prediction"] = (raw < 50).astype(int)
            result["vainqueur_predit"] = pd.Series((raw < 50).astype(int)).map(
                {0: "Macron", 1: "Le Pen"}
            )
            result["proba_macron"] = (raw / 100).clip(0, 1).round(4)
            result["proba_lepen"]  = (1 - result["proba_macron"]).round(4)
        else:
            y_proba = self.predict_proba(X)
            result["prediction"] = raw
            result["vainqueur_predit"] = pd.Series(raw).map({0: "Macron", 1: "Le Pen"})
            if y_proba is not None and y_proba.ndim == 2:
                result["proba_macron"] = y_proba[:, 0].round(4)
                result["proba_lepen"]  = y_proba[:, 1].round(4)
        return result
