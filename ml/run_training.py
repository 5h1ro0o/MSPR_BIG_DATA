"""
Script principal d'entraînement des modèles ML.

Usage :
    python -X utf8 ml/run_training.py
    python -X utf8 ml/run_training.py --model rf
    python -X utf8 ml/run_training.py --model gb --search
    python -X utf8 ml/run_training.py --model lstm

    # Chemin explicite vers le dataset (surcharge DATA_PATH)
    python -X utf8 ml/run_training.py --data-path /chemin/vers/dataset.csv

Modèles :
    rf   → Random Forest
    gb   → Gradient Boosting
    lstm → LSTM  (TensorFlow requis)
    dt   → Decision Tree  (squelette)
    mlp  → MLP            (squelette)
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import warnings

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="Entrainement ML — Elections IDF 2022")
    p.add_argument("--model", default="all", help="rf | gb | lstm | dt | mlp | all")
    p.add_argument(
        "--target",
        default="classification_t2",
        help="classification_t2 | regression_macron | regression_marge",
    )
    p.add_argument(
        "--feature-set",
        default="pre_vote",
        help="pre_vote | post_t1 | full | hist_only",
    )
    p.add_argument("--search", action="store_true", help="Activer RandomizedSearchCV")
    p.add_argument(
        "--full-search", action="store_true", help="Grille complete (plus lent)"
    )
    p.add_argument("--n-iter", type=int, default=20, help="Iterations RandomizedSearch")
    p.add_argument(
        "--no-save", action="store_true", help="Ne pas sauvegarder les modeles"
    )
    p.add_argument(
        "--data-path",
        default=None,
        help="Chemin explicite vers le dataset CSV (ecrase DATA_PATH)",
    )
    return p.parse_args()


def _resolve_dataset(data_path_arg: str | None) -> Path:
    """
    Retourne le chemin absolu du dataset à utiliser.
    Priorité : --data-path > DATA_PATH (ml/config.py).
    Lève FileNotFoundError si le fichier n'existe pas.
    """
    from ml.config import DATA_PATH

    if data_path_arg:
        path = Path(data_path_arg).resolve()
    else:
        path = DATA_PATH.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {path}\n"
            "  Lancez d'abord l'ETL : python -X utf8 etl_pipeline.py\n"
            "  Ou passez --data-path /chemin/vers/dataset.csv"
        )
    return path


def main():
    args = parse_args()
    model = args.model.lower()
    save = not args.no_save

    data_path = _resolve_dataset(args.data_path)

    if args.data_path:
        import ml.config as _cfg

        _cfg.DATA_PATH = data_path

    size_mb = data_path.stat().st_size / 1_048_576
    print(f"\n{'='*60}")
    print("  Pipeline ML — Elections IDF 2022")
    print(f"  Modele(s)   : {model}")
    print(f"  Cible       : {args.target}")
    print(f"  Feature set : {args.feature_set}")
    print(f"  Dataset     : {data_path.name}  ({size_mb:.1f} Mo)")
    print(f"  Chemin      : {data_path}")
    print(f"{'='*60}\n")

    from ml.training.trainer import (
        compare_all_models,
        train_gradient_boosting,
        train_lstm,
        train_random_forest,
    )

    results = {}

    if model in ("all", "rf", "random_forest"):
        print("\n[1/3] Random Forest")
        print("-" * 40)
        try:
            metrics = train_random_forest(
                feature_set=args.feature_set,
                target=args.target,
                fast_search=not args.full_search,
                n_iter=args.n_iter,
                save=save,
            )
            results["random_forest"] = metrics
            acc = metrics.get("val_accuracy", metrics.get("test_accuracy", "N/A"))
            print(f"  RF OK — Accuracy: {acc}")
        except Exception as e:
            print(f"  RF ERREUR : {e}")
            import traceback

            traceback.print_exc()

    if model in ("all", "gb", "gradient_boosting"):
        print("\n[2/3] Gradient Boosting")
        print("-" * 40)
        try:
            metrics = train_gradient_boosting(
                feature_set=args.feature_set,
                target=args.target,
                use_search=args.search or args.full_search,
                n_iter=args.n_iter,
                save=save,
            )
            results["gradient_boosting"] = metrics
            acc = metrics.get("val_accuracy", metrics.get("test_accuracy", "N/A"))
            print(f"  GB OK — Accuracy: {acc}")
        except Exception as e:
            print(f"  GB ERREUR : {e}")
            import traceback

            traceback.print_exc()

    if model in ("all", "lstm"):
        print("\n[3/3] LSTM")
        print("-" * 40)
        try:
            metrics = train_lstm(
                target=args.target,
                save=save,
            )
            if metrics:
                results["lstm"] = metrics
                print(
                    f"  LSTM OK — Val AUC: {metrics.get('val_auc', metrics.get('best_val_auc', 'N/A'))}"
                )
        except Exception as e:
            print(f"  LSTM ERREUR : {e}")
            import traceback

            traceback.print_exc()

    if model in ("dt", "decision_tree"):
        print("\n  [SKIP] Decision Tree — non encore implemente.")
    if model in ("mlp",):
        print("\n  [SKIP] MLP — non encore implemente.")

    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  Comparaison des {len(results)} modeles")
        print(f"{'='*60}")
        compare_all_models(results)

    from ml.config import ARTIFACTS

    print(f"\n{'='*60}")
    print("  RESUME FINAL")
    print(f"{'='*60}")
    print(f"  {'Modele':<22} {'Accuracy':>10} {'AUC':>8} {'F1':>8} {'Train(s)':>10}")
    print(f"  {'-'*60}")

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    for name, m in results.items():
        acc = m.get("val_accuracy", m.get("test_accuracy", "-"))
        auc = m.get(
            "val_auc",
            m.get("test_auc", m.get("test_roc_auc", m.get("cv_roc_auc", "-"))),
        )
        f1 = m.get("val_f1", m.get("test_f1", "-"))
        t = m.get("training_time_s", "-")
        print(f"  {name:<22} {fmt(acc):>10} {fmt(auc):>8} {fmt(f1):>8} {fmt(t):>10}")

    print(f"\n  Artefacts : {ARTIFACTS}")
    print(f"  Dataset   : {data_path}")
    print("\n  Dashboard : streamlit run ml/visualization/dashboard.py")


if __name__ == "__main__":
    main()
