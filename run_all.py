"""
run_all.py — Orchestrateur Docker
==================================
Enchaîne les 3 étapes au démarrage du container pipeline :

  1. FETCH   : télécharge les datasets manquants depuis data.gouv.fr
  2. ETL     : pipeline Extract → Transform → Quality → Load (CSV + PostgreSQL)
  3. TRAIN   : entraîne RF, Gradient Boosting (+ LSTM si TensorFlow disponible)
               et stocke prédictions + métriques en base

Usage :
    python -X utf8 run_all.py [--skip-fetch] [--skip-etl] [--skip-train]
    python -X utf8 run_all.py --only-train   # utile pour re-train rapide
"""

import argparse
import logging
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_all")


def step_fetch(data_root: Path) -> bool:
    log.info("━━━ ÉTAPE 1/3 : FETCH data.gouv.fr ━━━")
    t0 = time.time()
    try:
        from etl.extract.datagouv import fetch_all_datasets

        results = fetch_all_datasets(data_root)
        ok_count = sum(v for v in results.values())
        log.info(
            f"Fetch terminé en {time.time()-t0:.1f}s — {ok_count}/{len(results)} datasets OK"
        )
        return True
    except Exception as e:
        log.error(f"Fetch ERREUR : {e}", exc_info=True)
        return False


def step_etl() -> bool:
    log.info("━━━ ÉTAPE 2/3 : ETL (Extract → Transform → Load) ━━━")
    t0 = time.time()
    try:
        from orchestration.pipeline import run_pipeline

        result = run_pipeline()
        log.info(
            f"ETL terminé en {time.time()-t0:.1f}s — "
            f"total={result.get('total_duration_s', 0):.1f}s"
        )
        return True
    except Exception as e:
        log.error(f"ETL ERREUR : {e}", exc_info=True)
        return False


def step_train() -> bool:
    log.info("━━━ ÉTAPE 3/3 : ENTRAÎNEMENT ML ━━━")
    t0 = time.time()
    try:
        from ml.training.trainer import (
            train_random_forest,
            train_gradient_boosting,
            train_lstm,
            compare_all_models,
        )

        results: dict = {}

        for fset in ("pre_vote", "post_t1"):
            log.info(f"  Random Forest — feature_set={fset}")
            try:
                m = train_random_forest(
                    feature_set=fset,
                    target="classification_t2",
                    fast_search=True,
                    n_iter=20,
                )
                results[f"rf_{fset}"] = m
            except Exception as e:
                log.warning(f"  RF {fset} ERREUR: {e}")

            log.info(f"  Gradient Boosting — feature_set={fset}")
            try:
                m = train_gradient_boosting(
                    feature_set=fset, target="classification_t2", use_search=False
                )
                results[f"gb_{fset}"] = m
            except Exception as e:
                log.warning(f"  GB {fset} ERREUR: {e}")

        try:
            log.info("  LSTM …")
            m = train_lstm(target="classification_t2")
            if m:
                results["lstm"] = m
        except ImportError:
            log.info("  TensorFlow absent — LSTM ignoré")

        if len(results) > 1:
            compare_all_models(results)

        log.info(
            f"Entraînement terminé en {time.time()-t0:.1f}s — {len(results)} modèles"
        )
        return True
    except Exception as e:
        log.error(f"Entraînement ERREUR : {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Orchestrateur pipeline complet")
    parser.add_argument(
        "--skip-fetch", action="store_true", help="Saute le fetch data.gouv.fr"
    )
    parser.add_argument("--skip-etl", action="store_true", help="Saute le pipeline ETL")
    parser.add_argument(
        "--skip-train", action="store_true", help="Saute l'entraînement ML"
    )
    parser.add_argument(
        "--only-train",
        action="store_true",
        help="Lance uniquement l'entraînement (équivalent --skip-fetch --skip-etl)",
    )
    args = parser.parse_args()

    if args.only_train:
        args.skip_fetch = True
        args.skip_etl = True

    from config.settings import settings

    data_root = settings.data_root
    log.info(f"DATA_ROOT = {data_root}")

    t_total = time.time()
    success = True

    if not args.skip_fetch:
        ok = step_fetch(data_root)
        if not ok:
            log.warning("Fetch partiel — l'ETL peut échouer si des fichiers manquent")

    if not args.skip_etl:
        ok = step_etl()
        if not ok:
            log.error("ETL échoué — arrêt du pipeline")
            sys.exit(1)

    if not args.skip_train:
        ok = step_train()
        success = success and ok

    elapsed = time.time() - t_total
    status = "SUCCESS" if success else "PARTIAL"
    log.info(f"━━━ PIPELINE {status} | {elapsed:.0f}s total ━━━")

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
