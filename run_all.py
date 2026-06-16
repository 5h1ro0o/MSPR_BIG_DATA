"""
run_all.py — Orchestrateur Docker
==================================
Enchaîne les 4 étapes au démarrage du container pipeline :

  1. FETCH    : télécharge les datasets manquants depuis data.gouv.fr
  2. ETL      : pipeline Extract → Transform → Quality → Load (CSV + PostgreSQL)
  3. DATAMART : alimente le schéma en étoile (datamart analytique)
  4. TRAIN    : entraîne RF, Gradient Boosting (+ LSTM si TensorFlow disponible),
                génère projections T+1/T+2/T+3 et stocke métriques en base

Usage :
    python -X utf8 run_all.py
    python -X utf8 run_all.py --only-train   # utile pour re-train rapide
"""

import argparse
import datetime
import json
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
    log.info("━━━ ÉTAPE 1/4 : FETCH data.gouv.fr ━━━")
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


def step_etl() -> tuple[bool, dict]:
    log.info("━━━ ÉTAPE 2/4 : ETL (Extract → Transform → Load) ━━━")
    t0 = time.time()
    try:
        from orchestration.pipeline import run_pipeline

        result = run_pipeline()
        log.info(
            f"ETL terminé en {time.time()-t0:.1f}s — "
            f"total={result.get('total_duration_s', 0):.1f}s"
        )
        return True, result
    except Exception as e:
        log.error(f"ETL ERREUR : {e}", exc_info=True)
        return False, {}


def step_datamart() -> bool:
    log.info("━━━ ÉTAPE 3/4 : DATAMART (schéma en étoile) ━━━")
    t0 = time.time()
    try:
        from db.populate_datamart import run as populate_datamart

        ok = populate_datamart()
        elapsed = time.time() - t0
        if ok:
            log.info(f"Datamart alimenté en {elapsed:.1f}s")
        else:
            log.warning(
                "Datamart non alimenté (CSV ou DB indisponible) — pipeline continue"
            )
        return True  # non-bloquant : le frontend fonctionne sans le datamart
    except Exception as e:
        log.warning(f"Datamart ERREUR (non-bloquant) : {e}")
        return True


def step_train() -> bool:
    log.info("━━━ ÉTAPE 4/4 : ENTRAÎNEMENT ML ━━━")
    t0 = time.time()
    try:
        from ml.training.trainer import (
            compare_all_models,
            train_decision_tree,
            train_gradient_boosting,
            train_lstm,
            train_mlp,
            train_random_forest,
        )

        results: dict = {}

        # Régression sur le % Macron T2 — en IDF 2022, Macron a gagné dans TOUTES les
        # communes (vote majoritaire), donc cible_t2_vainqueur = toujours 0 (1 seule classe).
        # La classification est impossible sur ce dataset IDF.
        # La régression prédit la marge de Macron (~50-90% selon les communes),
        # ce qui donne des métriques R²/MAE/RMSE significatives et réelles.
        TARGET = "regression_macron"

        for fset in ("pre_vote", "post_t1"):
            log.info(f"  Random Forest — feature_set={fset}")
            try:
                m = train_random_forest(
                    feature_set=fset,
                    target=TARGET,
                    fast_search=True,
                    n_iter=15,   # réduit pour éviter OOM Docker (150 fits × RF vs 300)
                )
                results[f"rf_{fset}"] = m
            except Exception as e:
                log.warning(f"  RF {fset} ERREUR: {e}")

            log.info(f"  Gradient Boosting — feature_set={fset}")
            try:
                m = train_gradient_boosting(
                    feature_set=fset, target=TARGET, use_search=False
                )
                results[f"gb_{fset}"] = m
            except Exception as e:
                log.warning(f"  GB {fset} ERREUR: {e}")

        if "regression" in TARGET:
            log.info("  LSTM ignoré — supporte uniquement la classification (stratify incompatible avec régression)")
        else:
            try:
                log.info("  LSTM …")
                m = train_lstm(target=TARGET)
                if m:
                    results["lstm"] = m
            except ImportError:
                log.info("  TensorFlow absent — LSTM ignoré")

        log.info("  Decision Tree — feature_set=pre_vote")
        try:
            m = train_decision_tree(feature_set="pre_vote", target=TARGET)
            results["decision_tree"] = m
        except Exception as e:
            log.warning(f"  DT ERREUR: {e}")

        log.info("  MLP — feature_set=pre_vote")
        try:
            m = train_mlp(
                feature_set="pre_vote",
                target=TARGET,
                use_grid_search=False,
            )
            results["mlp"] = m
        except Exception as e:
            log.warning(f"  MLP ERREUR: {e}")

        # Projections temporelles T+1/T+2/T+3 (après entraînement GB pré-vote)
        log.info("  Projections temporelles (T+1 / T+2 / T+3)…")
        try:
            from ml.projection.generate_projections import generate as gen_projections

            gen_projections()
        except Exception as e:
            log.warning(f"  Projections ERREUR: {e}")

        if len(results) > 1:
            compare_all_models(results)

        log.info(
            f"Entraînement terminé en {time.time()-t0:.1f}s — {len(results)} modèles"
        )
        return True
    except Exception as e:
        log.error(f"Entraînement ERREUR : {e}", exc_info=True)
        return False


def _write_run_artifact(t_start: float, elapsed: float, success: bool, etl_meta: dict) -> None:
    """Écrit ml/artifacts/pipeline_run.json pour le frontend."""
    try:
        start_dt = datetime.datetime.utcfromtimestamp(t_start)
        run_id = etl_meta.get("run_id") or start_dt.strftime("%Y%m%dT%H%M%S") + "_000000"
        artifact = {
            "id":       run_id,
            "start":    start_dt.strftime("%d/%m %H:%M"),
            "dur":      f"{int(elapsed // 60)}m {int(elapsed % 60)}s",
            "status":   "SUCCESS" if success else "PARTIAL",
            "communes": etl_meta.get("n_communes", 0),
            "qc":       "OK" if etl_meta.get("qc_passed", False) else "KO",
            "nulls":    etl_meta.get("n_nulls", 0),
        }
        artifacts_dir = PROJECT_ROOT / "ml" / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        out = artifacts_dir / "pipeline_run.json"
        out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(f"Artifact run écrit : {out}")
    except Exception as e:
        log.warning(f"Impossible d'écrire pipeline_run.json : {e}")


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

    etl_meta: dict = {}
    if not args.skip_etl:
        ok, etl_meta = step_etl()
        if not ok:
            log.error("ETL échoué — arrêt du pipeline")
            sys.exit(1)
        step_datamart()

    if not args.skip_train:
        ok = step_train()
        success = success and ok

    elapsed = time.time() - t_total
    status = "SUCCESS" if success else "PARTIAL"
    log.info(f"━━━ PIPELINE {status} | {elapsed:.0f}s total ━━━")

    _write_run_artifact(t_total, elapsed, success, etl_meta)

    log.info("")
    log.info("┌─────────────────────────────────────────────────────────────┐")
    log.info("│                        ✓  READY                            │")
    log.info("│   Frontend  →  http://localhost:3000                        │")
    log.info("│   Grafana   →  http://localhost:3001                        │")
    log.info("└─────────────────────────────────────────────────────────────┘")
    log.info("")

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
