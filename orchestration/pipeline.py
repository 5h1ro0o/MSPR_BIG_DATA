"""
Pipeline principal — peut être exécuté directement (python pipeline.py)
ou via Prefect (décorateurs @flow / @task).

Usage direct :
    python -X utf8 orchestration/pipeline.py

Usage Prefect :
    prefect deploy orchestration/pipeline.py:etl_flow
"""

import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from etl.extract import (
    extract_candidats_raw,
    extract_chomage_historique,
    extract_demographique,
    extract_elections,
    extract_emploi_2022,
    extract_pauvrete,
)

# Nouvelles sources — optionnelles (ignorées silencieusement si fichier absent)
try:
    from etl.extract.associations import extract_associations
    from etl.extract.entreprises import extract_entreprises
    from etl.extract.securite import extract_securite

    _NEW_SOURCES_AVAILABLE = True
except ImportError:
    _NEW_SOURCES_AVAILABLE = False
from etl.load import load_to_db, write_gold_csv
from etl.load.db_loader import check_db_connection
from etl.quality import run_quality_checks
from etl.quality.ge_suite import run_ge_suite
from etl.transform import (
    assemble_dataset,
    transform_chomage_hist,
    transform_cibles,
    transform_demographique,
    transform_emploi_2022,
    transform_historique,
    transform_participation,
    transform_pauvrete,
)
from monitoring.logger import get_logger, setup_logger
from monitoring.metrics import PipelineMetrics

setup_logger(settings.log_level, settings.log_dir)
log = get_logger(__name__)


def _run_extract(name: str, fn, *args):
    """Wrapper thread-safe pour une extraction : retourne (name, DataFrame|None)."""
    log.debug(f"[extract] {name} …")
    data = fn(*args)
    n = len(data) if data is not None else 0
    log.debug(f"[extract] {name} → {n} lignes")
    return name, data


def _persist_quality_checks(
    qc_report,
    ge_result: dict,
    run_id: str,
    database_url: str,
    started_at: "datetime" = None,
) -> None:
    """Écrit le run dans audit.pipeline_runs puis les checks dans audit.quality_checks."""
    try:
        from sqlalchemy import create_engine, text

        checks = [
            (
                "min_communes_1200",
                qc_report.stats.get("n_communes", 0) >= 1200,
                f"n={qc_report.stats.get('n_communes', 0)}",
            ),
            (
                "no_nulls",
                qc_report.stats.get("n_nulls", 1) == 0,
                f"nulls={qc_report.stats.get('n_nulls', '?')}",
            ),
            (
                "no_duplicates",
                qc_report.stats.get("n_duplicates", 1) == 0,
                f"dups={qc_report.stats.get('n_duplicates', '?')}",
            ),
            (
                "all_idf_depts",
                len(qc_report.errors) == 0
                or not any("dept" in e.lower() for e in qc_report.errors),
                f"depts={qc_report.stats.get('depts_found', [])}",
            ),
            (
                "cible_cols_present",
                not any("cible" in e.lower() for e in qc_report.errors),
                "colonnes cibles présentes",
            ),
            (
                "t2_pct_coherence",
                not any("t2" in e.lower() for e in qc_report.errors),
                "cohérence T2 OK",
            ),
            (
                "value_ranges_ok",
                not any(
                    "plage" in e.lower() or "range" in e.lower()
                    for e in qc_report.errors
                ),
                "plages de valeurs valides",
            ),
            (
                "qc_global_passed",
                qc_report.passed,
                (
                    "OK"
                    if qc_report.passed
                    else f"Erreurs: {'; '.join(qc_report.errors[:3])}"
                ),
            ),
        ]

        ge_total = ge_result.get("total", 0)
        ge_passed = ge_result.get("passed", 0)
        if not ge_result.get("skipped"):
            checks.append(
                (
                    f"great_expectations_{ge_passed}_sur_{ge_total}",
                    ge_result.get("success", False),
                    f"{ge_passed}/{ge_total} expectations passées",
                )
            )

        engine = create_engine(database_url, pool_pre_ping=True)
        now = datetime.utcnow()
        t_start = started_at or now
        with engine.begin() as conn:
            # Upsert pipeline_runs first so quality_checks FK is satisfied
            conn.execute(
                text(
                    "INSERT INTO audit.pipeline_runs "
                    "(run_id, started_at, ended_at, status, qc_passed) "
                    "VALUES (:rid, :sa, :ea, :st, :qp) "
                    "ON CONFLICT (run_id) DO UPDATE SET "
                    "ended_at=EXCLUDED.ended_at, status=EXCLUDED.status, qc_passed=EXCLUDED.qc_passed"
                ),
                {
                    "rid": run_id,
                    "sa": t_start,
                    "ea": now,
                    "st": "SUCCESS" if qc_report.passed else "QC_FAIL",
                    "qp": bool(qc_report.passed),
                },
            )
            for check_name, passed, details in checks:
                conn.execute(
                    text(
                        "INSERT INTO audit.quality_checks (run_id, check_name, passed, details, checked_at) "
                        "VALUES (:rid, :cn, :p, :d, :at)"
                    ),
                    {
                        "rid": run_id,
                        "cn": check_name,
                        "p": bool(passed),
                        "d": details,
                        "at": now,
                    },
                )
        log.info(f"[QC] {len(checks)} contrôles persistés dans audit.quality_checks")
    except Exception as exc:
        log.warning(f"[QC] Impossible d'écrire dans audit.quality_checks : {exc}")


def run_pipeline() -> dict:
    """
    Exécute le pipeline ETL complet :
      Extract (parallèle) → Transform → Quality (checks.py + Great Expectations)
      → Load (CSV + optionnel PostgreSQL)

    Returns:
        Dictionnaire de métriques du run.
    """
    pipeline_started_at = datetime.utcnow()
    run_id = pipeline_started_at.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
    metrics = PipelineMetrics(run_id)

    log.info(f"=== PIPELINE START | run_id={run_id} ===")

    # ── Phase 1 : Extractions PARALLÈLES ─────────────────────────────────────
    # Les 5 sources principales sont indépendantes (fichiers distincts).
    # ThreadPoolExecutor les exécute simultanément — traitement distribué.
    step = metrics.start_step("extract_parallel")
    log.debug("[extract] Lancement de 5 extractions en parallèle …")

    primary_tasks = {
        "elections": (extract_elections, settings.elections_dir),
        "demographique": (extract_demographique, settings.demographique_file),
        "pauvrete": (extract_pauvrete, settings.pauvrete_file),
        "chomage_hist": (extract_chomage_historique, settings.chomage_hist_file),
        "emploi": (extract_emploi_2022, settings.emploi_file),
    }

    extracts: dict = {}
    with ThreadPoolExecutor(max_workers=5, thread_name_prefix="etl_extract") as pool:
        futures = {
            pool.submit(_run_extract, name, fn, *args): name
            for name, (fn, *args) in primary_tasks.items()
        }
        for future in as_completed(futures):
            name, data = future.result()
            extracts[name] = data

    step.rows_out = sum(len(v) for v in extracts.values())
    step.finish()

    raw_elections = extracts["elections"]
    raw_demo = extracts["demographique"]
    raw_pauvrete = extracts["pauvrete"]
    raw_chomage = extracts["chomage_hist"]
    raw_emploi = extracts["emploi"]

    # ── Phase 2 : Nouvelles sources optionnelles (parallèles entre elles) ────
    if _NEW_SOURCES_AVAILABLE:
        step = metrics.start_step("extract_new_sources_parallel")
        log.debug("[extract] Nouvelles sources optionnelles en parallèle …")

        optional_tasks = {
            "securite": (extract_securite, settings.data_root),
            "entreprises": (extract_entreprises, settings.data_root),
            "associations": (extract_associations, settings.data_root),
        }

        opt_results: dict = {}
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="etl_opt") as pool:
            futures = {
                pool.submit(_run_extract, name, fn, *args): name
                for name, (fn, *args) in optional_tasks.items()
            }
            for future in as_completed(futures):
                name, data = future.result()
                opt_results[name] = data

        step.rows_out = sum(len(v) for v in opt_results.values() if v is not None)
        step.finish()

    # ── Phase 3 : Extractions candidats (même fichier source → séquentiel) ──
    step = metrics.start_step("extract_candidats_historique")
    raw_cands_hist = extract_candidats_raw(
        settings.candidats_file,
        elections_filter=[
            "2012_pres_t1",
            "2012_pres_t2",
            "2017_pres_t1",
            "2017_pres_t2",
        ],
        chunk_size=settings.chunk_size,
    )
    step.rows_out = len(raw_cands_hist)
    step.finish()

    step = metrics.start_step("extract_candidats_2022")
    raw_cands_2022 = extract_candidats_raw(
        settings.candidats_file,
        elections_filter=["2022_pres_t1", "2022_pres_t2"],
        chunk_size=settings.chunk_size,
    )
    step.rows_out = len(raw_cands_2022)
    step.finish()

    # ── Phase 4 : Transformations (séquentielles — dépendances entre étapes) ─
    step = metrics.start_step("transform_participation")
    silver_participation = transform_participation(raw_elections)
    step.rows_out = len(silver_participation)
    step.finish()

    step = metrics.start_step("transform_demographique")
    silver_demo = transform_demographique(raw_demo)
    step.rows_out = len(silver_demo)
    step.finish()

    step = metrics.start_step("transform_pauvrete")
    silver_pauvrete = transform_pauvrete(raw_pauvrete)
    step.rows_out = len(silver_pauvrete)
    step.finish()

    step = metrics.start_step("transform_chomage")
    silver_chomage = transform_chomage_hist(raw_chomage)
    step.rows_out = len(silver_chomage)
    step.finish()

    step = metrics.start_step("transform_emploi")
    silver_emploi = transform_emploi_2022(raw_emploi)
    step.rows_out = len(silver_emploi)
    step.finish()

    step = metrics.start_step("transform_historique")
    silver_hist = transform_historique(raw_cands_hist)
    step.rows_out = len(silver_hist)
    step.finish()

    step = metrics.start_step("transform_cibles")
    silver_cibles = transform_cibles(raw_cands_2022)
    step.rows_out = len(silver_cibles)
    step.finish()

    step = metrics.start_step("assemble_gold")
    gold_dataset = assemble_dataset(
        participation=silver_participation,
        demographique=silver_demo,
        pauvrete=silver_pauvrete,
        chomage=silver_chomage,
        emploi=silver_emploi,
        historique=silver_hist,
        cibles=silver_cibles,
        iqr_factor=settings.iqr_factor,
    )
    step.rows_out = len(gold_dataset)
    step.cols_out = len(gold_dataset.columns)
    step.finish()

    # ── Phase 5 : Qualité — checks maison + Great Expectations (DQM) ─────────
    step = metrics.start_step("quality_checks")
    qc_report = run_quality_checks(gold_dataset)
    step.finish()

    if not qc_report.passed:
        log.error(f"Contrôles qualité ECHEC : {qc_report.errors}")

    step = metrics.start_step("ge_validation")
    ge_result = run_ge_suite(gold_dataset)
    step.rows_out = ge_result.get("passed", 0)
    step.finish()

    # ── Phase 6 : Load ────────────────────────────────────────────────────────
    step = metrics.start_step("load_csv")
    write_gold_csv(gold_dataset, settings.output_file)
    step.finish()

    db_ok = False
    try:
        db_ok = check_db_connection(settings.database_url)
        if db_ok:
            step = metrics.start_step("load_db")
            load_to_db(gold_dataset, settings.database_url)
            step.finish()
    except Exception as e:
        log.warning(f"Chargement PostgreSQL ignoré (non disponible) : {e}")

    # ── Persistance des quality checks dans audit.quality_checks ─────────────
    if db_ok:
        _persist_quality_checks(
            qc_report, ge_result, run_id, settings.database_url, pipeline_started_at
        )

    metrics_dict = metrics.to_dict()
    metrics_dict["run_id"] = run_id
    metrics_dict["n_communes"] = len(gold_dataset)
    metrics_dict["qc_passed"] = qc_report.passed
    metrics_dict["n_nulls"] = int(qc_report.stats.get("n_nulls", 0))
    metrics.save(settings.log_dir / "metrics")

    log.info(
        f"=== PIPELINE END | run_id={run_id} | "
        f"{len(gold_dataset)} communes × {len(gold_dataset.columns)} colonnes | "
        f"QC={'OK' if qc_report.passed else 'KO'} | "
        f"GE={ge_result.get('passed', 'n/a')}/{ge_result.get('total', 'n/a')} ==="
    )

    return metrics_dict


if __name__ == "__main__":
    result = run_pipeline()
    print(f"\nTotal duration: {result['total_duration_s']:.1f}s")
