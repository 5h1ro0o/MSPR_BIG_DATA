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
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from monitoring.logger import setup_logger, get_logger
from monitoring.metrics import PipelineMetrics

from etl.extract import (
    extract_elections,
    extract_candidats_raw,
    extract_demographique,
    extract_chomage_historique,
    extract_emploi_2022,
    extract_pauvrete,
)
from etl.transform import (
    transform_participation,
    transform_historique,
    transform_cibles,
    transform_demographique,
    transform_chomage_hist,
    transform_emploi_2022,
    transform_pauvrete,
    assemble_dataset,
)
from etl.load import write_gold_csv, load_to_db
from etl.load.db_loader import check_db_connection
from etl.quality import run_quality_checks

setup_logger(settings.log_level, settings.log_dir)
log = get_logger(__name__)

def run_pipeline() -> dict:
    """
    Exécute le pipeline ETL complet :
      Extract → Transform → Quality → Load (CSV + optionnel PostgreSQL)

    Returns:
        Dictionnaire de métriques du run.
    """
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
    metrics = PipelineMetrics(run_id)

    log.info(f"=== PIPELINE START | run_id={run_id} ===")

    step = metrics.start_step("extract_elections")
    raw_elections = extract_elections(settings.elections_dir)
    step.rows_out = len(raw_elections)
    step.finish()

    step = metrics.start_step("extract_demographique")
    raw_demo = extract_demographique(settings.demographique_file)
    step.rows_out = len(raw_demo)
    step.finish()

    step = metrics.start_step("extract_pauvrete")
    raw_pauvrete = extract_pauvrete(settings.pauvrete_file)
    step.rows_out = len(raw_pauvrete)
    step.finish()

    step = metrics.start_step("extract_chomage_hist")
    raw_chomage = extract_chomage_historique(settings.chomage_hist_file)
    step.rows_out = len(raw_chomage)
    step.finish()

    step = metrics.start_step("extract_emploi")
    raw_emploi = extract_emploi_2022(settings.emploi_file)
    step.rows_out = len(raw_emploi)
    step.finish()

    step = metrics.start_step("extract_candidats_historique")
    raw_cands_hist = extract_candidats_raw(
        settings.candidats_file,
        elections_filter=["2012_pres_t1", "2012_pres_t2", "2017_pres_t1", "2017_pres_t2"],
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

    step = metrics.start_step("quality_checks")
    qc_report = run_quality_checks(gold_dataset)
    step.finish()

    if not qc_report.passed:
        log.error(f"Contrôles qualité ECHEC : {qc_report.errors}")

    step = metrics.start_step("load_csv")
    write_gold_csv(gold_dataset, settings.output_file)
    step.finish()

    try:
        if check_db_connection(settings.database_url):
            step = metrics.start_step("load_db")
            load_to_db(gold_dataset, settings.database_url)
            step.finish()
    except Exception as e:
        log.warning(f"Chargement PostgreSQL ignoré (non disponible) : {e}")

    metrics_dict = metrics.to_dict()
    metrics.save(settings.log_dir / "metrics")

    log.info(
        f"=== PIPELINE END | run_id={run_id} | "
        f"{len(gold_dataset):,} communes × {len(gold_dataset.columns)} colonnes | "
        f"QC={'OK' if qc_report.passed else 'KO'} ==="
    )

    return metrics_dict

if __name__ == "__main__":
    result = run_pipeline()
    print(f"\nTotal duration: {result['total_duration_s']:.1f}s")
