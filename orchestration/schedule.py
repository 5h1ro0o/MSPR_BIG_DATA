"""
Planification du pipeline via Prefect (optionnel).
Lance le pipeline sur un schedule cron ou à la demande.

Prérequis :
    pip install prefect
    prefect server start          # démarrer le serveur local
    python orchestration/schedule.py  # enregistrer le déploiement
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from prefect import flow
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule

    from orchestration.pipeline import run_pipeline

    @flow(name="etl-elections-idf", log_prints=True)
    def etl_flow() -> dict:
        """Flow Prefect wrappant le pipeline ETL."""
        return run_pipeline()

    def deploy(cron: str = "0 2 * * *") -> None:
        """
        Déploie le flow avec un schedule cron (défaut : 2h du matin chaque jour).

        Args:
            cron: expression cron (ex. "0 2 * * 1" = lundi 2h)
        """
        deployment = Deployment.build_from_flow(
            flow=etl_flow,
            name="etl-elections-idf-daily",
            schedule=CronSchedule(cron=cron, timezone="Europe/Paris"),
            work_pool_name="default-agent-pool",
            tags=["elections", "idf", "etl"],
        )
        deployment.apply()
        print(f"Déploiement enregistré avec schedule : {cron}")

    if __name__ == "__main__":
        deploy()

except ImportError:
    print(
        "Prefect non installé. Pour activer l'orchestration :\n"
        "  pip install prefect\n"
        "  prefect server start\n"
        "  python orchestration/schedule.py"
    )
