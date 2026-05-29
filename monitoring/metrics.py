"""
Collecte de métriques légères pour chaque étape du pipeline.
Pas de dépendance externe — stocké dans un dict en mémoire,
sérialisable en JSON pour archivage ou envoi à Prometheus/Grafana.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from monitoring.logger import get_logger

log = get_logger(__name__)


@dataclass
class StepMetric:
    step: str
    start_ts: float = field(default_factory=time.time)
    end_ts: Optional[float] = None
    rows_in: int = 0
    rows_out: int = 0
    cols_out: int = 0
    nulls_before: int = 0
    nulls_after: int = 0
    outliers_capped: int = 0
    error: Optional[str] = None

    @property
    def duration_s(self) -> float:
        if self.end_ts:
            return round(self.end_ts - self.start_ts, 2)
        return 0.0

    def finish(self) -> None:
        self.end_ts = time.time()
        log.info(
            f"[{self.step}] {self.rows_out:,} lignes | "
            f"{self.cols_out} cols | {self.duration_s}s"
        )


class PipelineMetrics:
    """Agrège les métriques de toutes les étapes d'un run."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.started_at = datetime.utcnow().isoformat()
        self.steps: list[StepMetric] = []

    def start_step(self, step: str) -> StepMetric:
        m = StepMetric(step=step)
        self.steps.append(m)
        return m

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "steps": [asdict(s) for s in self.steps],
            "total_duration_s": sum(s.duration_s for s in self.steps),
        }

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"metrics_{self.run_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        log.info(f"Métriques sauvegardées : {path}")
