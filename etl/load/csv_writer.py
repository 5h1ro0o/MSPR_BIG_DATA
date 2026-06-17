"""
Chargement Gold — export CSV.
Écriture idempotente avec backup de l'ancienne version.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from monitoring.logger import get_logger

log = get_logger(__name__)


def write_gold_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    Sauvegarde le dataset gold en CSV.
    Si un fichier existe déjà, il est archivé avec un timestamp avant écrasement.

    Args:
        df:          DataFrame gold final
        output_path: chemin cible (ex. .../dataset_elections_2022_idf.csv)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        archive = output_path.with_name(f"{output_path.stem}_{ts}.csv")
        output_path.rename(archive)
        log.info(f"Ancienne version archivée : {archive.name}")

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    log.info(
        f"Dataset gold sauvegardé : {output_path} "
        f"({len(df):,} lignes × {len(df.columns)} colonnes)"
    )
