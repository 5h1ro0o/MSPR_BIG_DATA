"""
Extraction — Fichiers de participation électorale par département IDF.
Retourne les données BRUTES (Bronze) : concaténation + filtrage minimal.
"""

from pathlib import Path
import pandas as pd

from etl.helpers import norm_code
from monitoring.logger import get_logger

log = get_logger(__name__)

IDF_DEPTS = frozenset({"75", "77", "78", "91", "92", "93", "94", "95"})
DEPT_FILES = {
    "75": "elections_75.csv",
    "77": "elections_77.csv",
    "78": "elections_78.csv",
    "91": "elections_91.csv",
    "92": "elections_92.csv",
    "93": "elections_93.csv",
    "94": "elections_94.csv",
    "95": "elections_95.csv",
}


def extract_elections(elections_dir: Path) -> pd.DataFrame:
    """
    Charge tous les CSV de participation par département IDF.

    Args:
        elections_dir: répertoire contenant les fichiers elections_XX.csv

    Returns:
        DataFrame brut (bronze) — toutes élections, tous départements IDF.
    """
    frames = []
    for dept, filename in DEPT_FILES.items():
        path = elections_dir / filename
        if not path.exists():
            log.warning(f"Fichier manquant : {path}")
            continue
        df = pd.read_csv(path, low_memory=False, encoding="utf-8")
        df["code_commune"] = norm_code(df["code_commune"])
        frames.append(df)
        log.debug(f"Chargé {filename} : {len(df):,} lignes")

    if not frames:
        raise FileNotFoundError(f"Aucun fichier trouvé dans {elections_dir}")

    result = pd.concat(frames, ignore_index=True)
    log.info(
        f"Elections brutes : {len(result):,} lignes, {len(result.columns)} colonnes"
    )
    return result
