"""
Extraction — candidats_results.csv (2.2 GB, ~27M lignes).
Lecture en chunks avec filtre précoce pour économiser la mémoire.
"""
from pathlib import Path
import pandas as pd

from etl.helpers import norm_code, norm_nom
from monitoring.logger import get_logger

log = get_logger(__name__)

IDF_DEPTS = frozenset({"75", "77", "78", "91", "92", "93", "94", "95"})

USECOLS = ["id_election", "code_departement", "code_commune", "voix", "nom", "prenom"]

def extract_candidats_raw(
    candidats_file: Path,
    elections_filter: list[str],
    chunk_size: int = 200_000,
) -> pd.DataFrame:
    """
    Charge candidats_results.csv en filtrant immédiatement sur :
      - les élections listées dans elections_filter
      - les 8 départements IDF

    La lecture en chunks évite de charger 2.2 Go en mémoire.

    Args:
        candidats_file: chemin vers candidats_results.csv
        elections_filter: liste d'id_election à garder
                          (ex. ["2012_pres_t1", "2012_pres_t2", ...])
        chunk_size: taille des chunks pandas

    Returns:
        DataFrame filtré (~quelques milliers de lignes pour IDF seul).
    """
    if not candidats_file.exists():
        raise FileNotFoundError(f"Fichier introuvable : {candidats_file}")

    elections_set = set(elections_filter)
    kept_chunks = []

    log.info(f"Lecture par chunks de {candidats_file.name} (chunk={chunk_size:,})...")

    reader = pd.read_csv(
        candidats_file,
        sep=";",
        usecols=USECOLS,
        low_memory=False,
        dtype={"code_departement": str, "code_commune": str},
        chunksize=chunk_size,
    )

    total_read = 0
    for chunk in reader:
        total_read += len(chunk)
        chunk["code_departement"] = chunk["code_departement"].str.zfill(2)
        mask = (
            chunk["id_election"].isin(elections_set) &
            chunk["code_departement"].isin(IDF_DEPTS)
        )
        filtered = chunk[mask].copy()
        if not filtered.empty:
            filtered["code_commune"] = norm_code(filtered["code_commune"])
            filtered["nom_norm"] = filtered["nom"].apply(norm_nom)
            kept_chunks.append(filtered)

    log.info(f"Lu {total_read:,} lignes au total")

    if not kept_chunks:
        log.warning("Aucune donnée après filtrage IDF / élections")
        return pd.DataFrame(columns=USECOLS + ["nom_norm"])

    result = pd.concat(kept_chunks, ignore_index=True)
    log.info(f"Candidats filtrés : {len(result):,} lignes")
    return result
