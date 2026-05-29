"""
Contrôles qualité du dataset gold.
Retourne un rapport structuré — ne lève pas d'exception,
laisse l'orchestrateur décider de la criticité.
"""
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from monitoring.logger import get_logger

log = get_logger(__name__)

MIN_COMMUNES = 1200
MAX_NULL_RATIO = 0.0
EXPECTED_DEPTS = {"75", "77", "78", "91", "92", "93", "94", "95"}
CIBLE_COLS = [
    "cible_t2_vainqueur",
    "cible_t2_pct_macron",
    "cible_t2_pct_lepen",
    "cible_t2_marge",
    "cible_t1_premier",
]
T2_PCT_COLS = ["cible_t2_pct_macron", "cible_t2_pct_lepen"]

@dataclass
class QualityReport:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)
        log.error(f"[QC FAIL] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        log.warning(f"[QC WARN] {msg}")

def run_quality_checks(df: pd.DataFrame) -> QualityReport:
    """
    Exécute tous les contrôles qualité sur le dataset gold.

    Checks :
      1. Nombre de communes (>= MIN_COMMUNES)
      2. Départements IDF couverts
      3. Absence de nulls
      4. Absence de doublons sur code_commune
      5. Présence des colonnes cibles
      6. Cohérence vainqueur vs pourcentages T2
      7. Plages de valeurs (taux entre 0-100)
      8. Somme des % T2 ≈ 100

    Returns:
        QualityReport avec statut global et détails des erreurs/warnings.
    """
    report = QualityReport()
    report.stats["n_communes"] = len(df)
    report.stats["n_cols"] = len(df.columns)
    report.stats["n_nulls"] = int(df.isnull().sum().sum())
    report.stats["n_duplicates"] = int(df.duplicated(subset="code_commune").sum())

    if len(df) < MIN_COMMUNES:
        report.fail(f"Trop peu de communes : {len(df)} < {MIN_COMMUNES}")

    if "code_departement" in df.columns:
        depts = set(df["code_departement"].astype(str).str.zfill(2).unique())
        missing_depts = EXPECTED_DEPTS - depts
        if missing_depts:
            report.warn(f"Départements manquants : {missing_depts}")
        report.stats["depts_found"] = sorted(depts)

    null_count = df.isnull().sum().sum()
    if null_count > 0:
        top_nulls = df.isnull().sum().nlargest(5).to_dict()
        report.fail(f"{null_count} valeurs nulles — top colonnes : {top_nulls}")

    n_dup = df.duplicated(subset="code_commune").sum()
    if n_dup > 0:
        report.fail(f"{n_dup} doublons sur code_commune")

    for col in CIBLE_COLS:
        if col not in df.columns:
            report.fail(f"Colonne cible manquante : {col}")

    if all(c in df.columns for c in ["cible_t2_vainqueur", "cible_t2_pct_macron", "cible_t2_pct_lepen"]):
        expected_winner = (df["cible_t2_pct_macron"] >= df["cible_t2_pct_lepen"]).astype(int)
        expected_label = (~(df["cible_t2_pct_macron"] >= df["cible_t2_pct_lepen"])).astype(int)
        incoherent = (df["cible_t2_vainqueur"] != expected_label).sum()
        if incoherent > 0:
            report.fail(f"{incoherent} incohérences cible_t2_vainqueur vs pourcentages")
        else:
            report.stats["winner_coherence"] = "OK"

    pct_cols = [c for c in df.columns if "taux_" in c or "_pct_" in c]
    for col in pct_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        out_of_range = ((df[col] < -1) | (df[col] > 101)).sum()
        if out_of_range > 0:
            report.warn(f"{col} : {out_of_range} valeurs hors plage [0, 100]")

    if all(c in df.columns for c in T2_PCT_COLS):
        total_t2 = df[T2_PCT_COLS].sum(axis=1)
        bad = ((total_t2 < 95) | (total_t2 > 105)).sum()
        if bad > 0:
            report.warn(f"{bad} communes avec somme T2 hors [95, 105]%")

    if report.passed:
        log.info(
            f"Qualité OK — {len(df):,} communes, {null_count} nulls, "
            f"{report.stats.get('winner_coherence', 'N/A')} cohérence vainqueur"
        )
    else:
        log.error(f"Qualité KO — {len(report.errors)} erreur(s) : {report.errors}")

    return report
