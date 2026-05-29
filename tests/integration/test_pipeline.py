"""
Tests d'intégration — vérifient que le pipeline complet tourne sur les vraies données.
Ces tests sont lents (chargent candidats_results.csv) et doivent être lancés
explicitement avec : pytest tests/integration/ -v

Ils s'appuient sur les données réelles dans DATA_ROOT.
"""

import os
import pytest
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "C:/Users/matte/Desktop/dataset_mspr"))
SKIP_IF_NO_DATA = pytest.mark.skipif(
    not (DATA_ROOT / "candidats_results.csv").exists(),
    reason="Données réelles non disponibles (DATA_ROOT non configuré)",
)


@SKIP_IF_NO_DATA
def test_extract_elections():
    """Vérifie que l'extraction des elections retourne des données IDF valides."""
    from etl.extract.elections import extract_elections

    df = extract_elections(DATA_ROOT / "elections")
    assert len(df) > 0
    assert "code_commune" in df.columns
    assert "id_election" in df.columns


@SKIP_IF_NO_DATA
def test_transform_participation():
    """Vérifie la transformation participation sur données réelles."""
    from etl.extract.elections import extract_elections
    from etl.transform.participation import transform_participation

    raw = extract_elections(DATA_ROOT / "elections")
    silver = transform_participation(raw)

    assert len(silver) >= 1200
    assert silver["taux_participation_t1"].between(0, 100).all()
    assert silver["code_commune"].nunique() == len(silver)


@SKIP_IF_NO_DATA
def test_full_pipeline_smoke():
    """
    Smoke test du pipeline complet.
    Vérifie que le dataset final a le bon format sans valeurs nulles.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from orchestration.pipeline import run_pipeline

    metrics = run_pipeline()

    assert metrics["total_duration_s"] > 0
    assert any(s["step"] == "assemble_gold" for s in metrics["steps"])

    gold_step = next(s for s in metrics["steps"] if s["step"] == "assemble_gold")
    assert gold_step["rows_out"] >= 1200
    assert gold_step["cols_out"] >= 100
