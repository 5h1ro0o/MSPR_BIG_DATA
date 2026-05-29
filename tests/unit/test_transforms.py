"""
Tests unitaires — couches de transformation.
"""

import numpy as np

from etl.quality.checks import run_quality_checks
from etl.transform.candidats import pivot_candidats
from etl.transform.participation import transform_participation


class TestTransformParticipation:
    def test_produces_one_row_per_commune(self, sample_elections_raw):
        result = transform_participation(sample_elections_raw)
        assert len(result) == 2

    def test_taux_participation_range(self, sample_elections_raw):
        result = transform_participation(sample_elections_raw)
        assert (result["taux_participation_t1"] >= 0).all()
        assert (result["taux_participation_t1"] <= 100).all()

    def test_delta_participation_exists(self, sample_elections_raw):
        result = transform_participation(sample_elections_raw)
        assert "delta_participation_t2_t1" in result.columns

    def test_id_cols_present(self, sample_elections_raw):
        result = transform_participation(sample_elections_raw)
        for col in ["code_commune", "code_departement", "libelle_commune"]:
            assert col in result.columns


class TestPivotCandidats:
    def test_basic_pivot(self, sample_candidats_raw):
        result = pivot_candidats(
            sample_candidats_raw,
            "2017_pres_t2",
            {"MACRON": "macron", "LE PEN": "lepen"},
            "h17_t2",
        )
        assert "h17_t2_pct_macron" in result.columns
        assert "h17_t2_pct_lepen" in result.columns
        assert len(result) == 2

    def test_pct_sum_to_100(self, sample_candidats_raw):
        result = pivot_candidats(
            sample_candidats_raw,
            "2017_pres_t2",
            {"MACRON": "macron", "LE PEN": "lepen"},
            "h17_t2",
        )
        total = (
            result["h17_t2_pct_macron"]
            + result["h17_t2_pct_lepen"]
            + result["h17_t2_pct_autres"]
        )
        assert (abs(total - 100) < 0.1).all()

    def test_empty_election(self, sample_candidats_raw):
        result = pivot_candidats(
            sample_candidats_raw,
            "2099_pres_t1",
            {"MACRON": "macron"},
            "h99_t1",
        )
        assert "code_commune" in result.columns
        assert len(result) == 0


class TestQualityChecks:
    def test_passes_valid_dataset(self, sample_gold_df):
        report = run_quality_checks(sample_gold_df)
        assert isinstance(report.passed, bool)

    def test_detects_nulls(self, sample_gold_df):
        df = sample_gold_df.copy()
        df.loc[0, "cible_t2_pct_macron"] = np.nan
        report = run_quality_checks(df)
        assert not report.passed
        assert any("null" in e.lower() for e in report.errors)

    def test_detects_winner_incoherence(self, sample_gold_df):
        df = sample_gold_df.copy()
        df["cible_t2_vainqueur"] = 1
        report = run_quality_checks(df)
        assert not report.passed
        assert any("incoh" in e.lower() for e in report.errors)
