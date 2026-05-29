"""
Tests unitaires — fonctions utilitaires (helpers.py).
"""
import numpy as np
import pandas as pd
import pytest

from etl.helpers import norm_code, pct, cap_outliers, impute_missing, recalc_winner_marge

class TestNormCode:
    def test_integer_padded(self):
        s = pd.Series([75056, 9200, 1])
        result = norm_code(s)
        assert list(result) == ["75056", "09200", "00001"]

    def test_string_already_5(self):
        s = pd.Series(["75056", "92007"])
        assert list(norm_code(s)) == ["75056", "92007"]

    def test_strip_whitespace(self):
        s = pd.Series([" 75056 "])
        assert norm_code(s).iloc[0] == "75056"

class TestPct:
    def test_basic(self):
        num = pd.Series([50.0])
        den = pd.Series([100.0])
        assert pct(num, den).iloc[0] == 50.0

    def test_zero_denominator(self):
        num = pd.Series([10.0])
        den = pd.Series([0.0])
        result = pct(num, den)
        assert pd.isna(result.iloc[0])

    def test_rounding(self):
        num = pd.Series([1.0])
        den = pd.Series([3.0])
        result = pct(num, den, decimals=2)
        assert result.iloc[0] == 33.33

class TestCapOutliers:
    def test_no_change_within_bounds(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        original = df["x"].copy()
        result = cap_outliers(df, ["x"], factor=3.0)
        pd.testing.assert_series_equal(result["x"], original)

    def test_outlier_capped(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
        result = cap_outliers(df, ["x"], factor=1.5)
        assert result["x"].max() < 1000

    def test_skips_non_numeric(self):
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        result = cap_outliers(df, ["x"])
        assert list(result["x"]) == ["a", "b", "c"]

class TestImputeMissing:
    def test_imputes_nan_with_median(self):
        df = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0, 100.0, np.nan]})
        result = impute_missing(df, ["x"])
        assert not result["x"].isna().any()

    def test_imputes_nan_with_mean(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan]})
        result = impute_missing(df, ["x"])
        assert not result["x"].isna().any()

    def test_no_change_if_no_nan(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        original = df["x"].copy()
        result = impute_missing(df, ["x"])
        pd.testing.assert_series_equal(result["x"], original)

class TestRecalcWinnerMarge:
    def test_macron_wins(self):
        df = pd.DataFrame({
            "cible_t2_pct_macron": [60.0, 55.0],
            "cible_t2_pct_lepen": [40.0, 45.0],
        })
        result = recalc_winner_marge(df, "cible_t2", "macron", "lepen", 0, 1)
        assert list(result["cible_t2_vainqueur"]) == [0, 0]
        assert list(result["cible_t2_marge"]) == [20.0, 10.0]

    def test_lepen_wins(self):
        df = pd.DataFrame({
            "cible_t2_pct_macron": [40.0],
            "cible_t2_pct_lepen": [60.0],
        })
        result = recalc_winner_marge(df, "cible_t2", "macron", "lepen", 0, 1)
        assert result["cible_t2_vainqueur"].iloc[0] == 1

    def test_no_index_misalignment(self):
        """Vérifie que .values évite le désalignement d'index pandas."""
        df = pd.DataFrame({
            "cible_t2_pct_macron": [60.0, 40.0],
            "cible_t2_pct_lepen": [40.0, 60.0],
        }, index=[10, 20])
        result = recalc_winner_marge(df, "cible_t2", "macron", "lepen", 0, 1)
        assert list(result["cible_t2_vainqueur"]) == [0, 1]
