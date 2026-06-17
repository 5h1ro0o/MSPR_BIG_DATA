"""Tests unitaires — etl/transform/assembler.py."""

import numpy as np
import pandas as pd

from etl.transform.assembler import assemble_dataset


def _participation():
    return pd.DataFrame({
        "code_commune": ["75056", "92007", "93001"],
        "code_departement": ["75", "92", "93"],
        "libelle_departement": ["Paris", "Hauts-de-Seine", "Seine-Saint-Denis"],
        "libelle_commune": ["Paris", "Boulogne", "Bobigny"],
        "taux_participation_t1": [75.0, 72.0, 68.0],
        "taux_participation_t2": [78.0, 74.0, 70.0],
        "delta_participation_t2_t1": [3.0, 2.0, 2.0],
    })


def _cibles():
    return pd.DataFrame({
        "code_commune": ["75056", "92007", "93001"],
        "cible_t2_pct_macron": [80.0, 65.0, 55.0],
        "cible_t2_pct_lepen": [20.0, 35.0, 45.0],
        "cible_t2_vainqueur": [0, 0, 0],
        "cible_t2_marge": [60.0, 30.0, 10.0],
        "cible_t1_premier": ["macron", "macron", "melenchon"],
    })


def _empty():
    return pd.DataFrame(columns=["code_commune"])


def _assemble(**kwargs):
    defaults = dict(
        participation=_participation(),
        demographique=_empty(),
        pauvrete=_empty(),
        chomage=_empty(),
        emploi=_empty(),
        historique=_empty(),
        cibles=_cibles(),
    )
    defaults.update(kwargs)
    return assemble_dataset(**defaults)


class TestAssembleDatasetBasic:
    def test_returns_dataframe(self):
        result = _assemble()
        assert isinstance(result, pd.DataFrame)

    def test_correct_row_count(self):
        result = _assemble()
        assert len(result) == 3

    def test_code_commune_present(self):
        result = _assemble()
        assert "code_commune" in result.columns

    def test_target_columns_present(self):
        result = _assemble()
        assert "cible_t2_pct_macron" in result.columns
        assert "cible_t2_pct_lepen" in result.columns


class TestAssembleDeduplication:
    def test_removes_duplicate_communes(self):
        df = _participation()
        df = pd.concat([df, df]).reset_index(drop=True)
        result = _assemble(participation=df)
        assert result["code_commune"].nunique() == len(result)

    def test_keeps_first_occurrence(self):
        df = _participation().copy()
        df2 = df.copy()
        df2["taux_participation_t1"] = 99.0
        combined = pd.concat([df, df2]).reset_index(drop=True)
        result = _assemble(participation=combined)
        assert len(result) == 3


class TestAssembleNullHandling:
    def test_no_nulls_in_numeric_features(self):
        part = _participation().copy()
        part.loc[0, "taux_participation_t1"] = np.nan
        result = _assemble(participation=part)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert result[numeric_cols].isnull().sum().sum() == 0

    def test_string_nulls_filled_with_empty(self):
        part = _participation().copy()
        part.loc[0, "libelle_commune"] = np.nan
        result = _assemble(participation=part)
        str_cols = result.select_dtypes(include=["object"]).columns
        for col in str_cols:
            assert not result[col].isnull().any()

    def test_drops_rows_without_cibles(self):
        part = _participation().copy()
        extra = pd.DataFrame({
            "code_commune": ["99999"],
            "code_departement": ["99"],
            "libelle_departement": ["Inconnue"],
            "libelle_commune": ["Inconnue"],
            "taux_participation_t1": [50.0],
            "taux_participation_t2": [55.0],
            "delta_participation_t2_t1": [5.0],
        })
        result = _assemble(participation=pd.concat([part, extra], ignore_index=True))
        assert "99999" not in result["code_commune"].values


class TestAssembleWinnerRecalc:
    def test_winner_column_present(self):
        result = _assemble()
        assert "cible_t2_vainqueur" in result.columns

    def test_marge_column_present(self):
        result = _assemble()
        assert "cible_t2_marge" in result.columns

    def test_marge_positive(self):
        result = _assemble()
        assert (result["cible_t2_marge"] >= 0).all()

    def test_winner_recalculated_correctly(self):
        cibles = _cibles().copy()
        cibles.loc[0, "cible_t2_pct_macron"] = 40.0
        cibles.loc[0, "cible_t2_pct_lepen"] = 60.0
        cibles.loc[0, "cible_t2_vainqueur"] = 0
        result = _assemble(cibles=cibles)
        paris_row = result[result["code_commune"] == "75056"].iloc[0]
        assert paris_row["cible_t2_vainqueur"] == 1


class TestAssembleDeptNormalization:
    def test_dept_code_zero_padded(self):
        part = _participation().copy()
        part["code_departement"] = part["code_departement"].astype(int)
        result = _assemble(participation=part)
        assert all(len(d) == 2 for d in result["code_departement"])

    def test_dept_code_is_string(self):
        result = _assemble()
        assert result["code_departement"].dtype == object


class TestAssembleIqrFactor:
    def test_custom_iqr_factor_accepted(self):
        result = _assemble(iqr_factor=1.5)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
