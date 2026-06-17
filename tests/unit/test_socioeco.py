"""Tests unitaires — etl/transform/socioeco.py + candidats historique/cibles."""

import numpy as np
import pandas as pd
import pytest

from etl.transform.socioeco import (
    transform_chomage_hist,
    transform_demographique,
    transform_emploi_2022,
    transform_pauvrete,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _demo_raw():
    return pd.DataFrame({
        "code_commune": ["75056", "92007"],
        "pop_totale": [1000, 800],
        "pop_0014": [150, 120],
        "pop_1529": [200, 160],
        "pop_3044": [250, 200],
        "pop_4559": [200, 160],
        "pop_6074": [100, 80],
        "pop_7589": [80, 64],
        "pop_90p": [20, 16],
        "pop_hommes": [480, 384],
        "pop_femmes": [520, 416],
        "pop_15p": [850, 680],
        "csp_cadres": [200, 160],
        "csp_prof_intermediaires": [150, 120],
        "csp_employes": [180, 144],
        "csp_ouvriers": [100, 80],
        "csp_retraites": [120, 96],
        "csp_autres_inactifs": [100, 80],
    })


def _chomage_raw():
    return pd.DataFrame({
        "code_commune": ["75056", "92007", "93001"],
        "chom2020": [100, 80, 200],
        "chom2019": [95, 75, 190],
        "chom2018": [90, 70, 180],
        "txchom2011": [8.5, 7.2, 15.0],
        "txchom2010": [8.0, 7.0, 14.5],
        "txchom2009": [7.5, 6.5, 14.0],
    })


def _emploi_baseccc():
    return pd.DataFrame({
        "CODGEO": ["75056", "92007", "78000"],
        "P22_ACT1564": [500, 400, 300],
        "P22_CHOM1564": [50, 30, 45],
    })


def _pauvrete_raw():
    return pd.DataFrame({
        "CODGEO": ["75056", "92007"],
        "MED17": [30000, 35000],
        "TP6017": [10.5, 8.2],
        "D117": [15000, 18000],
        "D917": [60000, 70000],
        "RD17": [4.0, 3.9],
        "PIMP17": [65.0, 70.0],
    })


# ── transform_demographique ───────────────────────────────────────────────────


class TestTransformDemographique:
    def test_empty_input_returns_single_col(self):
        result = transform_demographique(pd.DataFrame())
        assert list(result.columns) == ["code_commune"]

    def test_missing_pop_totale_returns_empty(self):
        result = transform_demographique(pd.DataFrame({"code_commune": ["75056"]}))
        assert list(result.columns) == ["code_commune"]

    def test_produces_pct_columns(self):
        result = transform_demographique(_demo_raw())
        pct_cols = [c for c in result.columns if c.startswith("pct_")]
        assert len(pct_cols) > 0

    def test_age_pct_between_0_100(self):
        result = transform_demographique(_demo_raw())
        for col in [c for c in result.columns if "pct_pop" in c]:
            assert (result[col].dropna() >= 0).all()
            assert (result[col].dropna() <= 100).all()

    def test_senior_pct_created(self):
        result = transform_demographique(_demo_raw())
        assert "pct_pop_senior_60p" in result.columns

    def test_csp_precaires_created(self):
        result = transform_demographique(_demo_raw())
        assert "pct_csp_precaires" in result.columns

    def test_ratio_hommes_femmes_created(self):
        result = transform_demographique(_demo_raw())
        assert "ratio_hommes_femmes" in result.columns

    def test_raw_effectifs_dropped(self):
        result = transform_demographique(_demo_raw())
        assert "pop_0014" not in result.columns
        assert "pop_hommes" not in result.columns

    def test_code_commune_preserved(self):
        result = transform_demographique(_demo_raw())
        assert "code_commune" in result.columns
        assert list(result["code_commune"]) == ["75056", "92007"]

    def test_row_count_preserved(self):
        result = transform_demographique(_demo_raw())
        assert len(result) == 2


# ── transform_chomage_hist ───────────────────────────────────────────────────


class TestTransformChomageHist:
    def test_returns_dataframe(self):
        result = transform_chomage_hist(_chomage_raw())
        assert isinstance(result, pd.DataFrame)

    def test_code_commune_present(self):
        result = transform_chomage_hist(_chomage_raw())
        assert "code_commune" in result.columns

    def test_evolution_column_created(self):
        result = transform_chomage_hist(_chomage_raw())
        assert "evol_chomage_2018_2020_pct" in result.columns

    def test_renamed_columns(self):
        result = transform_chomage_hist(_chomage_raw())
        assert "nb_chomeurs_2020" in result.columns
        assert "taux_chomage_2011" in result.columns

    def test_raw_columns_dropped(self):
        result = transform_chomage_hist(_chomage_raw())
        assert "chom2020" not in result.columns
        assert "txchom2011" not in result.columns

    def test_row_count_preserved(self):
        result = transform_chomage_hist(_chomage_raw())
        assert len(result) == 3

    def test_evolution_sign(self):
        df = _chomage_raw()
        result = transform_chomage_hist(df)
        paris = result[result["code_commune"] == "75056"].iloc[0]
        assert paris["evol_chomage_2018_2020_pct"] > 0

    def test_missing_columns_graceful(self):
        minimal = pd.DataFrame({
            "code_commune": ["75056"],
            "chom2020": [100],
        })
        result = transform_chomage_hist(minimal)
        assert "code_commune" in result.columns


# ── transform_emploi_2022 ─────────────────────────────────────────────────────


class TestTransformEmploi2022:
    def test_empty_returns_code_commune_col(self):
        result = transform_emploi_2022(pd.DataFrame())
        assert "code_commune" in result.columns

    def test_baseccc_format(self):
        result = transform_emploi_2022(_emploi_baseccc())
        assert "code_commune" in result.columns
        assert "ds_taux_chomage_2022" in result.columns

    def test_chomage_rate_between_0_100(self):
        result = transform_emploi_2022(_emploi_baseccc())
        rates = result["ds_taux_chomage_2022"].dropna()
        assert (rates >= 0).all()
        assert (rates <= 100).all()

    def test_filters_idf_depts(self):
        df = _emploi_baseccc().copy()
        df.loc[2, "CODGEO"] = "31000"
        result = transform_emploi_2022(df)
        assert "31000" not in result["code_commune"].values

    def test_unknown_format_returns_empty(self):
        df = pd.DataFrame({"unknown_col": [1, 2]})
        result = transform_emploi_2022(df)
        assert "code_commune" in result.columns
        assert len(result) == 0


# ── transform_pauvrete ────────────────────────────────────────────────────────


class TestTransformPauvrete:
    def test_empty_returns_code_commune_col(self):
        result = transform_pauvrete(pd.DataFrame())
        assert "code_commune" in result.columns

    def test_renames_codgeo(self):
        result = transform_pauvrete(_pauvrete_raw())
        assert "code_commune" in result.columns
        assert "CODGEO" not in result.columns

    def test_known_columns_renamed(self):
        result = transform_pauvrete(_pauvrete_raw())
        assert "revenu_median_2017" in result.columns
        assert "taux_pauvrete_2017" in result.columns

    def test_row_count_preserved(self):
        result = transform_pauvrete(_pauvrete_raw())
        assert len(result) == 2

    def test_interdecile_computed_if_missing(self):
        df = _pauvrete_raw().drop(columns=["RD17"])
        result = transform_pauvrete(df)
        assert "rapport_interdecile_2017" in result.columns

    def test_values_reasonable(self):
        result = transform_pauvrete(_pauvrete_raw())
        assert (result["revenu_median_2017"] > 0).all()


# ── Tests supplémentaires candidats (transform_historique / transform_cibles) ─


class TestTransformHistorique:
    def _make_raw(self):
        rows = []
        for election, cands in [
            ("2012_pres_t1", [("HOLLANDE", 1000), ("SARKOZY", 800), ("LE PEN", 400)]),
            ("2012_pres_t2", [("HOLLANDE", 1200), ("SARKOZY", 900)]),
            ("2017_pres_t1", [("MACRON", 1100), ("LE PEN", 600), ("MELENCHON", 500)]),
            ("2017_pres_t2", [("MACRON", 1300), ("LE PEN", 700)]),
        ]:
            for commune in ["75056", "92007"]:
                for nom, voix in cands:
                    rows.append({
                        "id_election": election,
                        "code_commune": commune,
                        "nom_norm": nom,
                        "voix": voix,
                    })
        return pd.DataFrame(rows)

    def test_returns_dataframe(self):
        from etl.transform.candidats import transform_historique
        result = transform_historique(self._make_raw())
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_commune(self):
        from etl.transform.candidats import transform_historique
        result = transform_historique(self._make_raw())
        assert result["code_commune"].nunique() == len(result)

    def test_h17_columns_present(self):
        from etl.transform.candidats import transform_historique
        result = transform_historique(self._make_raw())
        assert "h17_t2_pct_macron" in result.columns
        assert "h17_t2_pct_lepen" in result.columns


class TestTransformCibles:
    def _make_raw_2022(self):
        rows = []
        for election, cands in [
            ("2022_pres_t1", [("MACRON", 900), ("LE PEN", 500), ("MELENCHON", 800)]),
            ("2022_pres_t2", [("MACRON", 1200), ("LE PEN", 600)]),
        ]:
            for commune in ["75056", "92007"]:
                for nom, voix in cands:
                    rows.append({
                        "id_election": election,
                        "code_commune": commune,
                        "nom_norm": nom,
                        "voix": voix,
                    })
        return pd.DataFrame(rows)

    def test_returns_dataframe(self):
        from etl.transform.candidats import transform_cibles
        result = transform_cibles(self._make_raw_2022())
        assert isinstance(result, pd.DataFrame)

    def test_t1_premier_column(self):
        from etl.transform.candidats import transform_cibles
        result = transform_cibles(self._make_raw_2022())
        assert "cible_t1_premier" in result.columns

    def test_t2_pct_columns(self):
        from etl.transform.candidats import transform_cibles
        result = transform_cibles(self._make_raw_2022())
        assert "cible_t2_pct_macron" in result.columns
        assert "cible_t2_pct_lepen" in result.columns
