"""
Fixtures partagées pour les tests.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_elections_raw():
    """DataFrame simulant un fichier elections_XX.csv brut."""
    return pd.DataFrame(
        {
            "id_election": ["2022_pres_t1"] * 4 + ["2022_pres_t2"] * 4,
            "code_commune": ["75001", "75001", "75002", "75002"] * 2,
            "code_departement": ["75"] * 8,
            "libelle_departement": ["Paris"] * 8,
            "libelle_commune": ["Paris 1er"] * 4 + ["Paris 2e"] * 4,
            "inscrits": [1000, 500, 800, 400, 1000, 500, 800, 400],
            "abstentions": [200, 100, 160, 80, 180, 90, 144, 72],
            "votants": [800, 400, 640, 320, 820, 410, 656, 328],
            "blancs": [20, 10, 16, 8, 10, 5, 8, 4],
            "nuls": [10, 5, 8, 4, 5, 2, 4, 2],
            "exprimes": [770, 385, 616, 308, 805, 403, 644, 322],
            "code_bv": ["0001", "0002", "0001", "0002"] * 2,
        }
    )


@pytest.fixture
def sample_candidats_raw():
    """DataFrame simulant candidats_results.csv filtré IDF."""
    return pd.DataFrame(
        {
            "id_election": ["2017_pres_t2"] * 4,
            "code_departement": ["75", "75", "75", "75"],
            "code_commune": ["75056", "75056", "92007", "92007"],
            "voix": [5000, 500, 3000, 2000],
            "nom": ["MACRON", "LE PEN", "MACRON", "LE PEN"],
            "prenom": ["Emmanuel", "Marine", "Emmanuel", "Marine"],
            "nom_norm": ["MACRON", "LE PEN", "MACRON", "LE PEN"],
        }
    )


@pytest.fixture
def sample_gold_df():
    """Dataset gold minimal pour les tests de qualité."""
    return pd.DataFrame(
        {
            "code_commune": ["75056", "92007", "93001"],
            "code_departement": ["75", "92", "93"],
            "libelle_departement": ["Paris", "Hauts-de-Seine", "Seine-Saint-Denis"],
            "libelle_commune": ["Paris", "Boulogne-Billancourt", "Bobigny"],
            "taux_participation_t1": [75.0, 72.0, 68.0],
            "taux_participation_t2": [78.0, 74.0, 70.0],
            "cible_t2_pct_macron": [80.0, 65.0, 55.0],
            "cible_t2_pct_lepen": [20.0, 35.0, 45.0],
            "cible_t2_vainqueur": [0, 0, 0],
            "cible_t2_marge": [60.0, 30.0, 10.0],
            "cible_t1_premier": ["macron", "macron", "melenchon"],
        }
    )
