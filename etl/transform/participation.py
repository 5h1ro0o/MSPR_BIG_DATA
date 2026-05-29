"""
Transformation — données de participation électorale 2022.
Entrée : DataFrame brut (bronze) de extract.elections.
Sortie : une ligne par commune avec indicateurs de participation T1/T2.
"""

import numpy as np
import pandas as pd

from etl.helpers import pct
from monitoring.logger import get_logger

log = get_logger(__name__)

ELECTIONS_2022 = {"2022_pres_t1", "2022_pres_t2"}

AGG_COLS = {
    "inscrits": "sum",
    "abstentions": "sum",
    "votants": "sum",
    "blancs": "sum",
    "nuls": "sum",
    "exprimes": "sum",
}


def transform_participation(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre sur 2022_pres_t1/t2, agrège par commune, calcule les taux,
    puis pivote T1/T2 en colonnes larges.

    Args:
        df_raw: données brutes de extract_elections()

    Returns:
        DataFrame silver — une ligne par commune.
    """
    pres = df_raw[df_raw["id_election"].isin(ELECTIONS_2022)].copy()

    commune_tour = pres.groupby(
        [
            "code_commune",
            "code_departement",
            "libelle_departement",
            "libelle_commune",
            "id_election",
        ],
        as_index=False,
    ).agg(AGG_COLS)

    commune_tour["taux_abstention"] = pct(
        commune_tour["abstentions"], commune_tour["inscrits"]
    )
    commune_tour["taux_participation"] = pct(
        commune_tour["votants"], commune_tour["inscrits"]
    )
    commune_tour["taux_blancs_votants"] = pct(
        commune_tour["blancs"], commune_tour["votants"]
    )
    commune_tour["taux_nuls_votants"] = pct(
        commune_tour["nuls"], commune_tour["votants"]
    )
    commune_tour["taux_exprimes_votants"] = pct(
        commune_tour["exprimes"], commune_tour["votants"]
    )

    def suffix_cols(df_tour: pd.DataFrame, suffix: str) -> pd.DataFrame:
        numeric = [
            "inscrits",
            "abstentions",
            "votants",
            "blancs",
            "nuls",
            "exprimes",
            "taux_abstention",
            "taux_participation",
            "taux_blancs_votants",
            "taux_nuls_votants",
            "taux_exprimes_votants",
        ]
        return df_tour.rename(columns={c: f"{c}_{suffix}" for c in numeric})

    id_cols = [
        "code_commune",
        "code_departement",
        "libelle_departement",
        "libelle_commune",
    ]

    t1 = suffix_cols(
        commune_tour[commune_tour["id_election"] == "2022_pres_t1"].copy(), "t1"
    )
    t2 = suffix_cols(
        commune_tour[commune_tour["id_election"] == "2022_pres_t2"].copy(), "t2"
    )

    t1_final = t1[id_cols + [c for c in t1.columns if c.endswith("_t1")]]
    t2_final = t2[["code_commune"] + [c for c in t2.columns if c.endswith("_t2")]]

    merged = t1_final.merge(t2_final, on="code_commune", how="outer")

    merged["delta_participation_t2_t1"] = (
        merged["taux_participation_t2"] - merged["taux_participation_t1"]
    ).round(2)

    merged["ratio_blancs_nuls_t2_t1"] = (
        (merged["blancs_t2"] + merged["nuls_t2"])
        / (merged["blancs_t1"] + merged["nuls_t1"]).replace(0, np.nan)
    ).round(3)

    log.info(f"Participation silver : {len(merged):,} communes")
    return merged
