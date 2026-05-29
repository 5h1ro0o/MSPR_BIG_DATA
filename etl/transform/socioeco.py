"""
Transformations — données socio-économiques :
  - Démographie / CSP / diplômes / logement / ménages (RP 2022)
  - Chômage historique IDF
  - Emploi 2022 (DS_RP_EMPLOI)
  - Pauvreté (Filosofi 2017)
"""

import numpy as np
import pandas as pd

from etl.helpers import pct
from monitoring.logger import get_logger

log = get_logger(__name__)

IDF_DEPTS = frozenset({"75", "77", "78", "91", "92", "93", "94", "95"})


def transform_demographique(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme les effectifs bruts INSEE en taux (%).
    Supprime ensuite les colonnes d'effectifs intermédiaires.
    """
    df = df_raw.copy()
    pop = df["pop_totale"].replace(0, np.nan)

    for col in [
        "pop_0014",
        "pop_1529",
        "pop_3044",
        "pop_4559",
        "pop_6074",
        "pop_7589",
        "pop_90p",
    ]:
        if col in df.columns:
            df[f"pct_{col}"] = pct(df[col], pop)

    df["pct_pop_senior_60p"] = pct(
        df.get("pop_6074", pd.Series(0, index=df.index))
        + df.get("pop_7589", pd.Series(0, index=df.index))
        + df.get("pop_90p", pd.Series(0, index=df.index)),
        pop,
    )

    if "pop_hommes" in df.columns and "pop_femmes" in df.columns:
        df["ratio_hommes_femmes"] = (
            df["pop_hommes"] / df["pop_femmes"].replace(0, np.nan)
        ).round(3)

    pop15p = df.get("pop_15p", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    for col in [
        "csp_agriculteurs",
        "csp_artisans_commercants",
        "csp_cadres",
        "csp_prof_intermediaires",
        "csp_employes",
        "csp_ouvriers",
        "csp_retraites",
        "csp_autres_inactifs",
    ]:
        if col in df.columns:
            df[f"pct_{col}"] = pct(df[col], pop15p)

    if {"pct_csp_ouvriers", "pct_csp_employes"}.issubset(df.columns):
        df["pct_csp_precaires"] = (
            df["pct_csp_ouvriers"] + df["pct_csp_employes"]
        ).round(2)

    if {"pct_csp_cadres", "pct_csp_prof_intermediaires"}.issubset(df.columns):
        df["pct_csp_cols_blancs"] = (
            df["pct_csp_cadres"] + df["pct_csp_prof_intermediaires"]
        ).round(2)

    pop_nscol = df.get("pop_nscol15p", pd.Series(np.nan, index=df.index)).replace(
        0, np.nan
    )
    for col in [
        "dipl_aucun",
        "dipl_bepc",
        "dipl_capbep",
        "dipl_bac",
        "dipl_sup_bac2",
        "dipl_sup_bac34",
        "dipl_sup_bac5",
    ]:
        if col in df.columns:
            df[f"pct_{col}"] = pct(df[col], pop_nscol)

    sup_cols = ["dipl_sup_bac2", "dipl_sup_bac34", "dipl_sup_bac5"]
    if all(c in df.columns for c in sup_cols):
        df["pct_dipl_superieur"] = pct(
            df["dipl_sup_bac2"] + df["dipl_sup_bac34"] + df["dipl_sup_bac5"],
            pop_nscol,
        )

    if "actifs_1564" in df.columns and "chomeurs_1564" in df.columns:
        df["taux_chomage_rp2022"] = pct(df["chomeurs_1564"], df["actifs_1564"])

    nb_men = df.get("nb_menages", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    for col in ["log_proprietaires", "log_locataires", "log_hlm"]:
        if col in df.columns:
            df[f"pct_{col}"] = pct(df[col], nb_men)

    if "nb_logements" in df.columns and "nb_logements_vacants" in df.columns:
        df["pct_log_vacants"] = pct(
            df["nb_logements_vacants"], df["nb_logements"].replace(0, np.nan)
        )

    for col in [
        "men_seuls",
        "men_familiaux",
        "men_couple_enfants",
        "men_monoparentaux",
    ]:
        if col in df.columns:
            df[f"pct_{col}"] = pct(df[col], nb_men)

    drop_abs = [
        "pop_0014",
        "pop_1529",
        "pop_3044",
        "pop_4559",
        "pop_6074",
        "pop_7589",
        "pop_90p",
        "pop_hommes",
        "pop_femmes",
        "pop_15p",
        "pop_nscol15p",
        "csp_agriculteurs",
        "csp_artisans_commercants",
        "csp_cadres",
        "csp_prof_intermediaires",
        "csp_employes",
        "csp_ouvriers",
        "csp_retraites",
        "csp_autres_inactifs",
        "dipl_aucun",
        "dipl_bepc",
        "dipl_capbep",
        "dipl_bac",
        "dipl_sup_bac2",
        "dipl_sup_bac34",
        "dipl_sup_bac5",
        "actifs_1564",
        "actifs_occupes_1564",
        "chomeurs_1564",
        "log_proprietaires",
        "log_locataires",
        "log_hlm",
        "nb_logements",
        "nb_logements_vacants",
        "nb_menages",
        "men_seuls",
        "men_familiaux",
        "men_couple_enfants",
        "men_monoparentaux",
    ]
    df.drop(columns=[c for c in drop_abs if c in df.columns], inplace=True)

    log.info(f"Demographique silver : {len(df):,} communes, {len(df.columns)} colonnes")
    return df


def transform_chomage_hist(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne et renomme les colonnes utiles du chômage historique."""
    df = df_raw.copy()

    cols_keep = [
        "code_commune",
        "chom2020",
        "chom2019",
        "chom2018",
        "txchom2011",
        "txchom2010",
        "txchom2009",
    ]
    df = df[[c for c in cols_keep if c in df.columns]].copy()

    if {"chom2020", "chom2018"}.issubset(df.columns):
        df["evol_chomage_2018_2020_pct"] = pct(
            df["chom2020"] - df["chom2018"],
            df["chom2018"].replace(0, np.nan),
        )

    df.rename(
        columns={
            "chom2020": "nb_chomeurs_2020",
            "txchom2011": "taux_chomage_2011",
            "txchom2010": "taux_chomage_2010",
            "txchom2009": "taux_chomage_2009",
        },
        inplace=True,
    )

    keep_final = [
        "code_commune",
        "nb_chomeurs_2020",
        "taux_chomage_2009",
        "taux_chomage_2010",
        "taux_chomage_2011",
        "evol_chomage_2018_2020_pct",
    ]
    result = df[[c for c in keep_final if c in df.columns]]
    log.info(f"Chomage hist silver : {len(result):,} communes")
    return result


def transform_emploi_2022(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre GEO_OBJECT=COM, PCS=_T, TIME_PERIOD=2022, IDF.
    Pivote sur EMPSTA_ENQ pour obtenir actifs_occupes, chômeurs, actifs_totaux.
    """
    df = df_raw.copy()
    df["GEO"] = df["GEO"].astype(str).str.strip().str.zfill(5)

    df = df[
        (df["GEO_OBJECT"] == "COM")
        & (df["PCS"].astype(str) == "_T")
        & (df["TIME_PERIOD"] == 2022)
        & (df["GEO"].str[:2].isin(IDF_DEPTS))
    ].copy()

    if df.empty:
        log.warning("DS_RP_EMPLOI : aucune donnée après filtrage")
        return pd.DataFrame(columns=["code_commune"])

    pivot = df.pivot_table(
        index="GEO", columns="EMPSTA_ENQ", values="OBS_VALUE", aggfunc="sum"
    ).reset_index()
    pivot.columns.name = None
    pivot.rename(columns={"GEO": "code_commune"}, inplace=True)

    pivot.rename(
        columns={
            "1": "ds_actifs_occupes_2022",
            "2": "ds_chomeurs_2022",
            "1T2": "ds_actifs_totaux_2022",
        },
        inplace=True,
    )

    if {"ds_chomeurs_2022", "ds_actifs_totaux_2022"}.issubset(pivot.columns):
        pivot["ds_taux_chomage_2022"] = pct(
            pivot["ds_chomeurs_2022"], pivot["ds_actifs_totaux_2022"]
        )

    keep = ["code_commune", "ds_taux_chomage_2022"]
    result = pivot[[c for c in keep if c in pivot.columns]]
    log.info(f"Emploi 2022 silver : {len(result):,} communes")
    return result


def transform_pauvrete(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne et renomme les colonnes Filosofi 2017 utiles."""
    df = df_raw.copy()

    col_map = {
        "CODGEO": "code_commune",
        "MED17": "revenu_median_2017",
        "TP6017": "taux_pauvrete_2017",
        "D117": "decile1_revenu_2017",
        "D917": "decile9_revenu_2017",
        "RD17": "rapport_interdecile_2017",
        "PIMP17": "part_menages_imposables_2017",
        "PCHO17": "part_rev_chomage_2017",
        "PPEN17": "part_rev_pension_retraite_2017",
        "PPSOC17": "part_prestations_sociales_2017",
        "PPFAM17": "part_allocations_familiales_2017",
    }
    available = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available).copy()

    if (
        "rapport_interdecile_2017" not in df.columns
        and "decile1_revenu_2017" in df.columns
        and "decile9_revenu_2017" in df.columns
    ):
        df["rapport_interdecile_2017"] = (
            df["decile9_revenu_2017"] / df["decile1_revenu_2017"].replace(0, np.nan)
        ).round(3)

    log.info(f"Pauvrete silver : {len(df):,} communes")
    return df
