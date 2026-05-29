"""
Configuration ML — features, cibles, hyperparamètres pour tous les modèles.
Source : dataset_elections_2022_idf.csv (1268 communes × 113 colonnes)
"""

from pathlib import Path

import os as _os

ML_DIR = Path(__file__).parent
ARTIFACTS = ML_DIR / "artifacts"
DATA_PATH = Path(
    _os.environ.get(
        "DATA_PATH", str(ML_DIR.parent.parent / "dataset_elections_2022_idf.csv")
    )
)

ARTIFACTS.mkdir(exist_ok=True)

TARGETS = {
    "classification_t2": "cible_t2_vainqueur",
    "regression_macron": "cible_t2_pct_macron",
    "regression_lepen": "cible_t2_pct_lepen",
    "regression_marge": "cible_t2_marge",
    "classification_t1": "cible_t1_premier",
}

LEAKAGE_KEYWORDS = [
    "abstentions_t2",
    "blancs_t2",
    "exprimes_t2",
    "inscrits_t2",
    "nuls_t2",
    "taux_abstention_t2",
    "taux_blancs_votants_t2",
    "taux_exprimes_votants_t2",
    "taux_nuls_votants_t2",
    "taux_participation_t2",
    "votants_t2",
    "delta_participation",
    "ratio_blancs_nuls_t2",
]

FEATURES_SOCIO = [
    "pct_pop_0014",
    "pct_pop_1529",
    "pct_pop_3044",
    "pct_pop_4559",
    "pct_pop_6074",
    "pct_pop_7589",
    "pct_pop_90p",
    "pct_pop_senior_60p",
    "ratio_hommes_femmes",
    "pct_csp_agriculteurs",
    "pct_csp_artisans_commercants",
    "pct_csp_cadres",
    "pct_csp_prof_intermediaires",
    "pct_csp_employes",
    "pct_csp_ouvriers",
    "pct_csp_retraites",
    "pct_csp_autres_inactifs",
    "pct_csp_precaires",
    "pct_csp_cols_blancs",
    "pct_dipl_aucun",
    "pct_dipl_bepc",
    "pct_dipl_capbep",
    "pct_dipl_bac",
    "pct_dipl_sup_bac2",
    "pct_dipl_sup_bac34",
    "pct_dipl_sup_bac5",
    "pct_dipl_superieur",
    "pct_log_proprietaires",
    "pct_log_locataires",
    "pct_log_hlm",
    "pct_log_vacants",
    "pct_men_seuls",
    "pct_men_familiaux",
    "pct_men_couple_enfants",
    "pct_men_monoparentaux",
    "revenu_median_2021",
    "revenu_median_2017",
    "taux_pauvrete_2017",
    "rapport_interdecile_2017",
    "part_menages_imposables_2017",
    "part_rev_chomage_2017",
    "part_rev_pension_retraite_2017",
    "part_prestations_sociales_2017",
    "part_allocations_familiales_2017",
    "taux_chomage_rp2022",
    "ds_taux_chomage_2022",
    "taux_chomage_2009",
    "taux_chomage_2010",
    "taux_chomage_2011",
    "nb_chomeurs_2020",
    "evol_chomage_2018_2020_pct",
    "pop_totale",
]

FEATURES_SOCIO_LSTM = [
    "pct_csp_cadres",
    "pct_csp_ouvriers",
    "pct_csp_employes",
    "pct_csp_retraites",
    "pct_csp_precaires",
    "pct_dipl_aucun",
    "pct_dipl_superieur",
    "pct_log_hlm",
    "pct_log_proprietaires",
    "taux_pauvrete_2017",
    "revenu_median_2021",
    "pct_pop_senior_60p",
    "pct_pop_0014",
    "taux_chomage_rp2022",
    "pop_totale",
]

FEATURES_HIST_2012 = [
    "h12_t1_pct_hollande",
    "h12_t1_pct_sarkozy",
    "h12_t1_pct_lepen",
    "h12_t1_pct_melenchon",
    "h12_t1_pct_bayrou",
    "h12_t1_pct_autres",
    "h12_t2_pct_hollande",
    "h12_t2_pct_sarkozy",
    "h12_t2_marge",
]
FEATURES_HIST_2017 = [
    "h17_t1_pct_macron",
    "h17_t1_pct_fillon",
    "h17_t1_pct_melenchon",
    "h17_t1_pct_lepen",
    "h17_t1_pct_hamon",
    "h17_t1_pct_autres",
    "h17_t2_pct_macron",
    "h17_t2_pct_lepen",
    "h17_t2_marge",
]

FEATURES_2022_T1 = [
    "cible_t1_pct_melenchon",
    "cible_t1_pct_macron",
    "cible_t1_pct_lepen",
    "cible_t1_pct_zemmour",
    "cible_t1_pct_pecresse",
    "cible_t1_pct_jadot",
    "cible_t1_pct_autres",
    "taux_participation_t1",
    "taux_abstention_t1",
]

FEATURE_SETS = {
    "pre_vote": FEATURES_SOCIO + FEATURES_HIST_2012 + FEATURES_HIST_2017,
    "post_t1": FEATURES_SOCIO
    + FEATURES_HIST_2012
    + FEATURES_HIST_2017
    + FEATURES_2022_T1,
    "full": FEATURES_SOCIO + FEATURES_HIST_2012 + FEATURES_HIST_2017 + FEATURES_2022_T1,
    "hist_only": FEATURES_HIST_2012 + FEATURES_HIST_2017,
    "lstm": (
        FEATURES_HIST_2012
        + FEATURES_HIST_2017
        + FEATURES_2022_T1
        + FEATURES_SOCIO_LSTM
        + ["h12_t2_vainqueur", "h17_t2_vainqueur"]
    ),
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced", None],
}
RF_PARAM_GRID_FAST = {
    "n_estimators": [100, 200],
    "max_depth": [None, 15],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"],
}

GB_BEST_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "subsample": 0.8,
    "max_features": "sqrt",
    "random_state": 42,
}
GB_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.03, 0.05, 0.1, 0.15],
    "max_depth": [3, 4, 5],
    "min_samples_split": [5, 10, 20],
    "min_samples_leaf": [3, 5, 10],
    "subsample": [0.7, 0.8, 1.0],
    "max_features": ["sqrt", "log2"],
}

LSTM_CONFIG = {
    "lstm_units_1": 64,
    "lstm_units_2": 32,
    "dense_units_1": 32,
    "dense_units_2": 16,
    "merged_units": 16,
    "dropout": 0.3,
    "dropout_socio": 0.2,
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32,
    "patience": 15,
    "lr_patience": 7,
    "lr_factor": 0.5,
    "min_lr": 1e-5,
}

LSTM_PERIOD_FEATURES = [
    "pct_lepen",
    "pct_gauche",
    "pct_droite",
    "pct_centre",
    "pct_melenchon",
    "pct_autres",
    "marge_t2",
    "vainqueur_t2",
]

CLASSIF_METRICS = [
    "accuracy",
    "f1_weighted",
    "roc_auc",
    "precision_weighted",
    "recall_weighted",
]
REGRESS_METRICS = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
