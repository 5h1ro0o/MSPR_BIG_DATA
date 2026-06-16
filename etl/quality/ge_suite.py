"""
Great Expectations — Suite de validation du dataset gold.
=========================================================
Outil DQM reconnu (https://greatexpectations.io) pour le nettoyage,
la validation et la documentation de la qualité des données.

Remplace / complète le module checks.py avec des Expectations déclaratives,
traçables et réutilisables.
"""

from __future__ import annotations

import logging

import pandas as pd

# ── Silencer AVANT l'import de GE ─────────────────────────────────────────────
# Les messages INFO (_docs_decorators, registry) sont émis pendant l'import
# lui-même. Le setLevel doit être positionné avant, et propagate=False empêche
# les sous-loggers GE de remonter jusqu'au handler root de logging.basicConfig.
for _ge_name in [
    "great_expectations",
    "great_expectations._docs_decorators",
    "great_expectations.expectations.registry",
]:
    _lg = logging.getLogger(_ge_name)
    _lg.setLevel(logging.CRITICAL)
    _lg.propagate = False
    _lg.addHandler(logging.NullHandler())

from monitoring.logger import get_logger

log = get_logger(__name__)

try:
    import great_expectations as gx

    HAS_GE = True
except ImportError:
    HAS_GE = False
    log.warning(
        "great-expectations non installé — validation GE désactivée. "
        "Lancez : pip install great-expectations>=0.18.0"
    )

DEPTS_IDF = {"75", "77", "78", "91", "92", "93", "94", "95"}


def build_suite(context) -> "gx.ExpectationSuite":
    """Définit les Expectations sur le dataset gold."""
    suite = context.suites.add(gx.ExpectationSuite(name="gold_suite"))

    # ── Table-level ──────────────────────────────────────────────────────────
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=1_200, max_value=1_400)
    )
    suite.add_expectation(gx.expectations.ExpectTableColumnCountToBeBetween(min_value=50, max_value=130))

    # ── Colonne clé : code_commune ────────────────────────────────────────────
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="code_commune"))
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="code_commune")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(column="code_commune")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValueLengthsToEqual(column="code_commune", value=5)
    )

    # ── Département IDF ───────────────────────────────────────────────────────
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="code_departement",
            value_set=list(DEPTS_IDF),
        )
    )

    # ── Variables cibles (T2 2022) ────────────────────────────────────────────
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="cible_t2_vainqueur")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="cible_t2_vainqueur", value_set=[0, 1]
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="cible_t2_pct_macron", min_value=0.0, max_value=100.0
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="cible_t2_pct_lepen", min_value=0.0, max_value=100.0
        )
    )

    # ── Indicateurs socio-économiques ─────────────────────────────────────────
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="taux_chomage_rp2022", min_value=0.0, max_value=60.0, mostly=0.95
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="revenu_median_2021",
            min_value=5_000.0,
            max_value=150_000.0,
            mostly=0.99,
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="taux_pauvrete_2017", min_value=0.0, max_value=100.0, mostly=0.95
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="pct_dipl_superieur", min_value=0.0, max_value=100.0
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="pct_csp_cadres", min_value=0.0, max_value=100.0
        )
    )

    return suite


def run_ge_suite(df: pd.DataFrame) -> dict:
    """
    Exécute la suite Great Expectations sur le DataFrame gold.

    Returns:
        dict {success, passed, failed, total, results}
    """
    if not HAS_GE:
        return {"success": True, "passed": 0, "failed": 0, "total": 0, "skipped": True}

    try:
        context = gx.get_context(mode="ephemeral")

        data_source = context.data_sources.add_pandas("elections_gold_source")
        data_asset = data_source.add_dataframe_asset("gold_dataset")
        batch_def = data_asset.add_batch_definition_whole_dataframe("full_batch")

        suite = build_suite(context)

        validation_def = context.validation_definitions.add(
            gx.ValidationDefinition(
                name="gold_validation",
                data=batch_def,
                suite=suite,
            )
        )

        result = validation_def.run(batch_parameters={"dataframe": df})

        passed = sum(1 for r in result.results if r.success)
        failed = sum(1 for r in result.results if not r.success)
        total = len(result.results)

        suffix = f" — {failed} ÉCHEC(S)" if failed else " ✓"
        log.info(f"Great Expectations : {passed}/{total} expectations passées{suffix}")
        for r in result.results:
            if not r.success:
                col = getattr(r.expectation_config, "kwargs", {}).get("column", "table")
                log.warning(
                    f"  GE FAIL | {r.expectation_config.type} | col={col} | {r.result}"
                )

        return {
            "success": result.success,
            "passed": passed,
            "failed": failed,
            "total": total,
            "results": [
                {
                    "expectation": r.expectation_config.type,
                    "success": r.success,
                    "column": getattr(r.expectation_config, "kwargs", {}).get(
                        "column", "table"
                    ),
                }
                for r in result.results
            ],
        }

    except Exception as exc:
        log.error(f"Great Expectations ERREUR : {exc}")
        return {
            "success": False,
            "passed": 0,
            "failed": 1,
            "total": 1,
            "error": str(exc),
        }
