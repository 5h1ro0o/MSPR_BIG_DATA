CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;
CREATE SCHEMA IF NOT EXISTS audit;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS silver.communes (
    code_commune        CHAR(5)      PRIMARY KEY,
    code_departement    CHAR(2)      NOT NULL,
    libelle_departement VARCHAR(100),
    libelle_commune     VARCHAR(100),
    created_at          TIMESTAMPTZ  DEFAULT NOW(),
    updated_at          TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS silver.participation_elections (
    id                  UUID         DEFAULT uuid_generate_v4() PRIMARY KEY,
    code_commune        CHAR(5)      NOT NULL REFERENCES silver.communes(code_commune),
    id_election         VARCHAR(30)  NOT NULL,
    inscrits            INTEGER,
    abstentions         INTEGER,
    votants             INTEGER,
    blancs              INTEGER,
    nuls                INTEGER,
    exprimes            INTEGER,
    taux_abstention     NUMERIC(5,2),
    taux_participation  NUMERIC(5,2),
    created_at          TIMESTAMPTZ  DEFAULT NOW(),
    UNIQUE (code_commune, id_election)
);

CREATE TABLE IF NOT EXISTS silver.resultats_candidats (
    id              UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
    code_commune    CHAR(5)     NOT NULL REFERENCES silver.communes(code_commune),
    id_election     VARCHAR(30) NOT NULL,
    nom_candidat    VARCHAR(100) NOT NULL,
    voix            INTEGER,
    pct_exprimes    NUMERIC(5,2),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (code_commune, id_election, nom_candidat)
);

CREATE TABLE IF NOT EXISTS silver.features_socioeconomiques (
    code_commune                    CHAR(5)      PRIMARY KEY REFERENCES silver.communes(code_commune),
    pop_totale                      INTEGER,
    pct_pop_0014                    NUMERIC(5,2),
    pct_pop_1529                    NUMERIC(5,2),
    pct_pop_3044                    NUMERIC(5,2),
    pct_pop_4559                    NUMERIC(5,2),
    pct_pop_6074                    NUMERIC(5,2),
    pct_pop_7589                    NUMERIC(5,2),
    pct_pop_90p                     NUMERIC(5,2),
    pct_pop_senior_60p              NUMERIC(5,2),
    ratio_hommes_femmes             NUMERIC(6,3),
    pct_csp_cadres                  NUMERIC(5,2),
    pct_csp_employes                NUMERIC(5,2),
    pct_csp_ouvriers                NUMERIC(5,2),
    pct_csp_precaires               NUMERIC(5,2),
    pct_csp_cols_blancs             NUMERIC(5,2),
    pct_dipl_superieur              NUMERIC(5,2),
    taux_chomage_rp2022             NUMERIC(5,2),
    revenu_median_2021              NUMERIC(10,2),
    revenu_median_2017              NUMERIC(10,2),
    taux_pauvrete_2017              NUMERIC(5,2),
    rapport_interdecile_2017        NUMERIC(6,3),
    pct_log_proprietaires           NUMERIC(5,2),
    pct_log_locataires              NUMERIC(5,2),
    pct_log_hlm                     NUMERIC(5,2),
    pct_men_seuls                   NUMERIC(5,2),
    created_at                      TIMESTAMPTZ  DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gold.ml_features (
    code_commune        CHAR(5)      PRIMARY KEY,
    created_at          TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit.pipeline_runs (
    run_id          VARCHAR(50)  PRIMARY KEY,
    started_at      TIMESTAMPTZ  NOT NULL,
    ended_at        TIMESTAMPTZ,
    status          VARCHAR(20)  NOT NULL DEFAULT 'RUNNING',
    n_communes      INTEGER,
    n_nulls         INTEGER,
    qc_passed       BOOLEAN,
    qc_errors       JSONB,
    metrics         JSONB,
    error_message   TEXT,
    created_at      TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit.quality_checks (
    id              UUID         DEFAULT uuid_generate_v4() PRIMARY KEY,
    run_id          VARCHAR(50)  REFERENCES audit.pipeline_runs(run_id),
    check_name      VARCHAR(100) NOT NULL,
    passed          BOOLEAN      NOT NULL,
    details         TEXT,
    checked_at      TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gold.model_predictions (
    id              UUID         DEFAULT uuid_generate_v4() PRIMARY KEY,
    run_id          VARCHAR(50)  NOT NULL,
    model_name      VARCHAR(50)  NOT NULL,
    feature_set     VARCHAR(50),
    target          VARCHAR(50)  NOT NULL,
    code_commune    CHAR(5),
    libelle_commune VARCHAR(100),
    code_departement CHAR(2),
    prediction      VARCHAR(100),
    probability     NUMERIC(8,6),
    ground_truth    VARCHAR(100),
    correct         BOOLEAN,
    created_at      TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pred_run_model
    ON gold.model_predictions (run_id, model_name);
CREATE INDEX IF NOT EXISTS idx_pred_commune
    ON gold.model_predictions (code_commune);

CREATE TABLE IF NOT EXISTS gold.model_metrics (
    id           UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
    run_id       VARCHAR(50) NOT NULL,
    model_name   VARCHAR(50) NOT NULL,
    feature_set  VARCHAR(50),
    target       VARCHAR(50) NOT NULL,
    metric_name  VARCHAR(50) NOT NULL,
    metric_value NUMERIC(12, 6),
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_metrics_run_model
    ON gold.model_metrics (run_id, model_name);

CREATE OR REPLACE VIEW gold.latest_model_metrics AS
SELECT DISTINCT ON (model_name, feature_set, target, metric_name)
    model_name, feature_set, target, metric_name, metric_value, run_id, created_at
FROM gold.model_metrics
ORDER BY model_name, feature_set, target, metric_name, created_at DESC;

DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'etl_writer') THEN
        CREATE ROLE etl_writer LOGIN PASSWORD 'change_me_etl_writer';
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'ds_reader') THEN
        CREATE ROLE ds_reader LOGIN PASSWORD 'change_me_ds_reader';
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'etl_admin') THEN
        CREATE ROLE etl_admin LOGIN PASSWORD 'change_me_etl_admin' SUPERUSER;
    END IF;
END $$;

GRANT USAGE ON SCHEMA bronze, silver, gold, audit TO etl_writer;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA bronze TO etl_writer;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA silver TO etl_writer;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA gold TO etl_writer;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA audit TO etl_writer;

GRANT USAGE ON SCHEMA silver, gold TO ds_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA silver TO ds_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA gold TO ds_reader;

CREATE INDEX IF NOT EXISTS idx_participation_commune
    ON silver.participation_elections (code_commune);
CREATE INDEX IF NOT EXISTS idx_participation_election
    ON silver.participation_elections (id_election);
CREATE INDEX IF NOT EXISTS idx_resultats_commune
    ON silver.resultats_candidats (code_commune);
CREATE INDEX IF NOT EXISTS idx_communes_dept
    ON silver.communes (code_departement);
CREATE INDEX IF NOT EXISTS idx_runs_status
    ON audit.pipeline_runs (status, started_at);

-- ============================================================
-- SCHÉMA EN ÉTOILE (DATAMART) — Modèle multidimensionnel
-- Couche analytique/décisionnelle distincte de la couche gold ML
-- Architecture : 1 table de faits centrale + 4 tables de dimensions
-- ============================================================
CREATE SCHEMA IF NOT EXISTS datamart;

-- ── Dimension Commune ─────────────────────────────────────────────────────────
-- Grain : 1 ligne par commune IDF (1 268 communes)
CREATE TABLE IF NOT EXISTS datamart.dim_commune (
    sk_commune          SERIAL       PRIMARY KEY,
    code_commune        CHAR(5)      NOT NULL UNIQUE,
    code_departement    CHAR(2)      NOT NULL,
    libelle_departement VARCHAR(100),
    libelle_commune     VARCHAR(100),
    -- Indicateurs socio-économiques de référence (millésime 2022)
    revenu_median       NUMERIC(10,2),
    taux_chomage        NUMERIC(6,2),
    taux_pauvrete       NUMERIC(6,2),
    pct_dipl_superieur  NUMERIC(6,2),
    pct_csp_cadres      NUMERIC(6,2),
    pct_log_hlm         NUMERIC(6,2),
    pop_totale          INTEGER,
    -- Nouvelles sources (nullable si données non encore chargées)
    taux_crimes_total   NUMERIC(8,2),
    nb_entreprises      INTEGER,
    nb_associations     INTEGER,
    updated_at          TIMESTAMPTZ  DEFAULT NOW()
);

-- ── Dimension Candidat ────────────────────────────────────────────────────────
-- Grain : 1 ligne par candidat ayant participé à au moins une élection couverte
CREATE TABLE IF NOT EXISTS datamart.dim_candidat (
    sk_candidat         SERIAL       PRIMARY KEY,
    nom_candidat        VARCHAR(100) NOT NULL,
    parti               VARCHAR(50),
    orientation         VARCHAR(30), -- gauche / centre / droite / extreme_gauche / extreme_droite
    couleur_hex         CHAR(7),
    UNIQUE (nom_candidat, parti)
);

-- ── Dimension Temps ───────────────────────────────────────────────────────────
-- Grain : 1 ligne par scrutin (année + tour)
CREATE TABLE IF NOT EXISTS datamart.dim_temps (
    sk_temps            SERIAL       PRIMARY KEY,
    annee               SMALLINT     NOT NULL,
    tour                SMALLINT     NOT NULL CHECK (tour IN (1, 2)),
    type_election       VARCHAR(50)  DEFAULT 'présidentielle',
    date_scrutin        DATE,
    label               VARCHAR(50)  NOT NULL, -- ex. "T1 2022"
    UNIQUE (annee, tour)
);

-- ── Dimension Contexte Socio-Économique ───────────────────────────────────────
-- Grain : 1 ligne par (commune, millésime) — permet comparaison historique
CREATE TABLE IF NOT EXISTS datamart.dim_contexte (
    sk_contexte         SERIAL       PRIMARY KEY,
    code_commune        CHAR(5)      NOT NULL,
    millesime           SMALLINT     NOT NULL, -- année de la donnée (ex. 2017, 2022)
    revenu_median       NUMERIC(10,2),
    taux_chomage        NUMERIC(6,2),
    taux_pauvrete       NUMERIC(6,2),
    pct_dipl_superieur  NUMERIC(6,2),
    pct_csp_cadres      NUMERIC(6,2),
    pct_pop_senior_60p  NUMERIC(6,2),
    pct_log_hlm         NUMERIC(6,2),
    UNIQUE (code_commune, millesime)
);

-- ── Table de Faits : Résultats Électoraux ─────────────────────────────────────
-- Grain : 1 ligne par (commune × candidat × tour × élection)
-- Mesures : votes exprimés, % obtenu, participation, résultats agrégés
CREATE TABLE IF NOT EXISTS datamart.fait_resultats_electoraux (
    sk_fait             BIGSERIAL    PRIMARY KEY,
    -- Clés étrangères vers les dimensions
    sk_commune          INTEGER      NOT NULL REFERENCES datamart.dim_commune(sk_commune),
    sk_candidat         INTEGER      NOT NULL REFERENCES datamart.dim_candidat(sk_candidat),
    sk_temps            INTEGER      NOT NULL REFERENCES datamart.dim_temps(sk_temps),
    sk_contexte         INTEGER      REFERENCES datamart.dim_contexte(sk_contexte),
    -- Mesures électorales
    nb_inscrits         INTEGER,
    nb_votants          INTEGER,
    nb_exprimes         INTEGER,
    nb_voix             INTEGER,
    pct_voix            NUMERIC(6,3), -- % des voix exprimées
    pct_inscrits        NUMERIC(6,3), -- % des inscrits
    taux_participation  NUMERIC(6,3),
    taux_abstention     NUMERIC(6,3),
    -- Résultat binaire T2 (Macron=0, Le Pen=1) — uniquement pour le T2
    vainqueur_t2        SMALLINT,
    marge_vainqueur_t2  NUMERIC(6,3),
    -- Prédiction modèle (jointure possible)
    proba_macron_gb     NUMERIC(6,4),
    proba_lepen_gb      NUMERIC(6,4),
    -- Audit
    loaded_at           TIMESTAMPTZ  DEFAULT NOW()
);

-- ── Pré-remplissage des dimensions statiques ─────────────────────────────────
INSERT INTO datamart.dim_temps (annee, tour, label, date_scrutin)
VALUES
    (2012, 1, 'T1 2012', '2012-04-22'),
    (2012, 2, 'T2 2012', '2012-05-06'),
    (2017, 1, 'T1 2017', '2017-04-23'),
    (2017, 2, 'T2 2017', '2017-05-07'),
    (2022, 1, 'T1 2022', '2022-04-10'),
    (2022, 2, 'T2 2022', '2022-04-24')
ON CONFLICT (annee, tour) DO NOTHING;

INSERT INTO datamart.dim_candidat (nom_candidat, parti, orientation, couleur_hex)
VALUES
    ('Emmanuel Macron',       'LREM/Renaissance', 'centre',          '#5B9BD5'),
    ('Marine Le Pen',         'RN',               'extreme_droite',  '#E15759'),
    ('Jean-Luc Mélenchon',    'LFI',              'extreme_gauche',  '#59A14F'),
    ('Éric Zemmour',          'Reconquête',       'extreme_droite',  '#9B59B6'),
    ('Valérie Pécresse',      'LR',               'droite',          '#F28E2B'),
    ('Yannick Jadot',         'EELV',             'gauche',          '#76B7B2'),
    ('François Hollande',     'PS',               'gauche',          '#E05C5C'),
    ('Nicolas Sarkozy',       'UMP/LR',           'droite',          '#3366CC'),
    ('François Fillon',       'LR',               'droite',          '#B07AA1'),
    ('Benoît Hamon',          'PS',               'gauche',          '#FF9DA7'),
    ('Autres candidats',      'divers',           'divers',          '#BAB0AC')
ON CONFLICT (nom_candidat, parti) DO NOTHING;

-- ── Index pour performances des requêtes analytiques ─────────────────────────
CREATE INDEX IF NOT EXISTS idx_fait_commune
    ON datamart.fait_resultats_electoraux (sk_commune);
CREATE INDEX IF NOT EXISTS idx_fait_temps
    ON datamart.fait_resultats_electoraux (sk_temps);
CREATE INDEX IF NOT EXISTS idx_fait_candidat
    ON datamart.fait_resultats_electoraux (sk_candidat);
CREATE INDEX IF NOT EXISTS idx_dim_commune_dept
    ON datamart.dim_commune (code_departement);
CREATE INDEX IF NOT EXISTS idx_contexte_commune
    ON datamart.dim_contexte (code_commune, millesime);
