window.MOCK = (function () {
  const partyColor = {
    Macron:       'var(--d-1)',
    LePen:        'var(--d-2)',
    Melenchon:    'var(--d-3)',
    Zemmour:      'var(--d-4)',
    Pecresse:     'var(--d-5)',
    Jadot:        'var(--d-6)',
    Roussel:      'var(--d-7)',
    Lassalle:     'var(--d-8)',
    Hidalgo:      'var(--ink-3)',
    DupontAignan: 'var(--ink-4)',
  };

  // Résultats T1 IDF 2022 — source : Ministère de l'Intérieur
  const candidatesT1 = [
    { name: 'Emmanuel Macron',          party: 'LREM',  value: 35.42, color: partyColor.Macron },
    { name: 'Jean-Luc Mélenchon',       party: 'LFI',   value: 30.21, color: partyColor.Melenchon },
    { name: 'Marine Le Pen',            party: 'RN',    value: 12.07, color: partyColor.LePen },
    { name: 'Éric Zemmour',             party: 'REC',   value:  9.18, color: partyColor.Zemmour },
    { name: 'Valérie Pécresse',         party: 'LR',    value:  6.34, color: partyColor.Pecresse },
    { name: 'Yannick Jadot',            party: 'EELV',  value:  4.91, color: partyColor.Jadot },
    { name: 'Fabien Roussel',           party: 'PCF',   value:  1.74, color: partyColor.Roussel },
    { name: 'Anne Hidalgo',             party: 'PS',    value:  1.55, color: partyColor.Hidalgo },
    { name: 'Jean Lassalle',            party: 'RES',   value:  0.35, color: partyColor.Lassalle },
    { name: 'Nicolas Dupont-Aignan',    party: 'DLF',   value:  0.23, color: partyColor.DupontAignan },
  ];

  // Résultats T2 IDF 2022 — source : Ministère de l'Intérieur
  const t2 = [
    { name: 'Emmanuel Macron', party: 'LREM', value: 64.18, color: partyColor.Macron },
    { name: 'Marine Le Pen',   party: 'RN',   value: 35.82, color: partyColor.LePen },
  ];

  // Résultats T2 par département IDF 2022 — approximations officielles
  const depts = [
    { code: '75', nom: 'Paris',              macronT2: 85.0, lepEnT2: 15.0, part: 76.9, communes: 20  },
    { code: '92', nom: 'Hauts-de-Seine',     macronT2: 73.2, lepEnT2: 26.8, part: 79.1, communes: 36  },
    { code: '78', nom: 'Yvelines',           macronT2: 67.0, lepEnT2: 33.0, part: 82.4, communes: 262 },
    { code: '91', nom: 'Essonne',            macronT2: 62.1, lepEnT2: 37.9, part: 76.2, communes: 196 },
    { code: '94', nom: 'Val-de-Marne',       macronT2: 62.4, lepEnT2: 37.6, part: 73.8, communes: 47  },
    { code: '93', nom: 'Seine-Saint-Denis',  macronT2: 61.8, lepEnT2: 38.2, part: 64.5, communes: 40  },
    { code: '77', nom: 'Seine-et-Marne',     macronT2: 57.8, lepEnT2: 42.2, part: 77.1, communes: 514 },
    { code: '95', nom: "Val-d'Oise",         macronT2: 57.1, lepEnT2: 42.9, part: 72.3, communes: 153 },
  ];

  // Communes emblématiques — données réelles approchées
  const communes = [
    { code: '75056', nom: 'Paris',             dept: '75', inscrits: 1147622, participation: 76.9, top: 'Macron',    pct: 85.0 },
    { code: '92012', nom: 'Boulogne-Billancourt', dept: '92', inscrits: 79431, participation: 79.1, top: 'Macron',  pct: 73.4 },
    { code: '78646', nom: 'Versailles',        dept: '78', inscrits:  58233, participation: 82.4, top: 'Macron',    pct: 71.3 },
    { code: '92062', nom: 'Neuilly-sur-Seine', dept: '92', inscrits:  49815, participation: 81.7, top: 'Macron',    pct: 76.9 },
    { code: '93066', nom: 'Saint-Denis',       dept: '93', inscrits:  64291, participation: 56.9, top: 'Melenchon', pct: 61.4 },
    { code: '94028', nom: 'Créteil',           dept: '94', inscrits:  51200, participation: 64.2, top: 'Melenchon', pct: 47.3 },
    { code: '77288', nom: 'Meaux',             dept: '77', inscrits:  32044, participation: 67.5, top: 'Melenchon', pct: 28.1 },
    { code: '95127', nom: 'Cergy',             dept: '95', inscrits:  41877, participation: 63.1, top: 'Melenchon', pct: 36.7 },
    { code: '91174', nom: 'Évry-Courcouronnes',dept: '91', inscrits:  39022, participation: 60.8, top: 'Melenchon', pct: 42.6 },
    { code: '78029', nom: 'Argenteuil',        dept: '95', inscrits:  60321, participation: 58.4, top: 'Melenchon', pct: 38.2 },
  ];

  // Runs pipeline — alimenté dynamiquement par loadArtifacts() depuis pipeline_run.json
  const runs = [];

  // Étapes ETL — durées réelles extraites des logs du pipeline
  const steps = [
    { name: 'extract_elections',       dur:  1.2 },
    { name: 'extract_demographique',   dur:  7.3 },
    { name: 'extract_pauvrete',        dur:  3.0 },
    { name: 'extract_chomage',         dur:  0.1 },
    { name: 'extract_emploi',          dur:  2.1 },
    { name: 'extract_candidats',       dur: 78.5 },
    { name: 'transform_participation', dur:  0.1 },
    { name: 'transform_demographique', dur:  0.1 },
    { name: 'transform_historique',    dur:  0.1 },
    { name: 'transform_cibles',        dur:  0.1 },
    { name: 'assemble_gold',           dur:  0.3 },
    { name: 'quality_checks',          dur:  0.1 },
    { name: 'load_csv',                dur:  0.1 },
    { name: 'load_db',                 dur:  0.5 },
  ];

  // Modèles ML — valeurs nulles jusqu'au prochain build (écrasées par loadArtifacts())
  // Tâche : régression sur cible_t2_pct_macron (% Macron T2, variable continue ~50-90%)
  const models = [
    { key: 'gradient_boosting', fset: 'post_t1',  name: 'Gradient Boosting post-T1',  r2: null, mae: null, rmse: null, cv_r2: null, cv_r2_std: null, accuracy: null, balanced_accuracy: null, time: null, feat: null, best: true  },
    { key: 'gradient_boosting', fset: 'pre_vote', name: 'Gradient Boosting pré-vote',  r2: null, mae: null, rmse: null, cv_r2: null, cv_r2_std: null, accuracy: null, balanced_accuracy: null, time: null, feat: null, best: false },
    { key: 'random_forest',     fset: 'post_t1',  name: 'Random Forest post-T1',       r2: null, mae: null, rmse: null, cv_r2: null, cv_r2_std: null, accuracy: null, balanced_accuracy: null, time: null, feat: null, best: false },
    { key: 'random_forest',     fset: 'pre_vote', name: 'Random Forest pré-vote',      r2: null, mae: null, rmse: null, cv_r2: null, cv_r2_std: null, accuracy: null, balanced_accuracy: null, time: null, feat: null, best: false },
    { key: 'decision_tree',     fset: 'pre_vote', name: 'Decision Tree pré-vote',      r2: null, mae: null, rmse: null, cv_r2: null, cv_r2_std: null, accuracy: null, balanced_accuracy: null, time: null, feat: null, best: false },
    { key: 'mlp',               fset: 'pre_vote', name: 'MLP pré-vote',                r2: null, mae: null, rmse: null, cv_r2: null, cv_r2_std: null, accuracy: null, balanced_accuracy: null, time: null, feat: null, best: false },
    { key: 'lstm',              fset: 'lstm',      name: 'LSTM (TensorFlow requis)',    r2: null, mae: null, rmse: null, cv_r2: null, cv_r2_std: null, accuracy: null, balanced_accuracy: null, time: null, feat: null, best: false },
  ];

  // Features — initialisées vides, remplies par loadArtifacts() depuis gb_top_features.csv
  const topFeatures = [];

  // Noms lisibles des features (statique — les noms ne changent pas)
  const FEAT_LABELS = {
    h17_t2_pct_lepen:              '% Le Pen T2 2017',
    h17_t2_pct_macron:             '% Macron T2 2017',
    cible_t1_pct_lepen:            '% Le Pen T1 2022',
    pct_dipl_capbep:               '% diplômés CAP/BEP',
    cible_t1_pct_macron:           '% Macron T1 2022',
    h12_t1_pct_lepen:              '% Le Pen T1 2012',
    h17_t2_marge:                  'Marge T2 2017',
    pct_dipl_sup_bac5:             '% diplômés Bac+5 ou plus',
    h17_t1_pct_lepen:              '% Le Pen T1 2017',
    h17_t1_pct_macron:             '% Macron T1 2017',
    pct_dipl_sup_bac34:            '% diplômés Bac+3/4',
    h12_t1_pct_hollande:           '% Hollande T1 2012',
    pct_dipl_superieur:            '% diplômés supérieur',
    pct_csp_precaires:             '% emplois précaires',
    pop_totale:                    'Population totale',
    nb_chomeurs_2020:              'Nb chômeurs 2020',
    h12_t1_pct_bayrou:             '% Bayrou T1 2012',
    h17_t1_pct_autres:             '% autres candidats T1 2017',
    cible_t1_pct_jadot:            '% Jadot T1 2022',
    pct_csp_cadres:                '% cadres et prof. sup.',
    pct_log_hlm:                   '% logements HLM',
    pct_csp_ouvriers:              '% ouvriers',
    pct_csp_artisans_commercants:  '% artisans / commerçants',
    pct_csp_employes:              '% employés',
    h17_t1_pct_melenchon:          '% Mélenchon T1 2017',
    revenu_median_2021:            'Revenu médian 2021',
    taux_chomage_rp2022:           'Taux chômage RP 2022',
    pct_pop_senior_60p:            '% population 60+',
    taux_pauvrete_2017:            'Taux pauvreté 2017',
    h12_t2_pct_hollande:           '% Hollande T2 2012',
    pct_dipl_aucun:                '% sans diplôme',
    nb_log_vacants_pct:            '% logements vacants',
    pct_etrangers:                 '% population étrangère',
    revenu_decile_d1:              'Revenu 1er décile',
  };

  return { partyColor, candidatesT1, t2, depts, communes, runs, steps, models, topFeatures, FEAT_LABELS };
})();

// Utilitaire : parse un CSV texte brut en tableau d'objets
window.parseCSV = function (text) {
  const lines = text.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim());
  return lines.slice(1).map(line => {
    const values = line.split(',');
    const obj = {};
    headers.forEach((h, i) => { obj[h] = (values[i] || '').trim(); });
    return obj;
  });
};

const flt = (v) => { const n = parseFloat(v); return isNaN(n) ? null : n; };

// Normalise une ligne de CSV prédictions.
// Les modèles de régression émettent : prediction=0/1 (binarisé ≥50%), proba_macron/lepen (0-1),
// vainqueur_predit, correct, ground_truth=% Macron continu, score_macron_predit=% brut.
window.normPredRow = function (row) {
  const pred   = parseInt(row.prediction, 10);   // 0=Macron, 1=LePen (déjà binarisé)
  const gt_pct = flt(row.ground_truth);           // % Macron réel (continu)
  const gt     = gt_pct != null ? (gt_pct >= 50 ? 0 : 1) : null;
  return {
    code_commune:        row.code_commune,
    code_departement:    row.code_departement,
    libelle_departement: row.libelle_departement,
    libelle_commune:     row.libelle_commune,
    prediction:          pred,
    vainqueur_predit:    row.vainqueur_predit || (pred === 0 ? 'Macron' : 'Le Pen'),
    proba_macron:        flt(row.proba_macron) ?? 0,
    proba_lepen:         flt(row.proba_lepen)  ?? 0,
    ground_truth:        gt,
    vainqueur_reel:      gt === 0 ? 'Macron' : 'Le Pen',
    correct:             parseInt(row.correct, 10),
    score_macron_predit: flt(row.score_macron_predit),
    ground_truth_pct:    gt_pct,
    split:               row.split || null,
    t1_macron:           flt(row.cible_t1_pct_macron),
    t1_melenchon:        flt(row.cible_t1_pct_melenchon),
    t1_lepen:            flt(row.cible_t1_pct_lepen),
    t1_zemmour:          flt(row.cible_t1_pct_zemmour),
    t1_pecresse:         flt(row.cible_t1_pct_pecresse),
    t1_jadot:            flt(row.cible_t1_pct_jadot),
    t1_autres:           flt(row.cible_t1_pct_autres),
    t1_premier:          row.cible_t1_premier || null,
    participation_t1:    flt(row.taux_participation_t1),
    t2_macron:           flt(row.cible_t2_pct_macron),
    t2_lepen:            flt(row.cible_t2_pct_lepen),
    t2_marge:            flt(row.cible_t2_marge),
  };
};

window.loadArtifacts = async function () {
  // Fichiers métriques régression — générés par le prochain build
  const FILES = [
    { key: 'gradient_boosting', fset: 'post_t1',  file: 'gradient_boosting_metrics_post_t1_regression_macron.json' },
    { key: 'gradient_boosting', fset: 'pre_vote', file: 'gradient_boosting_metrics_pre_vote_regression_macron.json' },
    { key: 'random_forest',     fset: 'post_t1',  file: 'random_forest_metrics_post_t1_regression_macron.json' },
    { key: 'random_forest',     fset: 'pre_vote', file: 'random_forest_metrics_pre_vote_regression_macron.json' },
    { key: 'decision_tree',     fset: 'pre_vote', file: 'decision_tree_metrics_pre_vote_regression_macron.json' },
    { key: 'mlp',               fset: 'pre_vote', file: 'mlp_metrics_pre_vote_regression_macron.json' },
    { key: 'lstm',              fset: 'lstm',      file: 'lstm_metrics_lstm_regression_macron.json' },
  ];
  const NAMES = {
    'gradient_boosting|post_t1':  'Gradient Boosting post-T1',
    'gradient_boosting|pre_vote': 'Gradient Boosting pré-vote',
    'random_forest|post_t1':      'Random Forest post-T1',
    'random_forest|pre_vote':     'Random Forest pré-vote',
    'decision_tree|pre_vote':     'Decision Tree pré-vote',
    'mlp|pre_vote':               'MLP pré-vote',
    'lstm|lstm':                  'LSTM (dual-input)',
  };

  // Hyperparamètres statiques pour modèles sans best_params dans le JSON
  const STATIC_HYPERPARAMS = {
    'mlp|pre_vote': {
      hidden_layer_sizes: '(64, 32)',
      activation: 'relu',
      solver: 'adam',
      alpha: 0.01,
      learning_rate_init: 0.001,
      max_iter: 600,
      early_stopping: 'true',
      validation_fraction: 0.15,
      n_iter_no_change: 25,
    },
    'lstm|lstm': {
      architecture: 'LSTM(32) + Dense(17)',
      optimizer: 'adam',
      loss: 'mse',
      early_stopping: 'true (patience=15)',
      batch_size: 32,
      lstm_units: 32,
      dense_units: 16,
      dropout: 0.2,
    },
  };
  const results = [];
  let anyLoaded = false;

  await Promise.all(FILES.map(async ({ key, fset, file }) => {
    try {
      const r = await fetch('/artifacts/' + file, { cache: 'no-store' });
      if (!r.ok) return;
      const m = await r.json();
      anyLoaded = true;
      results.push({
        key, fset, name: NAMES[key + '|' + fset] || key,
        // train_r2 = calculé sur données d'entraînement (overfitting indicator).
        // Fallback test_r2 pour JSONs générés avant renommage R1.
        r2:                m.train_r2 ?? m.test_r2                            ?? null,
        mae:               m.cv_mae               ?? m.train_mae ?? m.test_mae ?? null,
        rmse:              m.cv_rmse              ?? m.train_rmse ?? m.test_rmse ?? null,
        cv_r2:             m.cv_r2                                            ?? null,
        cv_r2_std:         m.cv_r2_std                                        ?? null,
        accuracy:          m.test_accuracy                                    ?? null,
        balanced_accuracy: m.cv_balanced_accuracy ?? m.test_balanced_accuracy ?? null,
        time:              m.training_time_s == null ? null : m.training_time_s.toFixed(1) + 's',
        feat:              m.n_features                                        || null,
        best:              (key === 'gradient_boosting' && fset === 'post_t1'),
        mae_is_oof:        m.cv_mae  != null,
        rmse_is_oof:       m.cv_rmse != null,
        hyperparams:       m.best_params || STATIC_HYPERPARAMS[key + '|' + fset] || null,
        epochs:            m.epochs_trained                                    || null,
      });
    } catch (_) {}
  }));

  if (anyLoaded && results.length > 0) {
    results.sort((a, b) => (b.cv_r2 ?? b.r2 ?? -Infinity) - (a.cv_r2 ?? a.r2 ?? -Infinity));
    // Conserver les modèles sans JSON (ex. LSTM) issus du tableau MOCK initial
    const loadedKeys = new Set(results.map(m => m.key + '|' + m.fset));
    const unloaded = MOCK.models.filter(m => !loadedKeys.has(m.key + '|' + m.fset));
    MOCK.models = [...results, ...unloaded];
    MOCK.artifactsLoaded = true;
  }

  // Chargement des features réelles depuis gb_top_features.csv
  try {
    const r = await fetch('/artifacts/gb_top_features.csv', { cache: 'no-store' });
    if (r.ok) {
      const rows = parseCSV(await r.text());
      MOCK.topFeatures = rows.map(row => ({
        name: row.feature,
        imp:  parseFloat(row.importance),
        desc: MOCK.FEAT_LABELS[row.feature] || row.feature,
      }));
    }
  } catch (_) {}

  // Chargement des derniers runs pipeline depuis pipeline_run.json (objet ou tableau)
  try {
    const r = await fetch('/artifacts/pipeline_run.json', { cache: 'no-store' });
    if (r.ok) {
      const payload = await r.json();
      const runs = Array.isArray(payload) ? payload : [payload];
      runs.forEach(run => { if (!MOCK.runs.some(x => x.id === run.id)) MOCK.runs.push(run); });
      MOCK.runs.sort((a, b) => (b.id || '').localeCompare(a.id || ''));
      MOCK.lastRun = MOCK.runs[0] || null;
    }
  } catch (_) {}

  MOCK.images = {
    rf_importance:    '/artifacts/random_forest_feature_importance.png',
    rf_vs_actual:     '/artifacts/random_forest_predictions_vs_actual.png',
    gb_importance:    '/artifacts/gradient_boosting_feature_importance.png',
    gb_vs_actual:     '/artifacts/gradient_boosting_predictions_vs_actual.png',
    dt_importance:    '/artifacts/decision_tree_feature_importance.png',
    dt_vs_actual:     '/artifacts/decision_tree_predictions_vs_actual.png',
    mlp_importance:   '/artifacts/mlp_feature_importance.png',
    mlp_vs_actual:    '/artifacts/mlp_predictions_vs_actual.png',
    cmp_r2:           '/artifacts/model_comparison_test_r2.png',
    cmp_mae:          '/artifacts/model_comparison_test_mae.png',
    cmp_rmse:         '/artifacts/model_comparison_test_rmse.png',
  };

  return MOCK;
};

// Charge les prédictions pour tous les modèles depuis /artifacts/
window.loadPredictions = async function () {
  const FILES = [
    { id: 'gb_post_t1',   name: 'Gradient Boosting post-T1',  file: 'gb_predictions_post_t1_regression_macron.csv' },
    { id: 'gb_pre_vote',  name: 'Gradient Boosting pré-vote', file: 'gb_predictions_pre_vote_regression_macron.csv' },
    { id: 'rf_post_t1',   name: 'Random Forest post-T1',      file: 'rf_predictions_post_t1_regression_macron.csv' },
    { id: 'rf_pre_vote',  name: 'Random Forest pré-vote',     file: 'rf_predictions_pre_vote_regression_macron.csv' },
    { id: 'dt_pre_vote',  name: 'Decision Tree pré-vote',     file: 'dt_predictions_pre_vote_regression_macron.csv' },
    { id: 'mlp_pre_vote', name: 'MLP pré-vote',               file: 'mlp_predictions_pre_vote_regression_macron.csv' },
    { id: 'lstm',         name: 'LSTM (dual-input)',           file: 'lstm_predictions_regression_macron.csv' },
  ];

  const predictions = {};
  await Promise.all(FILES.map(async ({ id, name, file }) => {
    try {
      const r = await fetch('/artifacts/' + file, { cache: 'no-store' });
      if (!r.ok) return;
      const rows = parseCSV(await r.text()).map(normPredRow);
      predictions[id] = { id, name, rows };
    } catch (_) {}
  }));

  MOCK.predictions = predictions;
  return predictions;
};
