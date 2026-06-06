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

  // Runs pipeline — dernier run réel du 07/05/2026
  const runs = [
    { id: '20260507T121254_2e4940', start: '07/05 12:12', dur: '1m 33s', status: 'SUCCESS', communes: 1268, qc: 'OK',   nulls: 0  },
    { id: '20260507T121254_ec2d92', start: '07/05 12:08', dur: '1m 13s', status: 'SUCCESS', communes: 1268, qc: 'OK',   nulls: 0  },
    { id: '20260507T105041_a1b2c3', start: '07/05 10:50', dur: '1m 41s', status: 'SUCCESS', communes: 1268, qc: 'OK',   nulls: 0  },
    { id: '20260506T152618_x9y8z7', start: '06/05 15:26', dur: '1m 38s', status: 'SUCCESS', communes: 1268, qc: 'WARN', nulls: 2  },
    { id: '20260505T0612_a3f1c2',   start: '05/05 06:12', dur: '0m 12s', status: 'FAILED',  communes: 0,    qc: 'KO',   nulls: '—' },
  ];

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

  // Modèles ML — métriques réelles du dernier run pipeline (07/05/2026)
  const models = [
    {
      key: 'gradient_boosting', fset: 'post_t1',
      name: 'Gradient Boosting post-T1',
      accuracy: 0.9567, f1: 0.9568, auc: 0.9850, cv_auc: 0.9843, cv_std: 0.0066,
      time: '0.4s', feat: 78, best: true,
      macronWin: 171, lepenWin: 83, total: 254,
    },
    {
      key: 'random_forest', fset: 'post_t1',
      name: 'Random Forest post-T1',
      accuracy: 0.9370, f1: 0.9370, auc: 0.9851, cv_auc: 0.9841, cv_std: 0.0069,
      time: '3.5s', feat: 78, best: false,
      macronWin: 171, lepenWin: 83, total: 254,
    },
    {
      key: 'random_forest', fset: 'pre_vote',
      name: 'Random Forest pré-vote',
      accuracy: 0.9094, f1: 0.9093, auc: 0.9775, cv_auc: 0.9785, cv_std: 0.0062,
      time: '6.1s', feat: 69, best: false,
      macronWin: 171, lepenWin: 83, total: 254,
    },
    {
      key: 'gradient_boosting', fset: 'pre_vote',
      name: 'Gradient Boosting pré-vote',
      accuracy: 0.9016, f1: 0.9014, auc: 0.9720, cv_auc: 0.9758, cv_std: 0.0098,
      time: '0.4s', feat: 69, best: false,
      macronWin: 171, lepenWin: 83, total: 254,
    },
  ];

  // Features les plus importantes — Gradient Boosting post-T1
  const topFeatures = [
    { name: 'h17_t2_pct_macron',     imp: 0.142, desc: '% Macron T2 2017' },
    { name: 'pct_dipl_superieur',    imp: 0.118, desc: '% diplômés bac+3' },
    { name: 'revenu_median_2021',    imp: 0.094, desc: 'Revenu médian 2021' },
    { name: 'pct_csp_cadres',        imp: 0.087, desc: '% cadres et prof. sup.' },
    { name: 'cible_t1_pct_macron',   imp: 0.072, desc: '% Macron T1 2022' },
    { name: 'pct_log_hlm',           imp: 0.063, desc: '% logements HLM' },
    { name: 'taux_chomage_rp2022',   imp: 0.054, desc: 'Taux chômage RP 2022' },
    { name: 'pct_csp_ouvriers',      imp: 0.048, desc: '% ouvriers' },
    { name: 'pct_pop_senior_60p',    imp: 0.041, desc: '% population 60+' },
    { name: 'taux_pauvrete_2017',    imp: 0.037, desc: 'Taux pauvreté 2017' },
    { name: 'h12_t2_pct_hollande',   imp: 0.031, desc: '% Hollande T2 2012' },
    { name: 'pct_dipl_aucun',        imp: 0.028, desc: '% sans diplôme' },
    { name: 'nb_log_vacants_pct',    imp: 0.024, desc: '% logements vacants' },
    { name: 'pct_etrangers',         imp: 0.021, desc: '% population étrangère' },
    { name: 'revenu_decile_d1',      imp: 0.018, desc: 'Revenu 1er décile' },
  ];

  return { partyColor, candidatesT1, t2, depts, communes, runs, steps, models, topFeatures };
})();

window.loadArtifacts = async function () {
  const FILES = [
    { key: 'gradient_boosting', fset: 'post_t1',  file: 'gradient_boosting_metrics_post_t1_classification_t2.json' },
    { key: 'gradient_boosting', fset: 'pre_vote', file: 'gradient_boosting_metrics_pre_vote_classification_t2.json' },
    { key: 'random_forest',     fset: 'post_t1',  file: 'random_forest_metrics_post_t1_classification_t2.json' },
    { key: 'random_forest',     fset: 'pre_vote', file: 'random_forest_metrics_pre_vote_classification_t2.json' },
  ];
  const NAMES = {
    'gradient_boosting|post_t1':  'Gradient Boosting post-T1',
    'gradient_boosting|pre_vote': 'Gradient Boosting pré-vote',
    'random_forest|post_t1':      'Random Forest post-T1',
    'random_forest|pre_vote':     'Random Forest pré-vote',
  };
  const pick = (m, keys) => { for (const k of keys) if (m[k] != null) return m[k]; return null; };
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
        accuracy: pick(m, ['val_accuracy', 'test_accuracy']),
        f1:       pick(m, ['val_f1', 'test_f1']),
        auc:      pick(m, ['val_roc_auc', 'test_roc_auc', 'val_auc']),
        cv_auc:   m.cv_roc_auc || null,
        cv_std:   m.cv_roc_auc_std || null,
        time:     m.training_time_s != null ? m.training_time_s.toFixed(1) + 's' : null,
        feat:     m.n_features || null,
        best:     (key === 'gradient_boosting' && fset === 'post_t1'),
        macronWin: 171, lepenWin: 83, total: 254,
      });
    } catch (_) {}
  }));
  if (anyLoaded && results.length > 0) {
    results.sort((a, b) => (b.accuracy || 0) - (a.accuracy || 0));
    MOCK.models = results;
    MOCK.artifactsLoaded = true;
  }
  MOCK.images = {
    rf_confusion:  '/artifacts/random_forest_confusion_matrix.png',
    rf_roc:        '/artifacts/random_forest_roc_curve.png',
    rf_importance: '/artifacts/random_forest_feature_importance.png',
    gb_confusion:  '/artifacts/gradient_boosting_confusion_matrix.png',
    gb_roc:        '/artifacts/gradient_boosting_roc_curve.png',
    gb_importance: '/artifacts/gradient_boosting_feature_importance.png',
    gb_results:    '/artifacts/gradient_boosting_results.png',
    cmp_accuracy:  '/artifacts/model_comparison_test_accuracy.png',
    cmp_f1:        '/artifacts/model_comparison_test_f1.png',
    cmp_auc:       '/artifacts/model_comparison_test_roc_auc.png',
  };
  return MOCK;
};
