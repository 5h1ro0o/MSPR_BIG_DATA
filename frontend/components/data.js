// Données de l'Electoral Observatory.
// Charge les vrais artefacts ML depuis /artifacts/ (quand servi par nginx après docker-compose up).
// Fallback sur les données mock si les fichiers ne sont pas encore générés.

window.MOCK = (function () {
  const partyColor = {
    'Macron':        'var(--d-1)',
    'Le Pen':        'var(--d-2)',
    'Melenchon':     'var(--d-3)',
    'Zemmour':       'var(--d-4)',
    'Pecresse':      'var(--d-5)',
    'Jadot':         'var(--d-6)',
    'Roussel':       'var(--d-7)',
    'Lassalle':      'var(--d-8)',
    'Hidalgo':       'var(--ink-3)',
    'Dupont-Aignan': 'var(--ink-4)',
    'Poutou':        'var(--ink-5)',
    'Arthaud':       'var(--ink-2)',
  };

  const candidatesT1 = [
    { name: 'Emmanuel Macron',           party: 'La R\u00e9publique en marche', value: 35.42, color: partyColor['Macron'] },
    { name: 'Jean-Luc M\u00e9lenchon',   party: 'La France insoumise',          value: 30.21, color: partyColor['Melenchon'] },
    { name: 'Marine Le Pen',             party: 'Rassemblement national',        value: 12.07, color: partyColor['Le Pen'] },
    { name: '\u00c9ric Zemmour',         party: 'Reconqu\u00eate',              value:  9.18, color: partyColor['Zemmour'] },
    { name: 'Val\u00e9rie P\u00e9cresse',party: 'Les R\u00e9publicains',        value:  6.34, color: partyColor['Pecresse'] },
    { name: 'Yannick Jadot',             party: 'Europe \u00c9cologie',          value:  4.91, color: partyColor['Jadot'] },
    { name: 'Anne Hidalgo',              party: 'Parti socialiste',              value:  1.55, color: partyColor['Hidalgo'] },
    { name: 'Fabien Roussel',            party: 'Parti communiste',              value:  1.74, color: partyColor['Roussel'] },
  ];

  const t2 = [
    { name: 'Emmanuel Macron', party: 'LREM', value: 64.18, color: partyColor['Macron'] },
    { name: 'Marine Le Pen',   party: 'RN',   value: 35.82, color: partyColor['Le Pen'] },
  ];

  const communes = [
    { code: '75001', nom: 'Paris 1er',              dept: '75', inscrits: 12041, participation: 78.4, top: 'Macron',     pct: 47.2 },
    { code: '75104', nom: 'Paris 4e',               dept: '75', inscrits: 17562, participation: 76.8, top: 'Melenchon',  pct: 38.1 },
    { code: '78646', nom: 'Versailles',             dept: '78', inscrits: 58233, participation: 82.4, top: 'Macron',     pct: 41.7 },
    { code: '93066', nom: 'Saint-Denis',            dept: '93', inscrits: 64291, participation: 56.9, top: 'Melenchon',  pct: 61.4 },
    { code: '92012', nom: 'Boulogne-Billancourt',   dept: '92', inscrits: 79431, participation: 79.1, top: 'Macron',     pct: 38.9 },
    { code: '94028', nom: 'Cr\u00e9teil',           dept: '94', inscrits: 51200, participation: 64.2, top: 'Melenchon',  pct: 47.3 },
    { code: '77288', nom: 'Meaux',                  dept: '77', inscrits: 32044, participation: 67.5, top: 'Le Pen',     pct: 24.8 },
    { code: '95127', nom: 'Cergy',                  dept: '95', inscrits: 41877, participation: 63.1, top: 'Melenchon',  pct: 36.7 },
    { code: '91174', nom: '\u00c9vry-Courcouronnes',dept: '91', inscrits: 39022, participation: 60.8, top: 'Melenchon',  pct: 42.6 },
    { code: '92062', nom: 'Neuilly-sur-Seine',      dept: '92', inscrits: 49815, participation: 81.7, top: 'Macron',     pct: 53.4 },
  ];

  const runs = [
    { id: '20260505T0612_a3f1c2', start: '06:12:04',      dur: '4m 18s', status: 'SUCCESS', communes: 1268, qc: 'OK',   nulls: 0     },
    { id: '20260504T0610_b81d4e', start: 'hier 06:10',    dur: '4m 31s', status: 'SUCCESS', communes: 1268, qc: 'OK',   nulls: 2     },
    { id: '20260503T0610_c19a77', start: '02 mai 06:10',  dur: '5m 02s', status: 'WARN',    communes: 1268, qc: 'WARN', nulls: 14    },
    { id: '20260502T0610_d76e51', start: '01 mai 06:10',  dur: '4m 24s', status: 'SUCCESS', communes: 1268, qc: 'OK',   nulls: 0     },
    { id: '20260501T0610_f0a223', start: '30 avr 06:10',  dur: '0m 12s', status: 'FAILED',  communes: 0,    qc: 'KO',   nulls: '\u2014' },
  ];

  const steps = [
    { name: 'extract_elections',       dur:  9.4, status: 'ok' },
    { name: 'extract_demographique',   dur:  6.1, status: 'ok' },
    { name: 'extract_pauvrete',        dur:  3.2, status: 'ok' },
    { name: 'extract_chomage',         dur:  4.8, status: 'ok' },
    { name: 'extract_emploi',          dur:  5.3, status: 'ok' },
    { name: 'extract_candidats',       dur: 31.7, status: 'ok' },
    { name: 'transform_participation', dur:  7.6, status: 'ok' },
    { name: 'transform_demographique', dur:  4.2, status: 'ok' },
    { name: 'transform_historique',    dur: 12.3, status: 'ok' },
    { name: 'transform_cibles',        dur:  8.7, status: 'ok' },
    { name: 'assemble_gold',           dur: 38.4, status: 'ok' },
    { name: 'quality_checks',          dur:  6.8, status: 'ok' },
    { name: 'load_csv',                dur:  2.1, status: 'ok' },
    { name: 'load_db',                 dur: 17.9, status: 'ok' },
  ];

  // Données mock par défaut — remplacées par les vrais JSON si disponibles
  const models = [
    { key: 'gradient_boosting', fset: 'pre_vote',  name: 'Gradient Boosting (pre-vote)',   accuracy: 0.927, f1: 0.918, auc: 0.971, cv_auc: 0.968, time: '2.4s',  feat: 41, best: true },
    { key: 'gradient_boosting', fset: 'post_t1',   name: 'Gradient Boosting (post-T1)',    accuracy: 0.963, f1: 0.958, auc: 0.989, cv_auc: 0.985, time: '2.1s',  feat: 58 },
    { key: 'random_forest',     fset: 'pre_vote',  name: 'Random Forest (pre-vote)',        accuracy: 0.911, f1: 0.902, auc: 0.962, cv_auc: 0.958, time: '1.8s',  feat: 41 },
    { key: 'random_forest',     fset: 'post_t1',   name: 'Random Forest (post-T1)',         accuracy: 0.948, f1: 0.941, auc: 0.981, cv_auc: 0.978, time: '1.6s',  feat: 58 },
    { key: 'lstm',              fset: 'lstm',      name: 'LSTM (s\u00e9quentiel)',           accuracy: 0.894, f1: 0.881, auc: 0.949, cv_auc: null,  time: '38s',   feat: 24 },
  ];

  const topFeatures = [
    { name: 'h17_t2_pct_macron',   imp: 0.142 },
    { name: 'pct_dipl_superieur',  imp: 0.118 },
    { name: 'revenu_median_2021',  imp: 0.094 },
    { name: 'pct_csp_cadres',      imp: 0.087 },
    { name: 'cible_t1_pct_macron', imp: 0.072 },
    { name: 'pct_log_hlm',         imp: 0.063 },
    { name: 'taux_chomage_rp2022', imp: 0.054 },
    { name: 'pct_csp_ouvriers',    imp: 0.048 },
    { name: 'pct_pop_senior_60p',  imp: 0.041 },
    { name: 'taux_pauvrete_2017',  imp: 0.037 },
  ];

  return { partyColor, candidatesT1, t2, communes, runs, steps, models, topFeatures };
})();

// ─────────────────────────────────────────────────────────────────────────────
// Chargement des vrais artefacts ML depuis /artifacts/ (nginx bind-mount)
// Appelé au montage des pages prediction.html et overview.html
// ─────────────────────────────────────────────────────────────────────────────
window.loadArtifacts = async function () {
  const FILES = [
    { key: 'gradient_boosting', fset: 'pre_vote',  file: 'gradient_boosting_metrics_pre_vote_classification_t2.json' },
    { key: 'gradient_boosting', fset: 'post_t1',   file: 'gradient_boosting_metrics_post_t1_classification_t2.json' },
    { key: 'random_forest',     fset: 'pre_vote',  file: 'random_forest_metrics_pre_vote_classification_t2.json' },
    { key: 'random_forest',     fset: 'post_t1',   file: 'random_forest_metrics_post_t1_classification_t2.json' },
    { key: 'lstm',              fset: 'lstm',       file: 'lstm_metrics_lstm_classification_t2.json' },
  ];

  const NAMES = {
    'gradient_boosting|pre_vote': 'Gradient Boosting (pre-vote)',
    'gradient_boosting|post_t1': 'Gradient Boosting (post-T1)',
    'random_forest|pre_vote':    'Random Forest (pre-vote)',
    'random_forest|post_t1':     'Random Forest (post-T1)',
    'lstm|lstm':                 'LSTM (s\u00e9quentiel)',
  };

  const pick = (m, keys) => {
    for (const k of keys) if (m[k] !== undefined && m[k] !== null) return m[k];
    return null;
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
        key, fset,
        name:    NAMES[key + '|' + fset] || key,
        accuracy: pick(m, ['test_accuracy', 'val_accuracy']),
        f1:       pick(m, ['test_f1',       'val_f1']),
        auc:      pick(m, ['test_roc_auc',  'test_auc', 'val_auc']),
        cv_auc:   pick(m, ['cv_roc_auc']),
        cv_std:   pick(m, ['cv_roc_auc_std']),
        time:     m['training_time_s'] != null ? m['training_time_s'].toFixed(1) + 's' : null,
        feat:     pick(m, ['n_features']),
        params:   m['best_params'] || null,
        epochs:   m['epochs_trained'] || null,
        raw:      m,
        best:     (key === 'gradient_boosting' && fset === 'post_t1'),
      });
    } catch (_) {}
  }));

  if (anyLoaded && results.length > 0) {
    // Trier : GB post_t1 en tête, puis par accuracy décroissant
    results.sort((a, b) => (b.accuracy || 0) - (a.accuracy || 0));
    MOCK.models = results;
    MOCK.artifactsLoaded = true;
  }

  // Images disponibles depuis /artifacts/
  MOCK.images = {
    rf_confusion:  '/artifacts/random_forest_confusion_matrix.png',
    rf_roc:        '/artifacts/random_forest_roc_curve.png',
    rf_importance: '/artifacts/random_forest_feature_importance.png',
    gb_confusion:  '/artifacts/gradient_boosting_confusion_matrix.png',
    gb_roc:        '/artifacts/gradient_boosting_roc_curve.png',
    gb_importance: '/artifacts/gradient_boosting_feature_importance.png',
    gb_results:    '/artifacts/gradient_boosting_results.png',
    lstm_curves:   '/artifacts/lstm_training_curves.png',
    lstm_confusion:'/artifacts/lstm_confusion_matrix.png',
    lstm_roc:      '/artifacts/lstm_roc_curve.png',
    cmp_accuracy:  '/artifacts/model_comparison_test_accuracy.png',
    cmp_f1:        '/artifacts/model_comparison_test_f1.png',
    cmp_auc:       '/artifacts/model_comparison_test_roc_auc.png',
  };

  return MOCK;
};
