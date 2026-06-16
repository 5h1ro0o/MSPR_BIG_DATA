window.Topbar = function Topbar({ active }) {
  const items = [
    { id: 'index',       href: 'index.html',       label: 'Accueil' },
    { id: 'overview',    href: 'overview.html',    label: 'Résultats 2022' },
    { id: 'comparison',  href: 'comparison.html',  label: 'Réalité vs Prédictions' },
    { id: 'map',         href: 'map.html',         label: 'Carte IDF' },
    { id: 'pipeline',    href: 'pipeline.html',    label: 'Pipeline ETL' },
    { id: 'prediction',  href: 'prediction.html',  label: 'Modèles ML' },
    { id: 'projections', href: 'projections.html', label: 'Projections 2025–2027' },
  ];
  return (
    <div className="topbar">
      <div className="topbar-inner">
        <nav className="nav">
          <a className="brand" href="index.html">
            <span className="brand-mark"></span>
            <span>MSPR</span>
          </a>
          <div className="nav-links">
            {items.map(it => (
              <a key={it.id} href={it.href} className={active === it.id ? 'active' : ''}>
                {it.label}
              </a>
            ))}
            <a href="http://localhost:3001" target="_blank" rel="noopener noreferrer"
               style={{ display: 'flex', alignItems: 'center', gap: 5, opacity: 0.85 }}
               title="Tableau de bord Grafana — outil BI analytique">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
              </svg>
              Grafana BI
            </a>
          </div>
          <div className="nav-actions">
            <span className="chip ok"><span className="dot"></span>Pipeline actif</span>
          </div>
        </nav>
      </div>
    </div>
  );
};

window.Footer = function Footer() {
  return (
    <footer className="footer">
      <div className="footer-card">
        <div>
          <div className="brand-row">
            <span className="brand-mark"></span>
            <span>MSPR</span>
          </div>
          <p style={{maxWidth: 360, color: 'var(--ink-3)', fontSize: 13, lineHeight: 1.55, margin: 0}}>
            Plateforme d'analyse, monitoring ETL et prédiction électorale.
            Données publiques INSEE et Ministère de l'Intérieur.
            Architecture Medallion bronze / silver / gold.
          </p>
          <div className="footer-meta">
            <span>MSPR TPRE813 · Île-de-France · 2022</span>
            <span>1 268 communes · 113 features</span>
          </div>
        </div>
        <div>
          <h4>Navigation</h4>
          <ul>
            <li><a href="overview.html">Vue d'ensemble</a></li>
            <li><a href="comparison.html">Réalité vs Prédictions</a></li>
            <li><a href="map.html">Carte Île-de-France</a></li>
            <li><a href="pipeline.html">Pipeline ETL</a></li>
            <li><a href="prediction.html">Modèles ML</a></li>
          </ul>
        </div>
        <div>
          <h4>Sources</h4>
          <ul>
            <li><a href="https://www.data.gouv.fr/fr/pages/donnees-des-elections/" target="_blank">data.gouv.fr — Élections</a></li>
            <li><a href="https://www.insee.fr/fr/statistiques/6036912" target="_blank">INSEE — RP 2022</a></li>
            <li><a href="https://www.insee.fr/fr/statistiques/6049648" target="_blank">Filosofi 2017</a></li>
            <li><a href="https://www.interieur.gouv.fr/" target="_blank">Min. Intérieur</a></li>
          </ul>
        </div>
        <div>
          <h4>Technique</h4>
          <ul>
            <li><a href="https://github.com/5h1ro0o/MSPR_BIG_DATA" target="_blank">GitHub</a></li>
            <li><a href="https://sonarcloud.io/project/overview?id=5h1ro0o_MSPR_BIG_DATA" target="_blank">SonarCloud</a></li>
            <li><a href="pipeline.html">Quality gates (14/14)</a></li>
            <li><a href="prediction.html">Modèles (RF · GB)</a></li>
            <li><a href="projections.html">Projections 2025–2027</a></li>
          </ul>
        </div>
      </div>
    </footer>
  );
};

window.Sparkline = function Sparkline({ data, width = 120, height = 32, color = 'var(--ink)' }) {
  if (!data || !data.length) return null;
  const min = Math.min(...data), max = Math.max(...data);
  const range = max - min || 1;
  const stepX = width / (data.length - 1);
  const pts = data.map((d, i) => [i * stepX, height - ((d - min) / range) * (height - 4) - 2]);
  const linePath = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
  const areaPath = linePath + ' L ' + width + ' ' + height + ' L 0 ' + height + ' Z';
  return (
    <svg className="spark" viewBox={'0 0 ' + width + ' ' + height} preserveAspectRatio="none" style={{height}}>
      <path d={areaPath} style={{fill: color, opacity: 0.08}} />
      <path d={linePath} style={{stroke: color, fill: 'none', strokeWidth: 1.5}} />
    </svg>
  );
};

window.BarList = function BarList({ rows, max }) {
  const m = max || Math.max(...rows.map(r => r.value));
  return (
    <div className="bar-list">
      {rows.map((r, i) => (
        <div className="bar-row" key={r.name + i}>
          <div className="rank">{String(i + 1).padStart(2, '0')}</div>
          <div className="nm">
            {r.name}
            {r.party && <small>{r.party}</small>}
          </div>
          <div className="br">
            <i style={{ width: ((r.value / m) * 100).toFixed(1) + '%', background: r.color || 'var(--ink)' }} />
          </div>
          <div className="pc">{r.value.toFixed(2)}<small>%</small></div>
        </div>
      ))}
    </div>
  );
};

window.StackBar = function StackBar({ segments }) {
  const total = segments.reduce((a, s) => a + s.value, 0);
  return (
    <>
      <div className="stack">
        {segments.map((s, i) => (
          <div key={i} className="seg" style={{ width: ((s.value / total) * 100) + '%', background: s.color }} />
        ))}
      </div>
      <div className="stack-legend">
        {segments.map((s, i) => (
          <span key={i} style={{'--lc': s.color}}>
            {s.name} <strong>{((s.value / total) * 100).toFixed(1)}%</strong>
          </span>
        ))}
      </div>
    </>
  );
};

Object.assign(window, {
  Topbar: window.Topbar, Footer: window.Footer,
  Sparkline: window.Sparkline, BarList: window.BarList, StackBar: window.StackBar,
});
