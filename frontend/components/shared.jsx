// Shared: Topbar (glass nav), Footer, Sparkline, BarList, StackBar.
// Loaded by every page.

window.Topbar = function Topbar({ active }) {
  const items = [
    { id: 'index',      href: 'index.html',      label: 'Accueil' },
    { id: 'overview',   href: 'overview.html',   label: 'Vue d\u2019ensemble' },
    { id: 'comparison', href: 'comparison.html', label: 'Candidats' },
    { id: 'map',        href: 'map.html',        label: 'Carte' },
    { id: 'pipeline',   href: 'pipeline.html',   label: 'Pipeline' },
    { id: 'prediction', href: 'prediction.html', label: 'Mod\u00e8les' },
  ];
  return (
    <div className="topbar">
      <div className="topbar-inner">
        <nav className="nav">
          <a className="brand" href="index.html">
            <span className="brand-mark"></span>
            <span>Observatory</span>
          </a>
          <div className="nav-links">
            {items.map(it => (
              <a key={it.id} href={it.href} className={active === it.id ? 'active' : ''}>
                {it.label}
              </a>
            ))}
          </div>
          <div className="nav-actions">
            <span className="chip ok"><span className="dot"></span>Pipeline en ligne</span>
            <button className="btn ghost" style={{padding: '7px 10px'}}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
              <span className="kbd">\u2318K</span>
            </button>
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
            <span>Electoral Observatory</span>
          </div>
          <p style={{maxWidth: 360, color: 'var(--ink-3)', fontSize: 13, lineHeight: 1.55, margin: 0}}>
            Plateforme d&apos;analyse, monitoring ETL et pr\u00e9diction \u00e9lectorale.
            Donn\u00e9es publiques INSEE et Minist\u00e8re de l&apos;Int\u00e9rieur.
            Architecture Medallion bronze / silver / gold.
          </p>
          <div className="footer-meta">
            <span>MSPR \u00c9preuve E6.2 \u00b7 2026</span>
            <span>v 0.4.2</span>
          </div>
        </div>
        <div>
          <h4>Sections</h4>
          <ul>
            <li><a href="overview.html">Vue d&apos;ensemble</a></li>
            <li><a href="comparison.html">Candidats &amp; partis</a></li>
            <li><a href="map.html">Carte d&apos;\u00cele-de-France</a></li>
            <li><a href="pipeline.html">Pipeline ETL</a></li>
            <li><a href="prediction.html">Mod\u00e8les ML</a></li>
          </ul>
        </div>
        <div>
          <h4>Donn\u00e9es</h4>
          <ul>
            <li><a href="#">INSEE \u2014 RP 2022</a></li>
            <li><a href="#">Filosofi 2017</a></li>
            <li><a href="#">data.gouv.fr</a></li>
            <li><a href="#">Min. Int\u00e9rieur</a></li>
          </ul>
        </div>
        <div>
          <h4>Ressources</h4>
          <ul>
            <li><a href="#">Documentation</a></li>
            <li><a href="#">M\u00e9thodologie</a></li>
            <li><a href="#">Schema PostgreSQL</a></li>
            <li><a href="#">Quality gates</a></li>
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
      <path className="area" d={areaPath} style={{fill: color, opacity: 0.08}} />
      <path className="line" d={linePath} style={{stroke: color}} />
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
            <i style={{
              width: ((r.value / m) * 100).toFixed(1) + '%',
              background: r.color || 'var(--ink)'
            }}/>
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
        {segments.map((s, i) => {
          const pct = (s.value / total) * 100;
          return <div key={i} className="seg" style={{width: pct + '%', background: s.color}}/>;
        })}
      </div>
      <div className="stack-legend">
        {segments.map((s, i) => (
          <span key={i} style={{'--lc': s.color}}>{s.name} <strong>{((s.value/total)*100).toFixed(1)}%</strong></span>
        ))}
      </div>
    </>
  );
};

Object.assign(window, {
  Topbar: window.Topbar, Footer: window.Footer,
  Sparkline: window.Sparkline, BarList: window.BarList, StackBar: window.StackBar,
});
