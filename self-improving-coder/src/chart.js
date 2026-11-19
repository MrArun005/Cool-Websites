// scores.csv -> chart.svg. Pure string building, no external libraries or services.
import { loadConfig, statePath } from './config.js';
import { readScores } from './metrics.js';
import { writeText } from './util.js';

const escXml = (s) => String(s).replace(/[<>&"]/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;' }[c]));

/** One point per run per series. Holdout line = the champion's score after the gate decided. */
export function seriesFromScores(rows) {
  const runs = [...new Set(rows.map((r) => r.runId))];
  const byRun = (split) => runs.map((runId, i) => {
    const r = rows.find((x) => x.runId === runId && x.split === split);
    return r ? { x: i, y: r.passRate, n: r.n, decision: r.gateDecision, runId, timestamp: r.timestamp } : null;
  }).filter(Boolean);
  const holdout = runs.map((runId, i) => {
    const champ = rows.find((x) => x.runId === runId && x.split === 'holdout-champion');
    const cand = rows.find((x) => x.runId === runId && x.split === 'holdout-candidate');
    if (!champ) return null;
    const accepted = cand && cand.gateDecision === 'accept';
    const pick = accepted ? cand : champ;
    return { x: i, y: pick.passRate, n: pick.n, decision: champ.gateDecision, runId, timestamp: champ.timestamp };
  }).filter(Boolean);
  const baselinePts = holdout.filter((p) => p.decision === 'baseline');
  const baseline = baselinePts.length ? baselinePts.reduce((s, p) => s + p.y, 0) / baselinePts.length : null;
  return { runs, train: byRun('train'), holdout, baseline };
}

export function renderChart(rows, chartCfg) {
  const { width: W, height: H, margin: m, yTicks, colors: c, title, fontSize: fs, titleFontSize, legendItemWidth, pointRadius } = chartCfg;
  const gap = fs / 2; // small spacing derived from the font size
  const { runs, train, holdout, baseline } = seriesFromScores(rows);
  const iw = W - m.left - m.right;
  const ih = H - m.top - m.bottom;
  const xs = (i) => m.left + (runs.length <= 1 ? iw / 2 : (i / (runs.length - 1)) * iw);
  const ys = (v) => m.top + ih - v * ih;
  const parts = [];
  parts.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="system-ui, -apple-system, Segoe UI, sans-serif" font-size="${fs}">`);
  parts.push(`<rect width="${W}" height="${H}" fill="${c.background}"/>`);
  parts.push(`<text x="${m.left}" y="${titleFontSize + gap}" fill="${c.text}" font-size="${titleFontSize}" font-weight="600">${escXml(title)}</text>`);
  for (let t = 0; t <= yTicks; t++) {
    const v = t / yTicks;
    parts.push(`<line x1="${m.left}" x2="${W - m.right}" y1="${ys(v)}" y2="${ys(v)}" stroke="${c.grid}"/>`);
    parts.push(`<text x="${m.left - gap}" y="${ys(v) + gap / 2}" text-anchor="end" fill="${c.axis}">${Math.round(v * 100)}%</text>`);
  }
  runs.forEach((runId, i) => {
    const r = rows.find((x) => x.runId === runId);
    parts.push(`<text x="${xs(i)}" y="${H - m.bottom + fs * 1.5}" text-anchor="middle" fill="${c.axis}">${escXml((r?.timestamp ?? '').slice(5, 10) || i + 1)}</text>`);
  });
  parts.push(`<text x="${m.left + iw / 2}" y="${H - gap}" text-anchor="middle" fill="${c.axis}">run date (MM-DD)</text>`);
  if (baseline !== null) {
    parts.push(`<line x1="${m.left}" x2="${W - m.right}" y1="${ys(baseline)}" y2="${ys(baseline)}" stroke="${c.baseline}" stroke-dasharray="6 4" stroke-width="1.5"/>`);
    parts.push(`<text x="${m.left + gap}" y="${ys(baseline) + fs + gap / 2}" text-anchor="start" fill="${c.baseline}">baseline ${(baseline * 100).toFixed(1)}%</text>`);
  }
  const line = (pts, color, dash = '') => pts.length
    ? `<polyline fill="none" stroke="${color}" stroke-width="2" ${dash ? `stroke-dasharray="${dash}"` : ''} points="${pts.map((p) => `${xs(p.x)},${ys(p.y)}`).join(' ')}"/>`
    : '';
  parts.push(line(train, c.train, '3 3'));
  parts.push(line(holdout, c.holdout));
  for (const p of holdout) {
    const color = p.decision === 'accept' ? c.accept : p.decision === 'reject' ? c.reject : c.holdout;
    parts.push(`<circle cx="${xs(p.x)}" cy="${ys(p.y)}" r="${pointRadius}" fill="${color}"><title>${escXml(`${p.runId}: ${(p.y * 100).toFixed(1)}% (n=${p.n}, ${p.decision})`)}</title></circle>`);
  }
  // legend
  const lx = m.left;
  const ly = titleFontSize + gap * 3;
  const legend = [[c.holdout, 'holdout (champion)'], [c.train, 'train batch'], [c.accept, 'accepted'], [c.reject, 'rejected'], [c.baseline, 'baseline']];
  legend.forEach(([col, label], i) => {
    const x = lx + i * legendItemWidth;
    parts.push(`<rect x="${x}" y="${ly}" width="${gap * 2}" height="${gap * 2}" rx="${gap / 3}" fill="${col}"/><text x="${x + gap * 3}" y="${ly + gap * 2}" fill="${c.text}">${label}</text>`);
  });
  if (!runs.length) parts.push(`<text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="${c.axis}">no runs yet</text>`);
  parts.push('</svg>');
  return parts.filter(Boolean).join('\n') + '\n';
}

export function writeChart(cfg) {
  const svg = renderChart(readScores(statePath(cfg, 'scores')), cfg.chart);
  writeText(statePath(cfg, 'chart'), svg);
  return svg;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  writeChart(loadConfig({ dryRun: process.argv.includes('--dry-run') }));
}
