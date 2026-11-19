// scores.csv of every variant -> one SVG per track: one line per variant with a shaded confidence band,
// gate decisions as markers, and the starting baseline (mean of baseline runs) as a dashed line.
// Pure string building, no external libraries or services.
import fs from 'node:fs';
import path from 'node:path';
import { loadConfig, rootPath } from './config.js';
import { readScores } from './metrics.js';
import { variantStateDir } from './variants.js';
import { writeText } from './util.js';

const escXml = (s) => String(s).replace(/[<>&"']/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&#39;' }[c]));
const DATE_LEN = 10; // YYYY-MM-DD

/** Holdout point per run: the candidate's row if it was accepted (it became champion), else the champion's. */
export function holdoutSeries(rows) {
  const runs = [...new Set(rows.map((r) => r.runId))];
  return runs.map((runId) => {
    const champ = rows.find((x) => x.runId === runId && x.split === 'holdout-champion');
    if (!champ) return null;
    const cand = rows.find((x) => x.runId === runId && x.split === 'holdout-candidate');
    const p = cand && cand.gateDecision === 'accept' ? cand : champ;
    return { runId, date: champ.timestamp.slice(0, DATE_LEN), timestamp: champ.timestamp, y: p.passRate, low: p.ciLow ?? p.passRate, high: p.ciHigh ?? p.passRate, n: p.n, repeats: p.repeats, decision: champ.gateDecision };
  }).filter(Boolean);
}

/** Read every variant of `track` from disk. */
export function loadTrackSeries(cfg, track) {
  return cfg.variants.filter((v) => v.track === track).map((v) => {
    const file = path.resolve(cfg.root, variantStateDir(cfg, v.name), cfg.paths.scores);
    return { name: v.name, label: v.label, color: v.color, rows: readScores(file) };
  });
}

export function renderChart(series, chartCfg, title) {
  const { width: W, height: H, margin: m, yTicks, colors: c, fontSize: fs, titleFontSize, legendItemWidth, pointRadius, bandOpacity } = chartCfg;
  const gap = fs / 2;
  const lines = series.map((s) => ({ ...s, pts: holdoutSeries(s.rows) })).filter((s) => s.pts.length);
  const dates = [...new Set(lines.flatMap((s) => s.pts.map((p) => p.date)))].sort();
  const baselinePts = lines.flatMap((s) => s.pts.filter((p) => p.decision === 'baseline'));
  const baseline = baselinePts.length ? baselinePts.reduce((a, p) => a + p.y, 0) / baselinePts.length : null;
  const legendRows = Math.max(1, Math.ceil(lines.length * legendItemWidth / (W - m.left - m.right)));
  const top = m.top + (legendRows - 1) * fs * 1.5;
  const iw = W - m.left - m.right;
  const ih = H - top - m.bottom;
  const xs = (d) => m.left + (dates.length <= 1 ? iw / 2 : (dates.indexOf(d) / (dates.length - 1)) * iw);
  const ys = (v) => top + ih - v * ih;
  const out = [];
  out.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="system-ui, -apple-system, Segoe UI, sans-serif" font-size="${fs}" role="img" aria-label="${escXml(title)}">`);
  out.push(`<rect width="${W}" height="${H}" fill="${c.background}"/>`);
  out.push(`<text x="${m.left}" y="${titleFontSize + gap}" fill="${c.text}" font-size="${titleFontSize}" font-weight="600">${escXml(title)}</text>`);
  for (let t = 0; t <= yTicks; t++) {
    const v = t / yTicks;
    out.push(`<line x1="${m.left}" x2="${W - m.right}" y1="${ys(v)}" y2="${ys(v)}" stroke="${c.grid}"/>`);
    out.push(`<text x="${m.left - gap}" y="${ys(v) + gap / 2}" text-anchor="end" fill="${c.axis}">${Math.round(v * 100)}%</text>`);
  }
  dates.forEach((d) => out.push(`<text x="${xs(d)}" y="${H - m.bottom + fs * 1.5}" text-anchor="middle" fill="${c.axis}">${escXml(d.slice(5))}</text>`));
  out.push(`<text x="${m.left + iw / 2}" y="${H - gap}" text-anchor="middle" fill="${c.axis}">run date (MM-DD) · bands = 95% bootstrap CI over holdout problems</text>`);
  if (baseline !== null) {
    out.push(`<line x1="${m.left}" x2="${W - m.right}" y1="${ys(baseline)}" y2="${ys(baseline)}" stroke="${c.baseline}" stroke-dasharray="6 4" stroke-width="1.5"/>`);
    out.push(`<text x="${m.left + gap}" y="${ys(baseline) + fs + gap / 2}" fill="${c.baseline}">starting baseline ${(baseline * 100).toFixed(1)}%</text>`);
  }
  for (const s of lines) {
    // band: upper edge left->right, lower edge right->left
    const byDate = new Map(s.pts.map((p) => [p.date, p])); // last run of a day wins
    const pts = [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date));
    if (pts.length > 1) {
      const poly = [...pts.map((p) => `${xs(p.date)},${ys(p.high)}`), ...[...pts].reverse().map((p) => `${xs(p.date)},${ys(p.low)}`)].join(' ');
      out.push(`<polygon class="band" data-variant="${escXml(s.name)}" points="${poly}" fill="${s.color}" fill-opacity="${bandOpacity}" stroke="none"/>`);
    } else {
      const p = pts[0];
      out.push(`<line class="band" data-variant="${escXml(s.name)}" x1="${xs(p.date)}" x2="${xs(p.date)}" y1="${ys(p.low)}" y2="${ys(p.high)}" stroke="${s.color}" stroke-opacity="${bandOpacity * 3}" stroke-width="${pointRadius * 2}"/>`);
    }
    out.push(`<polyline data-variant="${escXml(s.name)}" fill="none" stroke="${s.color}" stroke-width="2" points="${pts.map((p) => `${xs(p.date)},${ys(p.y)}`).join(' ')}"/>`);
    for (const p of pts) {
      const fill = p.decision === 'accept' ? c.accept : p.decision === 'reject' ? c.reject : s.color;
      out.push(`<circle cx="${xs(p.date)}" cy="${ys(p.y)}" r="${pointRadius}" fill="${fill}" stroke="${c.background}"><title>${escXml(`${s.label} ${p.date}: ${(p.y * 100).toFixed(1)}% [${(p.low * 100).toFixed(1)}–${(p.high * 100).toFixed(1)}] n=${p.n}×${p.repeats ?? 1}, gate ${p.decision}`)}</title></circle>`);
    }
  }
  const perRow = Math.max(1, Math.floor(iw / legendItemWidth));
  lines.forEach((s, i) => {
    const x = m.left + (i % perRow) * legendItemWidth;
    const y = titleFontSize + gap * 3 + Math.floor(i / perRow) * fs * 1.5;
    out.push(`<rect x="${x}" y="${y}" width="${gap * 2}" height="${gap * 2}" rx="${gap / 3}" fill="${s.color}"/><text x="${x + gap * 3}" y="${y + gap * 2}" fill="${c.text}">${escXml(s.label)}</text>`);
  });
  if (!lines.length) out.push(`<text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="${c.axis}">no runs yet</text>`);
  out.push('</svg>');
  return out.join('\n') + '\n';
}

/** Write every chart in cfg.charts into cfg.paths.chartsDir. Returns { file: svg }. */
export function writeCharts(cfg) {
  const written = {};
  for (const ch of cfg.charts) {
    const svg = renderChart(loadTrackSeries(cfg, ch.track), cfg.chart, ch.title);
    const file = rootPath(cfg, cfg.paths.chartsDir, ch.file);
    if (!fs.existsSync(path.dirname(file))) fs.mkdirSync(path.dirname(file), { recursive: true });
    writeText(file, svg);
    written[ch.file] = svg;
  }
  return written;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  writeCharts(loadConfig({ dryRun: process.argv.includes('--dry-run') }));
}
