// Prints a plain-text summary of scores.csv: per-run holdout scores, gate decisions, baseline noise.
// Used to write the README's "what improved and why" section from real data.
import { loadConfig, statePath } from './config.js';
import { readScores } from './metrics.js';
import { seriesFromScores } from './chart.js';

const cfg = loadConfig({ dryRun: process.argv.includes('--dry-run') });
const rows = readScores(statePath(cfg, 'scores'));
const { holdout, train, baseline } = seriesFromScores(rows);
const pct = (v) => `${(v * 100).toFixed(1)}%`;
if (!holdout.length) {
  console.log('No holdout scores yet.');
} else {
  const base = holdout.filter((p) => p.decision === 'baseline');
  if (base.length) console.log(`Baseline: ${base.map((p) => `${pct(p.y)} (n=${p.n})`).join(', ')} -> mean ${pct(baseline)}, spread ${pct(Math.max(...base.map((p) => p.y)) - Math.min(...base.map((p) => p.y)))}`);
  for (const p of holdout) {
    const t = train.find((x) => x.runId === p.runId);
    console.log(`${p.timestamp.slice(0, 10)} ${p.runId}: holdout ${pct(p.y)} (n=${p.n}) gate=${p.decision}${t ? `, train ${pct(t.y)} (n=${t.n})` : ''}`);
  }
  const counts = holdout.reduce((m, p) => ({ ...m, [p.decision]: (m[p.decision] ?? 0) + 1 }), {});
  console.log('Gate decisions:', JSON.stringify(counts));
  const last = holdout.at(-1);
  if (baseline !== null) console.log(`Change vs baseline: ${(last.y - baseline >= 0 ? '+' : '')}${((last.y - baseline) * 100).toFixed(1)} points`);
}
