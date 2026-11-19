// One-command, offline reproduction with the mock provider:
//   npm run reproduce -- --variant=<name> --seed=<n> [--days=<d>] [--out=<dir>]
//   npm run reproduce -- --all --seed=<n>            every variant (used to build the demo dashboard)
// The synthetic mock (see src/llm/mock.js) is NOT a model: this checks that the pipeline, statistics,
// gate and reports behave as described, and that results are reproducible from a seed.
import fs from 'node:fs';
import path from 'node:path';
import { loadConfig, deepMerge, rootPath } from './config.js';
import { runCycle } from './run.js';
import { writeCharts } from './chart.js';
import { listVariants } from './variants.js';
import { argValue } from './util.js';
import { writeReport } from './report.js';
import { writeSite } from '../site/build.js';

// column widths of the console table (layout only)
const W = { pct: 5, day: 3, variant: 20, decision: 10, ci: 18 }; // allow-number: console layout

export function reproduceConfig(seed, { out, base } = {}) {
  const cfg = base ?? loadConfig({ env: {} });
  const r = cfg.reproduce;
  const outDir = path.resolve(cfg.root, out ?? path.join(r.outDir, `seed-${seed}`));
  return deepMerge(cfg, deepMerge(r.overrides, {
    split: { seed },
    llm: { seed, mock: { seed } },
    stats: { seed },
    paths: { stateDir: path.join(outDir, 'state'), chartsDir: outDir },
    curriculum: { fixture: { cacheDir: path.join(outDir, 'curriculum') } },
  }));
}

/** Run `days` daily cycles for the given variants with a fake clock. Returns per-run summaries. */
export async function reproduce({ seed, variants, days, out, base, fresh = true }) {
  const cfg = reproduceConfig(seed, { out, base });
  const r = cfg.reproduce;
  const outDir = path.dirname(cfg.paths.stateDir);
  if (fresh) fs.rmSync(outDir, { recursive: true, force: true });
  const start = Date.parse(r.startDate);
  const rows = [];
  const quiet = process.env.SIC_QUIET;
  process.env.SIC_QUIET = '1';
  try {
    for (let d = 0; d < (days ?? r.days); d++) {
      const now = () => new Date(start + d * r.dayMs);
      for (const variant of variants) {
        const s = await runCycle({ cfg, variant, deps: { now } });
        rows.push({ day: d + 1, variant, decision: s.decision, champion: s.champion, candidate: s.candidate, diff: s.diff, lessons: s.train.newLessons.length, requests: s.requestsUsed });
      }
    }
  } finally {
    if (quiet === undefined) delete process.env.SIC_QUIET; else process.env.SIC_QUIET = quiet;
  }
  writeCharts(cfg);
  return { cfg, rows, outDir };
}

const pct = (v) => (v === null || v === undefined ? '   -  ' : `${(v * 100).toFixed(1).padStart(W.pct)}%`);

export function formatRows(rows) {
  const lines = ['day  variant              decision    champion [95% CI]          candidate   diff [95% CI]'];
  for (const r of rows) {
    const c = r.champion;
    const ci = c ? `[${pct(c.ciLow)}, ${pct(c.ciHigh)}]` : '';
    const d = r.diff ? `${pct(r.diff.estimate)} [${pct(r.diff.low)}, ${pct(r.diff.high)}]` : '';
    lines.push(`${String(r.day).padStart(W.day)}  ${r.variant.padEnd(W.variant)} ${r.decision.padEnd(W.decision)} ${pct(c?.passRate)} ${ci.padEnd(W.ci)} ${pct(r.candidate?.passRate)}  ${d}`);
  }
  return lines.join('\n');
}

export function parseArgs(argv) {
  const get = (k) => argValue(argv, k);
  return { variant: get('variant'), seed: get('seed'), days: get('days'), out: get('out'), all: argv.includes('--all') };
}

async function main() {
  const a = parseArgs(process.argv.slice(2));
  const base = loadConfig({ env: process.env });
  if (!a.all && !a.variant) throw new Error(`usage: npm run reproduce -- --variant=<${listVariants(base).join('|')}> --seed=<n> [--days=<d>] [--out=<dir>]  (or --all)`);
  if (a.seed === undefined || !/^\d+$/.test(a.seed)) throw new Error('--seed=<non-negative integer> is required');
  const variants = a.all ? listVariants(base) : [a.variant];
  const { rows, outDir, cfg } = await reproduce({ seed: Number(a.seed), variants, days: a.days ? Number(a.days) : undefined, out: a.out, base });
  console.log(formatRows(rows));
  writeReport(cfg);
  writeSite(cfg, path.join(outDir, cfg.reproduce.siteFile));
  console.log(`\nstate written to ${path.relative(process.cwd(), outDir) || '.'} (mock provider: synthetic, NOT a real model)`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((e) => { console.error(e?.message ?? e); process.exit(1); });
}

export { rootPath };
