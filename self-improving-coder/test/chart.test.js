// Phase 6 acceptance: the chart renders one line + confidence band per variant (>= 3 variants).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
import { renderChart, holdoutSeries, loadTrackSeries, writeCharts } from '../src/chart.js';
import { appendScores, readScores, COLUMNS } from '../src/metrics.js';
import { loadConfig, ROOT } from '../src/config.js';
import { testConfig, tmpDir } from './helpers.js';

const FIXTURE_STATE = path.join(ROOT, 'test/fixtures/chart-state');
const fixtureCfg = () => loadConfig({ env: {}, overrides: { paths: { stateDir: FIXTURE_STATE, chartsDir: tmpDir() }, variants: [
  { name: 'baseline', label: 'No memory', track: 'multipl-e', levers: {}, budgetShare: 1, color: '#6b7280' },
  { name: 'lessons', label: 'Lessons', track: 'multipl-e', levers: { lessons: true }, budgetShare: 1, color: '#2563eb' },
  { name: 'lessons-skills', label: 'Lessons + skills', track: 'multipl-e', levers: { lessons: true, skills: true }, budgetShare: 1, color: '#16a34a' },
] } });

test('scores.csv round-trips with the documented columns', () => {
  const cfg = testConfig();
  const file = path.join(cfg.paths.stateDir, 'scores.csv');
  appendScores(file, [{ timestamp: 't', runId: 'r', split: 'train', passRate: 0.5, n: 4, promptVersion: 'v001', lessonsCount: 1, skillsCount: 2, requestsUsed: 9, gateDecision: 'accept', ciLow: 0.25, ciHigh: '' }]);
  assert.equal(fs.readFileSync(file, 'utf8').split('\n')[0], COLUMNS.join(','));
  const [row] = readScores(file);
  assert.equal(row.passRate, 0.5);
  assert.equal(row.ciLow, 0.25);
  assert.equal(row.ciHigh, null);
});

test('holdout series: accepted runs use the candidate score, others the champion score', () => {
  const rows = readScores(path.join(FIXTURE_STATE, 'variants/lessons/scores.csv'));
  const s = holdoutSeries(rows);
  assert.equal(s.length, 5);
  assert.deepEqual(s.map((p) => p.decision), ['baseline', 'baseline', 'accept', 'reject', 'accept']);
  assert.equal(s[2].y, 0.47); // accepted candidate
  assert.equal(s[3].y, 0.47); // rejected: champion stays
  assert.deepEqual([s[2].low, s[2].high, s[2].n, s[2].repeats], [0.40, 0.54, 50, 3]);
});

test('chart renders a line and a shaded CI band for each of three variants, baseline and gate markers', () => {
  const cfg = fixtureCfg();
  const series = loadTrackSeries(cfg, 'multipl-e');
  const svg = renderChart(series, cfg.chart, 'Holdout');
  assert.match(svg, /^<svg[^>]+xmlns="http:\/\/www.w3.org\/2000\/svg"/);
  for (const v of ['baseline', 'lessons', 'lessons-skills']) {
    assert.match(svg, new RegExp(`<polygon class="band" data-variant="${v}"[^>]+fill-opacity="${cfg.chart.bandOpacity}"`), `band for ${v}`);
    assert.match(svg, new RegExp(`<polyline data-variant="${v}"`), `line for ${v}`);
  }
  assert.equal((svg.match(/<polygon class="band"/g) ?? []).length, 3);
  // starting baseline = mean of the 6 baseline points
  const mean = (0.40 + 0.42 + 0.41 + 0.40 + 0.39 + 0.41) / 6;
  assert.match(svg, new RegExp(`starting baseline ${(mean * 100).toFixed(1)}%`));
  assert.match(svg, new RegExp(`fill="${cfg.chart.colors.accept}"`));
  assert.match(svg, new RegExp(`fill="${cfg.chart.colors.reject}"`));
  assert.match(svg, /Lessons \+ skills/);
});

test('writeCharts writes one file per track and copes with no data', () => {
  const cfg = fixtureCfg();
  const out = writeCharts(cfg);
  assert.ok(fs.existsSync(path.join(cfg.paths.chartsDir, 'chart.svg')));
  assert.match(out['chart-mutation.svg'], /no runs yet/);
});
