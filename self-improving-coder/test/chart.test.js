import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
import { renderChart, seriesFromScores } from '../src/chart.js';
import { appendScores, readScores, COLUMNS } from '../src/metrics.js';
import { testConfig } from './helpers.js';

const fixtureRows = () => readScores(new URL('./fixtures/scores.csv', import.meta.url).pathname);

test('scores.csv round-trips with the documented columns', () => {
  const cfg = testConfig();
  const file = path.join(cfg.paths.stateDir, 'scores.csv');
  appendScores(file, [{ timestamp: 't', runId: 'r', split: 'train', passRate: 0.5, n: 4, promptVersion: 'v001', lessonsCount: 1, skillsCount: 2, requestsUsed: 9, gateDecision: 'accept' }]);
  appendScores(file, [{ timestamp: 't2', runId: 'r2', split: 'train', passRate: 0.25, n: 4, promptVersion: 'v001', lessonsCount: 1, skillsCount: 2, requestsUsed: 9, gateDecision: 'keep' }]);
  assert.equal(fs.readFileSync(file, 'utf8').split('\n')[0], COLUMNS.join(','));
  assert.deepEqual(readScores(file).map((r) => r.passRate), [0.5, 0.25]);
});

test('chart renders train and holdout lines, baseline and gate markers from fixture data', () => {
  const rows = fixtureRows();
  const s = seriesFromScores(rows);
  assert.equal(s.runs.length, 5);
  assert.equal(s.holdout.length, 5);
  assert.equal(s.train.length, 3);
  assert.ok(Math.abs(s.baseline - 0.4125) < 1e-9); // mean of the two baseline runs
  assert.equal(s.holdout[2].y, 0.475); // accepted -> candidate score
  assert.equal(s.holdout[3].y, 0.475); // rejected -> champion score
  const cfg = testConfig();
  const svg = renderChart(rows, cfg.chart);
  assert.match(svg, /^<svg[^>]+xmlns="http:\/\/www.w3.org\/2000\/svg"/);
  assert.equal((svg.match(/<polyline/g) ?? []).length, 2);
  assert.match(svg, new RegExp(`stroke="${cfg.chart.colors.holdout}"`));
  assert.match(svg, new RegExp(`stroke="${cfg.chart.colors.train}"`));
  assert.match(svg, /baseline 41\.3%/);
  assert.match(svg, new RegExp(`fill="${cfg.chart.colors.accept}"><title>`));
  assert.match(svg, new RegExp(`fill="${cfg.chart.colors.reject}"><title>`));
});

test('chart handles no data', () => {
  assert.match(renderChart([], testConfig().chart), /no runs yet/);
});
