// Smoke mode: a tiny real-model run in CI that measures seconds per request and projects whether the full
// budget fits the job's time limit, without touching committed state.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadConfig } from '../src/config.js';
import { variantConfig } from '../src/variants.js';
import { stepSummary } from '../src/ci/step-summary.js';
import { runCycle } from '../src/run.js';
import { testConfig } from './helpers.js';

test('SIC_SMOKE=1 applies the smoke overlay: separate state dir, tiny budget, optimizer every run', () => {
  const normal = loadConfig({ env: {} });
  const smoke = loadConfig({ env: { SIC_SMOKE: '1' } });
  assert.notEqual(smoke.paths.stateDir, normal.paths.stateDir);
  assert.ok(smoke.budget.totalRequestsPerRun < normal.budget.totalRequestsPerRun);
  assert.equal(smoke.gate.baselineRuns, 0);
  assert.equal(smoke.promptOptimizer.everyNRuns, 1);
  assert.equal(loadConfig({ env: { SIC_SMOKE: '' } }).paths.stateDir, normal.paths.stateDir);
});

test('every attempt row and the run summary carry measured model latency', async () => {
  const cfg = testConfig();
  const s = await runCycle({ cfg });
  assert.ok(Number.isFinite(s.llmMs) && s.llmMs >= 0);
  assert.equal(s.budgetPerRun, variantConfig(cfg, cfg.defaultVariant).budget.requestsPerRun);
});

test('step summary projects the full budget from measured latency and flags when it does not fit', () => {
  const full = variantConfig(loadConfig({ env: {} }), 'lessons');
  const base = { decision: 'keep', champion: null, candidate: null, train: { passed: 1, problems: 2, newLessons: [], newSkills: [] }, tokensIn: 10, tokensOut: 5 };
  const fast = stepSummary(full, { ...base, requestsUsed: 10, llmMs: 10 * 2000 }); // 2 s per request
  assert.match(fast, /seconds per request \(measured\) \| 2\.00/);
  assert.match(fast, /fits yes/);
  const slow = stepSummary(full, { ...base, requestsUsed: 10, llmMs: 10 * 60000 }); // 60 s per request
  assert.match(slow, /fits \*\*no/);
  assert.match(stepSummary(full, null), /No run summary/);
});
