import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadConfig, applyEnvOverrides } from '../src/config.js';

test('env overrides use SIC_ + upper snake path, JSON-parsed', () => {
  const cfg = loadConfig({ env: { SIC_LLM__MODEL: 'other-model', SIC_BUDGET__TOTAL_REQUESTS_PER_RUN: '7', SIC_FEATURES__PROMPT_OPTIMIZER: 'false', SIC_EVAL__REPEATS: '' } });
  assert.equal(cfg.llm.model, 'other-model');
  assert.equal(cfg.budget.totalRequestsPerRun, 7);
  assert.equal(cfg.features.promptOptimizer, false);
  assert.equal(cfg.eval.repeats, loadConfig({ env: {} }).eval.repeats); // empty values are ignored
});

test('SIC_CONFIG_OVERRIDES deep-merges a JSON object', () => {
  const cfg = loadConfig({ env: { SIC_CONFIG_OVERRIDES: '{"llm":{"model":"m2"},"eval":{"repeats":5}}' } });
  assert.equal(cfg.llm.model, 'm2');
  assert.equal(cfg.eval.repeats, 5);
  assert.equal(cfg.eval.temperature, loadConfig({ env: {} }).eval.temperature);
});

test('whole objects can be overridden with JSON', () => {
  const out = applyEnvOverrides({ a: { b: 1 } }, { SIC_A: '{"b":2,"c":3}' });
  assert.deepEqual(out.a, { b: 2, c: 3 });
});

test('dry-run overlay switches to the mock provider', () => {
  const cfg = loadConfig({ dryRun: true, env: {} });
  assert.equal(cfg.llm.provider, 'mock');
  assert.equal(cfg.dataset.useFixture, true);
});

test('invalid config is rejected with a clear message', () => {
  assert.throws(() => loadConfig({ env: { SIC_LLM__PROVIDER: 'nope' } }), /llm.provider/);
  assert.throws(() => loadConfig({ env: { SIC_BUDGET__HOLDOUT_FRACTION: '2' } }), /holdoutFraction/);
  assert.throws(() => loadConfig({ env: { SIC_CONFIG_OVERRIDES: '{"variants":[{"name":"x","track":"nope","budgetShare":1}]}' } }), /unknown track/);
});
