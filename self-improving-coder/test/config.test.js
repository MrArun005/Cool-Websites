import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadConfig, applyEnvOverrides } from '../src/config.js';

test('env overrides use SIC_ + upper snake path, JSON-parsed', () => {
  const cfg = loadConfig({ env: { SIC_LLM__MODEL: 'other-model', SIC_BUDGET__REQUESTS_PER_RUN: '7', SIC_BUDGET__RESERVE_FOR_HOLDOUT: '2', SIC_FEATURES__PROMPT_OPTIMIZER: 'true' } });
  assert.equal(cfg.llm.model, 'other-model');
  assert.equal(cfg.budget.requestsPerRun, 7);
  assert.equal(cfg.features.promptOptimizer, true);
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
  assert.throws(() => loadConfig({ env: { SIC_BUDGET__RESERVE_FOR_HOLDOUT: '9999' } }), /reserveForHoldout/);
});
