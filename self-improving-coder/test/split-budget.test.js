import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
import { makeSplit, loadOrCreateSplit } from '../src/dataset/split.js';
import { createBudget, budgeted, BudgetExhausted } from '../src/budget.js';
import { entryPointOf, normalize } from '../src/dataset/fetch.js';
import { tmpDir, testConfig } from './helpers.js';
import { variantConfig, variantBudget } from '../src/variants.js';
import { loadConfig } from '../src/config.js';

const ids = Array.from({ length: 50 }, (_, i) => `p${i}`);

test('split is seeded, disjoint and complete', () => {
  const a = makeSplit(ids, { seed: 7, holdoutSize: 10 });
  const b = makeSplit([...ids].reverse(), { seed: 7, holdoutSize: 10 });
  assert.deepEqual(a, b);
  assert.equal(a.holdout.length, 10);
  assert.equal(a.train.length, 40);
  assert.equal(new Set([...a.train, ...a.holdout]).size, 50);
  assert.notDeepEqual(makeSplit(ids, { seed: 8, holdoutSize: 10 }).holdout, a.holdout);
});

test('split is persisted and reused; unknown ids are refused', () => {
  const file = path.join(tmpDir(), 'split.json');
  const first = loadOrCreateSplit(file, ids, { seed: 1, holdoutSize: 5 });
  const again = loadOrCreateSplit(file, ids, { seed: 999, holdoutSize: 20 });
  assert.deepEqual(again, first);
  assert.throws(() => loadOrCreateSplit(file, ids.slice(0, 10), { seed: 1, holdoutSize: 5 }), /unknown problems/);
  assert.ok(fs.existsSync(file));
});

test('budget: per-run cap, holdout reserve, persisted daily count', async () => {
  const cfg = variantConfig(testConfig({ budget: { requestsPerRun: 5, reserveForHoldout: 2, requestsPerDay: 6, maxRunMinutes: 60, reserveMinutesForHoldout: 0 } }), 'lessons-skills');
  const file = path.join(cfg.paths.stateDir, 'budget.json');
  const b = createBudget(cfg, { file });
  const llm = budgeted({ name: 'x', chat: async () => 'ok' }, b);
  await llm.chat([], { reserve: 2 });
  await llm.chat([], { reserve: 2 });
  await llm.chat([], { reserve: 2 });
  await assert.rejects(llm.chat([], { reserve: 2 }), BudgetExhausted); // train share used up
  await llm.chat([]); // holdout can still use the reserve
  await llm.chat([]);
  await assert.rejects(llm.chat([]), BudgetExhausted);
  b.save();
  const b2 = createBudget(cfg, { file });
  assert.equal(b2.usedToday, 5);
  assert.equal(b2.remainingRun(), 1); // daily limit binds before the per-run limit
});

test('budget: time limit stops calls', async () => {
  const cfg = variantConfig(testConfig({ budget: { requestsPerRun: 100, reserveForHoldout: 0, requestsPerDay: 100, maxRunMinutes: 1, reserveMinutesForHoldout: 0 } }), 'lessons-skills');
  let t = 0;
  const b = createBudget(cfg, { now: () => new Date(t) });
  b.check();
  t = 61000;
  assert.throws(() => b.check(), /time/);
});

test('dataset normalisation finds the entry point', () => {
  assert.equal(entryPointOf('// helper\nfunction a(x){ }\n// doc\nfunction solve_it(x, y){\n'), 'solve_it');
  const p = normalize({ name: 'HumanEval_0_x', prompt: 'function x(a){\n', tests: 't' }, 'humaneval-js');
  assert.deepEqual(p, { id: 'HumanEval_0_x', source: 'humaneval-js', prompt: 'function x(a){\n', tests: 't', entryPoint: 'x' });
});

test('daily budget is split across variants by budgetShare; each variant has its own state dir', () => {
  const cfg = loadConfig({ env: {}, overrides: { variants: [
    { name: 'a', track: 'multipl-e', levers: {}, budgetShare: 1 },
    { name: 'b', track: 'multipl-e', levers: { lessons: true }, budgetShare: 3 },
  ], budget: { totalRequestsPerRun: 400, totalRequestsPerDay: 800, holdoutFraction: 0.5 } } });
  assert.deepEqual([variantBudget(cfg, 'a').requestsPerRun, variantBudget(cfg, 'b').requestsPerRun], [100, 300]);
  assert.deepEqual([variantBudget(cfg, 'a').requestsPerDay, variantBudget(cfg, 'b').reserveForHoldout], [200, 150]);
  assert.notEqual(variantConfig(cfg, 'a').paths.stateDir, variantConfig(cfg, 'b').paths.stateDir);
  assert.deepEqual(variantConfig(cfg, 'b').variant.levers, { lessons: true, skills: false, promptOptimizer: false });
  assert.throws(() => variantConfig(cfg, 'zzz'), /unknown variant/);
});
