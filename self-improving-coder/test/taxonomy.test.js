// Phase 7 acceptance: every failure class is covered by a fixture that goes through the real judge.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { classify } from '../src/taxonomy.js';
import { runTests } from '../src/judge/run-tests.js';
import { loadConfig } from '../src/config.js';
import { runCycle } from '../src/run.js';
import { readJsonl } from '../src/recorder.js';
import { readJson } from '../src/util.js';
import { testConfig, vdir } from './helpers.js';

const cfg = loadConfig({ env: {} });
const judge = { ...cfg.judge, timeoutMs: 1500 };
const task = { id: 't', entryPoint: 'add', prompt: '// add two numbers\nfunction add(a, b){\n' };
const tests = "const assert = require('node:assert');\nfunction test(){ let candidate = add; assert.deepEqual(candidate(2, 3), 5); }\ntest();";

const FIXTURES = new URL('./fixtures/failures.json', import.meta.url).pathname;

test('each failure class has a fixture, and the real judge + rules classify it correctly', async () => {
  const fixtures = readJson(FIXTURES);
  const seen = new Set();
  for (const f of fixtures) {
    const result = await runTests(f.code, f.tests ?? tests, judge);
    const got = classify(result, f.task ?? task, f.code, cfg.taxonomy);
    assert.equal(got, f.expected, `${f.name}: got ${got} (${result.type}: ${result.output.slice(0, 200)})`);
    seen.add(got);
  }
  for (const c of [...cfg.taxonomy.classes, cfg.taxonomy.passClass]) assert.ok(seen.has(c), `fixture for ${c}`);
});

test('TAP (tape) output is classified like node:assert output', () => {
  const tapFail = { type: 'fail', output: "not ok 3 parses host\n  ---\n    operator: equal\n    expected: 'a'\n    actual: 'b'" };
  const tapThrow = { type: 'fail', output: 'not ok 4 TypeError: Cannot read properties of undefined\n  ---\n    operator: error' };
  const t = { entryPoint: 'parse', entryPoints: ['parse'] };
  const code = 'function parse (uri, opts) { return 1 }';
  assert.equal(classify(tapFail, t, code, cfg.taxonomy), 'wrong-output');
  assert.equal(classify(tapThrow, t, code, cfg.taxonomy), 'runtime-error');
  assert.equal(classify(tapFail, t, 'function other () {}', cfg.taxonomy), 'spec-misread'); // entry point missing
});

test('rules come from config: changing them changes classification', () => {
  const custom = { passClass: 'pass', rules: [{ class: 'everything' }] };
  assert.equal(classify({ type: 'fail', output: '' }, task, '', custom), 'everything');
  assert.throws(() => classify({ type: 'fail', output: '' }, task, '', { passClass: 'pass', rules: [{ class: 'x', checks: ['nope'] }] }), /unknown check/);
});

test('every row in attempts.jsonl has a failure class from the configured list', async () => {
  const c = testConfig();
  await runCycle({ cfg: c });
  await runCycle({ cfg: c });
  const rows = readJsonl(path.join(vdir(c), c.paths.attemptsLog));
  assert.ok(rows.length > 0);
  const allowed = new Set([...c.taxonomy.classes, c.taxonomy.passClass]);
  for (const r of rows) {
    assert.ok(allowed.has(r.failureClass), `row ${r.problemId}: ${r.failureClass}`);
    assert.equal(r.failureClass === c.taxonomy.passClass, r.result === 'pass');
    for (const k of ['requests', 'tokensIn', 'tokensOut']) assert.ok(Number.isFinite(r[k]) && r[k] >= 0, k);
  }
});
