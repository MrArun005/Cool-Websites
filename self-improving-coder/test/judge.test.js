import { test } from 'node:test';
import assert from 'node:assert/strict';
import { runTests, RESULT } from '../src/judge/run-tests.js';
import { loadConfig } from '../src/config.js';

const judge = { ...loadConfig({ env: {} }).judge, timeoutMs: 1500 };
const tests = (fn) => `const assert = require('node:assert');\nfunction test(){ let candidate = ${fn}; assert.deepEqual(candidate(2), 4); }\ntest();`;

test('pass', async () => {
  const r = await runTests('function dbl(x){ return x * 2; }', tests('dbl'), judge);
  assert.equal(r.type, RESULT.PASS);
});

test('assertion failure is "fail" and shows the assertion', async () => {
  const r = await runTests('function dbl(x){ return x + 1; }', tests('dbl'), judge);
  assert.equal(r.type, RESULT.FAIL);
  assert.match(r.output, /AssertionError/);
});

test('crash is "error"', async () => {
  const r = await runTests('function dbl(x){ return y; }', tests('dbl'), judge);
  assert.equal(r.type, RESULT.ERROR);
  assert.match(r.output, /ReferenceError/);
});

test('syntax error is "error"', async () => {
  const r = await runTests('function dbl(x){ return x * ; }', tests('dbl'), judge);
  assert.equal(r.type, RESULT.ERROR);
});

test('infinite loop is "timeout" and the process is killed', async () => {
  const t0 = Date.now();
  const r = await runTests('function dbl(x){ while(true){} }', tests('dbl'), judge);
  assert.equal(r.type, RESULT.TIMEOUT);
  assert.ok(Date.now() - t0 < judge.timeoutMs + 3000);
});

test('solution declaring its own `assert` does not clash with the tests', async () => {
  const r = await runTests("const assert = require('node:assert');\nfunction dbl(x){ return x * 2; }", tests('dbl'), judge);
  assert.equal(r.type, RESULT.PASS);
});

test('output is capped at maxOutputBytes', async () => {
  const r = await runTests('function dbl(x){ process.stderr.write("x".repeat(100000)); return x * 2; }', tests('dbl'), { ...judge, maxOutputBytes: 500 });
  assert.ok(r.output.length <= 600, `output length ${r.output.length}`);
});

test('child gets an empty environment (no tokens)', async () => {
  process.env.SIC_TEST_SECRET = 'leak-me';
  const r = await runTests('function dbl(x){ return Object.keys(process.env).length === 0 ? x * 2 : -1; }', tests('dbl'), judge);
  delete process.env.SIC_TEST_SECRET;
  assert.equal(r.type, RESULT.PASS);
});

test('permission model blocks spawning processes and writing files', async () => {
  const spawnTry = "function dbl(x){ require('node:child_process').execSync('echo hi'); return x * 2; }";
  assert.equal((await runTests(spawnTry, tests('dbl'), judge)).type, RESULT.ERROR);
  const writeTry = "function dbl(x){ require('node:fs').writeFileSync('/tmp/sic-pwned', 'x'); return x * 2; }";
  assert.equal((await runTests(writeTry, tests('dbl'), judge)).type, RESULT.ERROR);
  const readTry = "function dbl(x){ require('node:fs').readFileSync('/etc/hostname'); return x * 2; }";
  assert.equal((await runTests(readTry, tests('dbl'), judge)).type, RESULT.ERROR);
});
