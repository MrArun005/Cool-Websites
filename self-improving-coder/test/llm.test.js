import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createOpenAICompatible, backoffDelay } from '../src/llm/openai-compatible.js';
import { createLLM } from '../src/llm/index.js';
import { loadConfig } from '../src/config.js';
import { testConfig } from './helpers.js';

const cfg = loadConfig({ env: {} });
const ok = (content) => ({ ok: true, status: 200, json: async () => ({ choices: [{ message: { content } }] }) });
const err = (status) => ({ ok: false, status, text: async () => 'nope', headers: { get: () => null } });

test('retries 429 with backoff, then succeeds', async () => {
  const seq = [err(429), err(503), ok('hello')];
  const sleeps = [];
  const llm = createOpenAICompatible(cfg, { fetch: async () => seq.shift(), sleep: async (ms) => sleeps.push(ms), env: {} });
  assert.equal(await llm.chat([{ role: 'user', content: 'hi' }]), 'hello');
  assert.deepEqual(sleeps, [cfg.llm.retry.baseDelayMs, cfg.llm.retry.baseDelayMs * 2]);
});

test('does not retry non-retryable errors', async () => {
  let n = 0;
  const llm = createOpenAICompatible(cfg, { fetch: async () => { n++; return err(400); }, sleep: async () => {}, env: {} });
  await assert.rejects(llm.chat([]), /HTTP 400/);
  assert.equal(n, 1);
});

test('gives up after maxAttempts', async () => {
  let n = 0;
  const llm = createOpenAICompatible(cfg, { fetch: async () => { n++; return err(429); }, sleep: async () => {}, env: {} });
  await assert.rejects(llm.chat([]), /429/);
  assert.equal(n, cfg.llm.retry.maxAttempts);
});

test('sends model, temperature, max_tokens and optional bearer key', async () => {
  let seen;
  const llm = createOpenAICompatible(cfg, { fetch: async (url, init) => { seen = { url, init }; return ok('x'); }, env: { [cfg.llm.apiKeyEnv]: 'k' } });
  await llm.chat([{ role: 'user', content: 'hi' }], { temperature: 0 });
  const body = JSON.parse(seen.init.body);
  assert.equal(seen.url, cfg.llm.endpoint);
  assert.equal(body.model, cfg.llm.model);
  assert.equal(body.temperature, 0);
  assert.equal(body.max_tokens, cfg.llm.maxTokens);
  assert.equal(seen.init.headers.Authorization, 'Bearer k');
});

test('backoff is capped and honours Retry-After', () => {
  const r = { baseDelayMs: 100, maxDelayMs: 1000 };
  assert.equal(backoffDelay(1, r), 100);
  assert.equal(backoffDelay(10, r), 1000);
  assert.equal(backoffDelay(1, r, 0.5), 500);
});

test('mock returns buggy code until the fixing phrase is in the prompt', async () => {
  const llm = createLLM(testConfig());
  const meta = { kind: 'solve', problemId: 'fx_sum_to_n' };
  assert.match(await llm.chat([{ role: 'user', content: 'x' }], { meta }), /i < n/);
  assert.match(await llm.chat([{ role: 'user', content: 'include both ends' }], { meta }), /i <= n/);
});
