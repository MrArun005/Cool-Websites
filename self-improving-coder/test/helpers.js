import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { loadConfig } from '../src/config.js';
import { readJson } from '../src/util.js';

process.env.SIC_QUIET = '1';

export function tmpDir(prefix = 'sic-test-') {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

/** Dry-run config (mock LLM, fixture dataset) with state in a fresh temp dir. */
export function testConfig(overrides = {}) {
  const stateDir = tmpDir();
  const cfg = loadConfig({ dryRun: true, env: {}, overrides: { paths: { stateDir }, ...overrides } });
  return cfg;
}

export const fixtures = () => readJson(new URL('./fixtures/problems.json', import.meta.url).pathname);

/** Write a split.json so tests control exactly which problems are train vs holdout. */
export function writeSplit(cfg, train, holdout) {
  const file = path.join(cfg.paths.stateDir, cfg.paths.split);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify({ seed: 0, holdoutSize: holdout.length, train, holdout }));
}

/** Simple scripted LLM for tests: answer(meta, messages) -> string. Records calls. */
export function scriptedLLM(answer) {
  const calls = [];
  return { name: 'scripted', calls, async chat(messages, opts = {}) { calls.push({ messages, ...opts }); return answer(opts.meta ?? {}, messages); } };
}
