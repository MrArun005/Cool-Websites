import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { loadConfig, deepMerge } from '../src/config.js';
import { variantConfig } from '../src/variants.js';
import { readJson } from '../src/util.js';

process.env.SIC_QUIET = '1';

export function tmpDir(prefix = 'sic-test-') {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

export const TEST_VARIANT = 'lessons-skills';
const ONE_VARIANT = [{ name: TEST_VARIANT, label: 'Lessons + skills', track: 'multipl-e', levers: { lessons: true, skills: true }, budgetShare: 1, color: '#16a34a' }];

/**
 * Dry-run config (mock LLM, fixture dataset) with state in a fresh temp dir and a single variant,
 * so `budget.total*` equals that variant's budget. `budget.requestsPerRun` / `reserveForHoldout` style
 * overrides are translated to the variant-level equivalents.
 */
export function testConfig(overrides = {}) {
  const stateDir = tmpDir();
  const { budget: b, ...rest } = overrides;
  const budget = b ? {
    ...(b.requestsPerRun !== undefined ? { totalRequestsPerRun: b.requestsPerRun } : {}),
    ...(b.requestsPerDay !== undefined ? { totalRequestsPerDay: b.requestsPerDay } : {}),
    ...(b.reserveForHoldout !== undefined ? { holdoutFraction: b.reserveForHoldout / b.requestsPerRun } : {}),
    ...Object.fromEntries(Object.entries(b).filter(([k]) => !['requestsPerRun', 'requestsPerDay', 'reserveForHoldout'].includes(k))),
  } : undefined;
  const base = loadConfig({ dryRun: true, env: {}, overrides: { paths: { stateDir, chartsDir: stateDir }, variants: ONE_VARIANT, defaultVariant: TEST_VARIANT } });
  return deepMerge(base, { ...rest, ...(budget ? { budget } : {}) });
}

/** Absolute state dir of a variant. */
export function vdir(cfg, variant = cfg.defaultVariant) {
  return variantConfig(cfg, variant).paths.stateDir;
}

export const fixtures = () => readJson(new URL('./fixtures/problems.json', import.meta.url).pathname);

/** Write a split.json so tests control exactly which problems are train vs holdout. */
export function writeSplit(cfg, train, holdout, variant = cfg.defaultVariant) {
  const file = path.join(vdir(cfg, variant), cfg.paths.split);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify({ seed: 0, holdoutSize: holdout.length, train, holdout }));
}

/** Simple scripted LLM for tests: answer(meta, messages) -> string. Records calls. */
export function scriptedLLM(answer) {
  const calls = [];
  return { name: 'scripted', calls, lastUsage: { tokensIn: 1, tokensOut: 1 }, async chat(messages, opts = {}) { calls.push({ messages, ...opts }); return answer(opts.meta ?? {}, messages); } };
}
