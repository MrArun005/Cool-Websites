import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { runCycle } from '../src/run.js';
import { createLLM } from '../src/llm/index.js';
import { readJson } from '../src/util.js';
import { COLUMNS } from '../src/metrics.js';
import { ROOT, loadConfig } from '../src/config.js';
import { testConfig, writeSplit, tmpDir, vdir } from './helpers.js';

test('`node src/run.js --dry-run` runs a full cycle for every variant and writes scores.csv + charts', () => {
  const stateDir = tmpDir();
  const chartsDir = tmpDir();
  const cacheDir = tmpDir();
  execFileSync(process.execPath, ['src/run.js', '--dry-run'], { cwd: ROOT, env: { ...process.env, SIC_QUIET: '1',
    SIC_CONFIG_OVERRIDES: JSON.stringify({ paths: { stateDir, chartsDir }, curriculum: { fixture: { cacheDir } } }) } });
  const cfg = loadConfig({ dryRun: true, env: {} });
  for (const v of cfg.variants) {
    const dir = path.join(stateDir, 'variants', v.name);
    const scores = fs.readFileSync(path.join(dir, 'scores.csv'), 'utf8').trim().split('\n');
    assert.equal(scores[0], COLUMNS.join(','), v.name);
    assert.ok(scores.length >= 2, v.name);
    for (const f of ['champion.json', 'lessons.json', 'split.json', 'problem-status.json', 'run-state.json', 'last-run.json', 'attempts.jsonl', 'evals.jsonl', 'gate-log.jsonl']) {
      if (f === 'split.json' && v.track !== 'multipl-e') continue;
      assert.ok(fs.existsSync(path.join(dir, f)), `${v.name}/${f}`);
    }
  }
  for (const c of cfg.charts) assert.match(fs.readFileSync(path.join(chartsDir, c.file), 'utf8'), /<polyline/);
});

test('running out of budget mid-training stops gracefully and still writes state + scores', async () => {
  const cfg = testConfig({ budget: { requestsPerRun: 6, reserveForHoldout: 4, requestsPerDay: 1000, maxRunMinutes: 60, reserveMinutesForHoldout: 0 } });
  writeSplit(cfg, ['fx_sum_to_n', 'fx_count_even_in_range', 'fx_max_of_list', 'fx_first_char'], ['fx_reverse_words', 'fx_double_all']);
  const llm = createLLM(cfg);
  const s = await runCycle({ cfg, deps: { llm } });
  assert.equal(llm.calls.length, 6);
  assert.ok(s.train.problems >= 1);
  assert.equal(s.champion.n, 2); // 4 reserved requests -> 2 holdout problems x (champion + candidate)
  assert.ok(fs.existsSync(path.join(vdir(cfg), cfg.paths.scores)));
});

test('daily limit hit before holdout eval: run ends with no-eval and keeps changes pending', async () => {
  const cfg = testConfig({ budget: { requestsPerRun: 50, reserveForHoldout: 0, requestsPerDay: 3, maxRunMinutes: 60, reserveMinutesForHoldout: 0 } });
  writeSplit(cfg, ['fx_reverse_words', 'fx_double_all', 'fx_join_dash'], ['fx_vowel_count', 'fx_is_palindrome']);
  const s = await runCycle({ cfg });
  assert.equal(s.decision, 'no-eval');
  const champion = readJson(path.join(vdir(cfg), cfg.paths.champion));
  assert.deepEqual(champion.skillIds, []);
  assert.equal(champion.pending.skillIds.length, 3);
});

test('holdout problems are never trained on or reflected on', async () => {
  const cfg = testConfig();
  const holdout = ['fx_sum_to_n', 'fx_product_range'];
  writeSplit(cfg, ['fx_count_even_in_range', 'fx_max_of_list'], holdout);
  const llm = createLLM(cfg);
  await runCycle({ cfg, deps: { llm } });
  for (const c of llm.calls) {
    if (holdout.includes(c.meta.problemId)) {
      assert.equal(c.meta.kind, 'solve');
      assert.equal(c.temperature, cfg.llm.evalTemperature);
    }
  }
  const lessons = readJson(path.join(vdir(cfg), cfg.paths.lessons)).items;
  assert.ok(lessons.every((l) => !holdout.includes(l.sourceProblem)));
  const champion = readJson(path.join(vdir(cfg), cfg.paths.champion));
  assert.ok([...champion.skillIds, ...champion.pending.skillIds].every((id) => !holdout.includes(id)));
});

// Phase 5 acceptance: with the flag off, behaviour is identical to phase 4.
async function snapshot(features, evolvePrompt) {
  const cfg = testConfig({ features, promptOptimizer: { everyNRuns: 1 }, variants: [{ name: 'lessons-skills', track: 'multipl-e', levers: { lessons: true, skills: true, promptOptimizer: true }, budgetShare: 1, color: '#000' }] });
  writeSplit(cfg, ['fx_sum_to_n', 'fx_count_even_in_range', 'fx_max_of_list', 'fx_first_char'], ['fx_product_range', 'fx_reverse_words']);
  const llm = createLLM(cfg);
  const fixedNow = () => new Date('2026-11-20T03:17:00Z');
  const summary = await runCycle({ cfg, deps: { llm, now: fixedNow, evolvePrompt } });
  const dir = vdir(cfg);
  const files = {};
  for (const f of fs.readdirSync(dir, { recursive: true }).sort()) {
    const p = path.join(dir, f);
    if (fs.statSync(p).isFile()) files[f] = fs.readFileSync(p, 'utf8');
  }
  return { summary, files, calls: llm.calls.map((c) => ({ meta: c.meta, temperature: c.temperature, messages: c.messages })) };
}

test('promptOptimizer flag off: optimizer never called, output identical to a run without it', async () => {
  let called = 0;
  const spy = async () => { called++; return null; };
  const off = await snapshot({ promptOptimizer: false }, spy);
  const plain = await snapshot({ promptOptimizer: false }, undefined);
  // the run-dependent bits (temp dir names inside logs) are not part of behaviour
  for (const snap of [off, plain]) for (const k of Object.keys(snap.files)) snap.files[k] = snap.files[k].replace(/sic-test-\w+/g, 'TMP').replace(/"(durationMs|llmMs)":\d+/g, '"$1":0');
  assert.equal(called, 0);
  assert.deepEqual(off.calls, plain.calls);
  assert.deepEqual(off.files, plain.files);
  assert.deepEqual(off.summary, plain.summary);
  assert.ok(!off.calls.some((c) => c.meta.kind === 'evolve'));
});

test('promptOptimizer flag on: a proposed prompt goes through the same holdout gate', async () => {
  const proposed = [];
  const evolve = async ({ cfg }) => {
    const { writePrompt, nextPromptVersion } = await import('../src/memory.js');
    const v = nextPromptVersion(cfg);
    writePrompt(cfg, v, 'You are a careful JavaScript programmer. Reply with one code block.');
    proposed.push(v);
    return v;
  };
  const on = await snapshot({ promptOptimizer: true }, evolve);
  assert.deepEqual(proposed, ['v002']);
  const cand = on.calls.filter((c) => c.temperature === 0 && c.messages[0].content.startsWith('You are a careful'));
  assert.ok(cand.length > 0, 'candidate holdout eval used the proposed prompt');
  assert.ok(['accept', 'keep', 'reject'].includes(on.summary.decision));
  const champion = JSON.parse(on.files['champion.json']);
  if (on.summary.decision === 'accept') assert.equal(champion.promptVersion, 'v002');
  else assert.equal(champion.promptVersion, 'v001');
});
