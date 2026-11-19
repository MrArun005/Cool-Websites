// Phase 3 acceptance: accept / keep / reject, and a rejected candidate leaves champion.json unchanged.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { decide, applyDecision, DECISION } from '../src/gate.js';
import { runCycle } from '../src/run.js';
import { readJson } from '../src/util.js';
import { testConfig, writeSplit, fixtures, scriptedLLM, vdir } from './helpers.js';

const g = { minImprovement: 0.05, tolerance: 0.02 };

test('decide: accept / keep / reject boundaries', () => {
  assert.equal(decide(0.5, 0.55, g), DECISION.ACCEPT);
  assert.equal(decide(0.5, 0.549, g), DECISION.KEEP);
  assert.equal(decide(0.5, 0.48, g), DECISION.KEEP);
  assert.equal(decide(0.5, 0.479, g), DECISION.REJECT);
  assert.equal(decide(0.1, 0.15, g), DECISION.ACCEPT); // float-safe
});

const champ = () => ({ promptVersion: 'v001', lessonIds: ['L1'], skillIds: ['s1'], holdout: null, pending: { lessonIds: [], skillIds: [], promptVersion: null, sinceRun: null } });
const cand = { promptVersion: 'v002', lessonIds: ['L1', 'L2'], skillIds: ['s1', 's2'] };
const opts = { runIndex: 5, holdout: { passRate: 0.6, n: 10, runId: 'r' }, maxPendingRuns: 2 };

test('applyDecision accept: candidate becomes champion', () => {
  const c = applyDecision(champ(), cand, DECISION.ACCEPT, opts);
  assert.deepEqual([c.promptVersion, c.lessonIds, c.skillIds, c.holdout.passRate], ['v002', ['L1', 'L2'], ['s1', 's2'], 0.6]);
});

test('applyDecision keep: champion unchanged, additions pending; stale pending dropped', () => {
  const c = applyDecision(champ(), cand, DECISION.KEEP, opts);
  assert.deepEqual(c.lessonIds, ['L1']);
  assert.deepEqual(c.pending, { lessonIds: ['L2'], skillIds: ['s2'], promptVersion: 'v002', sinceRun: 5 });
  applyDecision(c, cand, DECISION.KEEP, { ...opts, runIndex: 6 });
  assert.equal(c.pending.sinceRun, 5);
  applyDecision(c, cand, DECISION.KEEP, { ...opts, runIndex: 7 }); // 3 runs pending > maxPendingRuns 2
  assert.deepEqual(c.pending.lessonIds, []);
});

test('applyDecision reject: additions dropped', () => {
  const c = applyDecision(champ(), cand, DECISION.REJECT, opts);
  assert.deepEqual(c, { ...champ() });
});

test('a rejected candidate leaves champion.json byte-for-byte unchanged and drops its lessons', async () => {
  const cfg = testConfig({ gate: { baselineRuns: 0, minImprovement: 0.05, tolerance: 0 } });
  const train = ['fx_sum_to_n', 'fx_count_even_in_range'];
  const holdout = ['fx_reverse_words', 'fx_double_all', 'fx_join_dash', 'fx_vowel_count'];
  writeSplit(cfg, train, holdout);
  const champFile = path.join(vdir(cfg), cfg.paths.champion);
  const before = JSON.stringify({ promptVersion: 'v001', lessonIds: [], skillIds: [], holdout: { passRate: 1, n: 4, runId: 'old' }, pending: { lessonIds: [], skillIds: [], promptVersion: null, sinceRun: null } }, null, 2) + '\n';
  fs.writeFileSync(champFile, before);
  const fx = new Map(fixtures().map((p) => [p.id, p]));
  // Train: always wrong (so lessons are made). Holdout: right unless the prompt contains lessons -> candidate is worse.
  const llm = scriptedLLM((meta, messages) => {
    if (meta.kind === 'reflect') return '{"lesson":"A harmful lesson about words and lists","tags":["words","list","double","dash"]}';
    const p = fx.get(meta.problemId);
    const hasLessons = messages[1].content.includes('Lessons from past mistakes');
    const good = holdout.includes(meta.problemId) && !hasLessons;
    return '```javascript\n' + (good ? p.mock.correct : `function ${p.entryPoint}(){ return null; }`) + '\n```';
  });
  const summary = await runCycle({ cfg, deps: { llm } });
  assert.equal(summary.decision, DECISION.REJECT);
  assert.ok(summary.candidate.passRate < summary.champion.passRate);
  assert.equal(fs.readFileSync(champFile, 'utf8'), before);
  assert.deepEqual(readJson(path.join(vdir(cfg), cfg.paths.lessons)).items, []);
  const scores = fs.readFileSync(path.join(vdir(cfg), cfg.paths.scores), 'utf8');
  assert.match(scores, /holdout-candidate,.*,reject/);
});

test('baseline runs only measure the champion and accept nothing', async () => {
  const cfg = testConfig({ gate: { baselineRuns: 2 } });
  writeSplit(cfg, ['fx_sum_to_n'], ['fx_reverse_words', 'fx_double_all']);
  const s1 = await runCycle({ cfg });
  const s2 = await runCycle({ cfg });
  const s3 = await runCycle({ cfg });
  assert.deepEqual([s1.decision, s2.decision], [DECISION.BASELINE, DECISION.BASELINE]);
  assert.equal(s1.train.problems, 0);
  assert.equal(s1.candidate, null);
  assert.notEqual(s3.decision, DECISION.BASELINE);
  assert.ok(s3.train.problems > 0);
});
