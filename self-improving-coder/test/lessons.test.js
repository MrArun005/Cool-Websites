// Phase 2 acceptance: a problem fails on attempt 1, a lesson is stored, and that lesson
// appears in the prompt for a related problem.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
import { runCycle } from '../src/run.js';
import { createLLM } from '../src/llm/index.js';
import { readJson } from '../src/util.js';
import { pruneLessons } from '../src/memory.js';
import { testConfig, writeSplit, vdir } from './helpers.js';

test('lesson learned from a failure is reused on a related problem', async () => {
  const cfg = testConfig({ agent: { maxAttemptsPerProblem: 2 } });
  writeSplit(cfg, ['fx_sum_to_n', 'fx_count_even_in_range'], ['fx_product_range', 'fx_sum_squares_range']);
  const llm = createLLM(cfg);
  const summary = await runCycle({ cfg, deps: { llm } });

  const solves = llm.calls.filter((c) => c.meta.kind === 'solve' && c.temperature === cfg.llm.temperature);
  const first = solves[0].meta.problemId;
  const other = first === 'fx_sum_to_n' ? 'fx_count_even_in_range' : 'fx_sum_to_n';

  // attempt 1 on the first problem had no lessons and failed -> a reflect call -> a stored lesson
  assert.doesNotMatch(solves[0].messages[1].content, /Lessons from past mistakes/);
  assert.equal(llm.calls.filter((c) => c.meta.kind === 'reflect')[0].meta.problemId, first);
  const lessons = readJson(path.join(vdir(cfg), cfg.paths.lessons)).items;
  assert.equal(lessons.length, 1);
  assert.equal(lessons[0].sourceProblem, first);
  assert.match(lessons[0].text, /include both ends/);

  // the related problem's FIRST prompt already contains the lesson, and it passes first time
  const otherFirst = solves.find((c) => c.meta.problemId === other);
  assert.match(otherFirst.messages[1].content, /Lessons from past mistakes[\s\S]*include both ends/);
  assert.equal(solves.filter((c) => c.meta.problemId === other).length, 1);
  assert.equal(summary.train.passed, 2);

  // the lesson also carried over to the holdout candidate and the gate accepted it
  assert.equal(summary.decision, 'accept');
  assert.ok(summary.candidate.passRate > summary.champion.passRate);
  const lessonAfter = lessons[0];
  assert.ok(lessonAfter.uses >= 2 && lessonAfter.helped >= 2);
});

test('passing solutions are saved as skills (deduped by problem id)', async () => {
  const cfg = testConfig();
  writeSplit(cfg, ['fx_reverse_words', 'fx_double_all'], ['fx_join_dash', 'fx_vowel_count']);
  await runCycle({ cfg });
  // holdout is already at 100% so the gate says "keep": skills stay pending, not lost
  const live = (c) => [...c.skillIds, ...c.pending.skillIds].sort();
  const champion = readJson(path.join(vdir(cfg), cfg.paths.champion));
  assert.deepEqual(live(champion), ['fx_double_all', 'fx_reverse_words']);
  assert.deepEqual(fs.readdirSync(path.join(vdir(cfg), cfg.paths.skillsDir)).sort(), ['fx_double_all.js', 'fx_reverse_words.js']);
  // second cycle: solved problems are not retried, so no duplicates
  await runCycle({ cfg });
  const again = readJson(path.join(vdir(cfg), cfg.paths.champion));
  assert.deepEqual(live(again), ['fx_double_all', 'fx_reverse_words']);
});

test('pruning drops never-helped, then least-used lessons first', () => {
  const mk = (id, uses, helped) => [id, { id, uses, helped, text: id, tags: [] }];
  const mem = {
    lessons: new Map([mk('L1', 5, 2), mk('L2', 9, 0), mk('L3', 1, 0), mk('L4', 3, 1)]),
    champion: { lessonIds: ['L1', 'L2', 'L3'], pending: { lessonIds: ['L4'] } },
  };
  assert.deepEqual(pruneLessons(mem, 2), ['L3', 'L2']);
  assert.deepEqual(mem.champion.lessonIds, ['L1']);
  assert.deepEqual(mem.champion.pending.lessonIds, ['L4']);
});
