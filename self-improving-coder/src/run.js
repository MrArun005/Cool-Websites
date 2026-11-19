// One improvement cycle: train on a batch, build a candidate, evaluate champion vs candidate
// on the holdout set, gate, prune, record scores, redraw the chart.
//   node src/run.js            real run (config llm.provider)
//   node src/run.js --dry-run  mock provider + fixture dataset, state in .dry-run/
import fs from 'node:fs';
import path from 'node:path';
import { loadConfig, statePath, rootPath } from './config.js';
import { createLLM } from './llm/index.js';
import { loadProblems } from './dataset/fetch.js';
import { loadOrCreateSplit } from './dataset/split.js';
import { createBudget, budgeted, BudgetExhausted } from './budget.js';
import { solve } from './agent/solve.js';
import { reflect } from './agent/reflect.js';
import { problemKeywords } from './agent/retrieve.js';
import { runTests, RESULT } from './judge/run-tests.js';
import { decide, applyDecision, DECISION } from './gate.js';
import { loadMemory, saveMemory, newLessonId, readPrompt, ensureSeedPrompt, pruneLessons } from './memory.js';
import { appendScores } from './metrics.js';
import { writeChart } from './chart.js';
import { seededShuffle, hash32, writeJson, log } from './util.js';

export function loadTemplates(cfg) {
  const dir = rootPath(cfg, cfg.paths.templatesDir);
  const read = (k) => fs.readFileSync(path.join(dir, cfg.paths[k]), 'utf8');
  return { solveUser: read('solveUserTemplate'), reflectSystem: read('reflectSystemTemplate'), reflectUser: read('reflectUserTemplate') };
}

/** Memory view for a given set of ids. */
function view(mem, cfg, { promptVersion, lessonIds, skillIds }) {
  return {
    systemPrompt: readPrompt(cfg, promptVersion),
    lessons: lessonIds.map((id) => mem.lessons.get(id)).filter(Boolean),
    skills: skillIds.map((id) => mem.skills.get(id)).filter(Boolean),
  };
}

/** Train problems not yet solved by champion/pending, never-tried first, then failures due for retry. */
export function pickTrainOrder(split, mem, runIndex, cfg) {
  const solved = new Set([...mem.champion.skillIds, ...mem.champion.pending.skillIds]);
  const fresh = [];
  const retry = [];
  for (const id of split.train) {
    if (solved.has(id)) continue;
    const a = mem.attempts[id];
    if (!a) fresh.push(id);
    else if (runIndex - a.lastRun >= cfg.agent.retryFailedAfterRuns) retry.push(id);
  }
  const seed = cfg.split.seed + runIndex;
  const retrySorted = seededShuffle(retry, seed).sort((x, y) => mem.attempts[x].tries - mem.attempts[y].tries);
  return [...seededShuffle(fresh, seed), ...retrySorted];
}

async function trainPhase({ llm, cfg, mem, problemsById, order, templates, runIndex, working, budget }) {
  const reserve = cfg.budget.reserveForHoldout;
  const stats = { problems: 0, passed: 0, newLessons: [], newSkills: [] };
  let stop = false;
  for (const id of order) {
    if (stop) break;
    const problem = problemsById.get(id);
    let passed = false;
    let attempted = false;
    for (let attempt = 1; attempt <= cfg.agent.maxAttemptsPerProblem && !passed; attempt++) {
      if (budget.remainingForTrain(reserve) <= 0) { stop = true; break; }
      const memory = view(mem, cfg, working);
      let res;
      try {
        res = await solve({ llm, problem, memory, templates, cfg, temperature: cfg.llm.temperature, reserve });
      } catch (e) {
        if (e instanceof BudgetExhausted) { stop = true; break; }
        throw e;
      }
      attempted = true;
      const result = await runTests(res.code, problem.tests, cfg.judge);
      passed = result.type === RESULT.PASS;
      for (const l of res.usedLessons) { l.uses = (l.uses ?? 0) + 1; if (passed) l.helped = (l.helped ?? 0) + 1; }
      log(`train ${id} attempt ${attempt}: ${result.type}${res.usedLessons.length ? ` (lessons: ${res.usedLessons.map((l) => l.id).join(',')})` : ''}`);
      if (passed) {
        if (!mem.skills.has(id)) {
          mem.skills.set(id, { id, entryPoint: problem.entryPoint, keywords: [...problemKeywords(problem, cfg.retrieve)], code: res.code, createdRun: runIndex });
          working.skillIds.push(id);
          stats.newSkills.push(id);
        }
      } else {
        let lesson = null;
        try {
          lesson = await reflect({ llm, problem, code: res.code, result, templates, cfg, reserve });
        } catch (e) {
          if (!(e instanceof BudgetExhausted)) throw e;
          stop = true;
        }
        const live = new Set(working.lessonIds);
        const dupe = lesson && [...mem.lessons.values()].some((l) => live.has(l.id) && l.text.toLowerCase() === lesson.text.toLowerCase());
        if (lesson && !dupe) {
          const l = { id: newLessonId(mem), text: lesson.text, tags: lesson.tags, sourceProblem: id, sourceResult: result.type, createdRun: runIndex, uses: 0, helped: 0 };
          mem.lessons.set(l.id, l);
          working.lessonIds.push(l.id);
          stats.newLessons.push(l.id);
          log(`  lesson ${l.id}: ${l.text}`);
        }
        if (stop) break;
      }
    }
    if (!attempted) continue;
    stats.problems++;
    if (passed) stats.passed++;
    const prev = mem.attempts[id] ?? { tries: 0 };
    mem.attempts[id] = { tries: prev.tries + 1, lastRun: runIndex, passed };
  }
  return stats;
}

/** Holdout evaluation: one attempt per problem, eval temperature, no reflection, no memory updates. */
export async function evaluate({ llm, cfg, mem, problemsById, ids, state, templates }) {
  const memory = view(mem, cfg, state);
  let pass = 0;
  let n = 0;
  for (const id of ids) {
    const problem = problemsById.get(id);
    const res = await solve({ llm, problem, memory, templates, cfg, temperature: cfg.llm.evalTemperature, reserve: 0 });
    const result = await runTests(res.code, problem.tests, cfg.judge);
    n++;
    if (result.type === RESULT.PASS) pass++;
  }
  return { passRate: n ? pass / n : 0, n, pass };
}

function sameState(a, b) {
  const eq = (x, y) => x.length === y.length && x.every((v, i) => v === y[i]);
  return a.promptVersion === b.promptVersion && eq([...a.lessonIds].sort(), [...b.lessonIds].sort()) && eq([...a.skillIds].sort(), [...b.skillIds].sort());
}

/**
 * Run one cycle.
 * @param {{cfg: object, deps?: {llm?, evolvePrompt?, now?, fetch?}}} args
 */
export async function runCycle({ cfg, deps = {} }) {
  const now = deps.now ?? (() => new Date());
  const runId = (process.env.GITHUB_RUN_ID ? `gh${process.env.GITHUB_RUN_ID}-` : '') + now().toISOString().replace(/[-:]/g, '').slice(0, 15);
  const timestamp = now().toISOString();
  const problems = await loadProblems(cfg, deps);
  const problemsById = new Map(problems.map((p) => [p.id, p]));
  const split = loadOrCreateSplit(statePath(cfg, 'split'), problems.map((p) => p.id), cfg.split);
  const mem = loadMemory(cfg);
  ensureSeedPrompt(cfg, mem.champion, fs.readFileSync(rootPath(cfg, cfg.paths.seedPrompt), 'utf8'));
  const templates = loadTemplates(cfg);
  const budget = createBudget(cfg, { file: statePath(cfg, 'budget'), now });
  const llm = budgeted(deps.llm ?? createLLM(cfg, deps), budget);
  const runIndex = mem.runState.runCount + 1;
  const ch = mem.champion;
  const championState = { promptVersion: ch.promptVersion, lessonIds: [...ch.lessonIds], skillIds: [...ch.skillIds] };
  const baselineMode = mem.runState.baselineRunsDone < cfg.gate.baselineRuns;
  log(`run ${runId} (#${runIndex})${baselineMode ? ' [baseline]' : ''}: ${split.train.length} train / ${split.holdout.length} holdout problems`);

  // 1. Train on top of champion + pending changes.
  const working = {
    promptVersion: ch.pending.promptVersion ?? ch.promptVersion,
    lessonIds: [...ch.lessonIds, ...ch.pending.lessonIds],
    skillIds: [...ch.skillIds, ...ch.pending.skillIds],
  };
  let train = { problems: 0, passed: 0, newLessons: [], newSkills: [] };
  if (!baselineMode) {
    const order = pickTrainOrder(split, mem, runIndex, cfg);
    train = await trainPhase({ llm, cfg, mem, problemsById, order, templates, runIndex, working, budget });
    log(`train: ${train.passed}/${train.problems} passed, +${train.newLessons.length} lessons, +${train.newSkills.length} skills`);

    // Phase 5 (flag-gated): propose a new system prompt from train failures.
    if (cfg.features.promptOptimizer) {
      const evolvePrompt = deps.evolvePrompt ?? (await import('./agent/evolve.js')).evolvePrompt;
      const version = await evolvePrompt({ cfg, mem, split, problemsById, working, budget, llm, templates }).catch((e) => {
        if (e instanceof BudgetExhausted) return null;
        throw e;
      });
      if (version) { working.promptVersion = version; log(`prompt optimizer proposed ${version}`); }
    }
  }

  // 2. Holdout evaluation.
  const candidateState = working;
  const changed = !sameState(championState, candidateState);
  const evalCount = baselineMode || !changed ? 1 : 2;
  const size = Math.min(split.holdout.length, Math.floor(budget.remainingRun() / evalCount));
  const holdoutIds = size >= split.holdout.length ? split.holdout : seededShuffle(split.holdout, cfg.split.seed ^ hash32(runId)).slice(0, size).sort();
  let decision;
  let champScore = null;
  let candScore = null;
  if (size <= 0) {
    decision = baselineMode ? DECISION.BASELINE : changed ? DECISION.NO_EVAL : DECISION.NO_CHANGE;
  } else {
    try {
      champScore = await evaluate({ llm, cfg, mem, problemsById, ids: holdoutIds, state: championState, templates });
      log(`holdout champion: ${(champScore.passRate * 100).toFixed(1)}% (n=${champScore.n})`);
      if (baselineMode) decision = DECISION.BASELINE;
      else if (!changed) decision = DECISION.NO_CHANGE;
      else {
        candScore = await evaluate({ llm, cfg, mem, problemsById, ids: holdoutIds, state: candidateState, templates });
        log(`holdout candidate: ${(candScore.passRate * 100).toFixed(1)}% (n=${candScore.n})`);
        decision = decide(champScore.passRate, candScore.passRate, cfg.gate);
      }
    } catch (e) {
      if (!(e instanceof BudgetExhausted)) throw e;
      log(`holdout evaluation stopped early (${e.reason}); scores discarded`);
      champScore = null;
      candScore = null;
      decision = baselineMode ? DECISION.BASELINE : changed ? DECISION.NO_EVAL : DECISION.NO_CHANGE;
    }
  }
  log(`gate: ${decision}`);

  // 3. Gate + prune.
  if (baselineMode) {
    if (champScore) {
      mem.runState.baselineRunsDone++; // only completed measurements count as baseline runs
      ch.holdout = { passRate: champScore.passRate, n: champScore.n, runId };
    }
  } else if (decision !== DECISION.NO_CHANGE) {
    const holdout = candScore ? { passRate: candScore.passRate, n: candScore.n, runId } : ch.holdout;
    applyDecision(ch, candidateState, decision, { runIndex, holdout, maxPendingRuns: cfg.gate.maxPendingRuns });
  }
  const pruned = pruneLessons(mem, cfg.agent.maxLessons);
  mem.runState.runCount = runIndex;

  // 4. Persist state, scores, chart.
  saveMemory(cfg, mem);
  budget.save();
  const base = { timestamp, runId, lessonsCount: ch.lessonIds.length, skillsCount: ch.skillIds.length, requestsUsed: budget.usedRun, gateDecision: decision };
  const rows = [];
  if (train.problems) rows.push({ ...base, split: 'train', passRate: train.passed / train.problems, n: train.problems, promptVersion: candidateState.promptVersion });
  if (champScore) rows.push({ ...base, split: 'holdout-champion', passRate: champScore.passRate, n: champScore.n, promptVersion: championState.promptVersion });
  if (candScore) rows.push({ ...base, split: 'holdout-candidate', passRate: candScore.passRate, n: candScore.n, promptVersion: candidateState.promptVersion });
  if (rows.length) appendScores(statePath(cfg, 'scores'), rows);
  writeChart(cfg);
  const summary = { runId, runIndex, timestamp, decision, baselineMode, train, champion: champScore, candidate: candScore, pruned, requestsUsed: budget.usedRun, requestsToday: budget.usedToday };
  writeJson(statePath(cfg, 'lastRun'), summary);
  return summary;
}

async function main() {
  const dryRun = process.argv.includes('--dry-run');
  const cfg = loadConfig({ dryRun });
  const summary = await runCycle({ cfg });
  log(`done: ${summary.decision}, ${summary.requestsUsed} requests`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((e) => {
    console.error(e?.stack ?? e);
    process.exit(1);
  });
}
