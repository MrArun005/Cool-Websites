// One improvement cycle for one variant: train on a batch, build a candidate, evaluate champion and
// candidate on the same holdout tasks `eval.repeats` times, gate on a paired bootstrap CI, prune,
// record raw logs + scores, redraw charts.
//   node src/run.js --variant=<name>   one variant (what each CI matrix job runs)
//   node src/run.js                    every variant in config, one after another
//   node src/run.js --dry-run          mock LLM + fixture tasks, state in .dry-run/
import fs from 'node:fs';
import path from 'node:path';
import { loadConfig, statePath, rootPath } from './config.js';
import { createLLM } from './llm/index.js';
import { createBudget, budgeted, BudgetExhausted } from './budget.js';
import { solve } from './agent/solve.js';
import { reflect } from './agent/reflect.js';
import { problemKeywords } from './agent/retrieve.js';
import { RESULT } from './judge/run-tests.js';
import { decide, applyDecision, DECISION } from './gate.js';
import { loadMemory, saveMemory, newLessonId, readPrompt, ensureSeedPrompt, pruneLessons } from './memory.js';
import { appendScores } from './metrics.js';
import { writeCharts } from './chart.js';
import { createRecorder } from './recorder.js';
import { createTrack } from './tracks/index.js';
import { classify } from './taxonomy.js';
import { summarizeEval, pairedGroupCI } from './stats.js';
import { variantConfig, listVariants } from './variants.js';
import { seededShuffle, writeJson, log, argValue } from './util.js';

export function loadSharedTemplates(cfg) {
  const dir = rootPath(cfg, cfg.paths.templatesDir);
  return { reflectSystem: fs.readFileSync(path.join(dir, cfg.paths.reflectSystemTemplate), 'utf8') };
}

/** Memory view for a given set of ids, respecting the variant's levers. */
function view(mem, cfg, { promptVersion, lessonIds, skillIds }) {
  const lv = cfg.variant.levers;
  return {
    systemPrompt: readPrompt(cfg, promptVersion),
    lessons: lv.lessons ? lessonIds.map((id) => mem.lessons.get(id)).filter(Boolean) : [],
    skills: lv.skills ? skillIds.map((id) => mem.skills.get(id)).filter(Boolean) : [],
  };
}

const classOf = (cfg, result, task, code) => classify(result, task, code, cfg.taxonomy);

async function trainPhase({ llm, cfg, mem, track, templates, runIndex, working, budget, rec }) {
  const reserve = cfg.budget.reserveForHoldout;
  const lv = cfg.variant.levers;
  const stats = { problems: 0, passed: 0, newLessons: [], newSkills: [], levelEvents: [] };
  const doneThisRun = new Set();
  const solved = new Set(lv.skills
    ? [...mem.champion.skillIds, ...mem.champion.pending.skillIds]
    : Object.entries(mem.attempts).filter(([, a]) => a.passed).map(([id]) => id));
  let stop = false;
  while (!stop) {
    const id = track.nextTrainTask({ mem, runIndex, doneThisRun, solved });
    if (!id) break;
    doneThisRun.add(id);
    const task = track.tasks.get(id);
    let passed = false;
    let attempted = false;
    for (let attempt = 1; attempt <= cfg.agent.maxAttemptsPerProblem && !passed; attempt++) {
      if (budget.remainingForTrain(reserve) <= 0) { stop = true; break; }
      const memory = view(mem, cfg, working);
      let res;
      try {
        res = await solve({ llm, problem: task, memory, templates, cfg, temperature: cfg.llm.temperature, reserve, extract: track.extractCode });
      } catch (e) {
        if (e instanceof BudgetExhausted) { stop = true; break; }
        throw e;
      }
      attempted = true;
      const result = await track.judge(res.code, task);
      passed = result.type === RESULT.PASS;
      const failureClass = classOf(cfg, result, task, res.code);
      for (const l of res.usedLessons) { l.uses = (l.uses ?? 0) + 1; if (passed) l.helped = (l.helped ?? 0) + 1; }
      log(`[${cfg.variant.name}] train ${id} attempt ${attempt}: ${result.type} (${failureClass})${res.usedLessons.length ? ` lessons ${res.usedLessons.map((l) => l.id).join(',')}` : ''}`);
      let requests = 1;
      let tokensIn = res.usage.tokensIn;
      let tokensOut = res.usage.tokensOut;
      let reflection = null;
      let lessonId = null;
      if (passed) {
        if (lv.skills && !mem.skills.has(id)) {
          mem.skills.set(id, { id, entryPoint: task.entryPoint, keywords: [...problemKeywords(task, cfg.retrieve)], code: track.storeCode ? res.code : '', createdRun: runIndex });
          working.skillIds.push(id);
          stats.newSkills.push(id);
        }
      } else if (lv.lessons) {
        try {
          const r = await reflect({ llm, problem: task, code: res.code, result, templates, cfg, reserve });
          requests++;
          tokensIn += r.usage.tokensIn;
          tokensOut += r.usage.tokensOut;
          reflection = r.lesson;
        } catch (e) {
          if (!(e instanceof BudgetExhausted)) throw e;
          stop = true;
        }
        const live = new Set(working.lessonIds);
        const dupe = reflection && [...mem.lessons.values()].some((l) => live.has(l.id) && l.text.toLowerCase() === reflection.text.toLowerCase());
        if (reflection && !dupe) {
          const l = { id: newLessonId(mem), text: reflection.text, tags: reflection.tags, sourceProblem: id, sourceResult: result.type, sourceClass: failureClass, sourceAttempt: attempt, createdRun: runIndex, uses: 0, helped: 0 };
          mem.lessons.set(l.id, l);
          working.lessonIds.push(l.id);
          stats.newLessons.push(l.id);
          lessonId = l.id;
          log(`  lesson ${l.id}: ${l.text}`);
        }
      }
      rec.attempt({ phase: 'train', problemId: id, level: task.level ?? null, attempt, result: result.type, failureClass, requests, tokensIn, tokensOut,
        lessonIds: res.usedLessons.map((l) => l.id), skillIds: res.usedSkills.map((s) => s.id), promptVersion: working.promptVersion, durationMs: result.durationMs, lessonCreated: lessonId });
      rec.trace({ problemId: id, attempt, result: result.type, failureClass, code: track.storeCode ? res.code : null, output: result.output,
        lessonIds: res.usedLessons.map((l) => l.id), reflection, lessonCreated: lessonId, duplicateLesson: Boolean(reflection && !lessonId) });
      if (stop) break;
    }
    if (!attempted) continue;
    stats.problems++;
    if (passed) { stats.passed++; solved.add(id); }
    const prev = mem.attempts[id] ?? { tries: 0 };
    mem.attempts[id] = { tries: prev.tries + 1, lastRun: runIndex, passed };
    const ev = track.onTrainResult(id, passed, { runIndex });
    if (ev) { stats.levelEvents.push(ev); log(`[${cfg.variant.name}] curriculum: ${JSON.stringify(ev)}`); }
  }
  return stats;
}

/**
 * Holdout evaluation: `repeats` passes over the same tasks, each repeat with its own sampling seed,
 * eval temperature, no reflection, no memory updates. Returns per-problem arrays of 0/1.
 */
export async function evaluate({ llm, cfg, mem, track, ids, state, templates, repeats, rec, label, runIndex }) {
  const memory = view(mem, cfg, state);
  const perProblem = ids.map(() => []);
  for (let r = 0; r < repeats; r++) {
    const seed = evalSeed(cfg, runIndex, r);
    for (let i = 0; i < ids.length; i++) {
      const task = track.tasks.get(ids[i]);
      const res = await solve({ llm, problem: task, memory, templates, cfg, temperature: cfg.eval.temperature, seed, reserve: 0, extract: track.extractCode });
      const result = await track.judge(res.code, task);
      const pass = result.type === RESULT.PASS ? 1 : 0;
      perProblem[i].push(pass);
      rec.attempt({ phase: 'eval', state: label, repeat: r, problemId: ids[i], level: task.level ?? null, attempt: 1, result: result.type,
        failureClass: classOf(cfg, result, task, res.code), requests: 1, tokensIn: res.usage.tokensIn, tokensOut: res.usage.tokensOut,
        lessonIds: res.usedLessons.map((l) => l.id), skillIds: res.usedSkills.map((s) => s.id), promptVersion: state.promptVersion, durationMs: result.durationMs });
    }
  }
  return perProblem;
}

/**
 * Sampling seed of eval repeat `r` in run `runIndex`. Champion and candidate share the seeds of a run
 * (paired comparison); with eval.seedMode "per-run" every run draws fresh seeds, so an unchanged champion
 * is re-measured independently each day and run-to-run noise is visible.
 */
export function evalSeed(cfg, runIndex, r) {
  const perRun = cfg.eval.seedMode === 'per-run' ? (runIndex - 1) * cfg.eval.repeats : 0;
  return cfg.llm.seed + (perRun + r) * cfg.eval.seedStride;
}

function sameState(a, b) {
  const eq = (x, y) => x.length === y.length && x.every((v, i) => v === y[i]);
  return a.promptVersion === b.promptVersion && eq([...a.lessonIds].sort(), [...b.lessonIds].sort()) && eq([...a.skillIds].sort(), [...b.skillIds].sort());
}

/** Holdout ids and repeats that fit the remaining budget. The subset is seeded by the split seed only, so
 * every variant and every day uses the same tasks for the same size. */
export function planEval(holdout, budgetLeft, evalCount, cfg) {
  const repeats = cfg.eval.repeats;
  const n = Math.min(holdout.length, Math.floor(budgetLeft / (repeats * evalCount)));
  const ids = n >= holdout.length ? holdout : seededShuffle(holdout, cfg.split.seed).slice(0, Math.max(0, n)).sort();
  return { ids, repeats };
}

const statOpts = (cfg, salt) => ({ resamples: cfg.stats.bootstrapResamples, confidence: cfg.stats.confidence, seed: cfg.stats.seed + salt });

/**
 * Run one cycle for one variant.
 * @param {{cfg: object, variant: string, deps?: {llm?, evolvePrompt?, now?, fetch?, track?}}} args
 */
export async function runCycle({ cfg: baseCfg, variant, deps = {} }) {
  const cfg = variantConfig(baseCfg, variant ?? baseCfg.defaultVariant);
  const now = deps.now ?? (() => new Date());
  const runId = (process.env.GITHUB_RUN_ID ? `gh${process.env.GITHUB_RUN_ID}-` : '') + now().toISOString().replace(/[-:]/g, '').slice(0, 'YYYYMMDDTHHMMSS'.length);
  const timestamp = now().toISOString();
  const track = deps.track ?? (await createTrack(cfg, deps));
  const mem = loadMemory(cfg);
  ensureSeedPrompt(cfg, mem.champion, fs.readFileSync(rootPath(cfg, cfg.paths.seedPrompt), 'utf8'));
  const templates = { ...loadSharedTemplates(cfg), ...track.templates };
  const budget = createBudget(cfg, { file: statePath(cfg, 'budget'), now });
  const llm = budgeted(deps.llm ?? createLLM(cfg, { ...deps, oracle: track.oracle }), budget);
  const runIndex = mem.runState.runCount + 1;
  const ch = mem.champion;
  const v = cfg.variant;
  const rec = createRecorder(cfg, { runId, ts: timestamp, variant: v.name, track: track.name });
  const championState = { promptVersion: ch.promptVersion, lessonIds: [...ch.lessonIds], skillIds: [...ch.skillIds] };
  const baselineMode = mem.runState.baselineRunsDone < cfg.gate.baselineRuns;
  log(`[${v.name}] run ${runId} (#${runIndex})${baselineMode ? ' [baseline]' : ''}: ${track.train.length} train / ${track.holdout.length} holdout, budget ${cfg.budget.requestsPerRun}`);

  // 1. Train on top of champion + pending changes.
  const working = {
    promptVersion: ch.pending.promptVersion ?? ch.promptVersion,
    lessonIds: [...ch.lessonIds, ...ch.pending.lessonIds],
    skillIds: [...ch.skillIds, ...ch.pending.skillIds],
  };
  let train = { problems: 0, passed: 0, newLessons: [], newSkills: [], levelEvents: [] };
  let optimizer = null;
  if (!baselineMode) {
    train = await trainPhase({ llm, cfg, mem, track, templates, runIndex, working, budget, rec });
    log(`[${v.name}] train: ${train.passed}/${train.problems} passed, +${train.newLessons.length} lessons, +${train.newSkills.length} skills`);

    // Phase 5 (flag + lever gated): propose a new system prompt from train failures.
    if (cfg.features.promptOptimizer && runIndex % cfg.promptOptimizer.everyNRuns === 0) {
      const evolvePrompt = deps.evolvePrompt ?? (await import('./agent/evolve.js')).evolvePrompt;
      try {
        const version = await evolvePrompt({ cfg, mem, split: { train: track.train }, problemsById: track.tasks, working, budget, llm, templates });
        optimizer = { proposed: version ?? null };
        if (version) { working.promptVersion = version; log(`[${v.name}] prompt optimizer proposed ${version}`); }
      } catch (e) {
        // The optimizer is experimental: a crash there must not lose the rest of the cycle.
        optimizer = { error: String(e?.message ?? e).slice(0, cfg.recorder.maxOutputChars) };
        log(`[${v.name}] prompt optimizer failed: ${optimizer.error}`);
      }
    }
  }

  // 2. Holdout evaluation (repeated) on the same seeded subset for champion and candidate.
  const candidateState = working;
  const changed = !sameState(championState, candidateState);
  const evalCount = baselineMode || !changed ? 1 : 2;
  const plan = planEval(track.holdout, budget.remainingRun(), evalCount, cfg);
  let decision;
  let champ = null;
  let cand = null;
  let diff = null;
  if (!plan.ids.length) {
    decision = baselineMode ? DECISION.BASELINE : changed ? DECISION.NO_EVAL : DECISION.NO_CHANGE;
  } else {
    try {
      const champRaw = await evaluate({ llm, cfg, mem, track, ids: plan.ids, state: championState, templates, repeats: plan.repeats, rec, label: 'champion', runIndex });
      champ = summarizeEval(champRaw, statOpts(cfg, runIndex), cfg.stats.method);
      rec.evalBatch({ state: 'champion', promptVersion: championState.promptVersion, ids: plan.ids, repeats: plan.repeats, results: champRaw, lessonsCount: championState.lessonIds.length, skillsCount: championState.skillIds.length });
      log(`[${v.name}] holdout champion: ${(champ.passRate * 100).toFixed(1)}% [${(champ.ciLow * 100).toFixed(1)}, ${(champ.ciHigh * 100).toFixed(1)}] (n=${champ.n} x${champ.repeats})`);
      if (baselineMode) decision = DECISION.BASELINE;
      else if (!changed) decision = DECISION.NO_CHANGE;
      else {
        const candRaw = await evaluate({ llm, cfg, mem, track, ids: plan.ids, state: candidateState, templates, repeats: plan.repeats, rec, label: 'candidate', runIndex });
        cand = summarizeEval(candRaw, statOpts(cfg, runIndex), cfg.stats.method);
        rec.evalBatch({ state: 'candidate', promptVersion: candidateState.promptVersion, ids: plan.ids, repeats: plan.repeats, results: candRaw, lessonsCount: candidateState.lessonIds.length, skillsCount: candidateState.skillIds.length });
        const d = pairedGroupCI(champRaw, candRaw, statOpts(cfg, runIndex), cfg.stats.method);
        diff = { estimate: d.estimate, low: d.low, high: d.high };
        log(`[${v.name}] holdout candidate: ${(cand.passRate * 100).toFixed(1)}% (diff ${(diff.estimate * 100).toFixed(1)} [${(diff.low * 100).toFixed(1)}, ${(diff.high * 100).toFixed(1)}])`);
        decision = decide(champ.passRate, cand.passRate, cfg.gate, diff);
      }
    } catch (e) {
      if (!(e instanceof BudgetExhausted)) throw e;
      log(`[${v.name}] holdout evaluation stopped early (${e.reason}); scores discarded`);
      champ = null;
      cand = null;
      decision = baselineMode ? DECISION.BASELINE : changed ? DECISION.NO_EVAL : DECISION.NO_CHANGE;
    }
  }
  log(`[${v.name}] gate: ${decision}`);

  // 3. Gate + prune.
  const added = {
    lessonIds: candidateState.lessonIds.filter((id) => !championState.lessonIds.includes(id)),
    skillIds: candidateState.skillIds.filter((id) => !championState.skillIds.includes(id)),
    promptVersion: candidateState.promptVersion !== championState.promptVersion ? candidateState.promptVersion : null,
  };
  const pick = (s) => s && { passRate: s.passRate, ciLow: s.ciLow, ciHigh: s.ciHigh, n: s.n, repeats: s.repeats, sdAcrossRepeats: s.sdAcrossRepeats };
  if (baselineMode) {
    if (champ) {
      mem.runState.baselineRunsDone++; // only completed measurements count as baseline runs
      ch.holdout = { ...pick(champ), runId };
    }
  } else if (decision !== DECISION.NO_CHANGE) {
    const holdout = cand ? { ...pick(cand), runId } : ch.holdout;
    applyDecision(ch, candidateState, decision, { runIndex, holdout, maxPendingRuns: cfg.gate.maxPendingRuns });
  }
  rec.gate({ runIndex, decision, method: cfg.gate.method, baselineMode, champion: pick(champ), candidate: pick(cand), diff,
    added: { lessons: added.lessonIds.length, skills: added.skillIds.length, promptVersion: added.promptVersion, lessonIds: added.lessonIds },
    thresholds: { minImprovement: cfg.gate.minImprovement, tolerance: cfg.gate.tolerance }, optimizer, levelEvents: train.levelEvents });
  const pruned = pruneLessons(mem, cfg.agent.maxLessons);
  mem.runState.runCount = runIndex;

  // 4. Persist state, raw logs, scores.
  saveMemory(cfg, mem);
  budget.save();
  rec.flush();
  const base = { timestamp, runId, variant: v.name, track: track.name, lessonsCount: ch.lessonIds.length, skillsCount: ch.skillIds.length, requestsUsed: budget.usedRun, tokensIn: llm.tokensIn, tokensOut: llm.tokensOut, gateDecision: decision };
  const row = (split, s, promptVersion) => ({ ...base, split, passRate: s.passRate, n: s.n, repeats: s.repeats ?? 1, sd: s.sdAcrossRepeats ?? '', ciLow: s.ciLow ?? '', ciHigh: s.ciHigh ?? '', promptVersion });
  const rows = [];
  if (train.problems) rows.push(row('train', { passRate: train.passed / train.problems, n: train.problems }, candidateState.promptVersion));
  if (champ) rows.push(row('holdout-champion', champ, championState.promptVersion));
  if (cand) rows.push(row('holdout-candidate', cand, candidateState.promptVersion));
  if (rows.length) appendScores(statePath(cfg, 'scores'), rows, cfg.output.csvDecimals);
  const summary = { runId, runIndex, timestamp, variant: v.name, decision, baselineMode, train, champion: pick(champ), candidate: pick(cand), diff, pruned, requestsUsed: budget.usedRun, requestsToday: budget.usedToday, tokensIn: llm.tokensIn, tokensOut: llm.tokensOut };
  writeJson(statePath(cfg, 'lastRun'), summary);
  return summary;
}

export function parseArgs(argv) {
  return { dryRun: argv.includes('--dry-run'), variant: argValue(argv, 'variant') };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const cfg = loadConfig({ dryRun: args.dryRun });
  const names = args.variant ? [args.variant] : listVariants(cfg);
  for (const variant of names) {
    const s = await runCycle({ cfg, variant });
    log(`[${variant}] done: ${s.decision}, ${s.requestsUsed} requests, ${s.tokensIn}+${s.tokensOut} tokens`);
  }
  writeCharts(cfg);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((e) => {
    console.error(e?.stack ?? e);
    process.exit(1);
  });
}
