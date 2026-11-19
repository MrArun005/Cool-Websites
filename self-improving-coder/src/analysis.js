// Everything the report and the dashboard say is computed here, from the raw state files only.
// Deterministic: no clocks, no randomness except seeded bootstrap; stable sort orders everywhere.
import fs from 'node:fs';
import path from 'node:path';
import { variantStateDir } from './variants.js';
import { readJsonl } from './recorder.js';
import { readScores } from './metrics.js';
import { readJson } from './util.js';
import { mean, pairedGroupCI, groupCI, didCI } from './stats.js';

const byStr = (f) => (a, b) => String(f(a)).localeCompare(String(f(b)));

export function loadVariantState(cfg, v) {
  const dir = path.resolve(cfg.root, variantStateDir(cfg, v.name));
  const p = (k) => path.join(dir, cfg.paths[k]);
  const exists = fs.existsSync(dir);
  return {
    ...v,
    dir,
    exists,
    champion: exists ? readJson(p('champion'), null) : null,
    lessons: exists ? readJson(p('lessons'), { items: [] }).items : [],
    attempts: exists ? readJsonl(p('attemptsLog')) : [],
    traces: exists ? readJsonl(p('traces')) : [],
    evals: exists ? readJsonl(p('evals')) : [],
    gateLog: exists ? readJsonl(p('gateLog')) : [],
    scores: exists ? readScores(p('scores')) : [],
    curriculum: exists ? readJson(path.join(dir, cfg.curriculum.stateFile), null) : null,
  };
}

/** Raw results of an eval batch per problem: Map(problemId -> [0/1 per repeat]). */
const problemGroups = (batch) => new Map(batch.ids.map((id, i) => [id, batch.results[i]]));

/** Align several per-problem maps on their common problems. */
function align(...maps) {
  const ids = [...maps[0].keys()].filter((id) => maps.every((m) => m.has(id))).sort();
  return { ids, groups: maps.map((m) => ids.map((id) => m.get(id))) };
}
const pairedCI = (cfg, salt, a, b) => {
  const { ids, groups } = align(a, b);
  const d = pairedGroupCI(groups[0], groups[1], statOpts(cfg, salt), cfg.stats.method);
  return { ids, a: groups[0], b: groups[1], d };
};

function statOpts(cfg, salt) {
  return { resamples: cfg.stats.bootstrapResamples, confidence: cfg.stats.confidence, seed: cfg.stats.seed + salt };
}

/** Champion eval batch that represents each run (the accepted candidate's batch if the gate accepted). */
function runBatches(vs) {
  const out = [];
  for (const g of [...vs.gateLog].sort((x, y) => x.runIndex - y.runIndex)) {
    const champ = vs.evals.find((e) => e.runId === g.runId && e.state === 'champion');
    if (!champ) continue;
    const cand = vs.evals.find((e) => e.runId === g.runId && e.state === 'candidate');
    out.push({ runId: g.runId, ts: g.ts, runIndex: g.runIndex, decision: g.decision, batch: g.decision === 'accept' && cand ? cand : champ, champBatch: champ, candBatch: cand ?? null });
  }
  return out;
}

function summarizeBatch(batch, cfg, salt) {
  const ci = groupCI(batch.results, statOpts(cfg, salt), cfg.stats.method);
  return { passRate: mean(batch.results.map(mean)), ciLow: ci.low, ciHigh: ci.high, n: batch.ids.length, repeats: batch.repeats };
}

/** The starting point: all baseline-run results per problem, repeats of every baseline run pooled. */
function baselineGroups(runs) {
  const base = runs.filter((r) => r.decision === 'baseline');
  if (!base.length) return null;
  const acc = new Map();
  for (const r of base) for (const [id, xs] of problemGroups(r.batch)) acc.set(id, [...(acc.get(id) ?? []), ...xs]);
  return acc;
}

function costOf(vs) {
  const train = vs.attempts.filter((a) => a.phase === 'train');
  const evals = vs.attempts.filter((a) => a.phase === 'eval');
  const sum = (rows, k) => rows.reduce((s, r) => s + (r[k] ?? 0), 0);
  const solved = new Set(train.filter((a) => a.result === 'pass').map((a) => a.problemId));
  const trainRequests = sum(train, 'requests');
  return {
    trainRequests,
    evalRequests: sum(evals, 'requests'),
    tokensIn: sum(vs.attempts, 'tokensIn'),
    tokensOut: sum(vs.attempts, 'tokensOut'),
    trainSolved: solved.size,
    trainProblemsTried: new Set(train.map((a) => a.problemId)).size,
    solvedPer100Requests: trainRequests ? (100 * solved.size) / trainRequests : null,
    requestsPerSolved: solved.size ? trainRequests / solved.size : null,
    tokensPerSolved: solved.size ? (sum(train, 'tokensIn') + sum(train, 'tokensOut')) / solved.size : null,
    // wall-clock model time per request, as measured on the runner (MS -> s done by the caller)
    msPerRequest: (() => { const reqs = sum(vs.attempts, 'requests'); const ms = sum(vs.attempts, 'llmMs'); return reqs && ms ? ms / reqs : null; })(),
  };
}

/** Failure-class shares of champion holdout attempts, per run. */
function taxonomyByRun(vs, classes) {
  const runs = [...new Set(vs.attempts.filter((a) => a.phase === 'eval' && a.state === 'champion').map((a) => a.runId))].sort();
  return runs.map((runId) => {
    const rows = vs.attempts.filter((a) => a.runId === runId && a.phase === 'eval' && a.state === 'champion');
    const counts = Object.fromEntries(classes.map((c) => [c, rows.filter((r) => r.failureClass === c).length]));
    return { runId, ts: rows[0]?.ts, total: rows.length, counts, shares: Object.fromEntries(classes.map((c) => [c, rows.length ? counts[c] / rows.length : 0])) };
  });
}

/** Lesson genealogy from the logs: birth (trace), every retrieval (attempt rows), current status. */
function genealogy(vs) {
  const live = new Set([...(vs.champion?.lessonIds ?? [])]);
  const pending = new Set([...(vs.champion?.pending?.lessonIds ?? [])]);
  const births = vs.traces.filter((t) => t.lessonCreated);
  return births.map((b) => {
    const uses = vs.attempts.filter((a) => (a.lessonIds ?? []).includes(b.lessonCreated));
    const trainUses = uses.filter((u) => u.phase === 'train');
    const stored = vs.lessons.find((l) => l.id === b.lessonCreated);
    return {
      id: b.lessonCreated,
      text: b.reflection?.text ?? stored?.text ?? '',
      tags: b.reflection?.tags ?? stored?.tags ?? [],
      born: { runId: b.runId, ts: b.ts, problemId: b.problemId, attempt: b.attempt, result: b.result, failureClass: b.failureClass },
      status: live.has(b.lessonCreated) ? 'accepted' : pending.has(b.lessonCreated) ? 'pending' : 'dropped',
      trainRetrievals: trainUses.length,
      trainPasses: trainUses.filter((u) => u.result === 'pass').length,
      evalRetrievals: uses.length - trainUses.length,
      evalPasses: uses.filter((u) => u.phase === 'eval' && u.result === 'pass').length,
      usedOn: [...new Set(trainUses.map((u) => u.problemId))].sort(),
    };
  }).sort(byStr((l) => l.id));
}

/** Pairs of lessons retrieved together on failed train attempts more often than `minCount`. */
function conflicts(vs, minCount) {
  const pairs = new Map();
  const solo = new Map();
  for (const a of vs.attempts.filter((x) => x.phase === 'train')) {
    const ids = [...(a.lessonIds ?? [])].sort();
    for (const id of ids) {
      const s = solo.get(id) ?? { n: 0, pass: 0 };
      s.n++; if (a.result === 'pass') s.pass++;
      solo.set(id, s);
    }
    for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) {
      const k = `${ids[i]}+${ids[j]}`;
      const p = pairs.get(k) ?? { a: ids[i], b: ids[j], together: 0, failedTogether: 0 };
      p.together++; if (a.result !== 'pass') p.failedTogether++;
      pairs.set(k, p);
    }
  }
  return [...pairs.values()]
    .filter((p) => p.failedTogether >= minCount)
    .map((p) => ({ ...p, passRateTogether: 1 - p.failedTogether / p.together, passRateA: solo.get(p.a).pass / solo.get(p.a).n, passRateB: solo.get(p.b).pass / solo.get(p.b).n }))
    .filter((p) => p.passRateTogether < Math.min(p.passRateA, p.passRateB) || p.passRateTogether === 0)
    .sort((x, y) => y.failedTogether - x.failedTogether || byStr((p) => `${p.a}+${p.b}`)(x, y));
}

/** Replays: every train attempt that produced a lesson, with the next attempt (same problem, same run). */
function replays(vs, limit) {
  const out = [];
  for (const t of vs.traces.filter((x) => x.lessonCreated)) {
    const next = vs.traces.find((x) => x.runId === t.runId && x.problemId === t.problemId && x.attempt === t.attempt + 1);
    out.push({ id: `${vs.name}:${t.runId}:${t.problemId}:${t.attempt}`, variant: vs.name, track: vs.track, runId: t.runId, ts: t.ts, problemId: t.problemId,
      first: { attempt: t.attempt, result: t.result, failureClass: t.failureClass, code: t.code, output: t.output, lessonsUsed: t.lessonIds },
      reflection: t.reflection, lessonId: t.lessonCreated,
      next: next ? { attempt: next.attempt, result: next.result, failureClass: next.failureClass, code: next.code, output: next.output, lessonsUsed: next.lessonIds } : null });
  }
  // the demo moment first: fixed on the next attempt, then everything else; newest first within each
  out.sort((a, b) => Number(b.next?.result === 'pass') - Number(a.next?.result === 'pass') || byStr((r) => r.id)(b, a));
  return out.slice(0, limit);
}

function plateau(runs, cfg, salt) {
  const K = cfg.analysis.plateauRuns;
  const learning = runs.filter((r) => r.decision !== 'baseline');
  if (learning.length < K + 1) return { status: 'too-early', runsNeeded: K + 1, runsSoFar: learning.length };
  const recent = learning.slice(-K);
  const anchor = learning[learning.length - K - 1];
  const { ids, d } = pairedCI(cfg, salt, problemGroups(anchor.batch), problemGroups(recent.at(-1).batch));
  const p = { ids };
  const accepted = recent.filter((r) => r.decision === 'accept').length;
  return { status: d.low > 0 ? 'improving' : 'plateaued', K, accepted, diff: { estimate: d.estimate, low: d.low, high: d.high, n: p.ids.length }, since: anchor.ts };
}

/**
 * Floor/ceiling check: pooled starting (baseline-run) pass rate of every variant on the track. Near 0% there is
 * nothing to learn from passes; near 100% nothing left to improve; either way the comparison can't show much.
 */
export function difficultyCheck(vs, track, cfg) {
  const starts = vs.map((v) => v.runs.filter((r) => r.decision === 'baseline').map((r) => r.passRate)).flat();
  if (!starts.length) return { status: 'no-data' };
  const start = mean(starts);
  const { floorRate, ceilingRate, difficultyKnobs } = cfg.analysis;
  const knobs = difficultyKnobs[track] ?? {};
  if (start < floorRate) return { status: 'floor', start, limit: floorRate, hint: knobs.tooHard ?? null };
  if (start > ceilingRate) return { status: 'ceiling', start, limit: ceilingRate, hint: knobs.tooEasy ?? null };
  return { status: 'ok', start, floorRate, ceilingRate };
}

export function analyze(cfg) {
  const classes = cfg.taxonomy.classes;
  const variants = cfg.variants.map((v) => loadVariantState(cfg, v));
  const tracks = [...new Set(variants.map((v) => v.track))];
  const result = { tracks: {}, variants: {}, generatedFrom: { runs: 0, lastTs: null } };
  let salt = 0;
  for (const vs of variants) {
    salt++;
    const runs = runBatches(vs);
    const latest = runs.at(-1) ?? null;
    const base = baselineGroups(runs);
    let vsBaseline = null;
    if (latest && base) {
      const p = pairedCI(cfg, salt, base, problemGroups(latest.batch));
      vsBaseline = { baseline: mean(p.a.map(mean)), latest: mean(p.b.map(mean)), estimate: p.d.estimate, low: p.d.low, high: p.d.high, n: p.ids.length, significant: p.d.low > 0 || p.d.high < 0 };
    }
    result.variants[vs.name] = {
      name: vs.name, label: vs.label, track: vs.track, color: vs.color, levers: vs.levers, control: Boolean(vs.control),
      runs: runs.map((r) => ({ runId: r.runId, ts: r.ts, runIndex: r.runIndex, decision: r.decision, ...summarizeBatch(r.batch, cfg, salt) })),
      latest: latest ? { runId: latest.runId, ts: latest.ts, ...summarizeBatch(latest.batch, cfg, salt) } : null,
      vsBaseline,
      gateLog: [...vs.gateLog].sort((a, b) => a.runIndex - b.runIndex),
      cost: costOf(vs),
      taxonomy: taxonomyByRun(vs, classes),
      lessons: genealogy(vs),
      conflicts: conflicts(vs, cfg.analysis.minConflictCount),
      replays: replays(vs, cfg.analysis.maxReplaysPerVariant),
      plateau: plateau(runs, cfg, salt),
      curriculum: vs.curriculum ? { level: vs.curriculum.level, history: vs.curriculum.history } : null,
      liveLessons: vs.champion?.lessonIds?.length ?? 0,
      liveSkills: vs.champion?.skillIds?.length ?? 0,
      promptVersion: vs.champion?.promptVersion ?? null,
      _runs: runs,
      _base: base,
    };
    result.generatedFrom.runs += runs.length;
    for (const r of runs) if (!result.generatedFrom.lastTs || r.ts > result.generatedFrom.lastTs) result.generatedFrom.lastTs = r.ts;
  }
  // per track: each variant vs the control variant on the same day's problems
  for (const track of tracks) {
    const vs = Object.values(result.variants).filter((v) => v.track === track);
    const control = vs.find((v) => v.control) ?? null;
    const comparisons = [];
    for (const v of vs) {
      if (!control || v === control || !v._runs.length || !control._runs.length) continue;
      salt++;
      const p = pairedCI(cfg, salt, problemGroups(control._runs.at(-1).batch), problemGroups(v._runs.at(-1).batch));
      if (!p.ids.length) continue;
      const d = p.d;
      const cmp = { variant: v.name, control: control.name, controlRate: mean(p.a.map(mean)), variantRate: mean(p.b.map(mean)), estimate: d.estimate, low: d.low, high: d.high, n: p.ids.length, significant: d.low > 0 || d.high < 0 };
      // change since its own start minus the control's change over the same period (difference-in-differences)
      if (v._base && control._base) {
        const { ids, groups } = align(v._base, problemGroups(v._runs.at(-1).batch), control._base, problemGroups(control._runs.at(-1).batch));
        if (ids.length) {
          const [vStart, vLatest, cStart, cLatest] = groups;
          const dd = didCI(vStart, vLatest, cStart, cLatest, statOpts(cfg, salt), cfg.stats.method);
          v.vsBaselineAdjusted = { estimate: dd.estimate, low: dd.low, high: dd.high, n: ids.length, significant: dd.low > 0 || dd.high < 0 };
        }
      }
      comparisons.push(cmp);
    }
    // failure types: share at the start (first run) vs latest, learning variants vs control
    const taxonomyChange = classes.map((c) => {
      const delta = (v) => (v.taxonomy.length >= 2 ? v.taxonomy.at(-1).shares[c] - v.taxonomy[0].shares[c] : null);
      const learners = vs.filter((v) => !v.control).map((v) => ({ variant: v.name, delta: delta(v) })).filter((x) => x.delta !== null);
      const controlDelta = control ? delta(control) : null;
      const drop = cfg.analysis.taxonomyMinDrop;
      const fixedBy = learners.filter((l) => l.delta <= -drop && (controlDelta === null || l.delta < controlDelta - drop / 2)).map((l) => l.variant);
      const verdict = !learners.length ? 'no-data' : fixedBy.length ? 'reduced' : 'not-reduced';
      return { class: c, controlDelta, learners, fixedBy, verdict };
    });
    result.tracks[track] = { control: control?.name ?? null, variants: vs.map((v) => v.name), comparisons, taxonomyChange, difficulty: difficultyCheck(vs, track, cfg) };
  }
  for (const v of Object.values(result.variants)) { delete v._runs; delete v._base; }
  return result;
}
