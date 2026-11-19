// Mutation-repair track: "these tests of a real library fail, fix the function(s)". Tasks are killed
// mutants of a pinned third-party repo (src/mutate/generate.js); the judge is that repo's own test suite
// in the sandbox. Holdout = mutants from a seed the agent never trains on. A difficulty curriculum
// (levels 1-3) moves up when the rolling train pass rate at the current level crosses a threshold.
import fs from 'node:fs';
import path from 'node:path';
import { rootPath, statePath } from '../config.js';
import { loadCurriculum } from '../mutate/generate.js';
import { runSuite, curriculumCfg } from '../mutate/repo.js';
import { listFunctions, tryParse } from '../mutate/ast.js';
import { RESULT } from '../judge/run-tests.js';
import { readJson, writeJson, seededShuffle } from '../util.js';

/** All ```js blocks of a reply joined, or the reply itself when there are none. */
export function codeFromReply(reply) {
  const fences = [...String(reply).matchAll(/```[\w-]*[ \t]*\n([\s\S]*?)(?:```|$)/g)].map((m) => m[1]);
  return (fences.length ? fences.join('\n\n') : String(reply)).trim() + '\n';
}

/** Find named function declarations in the agent's code. Returns { name: source } or null if unparsable. */
export function functionsInCode(code, ecmaVersion) {
  if (!tryParse(code, ecmaVersion)) return null;
  const out = {};
  for (const f of listFunctions(code, ecmaVersion)) if (!(f.name in out)) out[f.name] = code.slice(f.start, f.end);
  return out;
}

/** Splice the agent's functions into the mutated files (replacing the mutated versions). */
export function patchedFiles(task, fixes) {
  const files = { ...task.mutatedFiles };
  for (const file of new Set(Object.values(task.located).map((l) => l.file))) if (!(file in files)) files[file] = null;
  for (const file of Object.keys(files)) {
    const locs = Object.values(task.located).filter((l) => l.file === file).sort((a, b) => b.start - a.start);
    let src = files[file];
    if (src === null) continue;
    for (const l of locs) if (fixes[l.name]) src = src.slice(0, l.start) + fixes[l.name] + src.slice(l.end);
    files[file] = src;
  }
  return Object.fromEntries(Object.entries(files).filter(([, v]) => v !== null));
}

export async function createMutationTrack(cfg, deps = {}) {
  const c = curriculumCfg(cfg);
  const tc = cfg.tracks.mutation;
  const tplDir = rootPath(cfg, cfg.paths.templatesDir);
  const templates = {
    solveUser: fs.readFileSync(path.join(tplDir, tc.solveUserTemplate), 'utf8'),
    reflectUser: fs.readFileSync(path.join(tplDir, tc.reflectUserTemplate), 'utf8'),
  };
  const cur = deps.curriculum ?? (await loadCurriculum(cfg));
  const { env } = cur;
  const tasks = new Map();
  for (const t of cur.tasks) {
    const locs = Object.values(t.located);
    const names = [...new Set(locs.map((l) => l.name))];
    const files = [...new Set(locs.map((l) => l.file))];
    tasks.set(t.id, {
      ...t,
      entryPoint: names[0],
      entryPoints: names,
      signatureSource: locs.map((l) => l.source).join('\n\n'),
      vars: { repo: c.name, files: files.join('`, `'), bugCount: String(t.operators.length), failures: t.failures, functions: locs.map((l) => l.source).join('\n\n') },
      // retrieval sees what a developer sees: function names, file names, failing test titles - never the operator
      keywordText: `${names.join(' ')} ${files.join(' ')} ${t.failures.split('\n').filter((l) => /^(# |not ok)/.test(l)).join(' ')}`,
      mock: { family: t.operators[0], correct: locs.map((l) => l.original).join('\n\n'), buggy: locs.map((l) => l.source).join('\n\n') },
    });
  }
  // manifest of seeds + operators only (no third-party code) for transparency and reproduction
  writeJson(path.join(cfg.paths.stateDir, c.manifestFile), cur.manifest);
  const train = cur.tasks.filter((t) => t.split === 'train').map((t) => t.id);
  const holdout = cur.tasks.filter((t) => t.split === 'holdout').map((t) => t.id);
  const levels = Object.keys(c.levels).map(Number).sort((a, b) => a - b);
  const stateFile = path.join(cfg.paths.stateDir, c.stateFile);
  const curState = readJson(stateFile, { level: c.startLevel, recent: [], history: [] });
  const order = new Map();

  return {
    name: 'mutation',
    templates,
    storeCode: tc.storeCode,
    tasks,
    train,
    holdout,
    get level() { return curState.level; },
    oracle: (id) => tasks.get(id)?.mock,
    /** Unsolved tasks at the current level (seeded order per run); falls back to lower levels when exhausted. */
    nextTrainTask({ mem, runIndex, doneThisRun, solved }) {
      for (const level of [curState.level, ...levels.filter((l) => l < curState.level).reverse()]) {
        if (!order.has(level)) {
          const ids = train.filter((id) => tasks.get(id).level === level);
          const fresh = ids.filter((id) => !mem.attempts[id]);
          const retry = ids.filter((id) => mem.attempts[id] && !solved.has(id) && runIndex - mem.attempts[id].lastRun >= cfg.agent.retryFailedAfterRuns);
          order.set(level, [...seededShuffle(fresh, cfg.split.seed + runIndex), ...seededShuffle(retry, cfg.split.seed + runIndex)]);
        }
        const next = order.get(level).find((id) => !doneThisRun.has(id) && !solved.has(id));
        if (next) return next;
      }
      return null;
    },
    /** Rolling pass rate at the current level; promote when it crosses the threshold. */
    onTrainResult(id, passed, { runIndex }) {
      const t = tasks.get(id);
      let event = null;
      if (t.level === curState.level) {
        curState.recent.push(passed ? 1 : 0);
        if (curState.recent.length > c.promotion.window) curState.recent.shift();
        const rate = curState.recent.reduce((a, b) => a + b, 0) / curState.recent.length;
        const maxLevel = Math.max(...levels);
        if (curState.recent.length >= c.promotion.window && rate >= c.promotion.threshold && curState.level < maxLevel) {
          event = { type: 'level-up', from: curState.level, to: curState.level + 1, rollingPassRate: rate, window: c.promotion.window, runIndex, afterTask: id };
          curState.history.push(event);
          curState.level += 1;
          curState.recent = [];
        }
      }
      writeJson(stateFile, curState);
      return event;
    },
    extractCode: (reply) => codeFromReply(reply),
    async judge(code, task) {
      const fixes = functionsInCode(code, c.ecmaVersion);
      if (!fixes) return { type: RESULT.ERROR, output: 'SyntaxError: the reply could not be parsed as JavaScript', durationMs: 0 };
      return runSuite(cfg, env, patchedFiles(task, fixes));
    },
  };
}

export { statePath };
