// MultiPL-E HumanEval + MBPP in JavaScript: "complete this function", judged by the problem's unit tests.
import fs from 'node:fs';
import path from 'node:path';
import { rootPath, statePath } from '../config.js';
import { loadProblems } from '../dataset/fetch.js';
import { loadOrCreateSplit } from '../dataset/split.js';
import { extractCode } from '../agent/solve.js';
import { runTests } from '../judge/run-tests.js';
import { seededShuffle } from '../util.js';

function commentText(prompt) {
  return prompt.split('\n').filter((l) => l.trim().startsWith('//')).join(' ');
}

export async function createMultiplETrack(cfg, deps = {}) {
  const tc = cfg.tracks['multipl-e'];
  const tplDir = rootPath(cfg, cfg.paths.templatesDir);
  const templates = {
    solveUser: fs.readFileSync(path.join(tplDir, tc.solveUserTemplate), 'utf8'),
    reflectUser: fs.readFileSync(path.join(tplDir, tc.reflectUserTemplate), 'utf8'),
  };
  const problems = await loadProblems(cfg, deps);
  const tasks = new Map(problems.map((p) => [p.id, {
    ...p,
    vars: { prompt: p.prompt.trimEnd() },
    keywordText: `${commentText(p.prompt)} ${p.entryPoint ?? ''}`,
  }]));
  const split = loadOrCreateSplit(statePath(cfg, 'split'), [...tasks.keys()], cfg.split);
  const mockById = new Map(problems.filter((p) => p.mock).map((p) => [p.id, p.mock]));
  let order = null;

  return {
    name: 'multipl-e',
    templates,
    storeCode: tc.storeCode,
    tasks,
    train: split.train,
    holdout: split.holdout,
    oracle: (id) => mockById.get(id),
    /** Never-attempted problems first (seeded shuffle per run), then failures due for a retry, fewest tries first. */
    nextTrainTask({ mem, runIndex, doneThisRun, solved }) {
      if (!order) {
        const fresh = [];
        const retry = [];
        for (const id of split.train) {
          if (solved.has(id)) continue;
          const a = mem.attempts[id];
          if (!a) fresh.push(id);
          else if (runIndex - a.lastRun >= cfg.agent.retryFailedAfterRuns) retry.push(id);
        }
        const seed = cfg.split.seed + runIndex;
        order = [...seededShuffle(fresh, seed), ...seededShuffle(retry, seed).sort((x, y) => mem.attempts[x].tries - mem.attempts[y].tries)];
      }
      return order.find((id) => !doneThisRun.has(id)) ?? null;
    },
    onTrainResult() { return null; },
    extractCode: (reply, task) => extractCode(reply, task),
    judge: (code, task) => runTests(code, task.tests, cfg.judge),
  };
}
