// Phase 5 (behind features.promptOptimizer): propose a new system prompt with GEPA via @ax-llm/ax.
// The proposal is NOT trusted: run.js puts it into the candidate state and it must pass the same
// holdout gate as lessons and skills. Optimizer examples come only from the TRAIN split.
import { BudgetExhausted } from '../budget.js';
import { extractCode } from './solve.js';
import { runTests, RESULT } from '../judge/run-tests.js';
import { readPrompt, writePrompt, nextPromptVersion } from '../memory.js';
import { seededShuffle, log } from '../util.js';

/** Wrap an ax AI instance so every chat call is budget-checked and counted, capped at `maxRequests`. */
export function budgetProxy(axAI, budget, reserve, maxRequests) {
  let used = 0;
  return new Proxy(axAI, {
    get(target, prop, recv) {
      const v = Reflect.get(target, prop, recv);
      if (prop !== 'chat' || typeof v !== 'function') return typeof v === 'function' ? v.bind(target) : v;
      return async (...args) => {
        if (used >= maxRequests) throw new BudgetExhausted('optimizer-cap');
        budget.check(reserve);
        budget.consume();
        used++;
        return v.apply(target, args);
      };
    },
  });
}

export function pickInstruction(result) {
  const p = result?.optimizedProgram;
  if (!p) return null;
  const map = p.componentMap ?? {};
  const key = Object.keys(map).find((k) => k.endsWith('::instruction'));
  const v = key ? map[key] : p.instruction;
  return typeof v === 'string' && v.trim() ? v.trim() : null;
}

export async function evolvePrompt({ cfg, mem, split, problemsById, working, budget, deps = {} }) {
  const oc = cfg.promptOptimizer;
  let axLib = deps.ax;
  if (!axLib) {
    try { axLib = await import('@ax-llm/ax'); } catch { log('prompt optimizer: @ax-llm/ax not installed, skipping'); return null; }
  }
  const { ai, ax, optimize } = axLib;
  // Prefer problems the agent recently failed, then anything else from train.
  const failed = split.train.filter((id) => mem.attempts[id] && !mem.attempts[id].passed);
  const rest = split.train.filter((id) => !failed.includes(id));
  const pool = [...seededShuffle(failed, cfg.split.seed), ...seededShuffle(rest, cfg.split.seed)];
  const need = oc.trainExamples + oc.validationExamples;
  if (pool.length < need) return null;
  const toEx = (id) => ({ problemPrompt: problemsById.get(id).prompt, problemId: id });
  const train = pool.slice(0, oc.trainExamples).map(toEx);
  const validation = pool.slice(oc.trainExamples, need).map(toEx);

  const apiURL = cfg.llm.endpoint.replace(/\/chat\/completions\/?$/, '');
  const base = ai({ name: 'openai-compatible', apiKey: process.env[cfg.llm.apiKeyEnv] || 'unused', apiURL, config: { model: cfg.llm.model, temperature: cfg.llm.temperature, maxTokens: cfg.llm.maxTokens } });
  const student = budgetProxy(base, budget, cfg.budget.reserveForHoldout, oc.maxRequests);
  const program = ax('problemPrompt:string "start of a JavaScript function to complete" -> code:string "the complete JavaScript function"');
  const current = readPrompt(cfg, working.promptVersion);
  program.setInstruction(current.trim());

  const metric = async ({ prediction, example }) => {
    const problem = problemsById.get(example.problemId);
    const code = extractCode(String(prediction?.code ?? ''), problem);
    const r = await runTests(code, problem.tests, cfg.judge);
    return r.type === RESULT.PASS ? 1 : 0;
  };

  const result = await optimize(program, train, metric, {
    studentAI: student, teacherAI: student,
    numTrials: oc.numTrials, minibatch: true, minibatchSize: oc.minibatchSize,
    earlyStoppingTrials: oc.earlyStoppingTrials, sampleCount: 1,
    validationExamples: validation, maxMetricCalls: oc.maxMetricCalls,
  });
  const instruction = pickInstruction(result);
  if (!instruction || instruction === current.trim()) return null;
  const version = nextPromptVersion(cfg);
  writePrompt(cfg, version, instruction);
  log(`prompt optimizer: baseline ${result.baselineScore} -> best ${result.bestScore} on train-validation`);
  return version;
}
