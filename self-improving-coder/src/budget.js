// Request counter. Checked BEFORE every LLM call. Persists the per-day count in state/budget.json
// so manual re-runs on the same day share the daily allowance.
import { readJson, writeJson, today, MS_PER_MINUTE } from './util.js';

export class BudgetExhausted extends Error {
  constructor(reason) { super(`budget exhausted: ${reason}`); this.reason = reason; }
}

export function createBudget(cfg, { file, now = () => new Date() } = {}) {
  const { requestsPerDay, requestsPerRun, maxRunMinutes, reserveMinutesForHoldout } = cfg.budget;
  const reserveMs = reserveMinutesForHoldout * MS_PER_MINUTE;
  const day = today(now());
  const stored = file ? readJson(file, { date: day, used: 0 }) : { date: day, used: 0 };
  let usedToday = stored.date === day ? stored.used : 0;
  let usedRun = 0;
  const startedAt = now().getTime();

  const self = {
    get usedRun() { return usedRun; },
    get usedToday() { return usedToday; },
    remainingRun() { return Math.max(0, Math.min(requestsPerRun - usedRun, requestsPerDay - usedToday)); },
    /** Requests available for training, keeping `reserve` back for holdout eval. */
    remainingForTrain(reserve = cfg.budget.reserveForHoldout) {
      if (self.timeLeft() <= reserveMs) return 0;
      return Math.max(0, self.remainingRun() - reserve);
    },
    timeLeft() { return maxRunMinutes * MS_PER_MINUTE - (now().getTime() - startedAt); },
    check(reserve = 0) {
      if (self.timeLeft() <= 0) throw new BudgetExhausted('time');
      if (usedToday >= requestsPerDay) throw new BudgetExhausted('daily');
      if (usedRun >= requestsPerRun - reserve) throw new BudgetExhausted(reserve ? 'train-share' : 'run');
      if (reserve && self.timeLeft() <= reserveMs) throw new BudgetExhausted('train-time');
    },
    consume() { usedRun++; usedToday++; },
    save() { if (file) writeJson(file, { date: day, used: usedToday }); },
  };
  return self;
}

/** Wrap an LLM so every call is budget-checked first and counted (requests, tokens, wall-clock latency). */
export function budgeted(llm, budget) {
  const self = {
    name: llm.name,
    lastUsage: { tokensIn: 0, tokensOut: 0 },
    tokensIn: 0,
    tokensOut: 0,
    llmMs: 0,
    async chat(messages, opts = {}) {
      budget.check(opts.reserve ?? 0);
      budget.consume();
      const t0 = performance.now();
      const out = await llm.chat(messages, opts);
      const latencyMs = Math.round(performance.now() - t0);
      self.lastUsage = { tokensIn: 0, tokensOut: 0, ...(llm.lastUsage ?? {}), latencyMs };
      self.tokensIn += self.lastUsage.tokensIn;
      self.tokensOut += self.lastUsage.tokensOut;
      self.llmMs += latencyMs;
      return out;
    },
  };
  return self;
}
