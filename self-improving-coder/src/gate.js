// Champion vs candidate decision on the same holdout problems.
export const DECISION = Object.freeze({
  ACCEPT: 'accept',     // candidate clearly better: becomes champion
  KEEP: 'keep',         // within noise: champion unchanged, candidate changes stay pending for the next run
  REJECT: 'reject',     // candidate worse: its new lessons/skills/prompt are dropped
  BASELINE: 'baseline', // measuring noise, no learning accepted yet
  NO_CHANGE: 'no-change',
  NO_EVAL: 'no-eval',   // no budget left to evaluate
});

export function decide(championRate, candidateRate, { minImprovement, tolerance }) {
  // Small epsilon so 0.1 + 0.025 style float error never flips a decision.
  const eps = 1e-9;
  if (candidateRate + eps >= championRate + minImprovement) return DECISION.ACCEPT;
  if (candidateRate + eps >= championRate - tolerance) return DECISION.KEEP;
  return DECISION.REJECT;
}

/** Apply a decision to the champion in place. `candidate` = { promptVersion, lessonIds, skillIds }. */
export function applyDecision(champion, candidate, decision, { runIndex, holdout, maxPendingRuns }) {
  const added = {
    lessonIds: candidate.lessonIds.filter((id) => !champion.lessonIds.includes(id)),
    skillIds: candidate.skillIds.filter((id) => !champion.skillIds.includes(id)),
    promptVersion: candidate.promptVersion !== champion.promptVersion ? candidate.promptVersion : null,
  };
  const emptyPending = { lessonIds: [], skillIds: [], promptVersion: null, sinceRun: null };
  if (decision === DECISION.ACCEPT) {
    champion.lessonIds = [...candidate.lessonIds];
    champion.skillIds = [...candidate.skillIds];
    champion.promptVersion = candidate.promptVersion;
    champion.holdout = holdout;
    champion.pending = emptyPending;
  } else if (decision === DECISION.KEEP || decision === DECISION.NO_EVAL) {
    const sinceRun = champion.pending.sinceRun ?? runIndex;
    if (runIndex - sinceRun + 1 > maxPendingRuns) champion.pending = emptyPending; // stale: drop
    else champion.pending = { ...added, sinceRun };
  } else if (decision === DECISION.REJECT) {
    champion.pending = emptyPending;
  }
  return champion;
}
