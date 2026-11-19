# BACKLOG.md

Improvement plan for the 3-day hardening pass (2026-11-19 to 2026-11-22). Order = what most reduces the
chance that the real week of runs is wasted or misread. Each item is done in its own commit with tests;
each item records the commit that finished it.

Rules for every item: `npm test` green, actionlint clean, no numbers/URLs/model names outside config,
DECISIONS.md updated for any non-obvious choice, README claims backed by generated numbers.

## Done

- [x] **Smoke mode** (`workflow_dispatch` input `smoke`): tiny real-model run per variant, state in `.smoke/`,
  nothing committed; job summary shows measured seconds per request and whether the full budget fits the time
  limit; exercises the prompt optimizer path. Model latency is recorded on every attempt row and shown in the cost tables.

## Next, in order

2. **Workflow robustness.** Retry the Ollama download/pull with backoff; a warm-up request that fails fast with a
   clear message if the model doesn't answer; a test that the workflow's step order and `if:` conditions keep smoke
   runs from committing.
3. **Multiple comparisons.** Holm-adjusted significance across the learning variants' control comparisons, reported
   next to the unadjusted CIs in plain words.
4. **Reflection quality metrics.** Rate of reflections that produced no usable lesson, duplicate-lesson rate,
   near-duplicate merging (token Jaccard above a configured threshold), and the expected/actual values given to the
   reflection prompt explicitly.
5. **Lesson effect estimate (observational, labelled as such).** Per lesson: pass rate of train attempts where it
   was retrieved vs comparable attempts without it.
6. **Gate re-test inflation.** Make the pending re-test rule explicit in numbers: simulate the false-accept rate
   of "keep up to N runs" with the power simulator and report it in docs/power.md.
7. **Dashboard:** curriculum level timeline, gate-log rows linked to their replays, per-variant anchors.
8. **Independent review** (a fresh reviewer, not anchored on my reading) of the sandbox, workflow permissions and the
   statistics; fix what it finds.
9. **Docs:** "how to read the dashboard" section; final pass on README/DECISIONS consistency.
