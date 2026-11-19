# DECISIONS.md

Every non-obvious choice, why it was made, and how to change it. Values live in `config/default.json`;
any of them can be overridden with `SIC_*` env vars or the `SIC_CONFIG_OVERRIDES` repo variable (README).

## 1. The LLM: GitHub Models is gone, so a local model on the runner

**The plan said** to use GitHub Models through the Actions `GITHUB_TOKEN`. **What I found** (19 Nov 2026):
GitHub's docs say *"As of July 30, 2026, GitHub Models has been fully retired. The playground, model
catalog, inference API, and bring your own key (BYOK) are no longer available to any customer."*

**Choice:** a small open-weights model served by **Ollama on the Actions runner itself**, called through its
OpenAI-compatible `/v1/chat/completions` on `127.0.0.1`. It's free (public-repo runners) and needs **no token or
secret at all**, so the job that runs model-written code only has `contents: read`. The client is a generic
`openai-compatible` provider, so another endpoint works by changing `llm.endpoint` (key via `SIC_LLM_API_KEY`).
Ollama's version and download URL are pinned in config; the workflow reads them through `src/ci/config-out.js`.
The context window is set explicitly (`runtime.ollamaContextLength`, 8192) because Ollama silently truncates longer
prompts, and mutation-track prompts (function + failing assertions + lessons) run to several hundred tokens or more.

## 2. Model: `qwen2.5-coder:0.5b`

Low tier on purpose (weaker start = room to improve) and fast on a 4-vCPU runner without a GPU; 1.5B is
roughly 3x slower per call. Change with `SIC_CONFIG_OVERRIDES={"llm":{"model":"..."}}`, ideally before the baseline runs.

## 3. Tracks

### 3a. MultiPL-E (HumanEval + MBPP in JavaScript)
Hugging Face `nuprl/MultiPL-E` Parquet files pinned to revision `28441b60…` (558 problems), parsed with
`hyparquet`. Holdout: 50 problems, seed `split.seed`, stored in each variant's `split.json` and never silently re-made.

### 3b. Mutation repair: the contamination-free track (WOW phase 8)
- **Repo: `fastify/fast-uri` @ `3df33a5ff84eca2bbd0de9bc38fd28c17a11c3e7`, BSD-3-Clause.** Why this one:
  permissive license; **zero runtime dependencies**; small (≈60 named functions in `index.js`, `lib/utils.js`,
  `lib/schemes.js`) but dense with comparisons, offsets and branches, which suits mutation operators; a large
  suite (≈3,300 assertions in ~30 files) that runs in **0.4 s** inside our sandbox, so every mutant and every
  agent attempt can be judged by the real suite; tests need only `tape` (pinned `5.8.1`, installed with
  `--ignore-scripts`). `ajv.test.js` is excluded because it needs `ajv`.
- **Operators and weights** (config `curriculum.operators`): flip-comparison 3, flip-logical 2, off-by-one 2,
  drop-branch 1, swap-arguments 1, wrong-return 1. Weighted toward the bugs small models actually make.
  Functions longer than `maxFunctionChars` (1500) are excluded so a fix fits in the model's output budget.
- **Killed mutants only:** the mutated library's own suite must fail (fail or error, not timeout).
  On the first full generation, 33 of 111 single-mutation candidates **survived** the suite; those are
  discarded (they're either equivalent mutants or holes in the suite).
- **Levels:** 1 = one mutation, 2 = two mutations in one function, 3 = one mutation in each of two functions.
  The agent moves up when its rolling train pass rate over the last `promotion.window` (10) tasks at its
  level reaches `promotion.threshold` (0.6). Level changes are logged in `curriculum-state.json` and the gate log.
- **Holdout** = mutants from `holdoutSeed`, generated first; train generation (from `trainSeed`) skips any
  mutant identical to a holdout one. Holdout covers all three levels (10 each), so the holdout score doesn't
  change meaning as the agent's train level rises.
- **No third-party code in the repo:** state keeps only `curriculum-manifest.json` (seed, level, operators,
  function names, hash of the mutated source). Everything is regenerated from the pinned commit; `reproduce()`
  checks the hash. Train traces for this track store test output and reflections but not code (`tracks.mutation.storeCode: false`),
  and the skills lever is off for this track because a skill would be third-party code.
- **What the agent sees:** the mutated function(s), the failing assertions (TAP `not ok` blocks with the
  test titles, expected/actual), and how many bugs to look for. Never the operator; retrieval keywords come from
  function/file names and failing test titles only (a test asserts this).

## 4. Variants (WOW phase 6)

Variants are data (`config.variants`): `baseline` (**control**: no lessons, no skills, no prompt changes),
`lessons`, `skills`, `lessons-skills`, `full` (+ prompt optimizer), `mutation-baseline` (control),
`mutation-lessons`. Each has its own `state/variants/<name>/`, its own budget share, and its own job in the matrix.
The workflow's matrix is read from config by the `plan` job, so adding a variant is a config change only.

Control variants exist because day-to-day differences are not only noise: sampling seeds change, the
Ollama build could change, the runner could change. **Every headline comparison is against the control on the
same day and the same problems**, plus a difference-in-differences (variant's change since its start minus the
control's change over the same period).

## 5. Evaluation and statistics

- **Repeats:** every holdout evaluation runs `eval.repeats` = 3 times at `eval.temperature` 0.2. Repeat `r` of
  run `k` uses seed `llm.seed + ((k-1)·repeats + r)` (`eval.seedMode: per-run`): champion and candidate share the
  seeds within a run (paired), and every run draws fresh seeds, so an unchanged champion is re-measured independently.
  _An earlier version reused the same seeds every day, which made an unchanged champion score identically every
  day and the baseline noise exactly zero. The demo exposed it; fixed._
- **CIs:** 95% percentile bootstrap over **problems** (`stats.method: problems`, 2,000 resamples). Raw 0/1
  results per problem and repeat are stored in `evals.jsonl`.
- **I tried a two-level (hierarchical) bootstrap and reverted it.** After a no-memory control looked
  "significantly" better than its own start in the 7-task demo, I suspected the problems-only bootstrap ignored
  within-problem noise and switched to resampling problems and then repeats. A simulation with the same CI code
  (`src/power.js`, `test/stats.test.js`) showed that was wrong. With no true effect, the problems-only CI excluded zero
  2.8% of the time (n=50; nominal one-sided 2.5%). The hierarchical CI did so 0.5% of the time, and detected a real
  +15 pt gain only 56% vs 80% of the time: it double-counts within-problem noise. The control's "significant"
  drift was a small-sample (n=7) false positive among many comparisons. Kept as an option, labelled conservative;
  a calibration test now pins the false-positive rate.
- **Power** ([docs/power.md](docs/power.md), generated by `npm run power`): at n=50 × 3 repeats a true gain of about
  15–20 points is detected ≥80% of the time; 5 repeats would get that to about 10 points at +67% eval cost;
  80 problems × 3 to about 15. I kept 50 × 3 because the runner's real speed is still unmeasured. The simulation
  assumes independent repeats with per-problem pass probabilities between 0.1 and 0.8. At low temperature a
  real model's repeats are more correlated, which lowers the value of extra repeats.

## 6. Gate

`gate.method: ci`. Accept if the candidate is ≥ `minImprovement` (2.5 pts) better **and** the paired bootstrap CI
of the difference excludes zero. Reject if the CI is entirely below zero, or the drop exceeds `tolerance` (2.5 pts).
Otherwise **keep**: the champion is unchanged and the candidate's new items stay pending into the next run, where
they're re-tested along with that run's additions. They're dropped after `maxPendingRuns` (3). Rejected items are
deleted. `champion.json` holds only accepted state (run counters live in `run-state.json`), so a rejected
candidate leaves it byte-for-byte unchanged (tested). The first `baselineRuns` (2) runs of every variant only
measure the starting point. Known cost: re-testing pending changes up to 3 times raises their chance of a lucky
accept; the report lists every accept with its CI so it can be checked.

## 7. Budget

`budget.totalRequestsPerRun` (2,940) is split across variants by `budgetShare` (equal: 420 each);
`holdoutFraction` 0.72 of each variant's share is reserved for evaluation (50 × 3 × champion + candidate = 300).
Variant jobs run on separate runners in parallel, so the split is about fairness and run time, not a shared quota.
The counter is checked **before** every model call; a time limit (`maxRunMinutes`, with `reserveMinutesForHoldout`)
stops training early enough to evaluate. If evaluation can't finish, its scores are discarded (`no-eval`) and state is still written.

## 8. Failure taxonomy (WOW phase 7)

Rules are config (`taxonomy.rules`), tried in order, first match wins: `timeout` → `syntax` (SyntaxError) →
`spec-misread` (entry point "is not defined"/"is not a function", or checks `entryPointMissing` / `arityMismatch`) →
`runtime-error` (a thrown JS error, including tape's `operator: error`) → `wrong-output` (assertion failed) → `other`
(non-zero exit with no recognisable error, e.g. `process.exit`). Each class has a fixture that goes through the real
judge. Classes are recorded on every attempt row. "Reduced by self-improvement" means a learning variant's share
dropped by ≥ `analysis.taxonomyMinDrop` and clearly more than the control's.

**Cost metrics:** requests include reflection calls; "solved" = distinct train problems passed at least once;
reported as solved per 100 requests, requests per solved and tokens per solved. Tokens come from Ollama's `usage`
field (the mock estimates them as characters / 4).

## 9. Dashboard and report (WOW phases 9-10)

- `site/build.js` writes one self-contained `site/index.html`: inline CSS/JS/SVG, light and dark, mobile layout.
  Headline numbers, charts and tables are static HTML (tested with scripts stripped); JS only adds the replay
  stepper and filters. Everything from state is HTML-escaped (tested with an injected `<script>` lesson).
- `src/report.js` writes `REPORT.md` from `src/analysis.js` only, with no clock and stable sort orders, so regenerating
  gives identical bytes (tested). It always includes: plateau check (no significant gain over the last
  `analysis.plateauRuns` runs), changes the gate blocked and their measured cost, conflicting lessons (pairs retrieved
  together on failed attempts with a lower pass rate together than either alone), and an author TODO.
- The README's results block is regenerated between markers by `npm run report -- --update-readme`.
- Workflow: `plan` → `improve` (matrix, `contents: read`) → `aggregate` (`contents: read`: merges artifacts with
  `src/ci/collect.js`, which accepts only configured variant names and no symlinks, then builds charts, report and site) →
  `commit` (`contents: write`, copies files only) and `pages` (`pages: write`, `id-token: write`).

## 9b. Floor and ceiling effects

A track whose starting pass rate is below 10% or above 90% (`analysis.floorRate` / `ceilingRate`) can show little
in either direction: with almost no passes there is little to build skills or lessons from, and near the top there
is nothing left to gain. The report and dashboard say so and name the knob to turn (`analysis.difficultyKnobs`).
The mutation track is the more likely to hit the floor with a 0.5B model; `curriculum.maxFunctionChars` and the
operator weights are its main difficulty controls.

## 10. Reproduction and the synthetic mock

`npm run reproduce -- --variant=<name> --seed=<n>` runs simulated days with a fake clock, entirely offline.
The mock's synthetic mode is **not a model**: each fixture problem belongs to a family with a typical bug
(`test/fixtures/synthetic-problems.json`, 39 problems generated and verified by `tools/make-synthetic-fixtures.cjs`).
Its pass probability rises when a lesson containing the family's fix phrase, or a solved problem of the same family,
is in the prompt. That gives realistic-looking learning dynamics for testing statistics, gate, report and dashboard.
Output is identical for the same seed (tested). Data from it is labelled "Demo data" / "synthetic mock" everywhere.

## 11. Earlier choices that still hold (PLAN.md phases 1-5)

- **Sandbox:** empty env, temp dir, process-group kill on timeout, heap cap, output cap, Node permission model
  (reads only in the temp dir plus pinned test dependencies; no writes, no child processes). Judge output drops
  Node-internal stack frames (config `judge.dropOutputLinePatterns`), which is cleaner for the dashboard and wastes less of the model's context.
- **Retrieval:** keyword overlap (cosine on token sets), no embeddings: offline, free, deterministic.
- **Prompt optimizer (phase 5):** `@ax-llm/ax` GEPA with train-only examples, budget-proxied, flag + lever gated,
  every `promptOptimizer.everyNRuns` runs, in the `full` variant only. Errors are caught and logged in the gate log.
  With the mock provider a deterministic stand-in proposes a prompt so the path is exercised offline.
  `@ax-llm/ax`'s postinstall writes files into the project, so installs use `--ignore-scripts` (also `.npmrc`).
- **Actions pinned to commit SHAs**; the cron is a literal (GitHub can't read it from a variable).
- **No hard-coded values:** a test parses every source file with acorn and fails on URLs, model names and
  numeric literals other than 0/1/2/100; the few fixed constants (PRNG/hash constants, unit conversions, the Node
  versions that renamed `--permission`, a float epsilon, ID formats) carry an `allow-number:` comment saying why.

## 12. Not yet measured

Installing Ollama in the build session was blocked, so **no real-model numbers exist yet**: no baseline, no
per-call speed, and no check that the GEPA path works against Ollama. The first scheduled runs will produce them;
REPORT.md and the dashboard fill in automatically. Update §5 (repeats) and §7 (budget) once the real per-call
time is known.
