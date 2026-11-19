# DECISIONS.md

Every non-obvious choice, why it was made, and how to change it. Values live in
`config/default.json`; any of them can be overridden with `SIC_*` env vars or the
`SIC_CONFIG_OVERRIDES` repo variable (see README).

## 1. The LLM: GitHub Models is gone, so a local model on the runner

**The plan said** to use GitHub Models through the Actions `GITHUB_TOKEN`.
**What I found** (19 Nov 2026): GitHub's docs say *"As of July 30, 2026, GitHub Models has been
fully retired. The playground, model catalog, inference API, and bring your own key (BYOK) are no
longer available to any customer."* The catalog endpoint returns an empty body.

**Choice:** run a small open-weights model with **Ollama on the GitHub Actions runner itself**, called
through its OpenAI-compatible `/v1/chat/completions` endpoint on `127.0.0.1`.

- Still costs nothing (public-repo runners are free) and needs **no token or secret at all**.
- Hard rule 4/7 got stronger: the job that runs LLM-written code now holds only `contents: read`.
- The provider is a generic `openai-compatible` client (`src/llm/openai-compatible.js`), so you can
  point `llm.endpoint` at any other OpenAI-style API later. If it needs a key, put it in a secret and
  expose it as `SIC_LLM_API_KEY` (name set by `llm.apiKeyEnv`). That would no longer be zero-cost
  unless the provider has a free tier, so it is not the default.
- Ollama version and download URL are pinned in config (`runtime.ollamaVersion`, `runtime.ollamaDownloadUrl`);
  the workflow reads them via `src/ci/config-out.js`, so the YAML has no versions or model names.

## 2. Model: `qwen2.5-coder:0.5b`

- The plan asked for a *low-tier* model on purpose: weaker start = visible gains.
- On a 4-vCPU runner with no GPU, a 0.5B model answers in a few seconds; 1.5B is roughly 3x slower and
  would cut the requests per run by about two thirds.
- Qwen2.5-Coder is instruction-tuned for code, follows the "one code block" format well enough, and is
  small enough (~400 MB) to cache between runs with `actions/cache`.
- To try a stronger model: set `SIC_CONFIG_OVERRIDES={"llm":{"model":"qwen2.5-coder:1.5b"}}` and lower the
  budget (see §4). Changing the model invalidates comparisons with earlier runs; do it before the baseline.

## 3. Dataset: MultiPL-E `humaneval-js` + `mbpp-js`, pinned

- Source: Hugging Face dataset `nuprl/MultiPL-E`, Parquet files, pinned to revision
  `28441b6024e71d4a1c1c0f6bf171c935cd5a43f2` via `/resolve/<sha>/` URLs (verified: 161 + 397 = 558 problems).
- The old raw JSON prompt files are no longer in the GitHub repo; the Parquet files are the canonical form now.
  They are parsed with `hyparquet` (pure JS, no native deps).
- Downloaded each run to `data/` (git-ignored). It's ~150 KB, so caching isn't worth the complexity.
- Each problem's `tests` is a CommonJS script using `node:assert`, so the judge writes `solution.cjs`.

## 4. Budget

There is no external rate limit any more; the budget is about **runner time**.

| key | value | why |
|---|---|---|
| `budget.requestsPerRun` | 360 | ~5–10 s per call for a 0.5B model on 4 vCPU → ~30–60 min of model time |
| `budget.reserveForHoldout` | 160 | 2 × `holdoutSize` (80): champion + candidate on the full holdout set |
| `budget.requestsPerDay` | 400 | lets one extra manual run happen the same day without doubling cost |
| `budget.maxRunMinutes` | 100 | wall clock for the cycle; the job timeout is 150 (var `SIC_IMPROVE_TIMEOUT_MINUTES`) |
| `budget.reserveMinutesForHoldout` | 40 | training stops early enough that the holdout eval can still finish |

The counter is checked **before** every LLM call (`src/budget.js`). If it runs out mid-training, training
stops and the cycle continues to holdout eval. If it runs out mid-eval, the scores are thrown away, the
decision is `no-eval`, and state is still written.

## 5. Split: 80 holdout / 478 train, seed 20261119

- 80 holdout problems makes one problem = 1.25 points, which is coarse but affordable twice per run.
- The split is stored in `state/split.json` and never silently re-made: if the dataset changes, the run
  fails and asks you to delete the file on purpose.
- If the budget can't cover the full holdout twice, both champion and candidate are scored on the **same**
  seeded random subset, and `n` is recorded.

## 6. Gate: `minImprovement` 0.025, `tolerance` 0.025, two baseline runs

- Accept if `candidate ≥ champion + 0.025` (2 problems on n=80). Reject if `candidate < champion − 0.025`.
- **"keep" is interpreted** as: champion unchanged, but the candidate's new lessons/skills/prompt stay
  *pending* and are carried into the next run, where they are re-tested together with that run's new
  items. Pending items older than `gate.maxPendingRuns` (3) are dropped. This lets small, individually
  unmeasurable gains add up without ever accepting something that measured worse.
- **Rejected items are deleted** (lessons from `lessons.json`, skill files from `state/skills/`). Rejected
  prompt versions stay in `state/prompts/` as a record but are not referenced.
- `champion.json` holds only the accepted state, so a rejected candidate leaves it byte-for-byte unchanged
  (a test proves this). Run counters live in `state/run-state.json`.
- Both champion and candidate are re-scored every run on the same problems, so the comparison is
  paired and not affected by day-to-day drift.
- **Baseline:** `gate.baselineRuns = 2`. The first two runs only score the champion on the holdout set
  (no training), to measure noise. Eval uses `temperature 0` and a fixed `seed`, so the two numbers
  should be close; any gap is the noise floor. If the gap is bigger than `tolerance`, raise
  `minImprovement`/`tolerance` to at least that gap before trusting accept decisions.

### Baseline numbers (fill in after the first two real runs)

| run | date | holdout pass rate | n |
|---|---|---|---|
| baseline 1 | _pending_ | _pending_ | 80 |
| baseline 2 | _pending_ | _pending_ | 80 |

Not measured in this session: installing and running Ollama here was blocked by the session's
permission policy, so the first real numbers will come from GitHub Actions. `npm run summary` prints them.

## 7. Agent details

- **Attempts:** 2 per train problem (`agent.maxAttemptsPerProblem`). Attempt 2 sees the lesson made from attempt 1.
- **Temperature:** 0.4 for training (some variety between attempts), 0 for holdout eval (repeatable).
- **Retrieval:** keyword overlap (cosine on token sets) between the problem's comments + function name and
  each lesson's tags + text, or each skill's keywords. No embeddings: offline, free, deterministic.
  Top 3 lessons, top 2 skills, `retrieve.minScore` 0.05. A problem never retrieves its own skill.
- **Lesson tags:** the model's tags plus the source problem's own keywords, so retrieval still works when a
  0.5B model returns bad JSON. If the reply has no JSON, its first line becomes the lesson.
- **Lesson size:** at most 40 words; exact duplicates (case-insensitive) are not stored twice.
- **Pruning:** above 60 live lessons, drop never-helped first, then least-used, then oldest.
  "helped" = the lesson was in the prompt of a passing train attempt.
- **Train order:** never-attempted problems first (seeded shuffle per run), then failures not retried for
  `agent.retryFailedAfterRuns` runs, fewest tries first. Problems already solved by a live skill are skipped.
- **Skills:** one file per problem in `state/skills/<id>.js`, first line is JSON metadata. Shown as
  "related functions you solved before" (truncated to 1200 chars).
- **Holdout isolation:** holdout problems are only ever solved at eval temperature, never reflected on,
  never saved as skills; a test checks this.

## 8. Sandbox

`src/judge/run-tests.js`: fresh temp dir per attempt (deleted after), `env: {}`, `cwd` = temp dir,
`--max-old-space-size=256`, process group killed on timeout (6 s), output capped at 16 KB.
**Extra, beyond the plan:** Node's permission model (`--permission`, or `--experimental-permission`
on older Node) with read access only to the temp dir. This blocks `child_process`, worker threads,
file writes and reads outside the temp dir. Tests check spawn, write and read are blocked.
The tests are wrapped in their own block (`;{ ... }`) so a solution that declares `const assert` doesn't
collide with the test file's `const assert`.

## 9. Workflow

- Cron `17 4 * * *` (04:17 UTC). GitHub can't read cron from a variable, so it's a literal in the YAML.
  The minute is off the hour because runs at :00 are the most delayed.
- Job timeouts come from repo variables with defaults (`SIC_IMPROVE_TIMEOUT_MINUTES` 150,
  `SIC_COMMIT_TIMEOUT_MINUTES` 10), because `timeout-minutes` is read before any step runs.
- Actions are pinned to commit SHAs (checkout v7.0.1, setup-node v7.1.0, cache v6.1.0,
  upload-artifact v7.0.2, download-artifact v8.0.2).
- The commit job refuses artifacts that contain symlinks and copies only `state/` and `chart.svg`.
- `npm ci --ignore-scripts`, plus `.npmrc` `ignore-scripts=true`: `@ax-llm/ax`'s postinstall script writes
  agent skill files into the project folder, which we don't want and never want to run in CI.
- A separate `ci.yml` runs `npm test` and the dry run on every push.

## 10. Phase 5: prompt evolution (off by default)

- `features.promptOptimizer: false`. When on, `src/agent/evolve.js` uses `@ax-llm/ax`'s `optimize()`
  (bootstrap few-shot → GEPA) with the same local model as student and teacher, pass/fail from the
  real judge as the metric, and **train problems only** for both its train and validation examples.
- The proposed instruction is saved as the next `state/prompts/vNNN.md` and must pass the same holdout
  gate as lessons and skills.
- The ax client is wrapped in a proxy that counts every call against the same budget, capped at
  `promptOptimizer.maxRequests` (120).
- `@ax-llm/ax` is an `optionalDependency`; if it's missing, the step is skipped.
- Caveat: ax wraps the instruction in its own prompt format during optimization, and we then use the
  instruction as our system prompt. The holdout gate is what protects us if that transfer doesn't work.
- Not tested against a real model in this session (see §6). Tests cover: flag off → byte-identical
  state and identical LLM calls to a run without the optimizer; flag on → the proposal goes through the gate.

## 11. Things the plan left open

- **Repo location:** this session could not create a new GitHub repository, so the project was built as a
  self-contained folder (`self-improving-coder/`) that is meant to become the root of its own repo.
- **"Train" line on the chart:** the pass rate on that run's train batch (up to 2 attempts, with learning
  happening during the run). It's not comparable to holdout and is drawn dashed and grey.
- **Holdout line:** the champion's score after the gate: the candidate's score if it was accepted, else
  the champion's fresh score.
- **Run id:** `gh<GITHUB_RUN_ID>-<UTC timestamp>`.
