// Writes a Markdown summary of one variant's run (from last-run.json) for the GitHub Actions job page:
// requests, tokens, measured seconds per model call and whether the configured budget fits the time limit.
//   node src/ci/step-summary.js --variant=<name> >> "$GITHUB_STEP_SUMMARY"
import fs from 'node:fs';
import { loadConfig, statePath } from '../config.js';
import { variantConfig } from '../variants.js';
import { argValue, MS_PER_SECOND, MS_PER_MINUTE } from '../util.js';

export function stepSummary(cfg, s) {
  const v = cfg.variant;
  if (!s) return `### ${v.name}\n\nNo run summary found (the run failed before writing state).\n`;
  const perCall = s.requestsUsed ? s.llmMs / s.requestsUsed : null;
  const projectedMin = perCall === null ? null : (perCall * cfg.budget.requestsPerRun) / MS_PER_MINUTE;
  const fits = projectedMin === null ? 'n/a' : projectedMin <= cfg.budget.maxRunMinutes ? 'yes' : '**no: lower the budget or raise the time limit**';
  const pct = (x) => (x ? `${(x.passRate * 100).toFixed(1)}% [${(x.ciLow * 100).toFixed(1)}, ${(x.ciHigh * 100).toFixed(1)}] n=${x.n}x${x.repeats}` : 'n/a');
  return [
    `### ${v.label} (\`${v.name}\`, ${v.track})`, '',
    '| | |', '|---|---|',
    `| gate decision | ${s.decision} |`,
    `| holdout champion | ${pct(s.champion)} |`,
    `| holdout candidate | ${pct(s.candidate)} |`,
    `| train | ${s.train.passed}/${s.train.problems} passed, +${s.train.newLessons.length} lessons, +${s.train.newSkills.length} skills |`,
    `| model requests | ${s.requestsUsed} (tokens in/out ${s.tokensIn}/${s.tokensOut}) |`,
    `| seconds per request (measured) | ${perCall === null ? 'n/a' : (perCall / MS_PER_SECOND).toFixed(2)} |`,
    `| projected minutes for this variant's configured budget (${cfg.budget.requestsPerRun} requests) | ${projectedMin === null ? 'n/a' : projectedMin.toFixed(1)} of ${cfg.budget.maxRunMinutes} allowed: fits ${fits} |`,
    '',
  ].join('\n');
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const base = loadConfig();
  const name = argValue(process.argv.slice(2), 'variant');
  // project against the FULL (non-smoke) budget, so a smoke run answers "will the real run fit?"
  const full = variantConfig(loadConfig({ env: { ...process.env, SIC_SMOKE: '' } }), name);
  const file = statePath(variantConfig(base, name), 'lastRun');
  const s = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : null;
  console.log(stepSummary(full, s));
}
