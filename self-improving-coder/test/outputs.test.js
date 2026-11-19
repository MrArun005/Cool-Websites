// Phase 9 + 10 acceptance: the dashboard builds from fixture state (headline numbers present without JS,
// a replay for every attempt that produced a lesson), REPORT.md regenerates byte-for-byte, and
// `npm run reproduce` is deterministic for a seed.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { reproduce, formatRows } from '../src/reproduce.js';
import { buildSite } from '../site/build.js';
import { writeReport, buildReport, readmeBlock, updateReadme } from '../src/report.js';
import { analyze } from '../src/analysis.js';
import { readJsonl } from '../src/recorder.js';
import { ROOT, loadConfig } from '../src/config.js';
import { tmpDir } from './helpers.js';

const VARIANTS = ['baseline', 'lessons-skills', 'mutation-lessons'];
const DAYS = 4;
let fixture;
async function state() {
  if (!fixture) fixture = await reproduce({ seed: 3, variants: VARIANTS, days: DAYS, out: tmpDir() });
  return fixture;
}

const stripScripts = (html) => html.replace(/<script>[\s\S]*?<\/script>/g, '');
const pct = (v) => `${(v * 100).toFixed(1)}%`;

test('site: headline numbers, chart and tables are plain HTML (work with JS disabled)', async () => {
  const { cfg } = await state();
  const { html, analysis } = buildSite(cfg);
  const noJs = stripScripts(html);
  assert.match(noJs, /^<!doctype html>/);
  assert.match(noJs, /<meta name="viewport"/);
  assert.match(noJs, /prefers-color-scheme:dark/);
  for (const name of VARIANTS) {
    const v = analysis.variants[name];
    assert.ok(v.latest, `${name} has a latest score`);
    assert.ok(noJs.includes(pct(v.latest.passRate)), `${name} headline ${pct(v.latest.passRate)} is in the static HTML`);
  }
  assert.equal((noJs.match(/<polygon class="band"/g) ?? []).length + (noJs.match(/<line class="band"/g) ?? []).length >= VARIANTS.length, true);
  for (const id of ['headline', 'lessons', 'replay', 'gate', 'taxonomy', 'cost']) assert.match(noJs, new RegExp(`<section id="${id}"`));
  assert.match(noJs, /Demo data/); // mock provider is labelled
});

test('site: there is a replay for every attempt that produced a lesson, with all four steps', async () => {
  const { cfg } = await state();
  const { html } = buildSite(cfg);
  let count = 0;
  for (const name of VARIANTS) {
    const dir = path.join(cfg.paths.stateDir, 'variants', name);
    for (const t of readJsonl(path.join(dir, 'traces.jsonl')).filter((x) => x.lessonCreated)) {
      count++;
      const id = `replay-${`${name}:${t.runId}:${t.problemId}:${t.attempt}`.replace(/[^\w-]/g, '-')}`;
      const m = html.match(new RegExp(`<article class="replay" id="${id}"[\\s\\S]*?</article>`));
      assert.ok(m, `replay ${id}`);
      assert.equal((m[0].match(/<li class="step/g) ?? []).length, 4);
      assert.ok(m[0].includes(t.lessonCreated));
    }
  }
  assert.ok(count > 0, 'fixture has lessons');
});

test('site: model-written text is escaped', async () => {
  const { cfg } = await state();
  const a = analyze(cfg);
  const v = Object.values(a.variants).find((x) => x.lessons.length);
  const lesson = v.lessons[0];
  const file = path.join(cfg.paths.stateDir, 'variants', v.name, 'traces.jsonl');
  const rows = readJsonl(file);
  const row = rows.find((r) => r.lessonCreated === lesson.id);
  row.reflection.text = '<img src=x onerror=alert(1)></script><script>alert(2)</script>';
  fs.writeFileSync(file, rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
  const { html } = buildSite(cfg);
  assert.ok(!html.includes('<img src=x'));
  assert.ok(html.includes('&lt;img src=x onerror=alert(1)&gt;'));
  assert.equal((html.match(/<script>/g) ?? []).length, 1);
  fixture = null; // state was modified; rebuild for later tests
});

test('report: deleting REPORT.md and regenerating gives the same bytes; mandatory sections present', async () => {
  const { cfg } = await state();
  const file = path.join(cfg.paths.chartsDir, cfg.paths.reportFile);
  const first = writeReport(cfg);
  fs.rmSync(file);
  writeReport(cfg);
  assert.equal(fs.readFileSync(file, 'utf8'), first);
  for (const h of ['## Setup', '## Variants', '## Results: holdout pass rate', '## Cost', '## Failure types', '## Curriculum levels', '## Gate decisions',
    '## Where it plateaued', '## Changes the gate blocked', '## Lessons', "## What I'd do with real compute", '## Reproduce']) assert.ok(first.includes(h), h);
  assert.match(first, /TODO \(author\)/);
  assert.match(first, /synthetic mock provider, not from a language model/);
  assert.match(first, /not statistically meaningful|statistically meaningful/);
  assert.match(first, /Which failure types does self-improvement fix here\?/);
});

test('report: numbers come from the data (headline matches analysis)', async () => {
  const { cfg } = await state();
  const a = analyze(cfg);
  const md = buildReport(cfg, a);
  for (const name of VARIANTS) assert.ok(md.includes(pct(a.variants[name].latest.passRate)), name);
});

test('report: README block is replaced between markers only', async () => {
  const { cfg } = await state();
  const block = readmeBlock(cfg, analyze(cfg));
  const before = 'intro\n<!-- report:summary:start -->\nold\n<!-- report:summary:end -->\noutro\n';
  const after = updateReadme(before, block);
  assert.ok(after.startsWith('intro\n<!-- report:summary:start -->'));
  assert.ok(after.endsWith('<!-- report:summary:end -->\noutro\n'));
  assert.ok(!after.includes('\nold\n'));
  assert.throws(() => updateReadme('no markers', block), /markers/);
});

test('report with no runs still renders every section', () => {
  const cfg = loadConfig({ env: {}, overrides: { paths: { stateDir: tmpDir(), chartsDir: tmpDir() } } });
  const md = buildReport(cfg, analyze(cfg));
  assert.match(md, /No evaluated runs yet/);
  assert.match(md, /## Where it plateaued/);
});

test('npm run reproduce is deterministic for a seed', () => {
  const run = (out) => execFileSync(process.execPath, ['src/reproduce.js', '--variant=lessons', '--seed=11', '--days=3', `--out=${out}`], { cwd: ROOT, env: { ...process.env, SIC_QUIET: '1' } }).toString().replace(/state written to .*/, '');
  const a = run(tmpDir());
  const b = run(tmpDir());
  assert.equal(a, b);
  assert.match(a, /lessons\s+baseline/);
  assert.throws(() => execFileSync(process.execPath, ['src/reproduce.js', '--variant=lessons'], { cwd: ROOT, stdio: 'pipe' }), /seed/);
});

test('formatRows shows CI and diff columns', () => {
  const s = formatRows([{ day: 1, variant: 'x', decision: 'accept', champion: { passRate: 0.5, ciLow: 0.4, ciHigh: 0.6 }, candidate: { passRate: 0.7 }, diff: { estimate: 0.2, low: 0.1, high: 0.3 } }]);
  assert.match(s, /50\.0%.*40\.0%.*60\.0%.*70\.0%.*20\.0%/);
});
