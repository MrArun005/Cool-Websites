// Phase 8 acceptance: generate.js produces N killed mutants from config, every mutant reproduces from its
// seed, the agent loop runs on them with the mock provider, and no third-party code is written to state.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { loadCurriculum, reproduce, candidate, loadTargets } from '../src/mutate/generate.js';
import { OPERATORS, applyEdits } from '../src/mutate/operators.js';
import { listFunctions } from '../src/mutate/ast.js';
import { curriculumCfg, runSuite, ensureRepo } from '../src/mutate/repo.js';
import { createMutationTrack, functionsInCode, patchedFiles } from '../src/tracks/mutation.js';
import { runCycle } from '../src/run.js';
import { readJsonl } from '../src/recorder.js';
import { readJson } from '../src/util.js';
import { ROOT } from '../src/config.js';
import { testConfig, tmpDir, vdir } from './helpers.js';

const MUT_VARIANT = { name: 'mutation-lessons', label: 'Mutation: lessons', track: 'mutation', levers: { lessons: true }, budgetShare: 1, color: '#2563eb' };
const cfgFor = (extra = {}) => testConfig({ variants: [MUT_VARIANT], defaultVariant: MUT_VARIANT.name, curriculum: { fixture: { cacheDir: tmpDir() } }, ...extra });

let shared;
async function curriculum() {
  if (!shared) { const cfg = cfgFor(); shared = { cfg, cur: await loadCurriculum(cfg) }; }
  return shared;
}

test('operators find sites and edits apply cleanly', () => {
  const src = 'function f (a, b) {\n  if (a < b && b > 0) return g(a, b)\n  else return 1\n}\n';
  const [fn] = listFunctions(src, 2022);
  const count = (op) => OPERATORS[op](fn, src).length;
  assert.equal(count('flip-comparison'), 2);
  assert.equal(count('flip-logical'), 1);
  assert.equal(count('swap-arguments'), 1);
  assert.equal(count('drop-branch'), 2); // empty then-branch, drop else
  assert.equal(count('off-by-one'), 3); // 0 -> 1 (no -1 below zero), 1 -> 2, 1 -> 0
  assert.equal(count('wrong-return'), 2);
  const [flip] = OPERATORS['flip-comparison'](fn, src);
  assert.match(applyEdits(src, [flip]), /a <= b/);
  assert.throws(() => applyEdits(src, [flip, flip]), /overlapping/);
});

test('generate --dry-run: N killed mutants per config, all reproducible from their seed', async () => {
  const { cfg, cur } = await curriculum();
  const c = curriculumCfg(cfg);
  const want = (counts) => Object.values(counts).reduce((a, b) => a + b, 0);
  assert.equal(cur.tasks.filter((t) => t.split === 'train').length, want(c.trainCount));
  assert.equal(cur.tasks.filter((t) => t.split === 'holdout').length, want(c.holdoutCount));
  for (const level of [1, 2, 3]) {
    const t = cur.tasks.find((x) => x.level === level);
    assert.equal(t.operators.length, c.levels[level].mutations);
    assert.equal(new Set(Object.keys(t.located)).size, c.levels[level].functions);
  }
  for (const m of cur.manifest) assert.ok(reproduce(m, cur.pristine, c), `${m.id} reproduces`);
  // a different seed gives a different mutant
  const a = candidate(cur.manifest[0].seed, 1, cur.pristine, c);
  const b = candidate(cur.manifest[0].seed + 1, 1, cur.pristine, c);
  assert.ok(!b || a.sourceHash !== b.sourceHash);
  // train never repeats a holdout mutant
  const holdoutHashes = new Set(cur.tasks.filter((t) => t.split === 'holdout').map((t) => t.sourceHash));
  assert.ok(cur.tasks.filter((t) => t.split === 'train').every((t) => !holdoutHashes.has(t.sourceHash)));
});

test('every kept mutant is killed by the repo\'s own tests, and restoring the original function fixes it', async () => {
  const { cfg, cur } = await curriculum();
  const env = ensureRepo(cfg);
  for (const t of cur.tasks.slice(0, 6)) {
    assert.notEqual((await runSuite(cfg, env, t.mutatedFiles)).type, 'pass', `${t.id} is killed`);
    const fixes = Object.fromEntries(Object.values(t.located).map((l) => [l.name, l.original]));
    assert.equal((await runSuite(cfg, env, patchedFiles(t, fixes))).type, 'pass', `${t.id} fixed by the original`);
  }
});

test('mutation track judge: original passes, mutant fails, garbage is a syntax error', async () => {
  const { cfg, cur } = await curriculum();
  const track = await createMutationTrack({ ...cfg, paths: { ...cfg.paths, stateDir: tmpDir() } }, { curriculum: cur });
  const task = track.tasks.get(track.train[0]);
  assert.equal((await track.judge(task.mock.correct, task)).type, 'pass');
  assert.notEqual((await track.judge(task.mock.buggy, task)).type, 'pass');
  const bad = await track.judge('function (', task);
  assert.equal(bad.type, 'error');
  assert.match(bad.output, /SyntaxError/);
  assert.deepEqual(Object.keys(functionsInCode('function a(){}\nfunction b(){}', 2022)), ['a', 'b']);
  // retrieval text never reveals the mutation operator
  for (const t of track.tasks.values()) for (const op of Object.keys(OPERATORS)) assert.ok(!t.keywordText.includes(op));
});

test('agent loop runs on mutants with the mock provider; lessons are learned; no third-party code in state', async () => {
  const cfg = cfgFor({ llm: { mock: { mode: 'synthetic' } } });
  const s1 = await runCycle({ cfg });
  const s2 = await runCycle({ cfg });
  assert.ok(s1.train.problems > 0);
  const dir = vdir(cfg);
  const attempts = readJsonl(path.join(dir, 'attempts.jsonl'));
  assert.ok(attempts.some((a) => a.phase === 'train') && attempts.some((a) => a.phase === 'eval'));
  assert.ok(attempts.every((a) => a.track === 'mutation'));
  assert.ok(readJsonl(path.join(dir, 'traces.jsonl')).every((t) => t.code === null));
  const manifest = readJson(path.join(dir, 'curriculum-manifest.json'));
  assert.ok(manifest.length > 0 && manifest.every((m) => !('mutatedFiles' in m) && !('located' in m)));
  // nothing from the fixture library's source is written into state
  const lib = fs.readFileSync(path.join(ROOT, 'test/fixtures/minirepo/lib/util.js'), 'utf8');
  const fnBodies = lib.split('\n\nfunction ').slice(1).map((b) => 'function ' + b.split('\n}')[0]);
  for (const f of fs.readdirSync(dir, { recursive: true })) {
    const p = path.join(dir, f);
    if (!fs.statSync(p).isFile()) continue;
    const text = fs.readFileSync(p, 'utf8');
    for (const body of fnBodies) assert.ok(!text.includes(body), `${f} contains library code`);
  }
  assert.ok([s1.decision, s2.decision].every(Boolean));
});

test('curriculum moves up a level when the rolling pass rate crosses the threshold, and logs it', async () => {
  const { cur } = await curriculum();
  const cfg = cfgFor({ curriculum: { promotion: { window: 2, threshold: 0.5 } } });
  const track = await createMutationTrack({ ...cfg, paths: { ...cfg.paths, stateDir: tmpDir() } }, { curriculum: cur });
  const l1 = track.train.filter((id) => track.tasks.get(id).level === 1);
  assert.equal(track.level, 1);
  assert.equal(track.onTrainResult(l1[0], false, { runIndex: 1 }), null);
  const ev = track.onTrainResult(l1[1], true, { runIndex: 1 });
  assert.deepEqual([ev.type, ev.from, ev.to, ev.rollingPassRate], ['level-up', 1, 2, 0.5]);
  assert.equal(track.level, 2);
  const mem = { attempts: {} };
  assert.equal(track.tasks.get(track.nextTrainTask({ mem, runIndex: 2, doneThisRun: new Set(), solved: new Set() })).level, 2);
});

test('`node src/mutate/generate.js --dry-run` prints the manifest', () => {
  const cache = tmpDir();
  const out = execFileSync(process.execPath, ['src/mutate/generate.js', '--dry-run'], { cwd: ROOT, env: { ...process.env, SIC_QUIET: '1', SIC_CONFIG_OVERRIDES: JSON.stringify({ curriculum: { fixture: { cacheDir: cache } } }) } }).toString();
  const parsed = JSON.parse(out);
  assert.equal(parsed.repo, 'minirepo');
  assert.equal(parsed.count, parsed.manifest.length);
  assert.ok(parsed.count > 0);
  assert.ok(parsed.manifest.every((m) => Number.isInteger(m.seed) && m.operators.length >= 1));
});
