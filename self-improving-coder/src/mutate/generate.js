// Mutation curriculum generator. A task = a seeded set of mutations to the third-party repo's functions
// that makes at least one of the repo's own tests fail ("killed mutant"). The agent's job: fix it.
//
// Everything about a mutant follows from (seed, level): which functions, which operators, which sites.
// Only seeds + operator lists + a hash of the mutated source are kept in state (curriculum-manifest.json);
// mutated third-party code is regenerated on demand and never committed.
//
//   node src/mutate/generate.js            generate from the pinned repo (needs network on first run)
//   node src/mutate/generate.js --dry-run  generate from the offline fixture repo
import fs from 'node:fs';
import path from 'node:path';
import { loadConfig, rootPath } from '../config.js';
import { listFunctions, tryParse } from './ast.js';
import { OPERATORS, applyEdits, overlaps } from './operators.js';
import { ensureRepo, runSuite, curriculumCfg } from './repo.js';
import { rng, hash32, readJson, writeJson, log } from '../util.js';
import { RESULT } from '../judge/run-tests.js';

export const fnKey = (f) => `${f.file}#${f.name}#${f.index}`;

/** Target functions with their mutation sites, from the pristine repo. */
export function loadTargets(env, c) {
  const files = {};
  const targets = [];
  for (const file of c.targetFiles) {
    const src = fs.readFileSync(path.join(env.repoDir, file), 'utf8');
    files[file] = src;
    for (const fn of listFunctions(src, c.ecmaVersion)) {
      if (fn.end - fn.start > c.maxFunctionChars) continue;
      const sites = {};
      for (const [op, weight] of Object.entries(c.operators)) {
        if (!(weight > 0)) continue;
        if (!OPERATORS[op]) throw new Error(`unknown mutation operator "${op}"`);
        sites[op] = OPERATORS[op](fn, src);
      }
      const total = Object.values(sites).reduce((s, a) => s + a.length, 0);
      if (total) targets.push({ file, name: fn.name, index: fn.index, start: fn.start, end: fn.end, sites });
    }
  }
  return { files, targets };
}

function weightedPick(r, items) {
  const total = items.reduce((s, [, w]) => s + w, 0);
  let u = r() * total;
  for (const [item, w] of items) { u -= w; if (u < 0) return item; }
  return items.at(-1)[0];
}

/** Deterministic mutant for (seed, level). Returns null if this seed can't produce one. */
export function candidate(seed, level, { files, targets }, c) {
  const L = c.levels[String(level)];
  if (!L) throw new Error(`unknown curriculum level ${level}`);
  if (targets.length < L.functions) return null;
  const r = rng(seed);
  const fns = [];
  for (let tries = 0; fns.length < L.functions && tries < targets.length * L.functions * 4; tries++) {
    const f = targets[Math.floor(r() * targets.length)];
    if (!fns.includes(f)) fns.push(f);
  }
  if (fns.length < L.functions) return null;
  const edits = [];
  for (let m = 0; m < L.mutations; m++) {
    const fn = fns[m % fns.length];
    const free = (s) => !edits.some((e) => e.file === fn.file && overlaps(e, s));
    const ops = Object.entries(fn.sites).filter(([, s]) => s.some(free)).map(([op]) => [op, c.operators[op]]);
    if (!ops.length) return null;
    const op = weightedPick(r, ops);
    const choices = fn.sites[op].filter(free);
    const site = choices[Math.floor(r() * choices.length)];
    edits.push({ ...site, file: fn.file, fn: fnKey(fn) });
  }
  const mutatedFiles = {};
  for (const file of new Set(edits.map((e) => e.file))) {
    mutatedFiles[file] = applyEdits(files[file], edits.filter((e) => e.file === file));
    if (!tryParse(mutatedFiles[file], c.ecmaVersion)) return null;
  }
  const sourceHash = hash32(Object.keys(mutatedFiles).sort().map((f) => `${f}\n${mutatedFiles[f]}`).join('\n'));
  return { seed, level, edits, fns, mutatedFiles, sourceHash };
}

const signature = (cand) => cand.edits.map((e) => `${e.file}:${e.start}:${e.end}:${e.replacement}`).sort().join('|');

/** Locate each target function in the mutated sources: { key: { file, start, end, source } }. */
export function locateFunctions(cand, files, c) {
  const out = {};
  for (const fn of cand.fns) {
    const src = cand.mutatedFiles[fn.file] ?? files[fn.file];
    const found = listFunctions(src, c.ecmaVersion).find((f) => f.name === fn.name && f.index === fn.index);
    out[fnKey(fn)] = { file: fn.file, name: fn.name, start: found.start, end: found.end, source: src.slice(found.start, found.end), original: files[fn.file].slice(fn.start, fn.end) };
  }
  return out;
}

/**
 * Keep what a developer would look at: for TAP output, each failing assertion ("not ok" + its indented
 * YAML block) under the name of its test; for anything else, the non-empty lines. Capped at maxChars.
 */
export function trimFailures(output, maxChars) {
  const lines = output.split('\n');
  if (!lines.some((l) => /^not ok \d+/.test(l))) return lines.filter((l) => l.trim()).join('\n').slice(0, maxChars);
  const out = [];
  let header = null;
  let inBlock = false;
  for (const l of lines) {
    if (/^# /.test(l) && !/^# (tests|pass|fail|ok)\b/.test(l)) { header = l; inBlock = false; continue; }
    if (/^not ok \d+/.test(l)) {
      if (header) { out.push(header); header = null; }
      out.push(l);
      inBlock = true;
      continue;
    }
    if (inBlock && /^\s+\S/.test(l) && !/^\s+(at[: ]|stack:|Error:|---|\.\.\.)/.test(l)) out.push(l);
    else if (!/^\s/.test(l)) inBlock = false;
    if (/^# (tests|pass|fail)\b/.test(l)) out.push(l);
  }
  return out.join('\n').slice(0, maxChars);
}

/** Generate `count` killed mutants for one split and level. `exclude` = signatures already used (e.g. by holdout). */
export async function generateLevel(cfg, env, pristine, { split, level, count, seedBase, exclude }) {
  const c = curriculumCfg(cfg);
  const kept = [];
  const stats = { tried: 0, invalid: 0, survived: 0, timeouts: 0, duplicates: 0 };
  for (let i = 0; kept.length < count && i < count * c.maxCandidatesPerTask; i++) {
    const seed = hash32(`${seedBase}:${level}:${i}`);
    const cand = candidate(seed, level, pristine, c);
    stats.tried++;
    if (!cand) { stats.invalid++; continue; }
    const sig = signature(cand);
    if (exclude.has(sig)) { stats.duplicates++; continue; }
    const res = await runSuite(cfg, env, cand.mutatedFiles);
    if (res.type === RESULT.PASS) { stats.survived++; continue; }
    if (res.type === RESULT.TIMEOUT && !c.keepTimeouts) { stats.timeouts++; continue; }
    exclude.add(sig);
    kept.push({ id: `mut-${split}-L${level}-${seed}`, split, level, seed, operators: cand.edits.map((e) => e.operator), functions: cand.edits.map((e) => e.fn),
      sourceHash: cand.sourceHash, killedAs: res.type, failures: trimFailures(res.output, c.failureOutputChars), cand });
  }
  return { kept, stats };
}

/** The committed, code-free description of a task. */
export const manifestEntry = (t) => ({ id: t.id, split: t.split, level: t.level, seed: t.seed, operators: t.operators, functions: t.functions, sourceHash: t.sourceHash, killedAs: t.killedAs });

/** Regenerate a task from its manifest entry and check it is the same mutant. */
export function reproduce(entry, pristine, c) {
  const cand = candidate(entry.seed, entry.level, pristine, c);
  return Boolean(cand) && cand.sourceHash === entry.sourceHash && JSON.stringify(cand.edits.map((e) => e.operator)) === JSON.stringify(entry.operators);
}

function cacheKey(cfg, env) {
  const c = curriculumCfg(cfg);
  const relevant = { ...c, fixture: undefined };
  return hash32(JSON.stringify(relevant) + (c.useFixtureRepo ? fs.readFileSync(path.join(env.repoDir, c.targetFiles[0]), 'utf8') : c.repo.commit));
}

/**
 * All curriculum tasks (holdout first, so train never repeats a holdout mutant). Cached in the
 * git-ignored cache dir; the cache holds third-party-derived code and is never committed.
 */
export async function loadCurriculum(cfg, { onProgress } = {}) {
  const c = curriculumCfg(cfg);
  const env = ensureRepo(cfg);
  const pristine = loadTargets(env, c);
  const key = cacheKey(cfg, env);
  const cacheFile = path.join(rootPath(cfg, c.cacheDir), `${c.name}-${key}-${c.tasksCacheFile}`);
  const cached = readJson(cacheFile, null);
  let tasks;
  let stats = null;
  if (cached?.key === key) {
    tasks = cached.tasks;
  } else {
    const baselineRun = await runSuite(cfg, env, {});
    if (baselineRun.type !== RESULT.PASS) throw new Error(`curriculum: the unmodified ${c.name} test suite does not pass in the sandbox:\n${baselineRun.output.slice(0, c.failureOutputChars)}`);
    const exclude = new Set();
    tasks = [];
    stats = {};
    for (const [split, seedBase, counts] of [['holdout', c.holdoutSeed, c.holdoutCount], ['train', c.trainSeed, c.trainCount]]) {
      for (const level of Object.keys(counts).map(Number).sort()) {
        const { kept, stats: s } = await generateLevel(cfg, env, pristine, { split, level, count: counts[level], seedBase, exclude });
        stats[`${split}-L${level}`] = { kept: kept.length, ...s };
        onProgress?.(split, level, kept.length, s);
        for (const t of kept) {
          const located = locateFunctions(t.cand, pristine.files, c);
          const { cand, ...rest } = t;
          tasks.push({ ...rest, mutatedFiles: cand.mutatedFiles, located });
        }
      }
    }
    writeJson(cacheFile, { key, tasks });
  }
  return { tasks, env, pristine, stats, manifest: tasks.map(manifestEntry) };
}

async function main() {
  const dryRun = process.argv.includes('--dry-run');
  const cfg = loadConfig({ dryRun });
  const c = curriculumCfg(cfg);
  const t0 = Date.now();
  const { tasks, pristine, stats, manifest } = await loadCurriculum(cfg, {
    onProgress: (split, level, kept, s) => log(`${split} level ${level}: kept ${kept} (tried ${s.tried}, survived ${s.survived}, invalid ${s.invalid}, timeouts ${s.timeouts})`),
  });
  const bad = manifest.filter((m) => !reproduce(m, pristine, c));
  if (bad.length) throw new Error(`${bad.length} mutants do not reproduce from their seed, e.g. ${bad[0].id}`);
  const byOp = {};
  for (const t of tasks) for (const op of t.operators) byOp[op] = (byOp[op] ?? 0) + 1;
  log(`${tasks.length} killed mutants from ${c.name} (${pristine.targets.length} target functions) in ${Date.now() - t0} ms; all reproduce from their seeds`);
  log(`operators: ${JSON.stringify(byOp)}`);
  if (stats) log(`generation stats: ${JSON.stringify(stats)}`);
  console.log(JSON.stringify({ repo: c.name, count: tasks.length, manifest }, null, 1));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((e) => { console.error(e?.stack ?? e); process.exit(1); });
}
