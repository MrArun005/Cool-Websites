// Downloads MultiPL-E JS problems (HumanEval + MBPP) pinned to a dataset revision,
// normalises them and caches to <cacheDir>/<problemsFile>.
import fs from 'node:fs';
import path from 'node:path';
import { parquetReadObjects } from 'hyparquet';
import { loadConfig, rootPath } from '../config.js';
import { readJson, writeJson, log } from '../util.js';

export function entryPointOf(prompt) {
  const names = [...prompt.matchAll(/function\s+([A-Za-z_$][\w$]*)\s*\(/g)].map((m) => m[1]);
  return names.at(-1);
}

export function normalize(row, source) {
  return {
    id: row.name,
    source,
    prompt: row.prompt,
    tests: row.tests,
    entryPoint: entryPointOf(row.prompt),
  };
}

export async function fetchDataset(cfg, { fetchFn = globalThis.fetch } = {}) {
  const out = [];
  for (const [source, urlTemplate] of Object.entries(cfg.dataset.source)) {
    const url = urlTemplate.replace('{revision}', cfg.dataset.revision);
    log(`fetching ${source} from ${url}`);
    const res = await fetchFn(url, { signal: AbortSignal.timeout(cfg.dataset.fetchTimeoutMs) });
    if (!res.ok) throw new Error(`dataset fetch failed: ${source} HTTP ${res.status}`);
    const buf = await res.arrayBuffer();
    const rows = await parquetReadObjects({ file: buf });
    for (const r of rows) out.push(normalize(r, source));
  }
  const bad = out.filter((p) => !p.entryPoint || !p.tests);
  if (bad.length) throw new Error(`dataset: ${bad.length} problems missing entry point/tests, e.g. ${bad[0].id}`);
  return out;
}

/** Return problems from cache, the test fixture (dry run), or download them. */
export async function loadProblems(cfg, deps = {}) {
  if (cfg.dataset.useFixture) return readJson(rootPath(cfg, cfg.dataset.fixturePath));
  const file = rootPath(cfg, cfg.dataset.cacheDir, cfg.dataset.problemsFile);
  if (fs.existsSync(file)) {
    const cached = readJson(file);
    if (cached.revision === cfg.dataset.revision) return cached.problems;
  }
  const problems = await fetchDataset(cfg, deps);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  writeJson(file, { revision: cfg.dataset.revision, problems });
  return problems;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const cfg = loadConfig();
  const problems = await loadProblems(cfg);
  log(`cached ${problems.length} problems`);
}
