// Seeded train/holdout split, persisted in state so it never changes between runs.
import { seededShuffle, hash32, readJson, writeJson } from '../util.js';

export function makeSplit(problemIds, { seed, holdoutSize }) {
  const ids = [...problemIds].sort();
  if (holdoutSize >= ids.length) throw new Error(`holdoutSize ${holdoutSize} >= problem count ${ids.length}`);
  const shuffled = seededShuffle(ids, seed);
  const holdout = shuffled.slice(0, holdoutSize).sort();
  const train = shuffled.slice(holdoutSize).sort();
  return { seed, holdoutSize, datasetHash: hash32(ids.join(',')), holdout, train };
}

/** Load the split from `file`, creating it if missing. Refuses to silently re-split. */
export function loadOrCreateSplit(file, problemIds, splitCfg) {
  const existing = readJson(file, null);
  if (existing) {
    const known = new Set(problemIds);
    const missing = [...existing.holdout, ...existing.train].filter((id) => !known.has(id));
    if (missing.length) throw new Error(`split.json references ${missing.length} unknown problems (dataset changed?). Delete it deliberately to re-split.`);
    return existing;
  }
  const split = makeSplit(problemIds, splitCfg);
  writeJson(file, split);
  return split;
}
