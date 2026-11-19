// Loads config/default.json, applies SIC_* env overrides and (optionally) the dryRun overlay.
// Env override syntax: SIC_<PATH> where PATH is the config key path in upper snake case,
// joined by double underscores, e.g. SIC_LLM__MODEL=<model-id>, SIC_BUDGET__REQUESTS_PER_RUN=100.
// Values are parsed as JSON when possible (numbers, booleans, arrays), otherwise kept as strings.
// Empty values are ignored (so unset GitHub repo variables do not blank out config).
// SIC_CONFIG_OVERRIDES may hold a JSON object that is deep-merged over the config (handy as one repo variable).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_PATH = path.join(ROOT, 'config', 'default.json');
const ENV_PREFIX = 'SIC_';

export function deepMerge(base, over) {
  if (!isPlainObject(base) || !isPlainObject(over)) return over === undefined ? base : over;
  const out = { ...base };
  for (const [k, v] of Object.entries(over)) out[k] = isPlainObject(v) && isPlainObject(base[k]) ? deepMerge(base[k], v) : v;
  return out;
}

function isPlainObject(v) {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

const camelToSnake = (s) => s.replace(/([a-z0-9])([A-Z])/g, '$1_$2').toUpperCase();

function parseValue(raw) {
  try { return JSON.parse(raw); } catch { return raw; }
}

// Walk the config tree and look up an env var for every leaf (and every object, for JSON blobs).
export function applyEnvOverrides(cfg, env, prefix = ENV_PREFIX) {
  const walk = (obj, keyPath) => {
    for (const [k, v] of Object.entries(obj)) {
      const p = [...keyPath, k];
      const name = prefix + p.map(camelToSnake).join('__');
      if (Object.prototype.hasOwnProperty.call(env, name) && env[name] !== '') {
        obj[k] = parseValue(env[name]);
      } else if (isPlainObject(v)) {
        walk(v, p);
      }
    }
  };
  const copy = structuredClone(cfg);
  walk(copy, []);
  return copy;
}

const REQUIRED = [
  ['llm.provider', 'string'], ['llm.model', 'string'], ['llm.endpoint', 'string'],
  ['llm.temperature', 'number'], ['llm.maxTokens', 'number'],
  ['llm.retry.maxAttempts', 'number'], ['llm.retry.baseDelayMs', 'number'], ['llm.retry.maxDelayMs', 'number'],
  ['budget.totalRequestsPerRun', 'number'], ['budget.totalRequestsPerDay', 'number'], ['budget.holdoutFraction', 'number'],
  ['budget.maxRunMinutes', 'number'], ['budget.reserveMinutesForHoldout', 'number'],
  ['dataset.cacheDir', 'string'], ['split.seed', 'number'], ['split.holdoutSize', 'number'],
  ['agent.maxAttemptsPerProblem', 'number'], ['agent.lessonsTopK', 'number'], ['agent.skillsTopK', 'number'],
  ['agent.maxLessons', 'number'], ['judge.timeoutMs', 'number'], ['judge.maxOutputBytes', 'number'],
  ['gate.minImprovement', 'number'], ['gate.tolerance', 'number'], ['gate.baselineRuns', 'number'],
  ['paths.stateDir', 'string'], ['features.promptOptimizer', 'boolean'],
];

export function get(obj, dotted) {
  return dotted.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);
}

export function validate(cfg) {
  const errors = [];
  for (const [key, type] of REQUIRED) {
    const v = get(cfg, key);
    if (typeof v !== type || (type === 'number' && !Number.isFinite(v))) errors.push(`${key} must be a ${type} (got ${JSON.stringify(v)})`);
  }
  if (!['openai-compatible', 'mock'].includes(cfg.llm?.provider)) errors.push(`llm.provider must be "openai-compatible" or "mock"`);
  if (cfg.budget && !(cfg.budget.holdoutFraction >= 0 && cfg.budget.holdoutFraction <= 1)) errors.push('budget.holdoutFraction must be in [0, 1]');
  if (!Array.isArray(cfg.variants) || !cfg.variants.length) errors.push('variants must be a non-empty array');
  for (const v of cfg.variants ?? []) if (!cfg.tracks?.[v.track]) errors.push(`variant ${v.name}: unknown track ${v.track}`);
  if (cfg.gate && cfg.gate.tolerance < 0) errors.push('gate.tolerance must be >= 0');
  if (errors.length) throw new Error('Invalid config:\n  ' + errors.join('\n  '));
  return cfg;
}

/**
 * @param {{dryRun?: boolean, env?: object, overrides?: object, file?: string}} opts
 */
export function loadConfig(opts = {}) {
  const env = opts.env ?? process.env;
  let cfg = JSON.parse(fs.readFileSync(opts.file ?? DEFAULT_PATH, 'utf8'));
  if (opts.dryRun) cfg = deepMerge(cfg, cfg.dryRun ?? {});
  if (env[ENV_PREFIX + 'CONFIG_OVERRIDES']) cfg = deepMerge(cfg, JSON.parse(env[ENV_PREFIX + 'CONFIG_OVERRIDES']));
  cfg = applyEnvOverrides(cfg, env);
  if (opts.overrides) cfg = deepMerge(cfg, opts.overrides);
  cfg.root = opts.root ?? ROOT;
  return validate(cfg);
}

/** Resolve a state file path (relative to stateDir) to an absolute path. */
export function statePath(cfg, key) {
  return path.resolve(cfg.root, cfg.paths.stateDir, cfg.paths[key]);
}

export function rootPath(cfg, ...parts) {
  return path.resolve(cfg.root, ...parts);
}
