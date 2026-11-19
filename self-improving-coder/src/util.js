import fs from 'node:fs';
import path from 'node:path';

export function readJson(file, fallback) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); } catch (e) {
    if (e.code === 'ENOENT' && fallback !== undefined) return structuredClone(fallback);
    throw e;
  }
}

export function writeJson(file, data) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2) + '\n');
  fs.renameSync(tmp, file);
}

export function writeText(file, text) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, text);
}

/** mulberry32: small, fast, seedable PRNG. */
export function rng(seed) {
  // mulberry32 constants (fixed by the algorithm, not tunables)
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0; // allow-number: mulberry32 constant
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1); // allow-number: mulberry32 constant
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); // allow-number: mulberry32 constants
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296; // allow-number: mulberry32 constant, 2^32
  };
}

export function seededShuffle(items, seed) {
  const r = rng(seed);
  const a = [...items];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** Stable 32-bit string hash (FNV-1a), used to derive seeds from strings. */
export function hash32(str) {
  let h = 0x811c9dc5; // allow-number: FNV-1a offset basis
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193); // allow-number: FNV-1a prime
  }
  return h >>> 0;
}

export function fill(template, vars) {
  return template.replace(/\{\{(\w+)\}\}/g, (_, k) => (vars[k] ?? ''));
}

/** Length of an ISO date "YYYY-MM-DD". */
export const ISO_DATE_LEN = 10; // allow-number: date format, not a tunable
export const MS_PER_MINUTE = 60000; // allow-number: unit conversion
export const MS_PER_SECOND = 1000; // allow-number: unit conversion
export const isoDay = (ts) => (ts ? String(ts).slice(0, ISO_DATE_LEN) : 'n/a');

export function today(now = new Date()) {
  return now.toISOString().slice(0, ISO_DATE_LEN);
}

/** --key=value command-line argument. */
export function argValue(argv, key) {
  const prefix = `--${key}=`;
  return argv.find((a) => a.startsWith(prefix))?.slice(prefix.length);
}

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function log(...args) {
  if (process.env.SIC_QUIET !== '1') console.log('[sic]', ...args);
}
