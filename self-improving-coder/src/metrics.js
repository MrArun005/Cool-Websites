import fs from 'node:fs';
import path from 'node:path';
import { loadConfig } from './config.js';

let cachedDecimals = null;
function loadDecimals() {
  if (cachedDecimals === null) cachedDecimals = loadConfig({ env: {} }).output.csvDecimals;
  return cachedDecimals;
}

export const COLUMNS = ['timestamp', 'runId', 'split', 'passRate', 'n', 'promptVersion', 'lessonsCount', 'skillsCount', 'requestsUsed', 'gateDecision', 'variant', 'track', 'repeats', 'sd', 'ciLow', 'ciHigh', 'tokensIn', 'tokensOut'];
const NUMERIC = ['passRate', 'n', 'lessonsCount', 'skillsCount', 'requestsUsed', 'repeats', 'sd', 'ciLow', 'ciHigh', 'tokensIn', 'tokensOut'];
const FIXED = ['passRate', 'sd', 'ciLow', 'ciHigh'];

const esc = (v) => {
  const s = String(v ?? '');
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

export function appendScores(file, rows, decimals = loadDecimals()) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const header = fs.existsSync(file) ? '' : COLUMNS.join(',') + '\n';
  const fmt = (c, v) => (FIXED.includes(c) && v !== '' && v !== undefined && v !== null && Number.isFinite(Number(v)) ? Number(v).toFixed(decimals) : v);
  const body = rows.map((r) => COLUMNS.map((c) => esc(fmt(c, r[c]))).join(',')).join('\n');
  fs.appendFileSync(file, header + body + '\n');
}

export function readScores(file) {
  if (!fs.existsSync(file)) return [];
  const lines = fs.readFileSync(file, 'utf8').trim().split('\n');
  const head = lines.shift()?.split(',') ?? [];
  return lines.filter(Boolean).map((line) => {
    const vals = line.split(','); // our values never contain commas
    const row = Object.fromEntries(head.map((h, i) => [h, vals[i]]));
    for (const k of NUMERIC) if (k in row) row[k] = row[k] === '' || row[k] === undefined ? null : Number(row[k]);
    return row;
  });
}
