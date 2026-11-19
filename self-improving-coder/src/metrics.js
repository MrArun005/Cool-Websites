import fs from 'node:fs';
import path from 'node:path';

export const COLUMNS = ['timestamp', 'runId', 'split', 'passRate', 'n', 'promptVersion', 'lessonsCount', 'skillsCount', 'requestsUsed', 'gateDecision'];

const esc = (v) => {
  const s = String(v ?? '');
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

export function appendScores(file, rows) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const header = fs.existsSync(file) ? '' : COLUMNS.join(',') + '\n';
  const body = rows.map((r) => COLUMNS.map((c) => esc(c === 'passRate' ? Number(r.passRate).toFixed(4) : r[c])).join(',')).join('\n');
  fs.appendFileSync(file, header + body + '\n');
}

export function readScores(file) {
  if (!fs.existsSync(file)) return [];
  const lines = fs.readFileSync(file, 'utf8').trim().split('\n');
  const head = lines.shift()?.split(',') ?? [];
  return lines.filter(Boolean).map((line) => {
    const vals = line.split(','); // our values never contain commas
    const row = Object.fromEntries(head.map((h, i) => [h, vals[i]]));
    row.passRate = Number(row.passRate);
    row.n = Number(row.n);
    return row;
  });
}
