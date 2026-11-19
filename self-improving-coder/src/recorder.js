// Append-only raw logs under each variant's state dir. Nothing here is aggregated: the report,
// chart and dashboard compute everything from these rows.
//   attempts.jsonl  every solve attempt (train and holdout eval) with failure class, tokens, requests
//   traces.jsonl    train attempts in detail (code if the track allows it, output, reflection, lesson)
//   evals.jsonl     raw holdout results: per problem, per repeat 0/1
//   gate-log.jsonl  every gate decision with the numbers behind it
import fs from 'node:fs';
import path from 'node:path';
import { statePath } from './config.js';

export function appendJsonl(file, rows) {
  if (!rows.length) return;
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.appendFileSync(file, rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
}

export function readJsonl(file) {
  if (!fs.existsSync(file)) return [];
  return fs.readFileSync(file, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
}

export function createRecorder(cfg, base) {
  const buf = { attempts: [], traces: [], evals: [], gateLog: [] };
  const trim = (s) => String(s ?? '').slice(0, cfg.recorder.maxOutputChars);
  return {
    attempt(row) { buf.attempts.push({ ...base, ...row }); },
    trace(row) { buf.traces.push({ ...base, ...row, output: trim(row.output) }); },
    evalBatch(row) { buf.evals.push({ ...base, ...row }); },
    gate(row) { buf.gateLog.push({ ...base, ...row }); },
    get attempts() { return buf.attempts; },
    flush() {
      appendJsonl(statePath(cfg, 'attemptsLog'), buf.attempts);
      appendJsonl(statePath(cfg, 'traces'), buf.traces);
      appendJsonl(statePath(cfg, 'evals'), buf.evals);
      appendJsonl(statePath(cfg, 'gateLog'), buf.gateLog);
      for (const k of Object.keys(buf)) buf[k] = [];
    },
  };
}
