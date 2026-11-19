// Runs untrusted, LLM-written code against unit tests in a child `node` process.
// - fresh temp dir per attempt, deleted afterwards
// - empty environment (no tokens), cwd = temp dir
// - Node permission model (fs reads limited to the temp dir, no fs writes, no child processes/workers)
// - hard timeout that kills the whole process group, capped captured output
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

export const RESULT = Object.freeze({ PASS: 'pass', FAIL: 'fail', ERROR: 'error', TIMEOUT: 'timeout' });

/** Build the program: solution, then the tests in their own block so their `const assert` never clashes. */
export function buildProgram(code, tests) {
  return `${code}\n\n;{\n${tests}\n}\n`;
}

function permissionFlags(dir) {
  // Node >= 22.13 / 23.5: --permission. Older 20.x/22.x: --experimental-permission.
  const [maj, min] = process.versions.node.split('.').map(Number);
  const flag = maj > 23 || (maj === 23 && min >= 5) || (maj === 22 && min >= 13) ? '--permission' : '--experimental-permission';
  return [flag, `--allow-fs-read=${dir}`];
}

export function classify({ code, signal, timedOut, stderr }) {
  if (timedOut) return RESULT.TIMEOUT;
  if (code === 0 && !signal) return RESULT.PASS;
  if (/AssertionError/.test(stderr)) return RESULT.FAIL;
  return RESULT.ERROR;
}

/**
 * @returns {Promise<{type: string, output: string, durationMs: number}>}
 */
export async function runTests(code, tests, judgeCfg) {
  const { timeoutMs, maxOutputBytes, maxMemoryMb, useNodePermissionModel, tmpRoot } = judgeCfg;
  const dir = fs.mkdtempSync(path.join(tmpRoot || os.tmpdir(), 'sic-judge-'));
  const file = path.join(dir, 'solution.cjs');
  fs.writeFileSync(file, buildProgram(code, tests));
  const args = [`--max-old-space-size=${maxMemoryMb}`, ...(useNodePermissionModel ? permissionFlags(dir) : []), file];
  const started = Date.now();
  try {
    return await new Promise((resolve) => {
      const child = spawn(process.execPath, args, { cwd: dir, env: {}, detached: true, stdio: ['ignore', 'pipe', 'pipe'] });
      let out = '';
      let err = '';
      let bytes = 0;
      const take = (which) => (chunk) => {
        if (bytes >= maxOutputBytes) return;
        const s = chunk.toString('utf8').slice(0, maxOutputBytes - bytes);
        bytes += Buffer.byteLength(s);
        if (which === 'out') out += s; else err += s;
      };
      child.stdout.on('data', take('out'));
      child.stderr.on('data', take('err'));
      let timedOut = false;
      const killTree = () => { try { process.kill(-child.pid, 'SIGKILL'); } catch { try { child.kill('SIGKILL'); } catch { /* gone */ } } };
      const timer = setTimeout(() => { timedOut = true; killTree(); }, timeoutMs);
      child.on('error', (e) => { err += `spawn error: ${e.message}`; });
      child.on('close', (code, signal) => {
        clearTimeout(timer);
        killTree(); // reap any stragglers in the group
        const type = classify({ code, signal, timedOut, stderr: err });
        const output = (timedOut ? `Timed out after ${timeoutMs} ms\n` : '') + (err || out);
        resolve({ type, output: output.replace(new RegExp(escapeRe(dir), 'g'), '<tmp>'), durationMs: Date.now() - started });
      });
    });
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
