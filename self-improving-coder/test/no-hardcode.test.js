// Phase 1 acceptance: no model names or URLs outside config/.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { ROOT } from '../src/config.js';

const ALLOWED_URLS = ['http://www.w3.org/2000/svg']; // XML namespace, not a network address
const MODEL_PATTERNS = [/\bgpt-[\w.-]+/i, /\bqwen[\w.:-]*/i, /\bllama[\d.:-]+/i, /\bphi-\d/i, /\bmistral-[\w.-]+/i, /\bgemma[\d.:-]+/i, /\bdeepseek-[\w.-]+/i, /\bclaude-[\w.-]+/i];

function files(dir) {
  return fs.readdirSync(dir, { recursive: true }).map((f) => path.join(dir, f)).filter((f) => f.endsWith('.js'));
}

test('src/ contains no URLs or model names', () => {
  const problems = [];
  for (const f of files(path.join(ROOT, 'src'))) {
    const text = fs.readFileSync(f, 'utf8');
    for (const url of text.match(/https?:\/\/[^\s'"`)]+/g) ?? []) if (!ALLOWED_URLS.includes(url)) problems.push(`${path.relative(ROOT, f)}: ${url}`);
    for (const re of MODEL_PATTERNS) {
      const m = text.match(re);
      if (m) problems.push(`${path.relative(ROOT, f)}: model name ${m[0]}`);
    }
  }
  assert.deepEqual(problems, []);
});

test('workflow takes model/runtime values from config, not literals', () => {
  const yml = fs.readFileSync(path.join(ROOT, '.github/workflows/improve.yml'), 'utf8');
  for (const re of MODEL_PATTERNS) assert.doesNotMatch(yml, re);
  assert.doesNotMatch(yml, /ollama\/releases\/download/);
});
