// Quality bar: no literal URLs, model names or numeric thresholds outside config/.
// Numbers are found with a real JS parser (acorn). Allowed: 0, 1, 2 (identity/halves/pairs), 100 (percent),
// or a literal on a line carrying an `// allow-number: <reason>` comment explaining why it is not a tunable.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import * as acorn from 'acorn';
import { ROOT } from '../src/config.js';

const ALLOWED_URLS = ['http://www.w3.org/2000/svg']; // XML namespace, not a network address
const MODEL_PATTERNS = [/\bgpt-[\w.-]+/i, /\bqwen[\w.:-]*/i, /\bllama[\d.:-]+/i, /\bphi-\d/i, /\bmistral-[\w.-]+/i, /\bgemma[\d.:-]+/i, /\bdeepseek-[\w.-]+/i, /\bclaude-[\w.-]+/i];
const ALLOWED_NUMBERS = new Set([0, 1, 2, 100]);
const SOURCE_DIRS = ['src', 'site'];

function sources() {
  return SOURCE_DIRS.flatMap((d) => fs.readdirSync(path.join(ROOT, d), { recursive: true }).map((f) => path.join(ROOT, d, f)))
    .filter((f) => f.endsWith('.js') && !f.includes(`${path.sep}node_modules${path.sep}`));
}

test('source contains no URLs or model names', () => {
  const problems = [];
  for (const f of sources()) {
    const text = fs.readFileSync(f, 'utf8');
    for (const url of text.match(/https?:\/\/[^\s'"`)]+/g) ?? []) if (!ALLOWED_URLS.includes(url)) problems.push(`${path.relative(ROOT, f)}: ${url}`);
    for (const re of MODEL_PATTERNS) {
      const m = text.match(re);
      if (m) problems.push(`${path.relative(ROOT, f)}: model name ${m[0]}`);
    }
  }
  assert.deepEqual(problems, []);
});

test('source contains no numeric thresholds or magic numbers (parsed with acorn)', () => {
  const problems = [];
  for (const f of sources()) {
    const text = fs.readFileSync(f, 'utf8');
    const lines = text.split('\n');
    const allowedLines = new Set();
    const ast = acorn.parse(text, { ecmaVersion: 'latest', sourceType: 'module', locations: true, allowHashBang: true,
      onComment: (block, value, start, end, startLoc) => { if (/allow-number:\s*\S/.test(value)) allowedLines.add(startLoc.line); } });
    const visit = (node) => {
      if (!node || typeof node.type !== 'string') return;
      if (node.type === 'Literal' && typeof node.value === 'number' && !ALLOWED_NUMBERS.has(node.value) && !allowedLines.has(node.loc.start.line)) {
        problems.push(`${path.relative(ROOT, f)}:${node.loc.start.line}: ${node.raw}  | ${lines[node.loc.start.line - 1].trim().slice(0, 90)}`);
      }
      for (const k of Object.keys(node)) {
        if (k === 'loc') continue;
        const v = node[k];
        if (Array.isArray(v)) v.forEach(visit); else if (v && typeof v.type === 'string') visit(v);
      }
    };
    visit(ast);
  }
  assert.deepEqual(problems, [], `\n${problems.join('\n')}`);
});

test('workflow takes model/runtime values and the variant list from config, not literals', () => {
  for (const wf of ['improve.yml', 'ci.yml']) {
    const yml = fs.readFileSync(path.join(ROOT, '.github/workflows', wf), 'utf8');
    for (const re of MODEL_PATTERNS) assert.doesNotMatch(yml, re);
    assert.doesNotMatch(yml, /ollama\/releases\/download/);
    assert.doesNotMatch(yml, /lessons-skills|mutation-baseline/);
  }
});
