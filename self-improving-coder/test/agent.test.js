import { test } from 'node:test';
import assert from 'node:assert/strict';
import { extractCode, buildMessages } from '../src/agent/solve.js';
import { parseLesson, clampWords } from '../src/agent/reflect.js';
import { retrieveLessons, retrieveSkills, tokenize } from '../src/agent/retrieve.js';
import { testConfig } from './helpers.js';

const problem = { id: 'p', entryPoint: 'add', prompt: '// Add two numbers\nfunction add(a, b){\n' };

test('extractCode: fenced full function', () => {
  assert.equal(extractCode('blah\n```javascript\nfunction add(a, b){ return a + b; }\n```\nmore', problem), 'function add(a, b){ return a + b; }\n');
});

test('extractCode: prefers the fence that defines the entry point', () => {
  const t = '```js\nconst x = 1;\n```\n```js\nfunction add(a,b){return a+b}\n```';
  assert.match(extractCode(t, problem), /function add/);
});

test('extractCode: body-only answer is wrapped in the prompt', () => {
  const code = extractCode('```\n  return a + b;\n```', problem);
  assert.equal(code, '// Add two numbers\nfunction add(a, b){\n  return a + b;\n\n}\n');
});

test('extractCode: unterminated fence and arrow function', () => {
  assert.match(extractCode('```javascript\nconst add = (a, b) => a + b;', problem), /^const add/);
});

test('parseLesson handles JSON, fenced JSON and plain text', () => {
  assert.deepEqual(parseLesson('{"lesson":"Use <=","tags":["range"]}'), { text: 'Use <=', tags: ['range'] });
  assert.deepEqual(parseLesson('```json\n{"lesson": "Check empty input", "tags": []}\n```'), { text: 'Check empty input', tags: [] });
  assert.deepEqual(parseLesson('Always check bounds.\nmore'), { text: 'Always check bounds.', tags: [] });
  assert.equal(parseLesson('   '), null);
  assert.equal(clampWords('a b c d', 2), 'a b…');
});

test('retrieval ranks lessons by keyword overlap and respects topK', () => {
  const cfg = testConfig({ agent: { lessonsTopK: 1 } });
  const lessons = [
    { id: 'L1', text: 'Sort before comparing', tags: ['sorting'], helped: 0 },
    { id: 'L2', text: 'Include both ends of an inclusive range', tags: ['range', 'inclusive'], helped: 0 },
  ];
  const p = { id: 'x', entryPoint: 'sum_range', prompt: '// Sum the inclusive range from a to b\nfunction sum_range(a, b){\n' };
  assert.deepEqual(retrieveLessons(p, lessons, cfg).map((l) => l.id), ['L2']);
  assert.ok(tokenize('sumRange of_numbers', cfg.retrieve).has('sum'));
});

test('a problem never retrieves its own skill', () => {
  const cfg = testConfig();
  const p = { id: 'x', entryPoint: 'sum_range', prompt: '// Sum range\nfunction sum_range(a, b){\n' };
  const skills = [{ id: 'x', keywords: ['sum', 'range'], code: '' }, { id: 'y', keywords: ['sum', 'range'], code: '' }];
  assert.deepEqual(retrieveSkills(p, skills, cfg).map((s) => s.id), ['y']);
});

test('prompt contains lessons and skills sections only when present', () => {
  const cfg = testConfig();
  const tpl = '{{lessons}}{{skills}}## Problem\n{{prompt}}';
  const none = buildMessages({ systemPrompt: 'S', userTemplate: tpl, problem, lessons: [], skills: [], cfg });
  assert.doesNotMatch(none[1].content, /Lessons/);
  const some = buildMessages({ systemPrompt: 'S', userTemplate: tpl, problem, lessons: [{ text: 'L!' }], skills: [{ code: 'function z(){}' }], cfg });
  assert.match(some[1].content, /- L!/);
  assert.match(some[1].content, /function z/);
});
