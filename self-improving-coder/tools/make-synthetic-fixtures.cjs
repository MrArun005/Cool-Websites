// Generates test/fixtures/synthetic-problems.json: small JS problems grouped in "families" that share a
// typical bug, for the synthetic mock (npm run reproduce, dashboard/report fixtures).
// Every correct solution is checked to pass and every buggy one to fail its tests.
// Usage: node tools/make-synthetic-fixtures.cjs
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const T = (name, cases) => `const assert = require('node:assert');\n\n\nfunction test() {\n  let candidate = ${name};\n${cases.map(([args, out]) => `  assert.deepEqual(candidate(${args}), ${JSON.stringify(out)});`).join('\n')}\n}\n\ntest();`;
const P = (doc, sig) => doc.split('\n').map((l) => '// ' + l).join('\n') + '\n' + sig + '{\n';
const out = [];
const add = (family, id, doc, sig, name, correct, buggy, cases) => out.push({ id: `syn_${family}_${id}`, source: 'synthetic', family, entryPoint: name, prompt: P(doc, sig), tests: T(name, cases), mock: { family, correct, buggy } });

// 1. inclusive ranges (bug: exclusive upper bound)
const terms = [['i', 'the integers'], ['i * i', 'the squares of the integers'], ['2 * i', 'twice the integers'], ['i * i * i', 'the cubes of the integers'], ['i % 3', 'the remainders mod 3 of the integers'], ['i + 1', 'the successors of the integers']];
terms.forEach(([t, desc], k) => {
  const name = `sum_range_${k}`;
  const f = (a, b) => { let s = 0; for (let i = a; i <= b; i++) s += eval(t); return s; };
  add('range', `sum${k}`, `Return the sum of ${desc} in the inclusive range from a to b.`, `function ${name}(a, b)`, name,
    `function ${name}(a, b){\n  let s = 0;\n  for (let i = a; i <= b; i++) s += ${t};\n  return s;\n}`,
    `function ${name}(a, b){\n  let s = 0;\n  for (let i = a; i < b; i++) s += ${t};\n  return s;\n}`,
    [['1, 4', f(1, 4)], ['3, 3', f(3, 3)], ['2, 6', f(2, 6)]]);
});
[[2, 'even'], [3, 'multiple of 3'], [5, 'multiple of 5'], [4, 'multiple of 4']].forEach(([m, desc], k) => {
  const name = `count_range_${k}`;
  const f = (a, b) => { let c = 0; for (let i = a; i <= b; i++) if (i % m === 0) c++; return c; };
  add('range', `count${k}`, `Count the integers in the inclusive range from a to b that are ${desc}.`, `function ${name}(a, b)`, name,
    `function ${name}(a, b){\n  let c = 0;\n  for (let i = a; i <= b; i++) if (i % ${m} === 0) c++;\n  return c;\n}`,
    `function ${name}(a, b){\n  let c = 0;\n  for (let i = a; i < b; i++) if (i % ${m} === 0) c++;\n  return c;\n}`,
    [[`${m}, ${m * 3}`, f(m, m * 3)], [`1, ${m}`, f(1, m)], [`${m * 2}, ${m * 2}`, 1]]);
});

// 2. running extremes (bug: start from 0 instead of the first element)
[['max', '>', 'largest'], ['min', '<', 'smallest']].forEach(([kind, op, word]) => {
  [['', 'number'], ['Math.abs', 'absolute value'], ['-', 'negated value'], ['2 *', 'doubled value']].forEach(([fx, d], k) => {
    const name = `${kind}_of_${k}`;
    const g = (x) => (fx === 'Math.abs' ? Math.abs(x) : fx === '-' ? -x : fx === '2 *' ? 2 * x : x);
    const expr = fx === 'Math.abs' ? 'Math.abs(x)' : fx === '-' ? '-x' : fx === '2 *' ? '2 * x' : 'x';
    const pick = (xs) => xs.map(g).reduce((a, b) => (kind === 'max' ? Math.max(a, b) : Math.min(a, b)));
    const pool = [[-5, -2, -9], [-1], [-4, -7], [3, 8, 1], [7], [6, 2]];
    // keep inputs where starting from 0 is wrong: all transformed values < 0 for max, > 0 for min
    const cases = pool.filter((xs) => xs.map(g).every((v) => (kind === 'max' ? v < 0 : v > 0)));
    if (cases.length < 2) return; // this transform can't expose the bug (e.g. max of absolute values)
    add('extreme', `${kind}${k}`, `Return the ${word} ${d} of the numbers in a non-empty list.`, `function ${name}(xs)`, name,
      `function ${name}(xs){\n  let best = ${expr.replace(/x/g, 'xs[0]')};\n  for (const x of xs) { const v = ${expr}; if (v ${op} best) best = v; }\n  return best;\n}`,
      `function ${name}(xs){\n  let best = 0;\n  for (const x of xs) { const v = ${expr}; if (v ${op} best) best = v; }\n  return best;\n}`,
      cases.map((xs) => [JSON.stringify(xs), pick(xs)]));
  });
});

// 3. empty input (bug: crashes / NaN on empty lists)
[['average', 'the average of the numbers', 'xs.reduce((a, b) => a + b, 0) / xs.length', 0, [[[2, 4], 3], [[5], 5]]],
 ['first_or_none', 'the first element', 'xs[0]', 'none', [[[7, 1], 7], [['a'], 'a']]],
 ['last_or_none', 'the last element', 'xs[xs.length - 1]', 'none', [[[7, 1], 1], [['a', 'b'], 'b']]],
 ['product', 'the product of the numbers', 'xs.reduce((a, b) => a * b)', 1, [[[2, 3], 6], [[4], 4]]],
 ['longest_word', 'the longest word', "xs.reduce((a, b) => (b.length > a.length ? b : a))", '', [[['ab', 'abc'], 'abc'], [['x'], 'x']]],
 ['max_len', 'the length of the longest string', 'Math.max(...xs.map((s) => s.length))', 0, [[['ab', 'abc'], 3], [['x'], 1]]],
].forEach(([name, desc, expr, dflt, cases]) => {
  add('empty', name, `Return ${desc} in the list. For an empty list return ${JSON.stringify(dflt)}.`, `function ${name}(xs)`, name,
    `function ${name}(xs){\n  if (xs.length === 0) return ${JSON.stringify(dflt)};\n  return ${expr};\n}`,
    `function ${name}(xs){\n  return ${name === 'first_or_none' || name === 'last_or_none' ? expr : expr.replace('reduce((a, b) => a * b)', 'reduce((a, b) => a * b)')};\n}`,
    [...cases.map(([xs, v]) => [JSON.stringify(xs), v]), ['[]', dflt]]);
});

// 4. case-insensitive text (bug: compares raw case)
[['count_char', 'Count how many times the letter c appears in s, ignoring upper/lower case.', '(s, c)', "[...s.toLowerCase()].filter((x) => x === c.toLowerCase()).length", "[...s].filter((x) => x === c).length", [['"Banana", "b"', 1], ['"aAa", "a"', 3]]],
 ['same_word', 'Return true if the two words are equal ignoring upper/lower case.', '(a, b)', 'a.toLowerCase() === b.toLowerCase()', 'a === b', [['"Cat", "cAT"', true], ['"dog", "cat"', false]]],
 ['starts_with', 'Return true if s starts with prefix p, ignoring upper/lower case.', '(s, p)', 's.toLowerCase().startsWith(p.toLowerCase())', 's.startsWith(p)', [['"Hello", "he"', true], ['"abc", "b"', false]]],
 ['count_vowels', 'Count the vowels a, e, i, o, u in s, ignoring upper/lower case.', '(s)', "[...s.toLowerCase()].filter((x) => 'aeiou'.includes(x)).length", "[...s].filter((x) => 'aeiou'.includes(x)).length", [['"AbE"', 2], ['"xyz"', 0]]],
 ['unique_words', 'Count the distinct words in a space-separated sentence, ignoring upper/lower case.', '(s)', "new Set(s.toLowerCase().split(' ')).size", "new Set(s.split(' ')).size", [['"a A b"', 2], ['"x y"', 2]]],
 ['is_palindrome_ci', 'Return true if s reads the same backwards, ignoring upper/lower case.', '(s)', "s.toLowerCase() === [...s.toLowerCase()].reverse().join('')", "s === [...s].reverse().join('')", [['"Abba"', true], ['"abc"', false]]],
].forEach(([name, doc, params, good, bad, cases]) => {
  add('case', name, doc, `function ${name}${params}`, name, `function ${name}${params}{\n  return ${good};\n}`, `function ${name}${params}{\n  return ${bad};\n}`, cases);
});

// 5. rounding (bug: truncates instead of rounding)
[['round_half', 'Return x rounded to the nearest integer (halves round up).', 'Math.round(x)', 'Math.floor(x)', [['2.7', 3], ['2.2', 2], ['-1.5', -1]]],
 ['round_tenth', 'Return x rounded to one decimal place.', 'Math.round(x * 10) / 10', 'Math.floor(x * 10) / 10', [['2.46', 2.5], ['1.04', 1]]],
 ['avg_rounded', 'Return the average of a and b rounded to the nearest integer.', 'Math.round((a + b) / 2)', 'Math.floor((a + b) / 2)', [['1, 2', 2], ['2, 4', 3]]],
 ['percent', 'Return part / whole as a whole-number percentage, rounded to the nearest integer.', 'Math.round((part / whole) * 100)', 'Math.floor((part / whole) * 100)', [['2, 3', 67], ['1, 4', 25]]],
].forEach(([name, doc, good, bad, cases]) => {
  const params = /a \+ b/.test(good) ? '(a, b)' : /part/.test(good) ? '(part, whole)' : '(x)';
  add('round', name, doc, `function ${name}${params}`, name, `function ${name}${params}{\n  return ${good};\n}`, `function ${name}${params}{\n  return ${bad};\n}`, cases);
});

// 6. easy problems (no shared bug; high base pass rate)
[['double_all', 'Double every number in a list.', '(xs)', 'xs.map((x) => x * 2)', [['[1, 2]', [2, 4]]]],
 ['reverse_words', 'Reverse the order of words in a sentence.', '(s)', "s.split(' ').reverse().join(' ')", [['"a b c"', 'c b a']]],
 ['join_dash', 'Join a list of words with dashes.', '(xs)', "xs.join('-')", [['["a", "b"]', 'a-b']]],
 ['is_even', 'Return true if n is even.', '(n)', 'n % 2 === 0', [['4', true], ['3', false]]],
 ['square', 'Return n squared.', '(n)', 'n * n', [['3', 9]]],
 ['char_count', 'Return the number of characters in s.', '(s)', 's.length', [['"abc"', 3]]],
].forEach(([name, doc, params, good, cases]) => {
  add('easy', name, doc, `function ${name}${params}`, name, `function ${name}${params}{\n  return ${good};\n}`, `function ${name}${params}{\n  return null;\n}`, cases);
});

// verify
const tmp = fs.mkdtempSync(path.join(require('os').tmpdir(), 'synfx-'));
for (const p of out) for (const k of ['correct', 'buggy']) {
  const f = path.join(tmp, 'x.cjs');
  fs.writeFileSync(f, p.mock[k] + '\n;{\n' + p.tests + '\n}\n');
  let ok; try { execFileSync(process.execPath, [f], { stdio: 'pipe', timeout: 3000 }); ok = true; } catch { ok = false; }
  if (ok !== (k === 'correct')) throw new Error(`fixture ${p.id}: ${k} solution ${ok ? 'passes' : 'fails'}`);
}
fs.rmSync(tmp, { recursive: true, force: true });
const dest = path.join(__dirname, '..', 'test', 'fixtures', 'synthetic-problems.json');
fs.writeFileSync(dest, JSON.stringify(out, null, 1) + '\n');
console.log(`wrote ${out.length} problems in ${new Set(out.map((p) => p.family)).size} families -> ${path.relative(process.cwd(), dest)}`);
