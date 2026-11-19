// Phase 6 acceptance: the confidence-interval code is checked against values with known answers.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mean, sd, quantile, bootstrapCI, pairedDiffCI, summarizeEval } from '../src/stats.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} expected ${b} ± ${tol}, got ${a}`);

test('mean and sample standard deviation (textbook values)', () => {
  const xs = [2, 4, 4, 4, 5, 5, 7, 9];
  assert.equal(mean(xs), 5);
  close(sd(xs), Math.sqrt(32 / 7), 1e-12); // sum of squares 32, n - 1 = 7 -> 2.138...
  assert.ok(Number.isNaN(sd([1])));
});

test('quantile matches R type 7', () => {
  const xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  assert.equal(quantile(xs, 0), 1);
  assert.equal(quantile(xs, 1), 10);
  close(quantile(xs, 0.25), 3.25, 1e-12); // R: quantile(1:10, .25) = 3.25
  close(quantile(xs, 0.9), 9.1, 1e-12); // R: quantile(1:10, .9) = 9.1
});

test('bootstrap CI of a constant sample is that constant', () => {
  const ci = bootstrapCI([0.7, 0.7, 0.7, 0.7], { resamples: 500, confidence: 0.95, seed: 1 });
  assert.deepEqual([ci.low, ci.high, ci.estimate], [0.7, 0.7, 0.7]);
});

test('bootstrap CI of {0,1}: exact resampling distribution is {0: .25, .5: .5, 1: .25}', () => {
  const ci = bootstrapCI([0, 1], { resamples: 20000, confidence: 0.95, seed: 7 });
  assert.equal(ci.low, 0);
  assert.equal(ci.high, 1);
});

test('bootstrap CI of 50 zeros + 50 ones matches the Binomial(100, 0.5)/100 quantiles (0.40, 0.60)', () => {
  // The bootstrap mean of this sample is exactly Binomial(100, .5)/100; its 2.5% / 97.5% quantiles are 40 and 60.
  const xs = [...Array(50).fill(0), ...Array(50).fill(1)];
  const ci = bootstrapCI(xs, { resamples: 20000, confidence: 0.95, seed: 3 });
  close(ci.low, 0.40, 0.011, 'low');
  close(ci.high, 0.60, 0.011, 'high');
  close(ci.estimate, 0.5, 1e-12);
});

test('bootstrap CI is reproducible from its seed and changes with it', () => {
  const xs = [0.1, 0.4, 0.35, 0.8, 0.9, 0.2, 0.55];
  const o = { resamples: 2000, confidence: 0.9 };
  assert.deepEqual(bootstrapCI(xs, { ...o, seed: 5 }), bootstrapCI(xs, { ...o, seed: 5 }));
  assert.notDeepEqual(bootstrapCI(xs, { ...o, seed: 5 }), bootstrapCI(xs, { ...o, seed: 6 }));
});

test('paired difference CI excludes 0 for a consistent improvement and includes it for none', () => {
  const a = Array.from({ length: 40 }, (_, i) => (i % 2 ? 1 : 0));
  const better = a.map((v, i) => (i % 4 === 0 ? 1 : v)); // fixes 10 of the 20 failures
  const ci = pairedDiffCI(a, better, { resamples: 4000, confidence: 0.95, seed: 2 });
  close(ci.estimate, 0.25, 1e-12);
  assert.ok(ci.low > 0);
  const same = pairedDiffCI(a, a, { resamples: 1000, confidence: 0.95, seed: 2 });
  assert.deepEqual([same.low, same.high], [0, 0]);
});

test('summarizeEval averages repeats per problem and reports spread across repeats', () => {
  const s = summarizeEval([[1, 1, 0], [0, 0, 0], [1, 1, 1], [0, 1, 0]], { resamples: 1000, confidence: 0.95, seed: 1 });
  close(s.passRate, (2 / 3 + 0 + 1 + 1 / 3) / 4, 1e-12);
  assert.equal(s.n, 4);
  assert.equal(s.repeats, 3);
  close(s.sdAcrossRepeats, sd([0.5, 0.75, 0.25]), 1e-12);
  assert.ok(s.ciLow <= s.passRate && s.passRate <= s.ciHigh);
});
