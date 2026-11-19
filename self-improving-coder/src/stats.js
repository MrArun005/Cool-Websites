// Small, dependency-free statistics: mean, sample standard deviation, quantiles,
// and seeded percentile-bootstrap confidence intervals.
import { rng } from './util.js';

export function mean(xs) {
  if (!xs.length) return NaN;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

/** Sample standard deviation (n - 1 denominator). NaN for fewer than two values. */
export function sd(xs) {
  if (xs.length < 2) return NaN;
  const m = mean(xs);
  let s = 0;
  for (const x of xs) s += (x - m) ** 2;
  return Math.sqrt(s / (xs.length - 1));
}

/** Quantile with linear interpolation between order statistics (Hyndman & Fan type 7, R's default). */
export function quantile(xs, q) {
  if (!xs.length) return NaN;
  const a = [...xs].sort((x, y) => x - y);
  const h = (a.length - 1) * q;
  const lo = Math.floor(h);
  const hi = Math.ceil(h);
  return a[lo] + (h - lo) * (a[hi] - a[lo]);
}

/**
 * Percentile bootstrap CI for a statistic of `xs` (default: the mean).
 * @param {number[]} xs
 * @param {{resamples: number, confidence: number, seed: number, statistic?: (xs:number[])=>number}} opts
 */
export function bootstrapCI(xs, { resamples, confidence, seed, statistic = mean }) {
  if (!xs.length) return { low: NaN, high: NaN, estimate: NaN };
  const r = rng(seed);
  const n = xs.length;
  const stats = new Array(resamples);
  const buf = new Array(n);
  for (let b = 0; b < resamples; b++) {
    for (let i = 0; i < n; i++) buf[i] = xs[Math.floor(r() * n)];
    stats[b] = statistic(buf);
  }
  const alpha = (1 - confidence) / 2;
  return { low: quantile(stats, alpha), high: quantile(stats, 1 - alpha), estimate: statistic(xs) };
}

/**
 * Paired bootstrap CI of mean(b - a) where a[i], b[i] are scores on the same item
 * (e.g. champion and candidate pass fractions on the same holdout problem).
 */
export function pairedDiffCI(a, b, opts) {
  if (a.length !== b.length) throw new Error('pairedDiffCI: arrays must have equal length');
  return bootstrapCI(b.map((v, i) => v - a[i]), opts);
}

/**
 * Summarise raw eval results: `perProblem` = array of arrays of 0/1 (one inner array per problem,
 * one entry per repeat). Bootstrap resamples problems (the unit that varies), not single attempts.
 */
export function summarizeEval(perProblem, opts) {
  const problemMeans = perProblem.map(mean);
  const repeats = Math.max(0, ...perProblem.map((r) => r.length));
  const perRepeat = [];
  for (let k = 0; k < repeats; k++) perRepeat.push(mean(perProblem.map((r) => r[k]).filter((v) => v !== undefined)));
  const ci = bootstrapCI(problemMeans, opts);
  return { passRate: mean(problemMeans), n: perProblem.length, repeats, sdAcrossRepeats: sd(perRepeat), ciLow: ci.low, ciHigh: ci.high, problemMeans };
}
