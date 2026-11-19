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
 * one entry per repeat). `method`: 'problems' resamples problem means; 'hierarchical' also resamples repeats.
 */
export function summarizeEval(perProblem, opts, method = 'problems') {
  const problemMeans = perProblem.map(mean);
  const repeats = Math.max(0, ...perProblem.map((r) => r.length));
  const perRepeat = [];
  for (let k = 0; k < repeats; k++) perRepeat.push(mean(perProblem.map((r) => r[k]).filter((v) => v !== undefined)));
  const ci = groupCI(perProblem, opts, method);
  return { passRate: mean(problemMeans), n: perProblem.length, repeats, sdAcrossRepeats: sd(perRepeat), ciLow: ci.low, ciHigh: ci.high, problemMeans };
}

/**
 * Two-level (hierarchical) bootstrap for repeated measurements: resample problems with replacement,
 * then resample each chosen problem's repeats with replacement. Because the spread of problem means
 * already contains within-problem noise, this double-counts it: it is CONSERVATIVE (simulation in
 * test/stats.test.js: ~0.5% one-sided false positives vs the nominal 2.5%, and much lower power).
 * Kept as an option (stats.method = "hierarchical"); the default resamples problems only.
 * `groups` = array (problems) of arrays (repeats) of values. With `pairedWith`, each problem's statistic is
 * mean(resampled b) - mean(resampled a), both resampled independently within the problem.
 */
export function hierarchicalCI(groups, { resamples, confidence, seed, pairedWith = null }) {
  const n = groups.length;
  if (!n) return { low: NaN, high: NaN, estimate: NaN };
  const r = rng(seed);
  const within = (xs) => {
    let s = 0;
    for (let i = 0; i < xs.length; i++) s += xs[Math.floor(r() * xs.length)];
    return s / xs.length;
  };
  const point = (i) => (pairedWith ? mean(groups[i]) - mean(pairedWith[i]) : mean(groups[i]));
  const estimate = mean(groups.map((_, i) => point(i)));
  const stats = new Array(resamples);
  for (let b = 0; b < resamples; b++) {
    let s = 0;
    for (let k = 0; k < n; k++) {
      const i = Math.floor(r() * n);
      s += pairedWith ? within(groups[i]) - within(pairedWith[i]) : within(groups[i]);
    }
    stats[b] = s / n;
  }
  const alpha = (1 - confidence) / 2;
  return { low: quantile(stats, alpha), high: quantile(stats, 1 - alpha), estimate };
}

/** CI for a set of repeated measurements using the configured method ("hierarchical" or "problems"). */
export function groupCI(groups, opts, method) {
  return method === 'hierarchical' ? hierarchicalCI(groups, opts) : bootstrapCI(groups.map(mean), opts);
}

/** Paired CI of mean(b - a) per problem using the configured method. */
export function pairedGroupCI(a, b, opts, method) {
  if (a.length !== b.length) throw new Error('pairedGroupCI: arrays must have equal length');
  return method === 'hierarchical' ? hierarchicalCI(b, { ...opts, pairedWith: a }) : pairedDiffCI(a.map(mean), b.map(mean), opts);
}

/**
 * Difference-in-differences per problem: (vLatest - vStart) - (cLatest - cStart), i.e. a variant's change
 * minus the control's change over the same period on the same problems. Each argument is an array
 * (problems, same order) of arrays (repeats). Hierarchical: resample problems, then repeats in each group.
 */
export function didCI(vStart, vLatest, cStart, cLatest, { resamples, confidence, seed }, method) {
  const n = vStart.length;
  if (!n) return { low: NaN, high: NaN, estimate: NaN };
  const point = (i) => (mean(vLatest[i]) - mean(vStart[i])) - (mean(cLatest[i]) - mean(cStart[i]));
  if (method !== 'hierarchical') return bootstrapCI(vStart.map((_, i) => point(i)), { resamples, confidence, seed });
  const r = rng(seed);
  const within = (xs) => { let s = 0; for (let i = 0; i < xs.length; i++) s += xs[Math.floor(r() * xs.length)]; return s / xs.length; };
  const stats = new Array(resamples);
  for (let b = 0; b < resamples; b++) {
    let s = 0;
    for (let k = 0; k < n; k++) {
      const i = Math.floor(r() * n);
      s += (within(vLatest[i]) - within(vStart[i])) - (within(cLatest[i]) - within(cStart[i]));
    }
    stats[b] = s / n;
  }
  const alpha = (1 - confidence) / 2;
  return { low: quantile(stats, alpha), high: quantile(stats, 1 - alpha), estimate: mean(vStart.map((_, i) => point(i))) };
}
