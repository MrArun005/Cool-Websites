// Deterministic fake LLM, two modes (config llm.mock.mode):
//
// "scripted" - behaviour per problem from the fixture file (llm.mock.scriptPath):
//    correct / buggy code, `fixedBy` (a phrase that, once present in the prompt, makes the mock
//    answer correctly) and the `lesson` / `lessonTags` returned when asked to reflect.
//
// "synthetic" - a seeded simulation used by `npm run reproduce`, the dashboard demo and the report
//    tests. Each problem belongs to a `family`. P(correct) = base + lessonBoost (if a lesson containing
//    the family's fix phrase is in the prompt) + skillBoost (if a solved problem of the same family is
//    shown) + promptBoost (if the system prompt contains llm.mock.synthetic.promptBoostPhrase).
//    Reflection returns the family's useful lesson with probability usefulLessonRate, else a vague one.
//    THIS IS NOT A MODEL. It exists so the whole pipeline can be exercised and reproduced offline.
//
// An `oracle` (problemId -> { correct, buggy, family, ... }) can be injected by tracks that know the
// answers (e.g. the mutation track knows each mutant's original function).
import fs from 'node:fs';
import path from 'node:path';
import { hash32, rng } from '../util.js';

export function estimateTokens(text, charsPerToken) {
  return Math.ceil(String(text).length / charsPerToken);
}

export function createMock(cfg, deps = {}) {
  const mc = cfg.llm.mock;
  const scriptFile = path.resolve(cfg.root, mc.scriptPath);
  const problems = deps.mockScript ?? (fs.existsSync(scriptFile) ? JSON.parse(fs.readFileSync(scriptFile, 'utf8')) : []);
  const byId = new Map(problems.map((p) => [p.id, { ...p.mock, entryPoint: p.entryPoint }]));
  const oracle = (id) => deps.oracle?.(id) ?? byId.get(id);
  const calls = [];
  const syn = mc.synthetic;

  function syntheticPass(o, all, opts) {
    const fam = o.family ?? 'none';
    const fix = syn.families?.[fam]?.fixPhrase;
    let p = syn.families?.[fam]?.base ?? syn.defaultBase;
    if (fix && all.includes(fix)) p += syn.lessonBoost;
    const sameFamily = [...byId.entries()].filter(([id, m]) => m.family === fam && id !== opts.meta?.problemId && m.correct);
    if (sameFamily.some(([, m]) => all.includes(m.correct.trim()))) p += syn.skillBoost; // a related solved function is shown
    if (syn.promptBoostPhrase && all.includes(syn.promptBoostPhrase)) p += syn.promptBoost;
    const u = rng(hash32(`${opts.seed ?? cfg.llm.seed}|${opts.meta?.problemId}|${opts.temperature}|${hash32(all)}|${mc.seed}`))();
    return u < Math.min(p, syn.maxPass);
  }

  return {
    name: 'mock',
    calls,
    lastUsage: { tokensIn: 0, tokensOut: 0 },
    async chat(messages, opts = {}) {
      const { meta = {} } = opts;
      calls.push({ meta, temperature: opts.temperature, seed: opts.seed, messages });
      const all = messages.map((m) => m.content).join('\n');
      const reply = this.answer(all, opts);
      this.lastUsage = { tokensIn: estimateTokens(all, mc.charsPerToken), tokensOut: estimateTokens(reply, mc.charsPerToken) };
      return reply;
    },
    answer(all, opts) {
      const { meta = {} } = opts;
      const o = oracle(meta.problemId);
      if (meta.kind === 'reflect') {
        if (mc.mode === 'synthetic' && o) {
          const fam = syn.families?.[o.family];
          const u = rng(hash32(`reflect|${meta.problemId}|${hash32(all)}|${mc.seed}`))();
          const useful = fam && u < syn.usefulLessonRate;
          const lesson = useful ? fam.lesson : syn.vagueLessons[Math.floor(u * syn.vagueLessons.length) % syn.vagueLessons.length];
          return '```json\n' + JSON.stringify({ lesson, tags: useful ? fam.tags : ['general'] }) + '\n```';
        }
        const lesson = o?.lesson ?? 'Read the problem statement carefully and handle edge cases.';
        return '```json\n' + JSON.stringify({ lesson, tags: o?.lessonTags ?? ['general'] }) + '\n```';
      }
      if (meta.kind === 'evolve') return deps.mockEvolveAnswer ?? 'You are a careful JavaScript programmer.';
      if (!o) {
        const fn = (all.match(/function\s+(\w+)\s*\(/g) ?? []).pop()?.match(/function\s+(\w+)/)?.[1] ?? 'f';
        return '```javascript\nfunction ' + fn + '() { return undefined; }\n```';
      }
      let good;
      if (mc.mode === 'synthetic') good = syntheticPass(o, all, opts);
      else good = o.fixedBy ? all.includes(o.fixedBy) : true;
      const code = good || !o.buggy ? o.correct : o.buggy;
      return 'Here is the solution:\n```javascript\n' + code + '\n```';
    },
  };
}
