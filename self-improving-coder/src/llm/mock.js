// Deterministic fake LLM. Behaviour is scripted per problem in a fixture file
// (config llm.mock.scriptPath). For each problem the script may define:
//   correct   - a correct solution
//   buggy     - a wrong solution returned until the prompt contains `fixedBy`
//   fixedBy   - a phrase; when any message contains it the mock returns `correct`
//   lesson, lessonTags - what the mock returns when asked to reflect on this problem
// Problems not in the script get a stub that fails the tests.
import fs from 'node:fs';
import path from 'node:path';

export function createMock(cfg, deps = {}) {
  const scriptFile = path.resolve(cfg.root, cfg.llm.mock.scriptPath);
  const problems = deps.mockScript ?? JSON.parse(fs.readFileSync(scriptFile, 'utf8'));
  const byId = new Map(problems.map((p) => [p.id, p]));
  const calls = [];

  return {
    name: 'mock',
    calls,
    async chat(messages, { temperature, meta = {} } = {}) {
      calls.push({ meta, temperature, messages });
      const p = byId.get(meta.problemId);
      const all = messages.map((m) => m.content).join('\n');
      if (meta.kind === 'reflect') {
        const lesson = p?.mock?.lesson ?? 'Read the problem statement carefully and handle edge cases.';
        const tags = p?.mock?.lessonTags ?? ['general'];
        return '```json\n' + JSON.stringify({ lesson, tags }) + '\n```';
      }
      if (meta.kind === 'evolve') return deps.mockEvolveAnswer ?? 'You are a careful JavaScript programmer.';
      if (!p) {
        const fn = (all.match(/function\s+(\w+)\s*\(/g) ?? []).pop()?.match(/function\s+(\w+)/)?.[1] ?? 'f';
        return '```javascript\nfunction ' + fn + '() { return undefined; }\n```';
      }
      const m = p.mock ?? {};
      const fixed = m.fixedBy ? all.includes(m.fixedBy) : true;
      const code = fixed || !m.buggy ? m.correct : m.buggy;
      return 'Here is the solution:\n```javascript\n' + code + '\n```';
    },
  };
}
