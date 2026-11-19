import { fill } from '../util.js';
import { retrieveLessons, retrieveSkills } from './retrieve.js';

export function formatLessons(lessons) {
  if (!lessons.length) return '';
  return '## Lessons from past mistakes\n' + lessons.map((l) => `- ${l.text}`).join('\n') + '\n\n';
}

export function formatSkills(skills, maxChars) {
  if (!skills.length) return '';
  return '## Related functions you solved before (for reference)\n' +
    skills.map((s) => '```javascript\n' + s.code.slice(0, maxChars) + '\n```').join('\n') + '\n\n';
}

export function buildMessages({ systemPrompt, userTemplate, problem, lessons, skills, cfg }) {
  const user = fill(userTemplate, {
    lessons: formatLessons(lessons),
    skills: formatSkills(skills, cfg.agent.maxSkillChars),
    prompt: problem.prompt.trimEnd(),
  });
  return [{ role: 'system', content: systemPrompt.trim() }, { role: 'user', content: user }];
}

/** Pull the code out of a model reply. Falls back to treating the reply as the function body. */
export function extractCode(text, problem) {
  const fences = [...String(text).matchAll(/```[\w-]*[ \t]*\n([\s\S]*?)(?:```|$)/g)].map((m) => m[1]);
  const ep = problem.entryPoint;
  const defines = (c) => new RegExp(`function\\s+${ep}\\s*\\(|(?:const|let|var)\\s+${ep}\\s*=`).test(c);
  let code = fences.find(defines) ?? fences[0] ?? String(text);
  if (!defines(code)) code = `${problem.prompt}${code}\n}`;
  return code.trim() + '\n';
}

/**
 * One solve attempt. `memory` = { systemPrompt, lessons, skills } visible to this attempt.
 * @returns {Promise<{code, messages, usedLessons, usedSkills}>}
 */
export async function solve({ llm, problem, memory, templates, cfg, temperature, reserve }) {
  const usedLessons = retrieveLessons(problem, memory.lessons, cfg);
  const usedSkills = retrieveSkills(problem, memory.skills, cfg);
  const messages = buildMessages({ systemPrompt: memory.systemPrompt, userTemplate: templates.solveUser, problem, lessons: usedLessons, skills: usedSkills, cfg });
  const reply = await llm.chat(messages, { temperature, reserve, meta: { kind: 'solve', problemId: problem.id } });
  return { code: extractCode(reply, problem), messages, usedLessons, usedSkills };
}
