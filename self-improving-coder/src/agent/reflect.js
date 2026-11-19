import { fill } from '../util.js';
import { problemKeywords } from './retrieve.js';

export function parseLesson(reply) {
  const text = String(reply);
  const objMatch = text.match(/\{[\s\S]*\}/);
  if (objMatch) {
    try {
      const obj = JSON.parse(objMatch[0]);
      if (typeof obj.lesson === 'string' && obj.lesson.trim()) return { text: obj.lesson, tags: Array.isArray(obj.tags) ? obj.tags.map(String) : [] };
    } catch { /* fall through */ }
  }
  const line = text.replace(/```\w*/g, '').split('\n').map((l) => l.trim()).find((l) => l.length > 0);
  return line ? { text: line, tags: [] } : null;
}

export function clampWords(text, maxWords) {
  const words = text.trim().replace(/\s+/g, ' ').split(' ');
  return words.length <= maxWords ? words.join(' ') : words.slice(0, maxWords).join(' ') + '…';
}

/**
 * Turn a failed attempt into a candidate lesson { text, tags }. Returns null if nothing usable came back.
 */
export async function reflect({ llm, problem, code, result, templates, cfg, reserve }) {
  const { maxLessonWords, maxLessonTags, reflectOutputChars } = cfg.agent;
  const messages = [
    { role: 'system', content: fill(templates.reflectSystem, { maxWords: maxLessonWords, maxTags: maxLessonTags }).trim() },
    { role: 'user', content: fill(templates.reflectUser, { prompt: problem.prompt.trimEnd(), code: code.trimEnd(), resultType: result.type, output: result.output.slice(0, reflectOutputChars) }) },
  ];
  const reply = await llm.chat(messages, { temperature: cfg.llm.temperature, reserve, meta: { kind: 'reflect', problemId: problem.id } });
  const parsed = parseLesson(reply);
  if (!parsed) return null;
  const kw = [...problemKeywords(problem, cfg.retrieve)];
  const tags = [...new Set([...parsed.tags.map((t) => t.toLowerCase().trim()).filter(Boolean), ...kw])].slice(0, maxLessonTags);
  return { text: clampWords(parsed.text, maxLessonWords), tags };
}
