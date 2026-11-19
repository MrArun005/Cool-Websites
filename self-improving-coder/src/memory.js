// Persistent agent memory under state/: champion, lessons, skills, prompts, attempts.
import fs from 'node:fs';
import path from 'node:path';
import { statePath } from './config.js';
import { readJson, writeJson, writeText } from './util.js';

export const EMPTY_CHAMPION = {
  promptVersion: 'v001',
  lessonIds: [],
  skillIds: [],
  holdout: null,
  pending: { lessonIds: [], skillIds: [], promptVersion: null, sinceRun: null },
};

const skillFile = (dir, id) => path.join(dir, `${id.replace(/[^\w.-]/g, '_')}.js`);

export function loadMemory(cfg) {
  const champion = { ...structuredClone(EMPTY_CHAMPION), ...readJson(statePath(cfg, 'champion'), EMPTY_CHAMPION) };
  const lessonsDoc = readJson(statePath(cfg, 'lessons'), { nextId: 1, items: [] });
  const attempts = readJson(statePath(cfg, 'attempts'), {});
  const runState = readJson(statePath(cfg, 'runState'), { runCount: 0, baselineRunsDone: 0 });
  const skillsDir = statePath(cfg, 'skillsDir');
  const skills = new Map();
  if (fs.existsSync(skillsDir)) {
    for (const f of fs.readdirSync(skillsDir).filter((f) => f.endsWith('.js')).sort()) {
      const s = parseSkill(fs.readFileSync(path.join(skillsDir, f), 'utf8'));
      if (s) skills.set(s.id, s);
    }
  }
  return { champion, runState, lessons: new Map(lessonsDoc.items.map((l) => [l.id, l])), nextLessonId: lessonsDoc.nextId, skills, attempts };
}

export function serializeSkill(s) {
  const meta = { id: s.id, entryPoint: s.entryPoint, keywords: s.keywords, createdRun: s.createdRun };
  return `// skill: ${JSON.stringify(meta)}\n${s.code.trimEnd()}\n`;
}

export function parseSkill(text) {
  const m = text.match(/^\/\/ skill: (.*)\n/);
  if (!m) return null;
  try { return { ...JSON.parse(m[1]), code: text.slice(m[0].length) }; } catch { return null; }
}

export function newLessonId(mem) {
  return `L${String(mem.nextLessonId++).padStart(4, '0')}`;
}

export function readPrompt(cfg, version) {
  return fs.readFileSync(path.join(statePath(cfg, 'promptsDir'), `${version}.md`), 'utf8');
}

export function writePrompt(cfg, version, text) {
  writeText(path.join(statePath(cfg, 'promptsDir'), `${version}.md`), text.trimEnd() + '\n');
}

export function nextPromptVersion(cfg) {
  const dir = statePath(cfg, 'promptsDir');
  const nums = fs.existsSync(dir) ? fs.readdirSync(dir).map((f) => f.match(/^v(\d+)\.md$/)?.[1]).filter(Boolean).map(Number) : [];
  return `v${String(Math.max(0, ...nums) + 1).padStart(3, '0')}`;
}

/** Ensure the seed prompt exists in state (copied from the champion version on first run). */
export function ensureSeedPrompt(cfg, champion, seedText) {
  const file = path.join(statePath(cfg, 'promptsDir'), `${champion.promptVersion}.md`);
  if (!fs.existsSync(file) && seedText) writeText(file, seedText);
}

export function saveMemory(cfg, mem) {
  const liveLessons = new Set([...mem.champion.lessonIds, ...mem.champion.pending.lessonIds]);
  const liveSkills = new Set([...mem.champion.skillIds, ...mem.champion.pending.skillIds]);
  writeJson(statePath(cfg, 'champion'), mem.champion);
  writeJson(statePath(cfg, 'lessons'), { nextId: mem.nextLessonId, items: [...mem.lessons.values()].filter((l) => liveLessons.has(l.id)) });
  writeJson(statePath(cfg, 'attempts'), mem.attempts);
  writeJson(statePath(cfg, 'runState'), mem.runState);
  const dir = statePath(cfg, 'skillsDir');
  fs.mkdirSync(dir, { recursive: true });
  const wanted = new Set();
  for (const s of mem.skills.values()) {
    if (!liveSkills.has(s.id)) continue;
    const f = skillFile(dir, s.id);
    wanted.add(path.basename(f));
    writeText(f, serializeSkill(s));
  }
  for (const f of fs.readdirSync(dir)) if (f.endsWith('.js') && !wanted.has(f)) fs.rmSync(path.join(dir, f));
}

/** Prune live lessons above maxLessons: never-helped first, then least-used, then oldest. Returns removed ids. */
export function pruneLessons(mem, maxLessons) {
  const ch = mem.champion;
  const live = [...ch.lessonIds, ...ch.pending.lessonIds].map((id) => mem.lessons.get(id)).filter(Boolean);
  if (live.length <= maxLessons) return [];
  const order = [...live].sort((a, b) => (a.helped - b.helped) || (a.uses - b.uses) || a.id.localeCompare(b.id));
  const removed = order.slice(0, live.length - maxLessons).map((l) => l.id);
  const drop = new Set(removed);
  ch.lessonIds = ch.lessonIds.filter((id) => !drop.has(id));
  ch.pending.lessonIds = ch.pending.lessonIds.filter((id) => !drop.has(id));
  for (const id of removed) mem.lessons.delete(id);
  return removed;
}
