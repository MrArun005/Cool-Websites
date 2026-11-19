// Keyword-overlap retrieval for lessons and skills. No embeddings: deterministic, offline and free.

export function tokenize(text, retrieveCfg) {
  const { minTokenLength, stopwords } = retrieveCfg;
  const stop = new Set(stopwords);
  const words = String(text)
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .toLowerCase()
    .split(/[^a-z0-9]+/);
  return new Set(words.filter((w) => w.length >= minTokenLength && !stop.has(w) && !/^\d+$/.test(w)));
}

/** Keywords of a task: the track's keywordText (e.g. a problem's comments + function name). */
export function problemKeywords(task, retrieveCfg) {
  if (task.keywordText !== undefined) return tokenize(task.keywordText, retrieveCfg);
  const comments = task.prompt.split('\n').filter((l) => l.trim().startsWith('//')).join(' ');
  return tokenize(`${comments} ${task.entryPoint ?? ''}`, retrieveCfg);
}

export function cosine(a, b) {
  if (!a.size || !b.size) return 0;
  let inter = 0;
  for (const x of a) if (b.has(x)) inter++;
  return inter / Math.sqrt(a.size * b.size);
}

function topK(items, scoreFn, k, minScore, tieBreak) {
  return items
    .map((it) => ({ it, score: scoreFn(it) }))
    .filter((x) => x.score >= minScore)
    .sort((a, b) => b.score - a.score || tieBreak(a.it, b.it))
    .slice(0, k)
    .map((x) => x.it);
}

export function lessonKeywords(lesson, retrieveCfg) {
  return tokenize(`${lesson.tags.join(' ')} ${lesson.text}`, retrieveCfg);
}

export function retrieveLessons(problem, lessons, cfg) {
  const pk = problemKeywords(problem, cfg.retrieve);
  return topK(lessons, (l) => cosine(lessonKeywords(l, cfg.retrieve), pk), cfg.agent.lessonsTopK, cfg.retrieve.minScore,
    (a, b) => (b.helped ?? 0) - (a.helped ?? 0) || a.id.localeCompare(b.id));
}

export function retrieveSkills(problem, skills, cfg) {
  const pk = problemKeywords(problem, cfg.retrieve);
  return topK(skills.filter((s) => s.id !== problem.id), (s) => cosine(new Set(s.keywords), pk), cfg.agent.skillsTopK,
    cfg.retrieve.minScore, (a, b) => a.id.localeCompare(b.id));
}
