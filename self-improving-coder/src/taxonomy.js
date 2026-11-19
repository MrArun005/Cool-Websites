// Rule-based failure taxonomy. Rules live in config (taxonomy.rules) and are tried in order;
// the first match wins. A rule matches when every criterion it specifies matches:
//   resultTypes   - judge result type is one of these (pass | fail | error | timeout)
//   outputPatterns / checks - at least one regex matches the judge output OR one named check is true
// Patterns may contain {{entryPoint}}, replaced by the task's (regex-escaped) function name.

const escapeRe = (s) => String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

function paramsOf(src, name) {
  const m = String(src).match(new RegExp(`function\\s+${escapeRe(name)}\\s*\\(([^)]*)\\)`));
  if (!m) return null;
  return m[1].split(',').map((s) => s.trim()).filter(Boolean).length;
}

/** Named checks a rule can use. Each gets { code, task, result }. */
export const CHECKS = {
  entryPointMissing: ({ code, task }) => {
    const names = task.entryPoints ?? [task.entryPoint];
    return names.some((n) => !new RegExp(`function\\s+${escapeRe(n)}\\s*\\(|(?:const|let|var)\\s+${escapeRe(n)}\\s*=`).test(code ?? ''));
  },
  arityMismatch: ({ code, task }) => {
    const names = task.entryPoints ?? [task.entryPoint];
    return names.some((n) => {
      const want = paramsOf(task.signatureSource ?? task.prompt ?? '', n);
      const got = paramsOf(code ?? '', n);
      return want !== null && got !== null && want !== got;
    });
  },
};

export function compileRules(rules) {
  return rules.map((r) => ({
    ...r,
    patternSources: r.outputPatterns ?? [],
  }));
}

/**
 * @param {{type: string, output: string}} result judge result
 * @param {{entryPoint?: string, entryPoints?: string[], prompt?: string}} task
 * @param {string} code the attempted code
 * @param {object[]} rules config taxonomy.rules
 * @returns {string} failure class, or `passClass` for passing results
 */
export function classify(result, task, code, { rules, passClass }) {
  if (result.type === 'pass') return passClass;
  const ep = escapeRe(task.entryPoint ?? (task.entryPoints ?? [])[0] ?? '');
  for (const rule of rules) {
    if (rule.resultTypes && !rule.resultTypes.includes(result.type)) continue;
    const pats = rule.outputPatterns ?? [];
    const checks = rule.checks ?? [];
    if (pats.length || checks.length) {
      const patHit = pats.some((p) => new RegExp(p.replaceAll('{{entryPoint}}', ep), 'm').test(result.output ?? ''));
      const checkHit = checks.some((c) => {
        if (!CHECKS[c]) throw new Error(`taxonomy: unknown check "${c}"`);
        return CHECKS[c]({ code, task, result });
      });
      if (!patHit && !checkHit) continue;
    }
    return rule.class;
  }
  throw new Error('taxonomy: no rule matched; the last rule should be a catch-all');
}
