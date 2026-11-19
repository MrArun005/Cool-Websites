// acorn-based helpers: parse a file, list named function declarations, walk nodes with parents.
import * as acorn from 'acorn';

export function parse(source, ecmaVersion) {
  return acorn.parse(source, { ecmaVersion, sourceType: 'script', allowHashBang: true, allowReturnOutsideFunction: true, allowAwaitOutsideFunction: true });
}

export function tryParse(source, ecmaVersion) {
  try { return parse(source, ecmaVersion); } catch { return null; }
}

/** Depth-first walk; `visit(node, parent, key)`. */
export function walk(node, visit, parent = null, key = null) {
  if (!node || typeof node.type !== 'string') return;
  visit(node, parent, key);
  for (const k of Object.keys(node)) {
    if (k === 'type' || k === 'start' || k === 'end' || k === 'loc' || k === 'range') continue;
    const v = node[k];
    if (Array.isArray(v)) for (const c of v) walk(c, visit, node, k);
    else if (v && typeof v.type === 'string') walk(v, visit, node, k);
  }
}

/**
 * Named function declarations in a file, in source order: { name, index, start, end, node }.
 * `index` disambiguates repeated names (e.g. nested helpers with the same name).
 */
export function listFunctions(source, ecmaVersion) {
  const ast = parse(source, ecmaVersion);
  const out = [];
  const seen = new Map();
  walk(ast, (n) => {
    if (n.type === 'FunctionDeclaration' && n.id?.name) {
      const idx = seen.get(n.id.name) ?? 0;
      seen.set(n.id.name, idx + 1);
      out.push({ name: n.id.name, index: idx, start: n.start, end: n.end, node: n });
    }
  });
  return out;
}
