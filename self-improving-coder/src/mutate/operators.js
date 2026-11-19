// Mutation operators. Each returns the candidate sites inside a function as edits
// { start, end, replacement, operator, detail } on the file's source. Which operators are used and
// how often is config (curriculum.operators: name -> weight).
import { walk } from './ast.js';

const COMPARISON_FLIP = { '<': '<=', '<=': '<', '>': '>=', '>=': '>', '===': '!==', '!==': '===', '==': '!=', '!=': '==' };
const LOGICAL_FLIP = { '&&': '||', '||': '&&' };

/** Edit that replaces the operator token between `left` and `right`. */
function operatorEdit(src, node, from, to, operator) {
  const gapStart = node.left.end;
  const gap = src.slice(gapStart, node.right.start);
  const at = gap.indexOf(from);
  if (at < 0) return null;
  return { start: gapStart + at, end: gapStart + at + from.length, replacement: to, operator, detail: `${from} -> ${to}` };
}

export const OPERATORS = {
  'flip-comparison'(fn, src) {
    const out = [];
    walk(fn.node.body, (n) => {
      if (n.type === 'BinaryExpression' && COMPARISON_FLIP[n.operator]) {
        const e = operatorEdit(src, n, n.operator, COMPARISON_FLIP[n.operator], 'flip-comparison');
        if (e) out.push(e);
      }
    });
    return out;
  },
  'flip-logical'(fn, src) {
    const out = [];
    walk(fn.node.body, (n) => {
      if (n.type === 'LogicalExpression' && LOGICAL_FLIP[n.operator]) {
        const e = operatorEdit(src, n, n.operator, LOGICAL_FLIP[n.operator], 'flip-logical');
        if (e) out.push(e);
      }
    });
    return out;
  },
  'off-by-one'(fn) {
    const out = [];
    walk(fn.node.body, (n, parent, key) => {
      if (n.type !== 'Literal' || typeof n.value !== 'number' || !Number.isInteger(n.value)) return;
      if (parent?.type === 'Property' && key === 'key') return;
      out.push({ start: n.start, end: n.end, replacement: String(n.value + 1), operator: 'off-by-one', detail: `${n.raw} -> ${n.value + 1}` });
      if (n.value > 0) out.push({ start: n.start, end: n.end, replacement: String(n.value - 1), operator: 'off-by-one', detail: `${n.raw} -> ${n.value - 1}` });
    });
    return out;
  },
  'drop-branch'(fn, src) {
    const out = [];
    walk(fn.node.body, (n) => {
      if (n.type !== 'IfStatement') return;
      out.push({ start: n.consequent.start, end: n.consequent.end, replacement: '{}', operator: 'drop-branch', detail: 'empty then-branch' });
      if (n.alternate) {
        // remove "else <alternate>" (keep the if/then part)
        const between = src.slice(n.consequent.end, n.alternate.start);
        if (/^\s*else\s*$/.test(between)) out.push({ start: n.consequent.end, end: n.alternate.end, replacement: '', operator: 'drop-branch', detail: 'drop else-branch' });
      }
    });
    return out;
  },
  'swap-arguments'(fn, src) {
    const out = [];
    walk(fn.node.body, (n) => {
      if ((n.type !== 'CallExpression' && n.type !== 'NewExpression') || n.arguments.length < 2) return;
      const [a, b] = n.arguments;
      if (a.type === 'SpreadElement' || b.type === 'SpreadElement') return;
      const ta = src.slice(a.start, a.end);
      const tb = src.slice(b.start, b.end);
      if (ta === tb) return;
      out.push({ start: a.start, end: b.end, replacement: `${tb}${src.slice(a.end, b.start)}${ta}`, operator: 'swap-arguments', detail: `swap ${ta.slice(0, 20)} / ${tb.slice(0, 20)}` });
    });
    return out;
  },
  'wrong-return'(fn) {
    const out = [];
    walk(fn.node.body, (n) => {
      // only returns of this function, not of nested functions
      if (n.type !== 'ReturnStatement' || !n.argument) return;
      const arg = n.argument;
      if (arg.type === 'Literal' && typeof arg.value === 'boolean') out.push({ start: arg.start, end: arg.end, replacement: String(!arg.value), operator: 'wrong-return', detail: `return ${arg.value} -> ${!arg.value}` });
      else if (!(arg.type === 'Identifier' && arg.name === 'undefined')) out.push({ start: arg.start, end: arg.end, replacement: 'undefined', operator: 'wrong-return', detail: 'return undefined' });
    });
    return out;
  },
};

/** Apply non-overlapping edits to `src` (any order). */
export function applyEdits(src, edits) {
  const sorted = [...edits].sort((a, b) => b.start - a.start);
  for (let i = 1; i < sorted.length; i++) if (sorted[i].end > sorted[i - 1].start) throw new Error('applyEdits: overlapping edits');
  let out = src;
  for (const e of sorted) out = out.slice(0, e.start) + e.replacement + out.slice(e.end);
  return out;
}

export const overlaps = (a, b) => a.start < b.end && b.start < a.end;
