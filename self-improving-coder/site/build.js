// Static dashboard: reads every variant's state (via src/analysis.js) and writes one self-contained
// site/index.html (inline CSS/JS/SVG, light + dark, mobile friendly). Headline numbers, charts and tables
// are plain HTML, so they work with JavaScript disabled; JS only adds the replay stepper and filters.
//   npm run site [-- --dry-run] [-- --out=<file>]
// Everything that comes from state (lessons, code, test output) is model- or test-written: always escaped.
import path from 'node:path';
import { loadConfig, rootPath } from '../src/config.js';
import { analyze } from '../src/analysis.js';
import { renderChart, loadTrackSeries } from '../src/chart.js';
import { writeText, isoDay } from '../src/util.js';

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (v) => (v === null || v === undefined || Number.isNaN(v) ? 'n/a' : `${(v * 100).toFixed(1)}%`);
const pts = (v) => (v === null || v === undefined || Number.isNaN(v) ? 'n/a' : `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toFixed(1)}`);
const day = isoDay;
const slug = (s) => String(s).replace(/[^\w-]/g, '-');

function badge(d) {
  if (!d) return '<span class="badge muted">no data</span>';
  if (!d.significant) return '<span class="badge muted" title="95% CI includes zero">not significant</span>';
  return d.estimate > 0 ? '<span class="badge good">significant gain</span>' : '<span class="badge bad">significant drop</span>';
}
const decisionBadge = (d) => `<span class="badge dec-${esc(d)}">${esc(d)}</span>`;

function tableHtml(head, rows, cls = '') {
  return `<div class="table-wrap"><table class="${cls}"><thead><tr>${head.map((h) => `<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

/** Stacked bars of failure-class shares per run for one variant. */
function taxonomySvg(v, classes, colors, sc) {
  const runs = v.taxonomy;
  if (!runs.length) return '<p class="muted">No evaluated runs yet.</p>';
  const { width: W, height: H, barGap } = sc.taxonomy;
  const pad = sc.taxonomy.padding;
  const bw = (W - pad * 2) / runs.length - barGap;
  const bars = runs.map((r, i) => {
    let y = H - pad;
    const x = pad + i * (bw + barGap);
    return classes.map((c) => {
      const h = r.shares[c] * (H - pad * 2);
      y -= h;
      return h > 0 ? `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" fill="${colors[c]}"><title>${esc(`${day(r.ts)} ${c}: ${pct(r.shares[c])} (${r.counts[c]} of ${r.total})`)}</title></rect>` : '';
    }).join('');
  }).join('');
  return `<svg viewBox="0 0 ${W} ${H}" class="tax" role="img" aria-label="${esc(`Failure classes per run for ${v.label}`)}">${bars}</svg>`;
}

function replayHtml(r, i) {
  const step = (n, title, body, cls = '') => `<li class="step ${cls}" data-step="${n}"><h4><span class="num">${n}</span>${title}</h4>${body}</li>`;
  const code = (c) => (c ? `<pre class="code"><code>${esc(c)}</code></pre>` : '<p class="muted">Code not stored for this track (third-party code is never committed).</p>');
  const out = (o) => (o ? `<pre class="out">${esc(o)}</pre>` : '');
  const res = (x) => `<span class="badge ${x.result === 'pass' ? 'good' : 'bad'}">${esc(x.result)}${x.failureClass && x.result !== 'pass' ? ` · ${esc(x.failureClass)}` : ''}</span>`;
  const steps = [
    (n) => step(n, `Attempt ${r.first.attempt} ${res(r.first)}`, `${code(r.first.code)}${out(r.first.output)}`),
    (n) => step(n, 'Reflection', r.reflection ? `<p>The model looked at the failure and wrote:</p><blockquote>${esc(r.reflection.text)}</blockquote><p class="tags">${(r.reflection.tags ?? []).map((t) => `<span class="tag">${esc(t)}</span>`).join('')}</p>` : '<p class="muted">No reflection.</p>'),
    (n) => step(n, `Lesson ${esc(r.lessonId)} stored`, `<p>Stored as <strong>${esc(r.lessonId)}</strong> and retrieved by keyword overlap for later attempts.</p>`),
    r.next ? (n) => step(n, `Attempt ${r.next.attempt} ${res(r.next)}`, `<p>Lessons in the prompt: ${(r.next.lessonsUsed ?? []).map((l) => `<code>${esc(l)}</code>`).join(', ') || 'none'}</p>${code(r.next.code)}${r.next.result === 'pass' ? '' : out(r.next.output)}`, r.next.result === 'pass' ? 'win' : '')
      : (n) => step(n, 'No further attempt', '<p class="muted">The attempt budget for this problem ran out in this run.</p>'),
  ];
  return `<article class="replay" id="replay-${slug(r.id)}" data-index="${i}" data-variant="${esc(r.variant)}" data-pass="${r.next?.result === 'pass' ? 1 : 0}">
<header><h3><code>${esc(r.problemId)}</code></h3><p class="muted">${esc(r.variant)} · ${day(r.ts)} · ${r.next?.result === 'pass' ? 'failed, learned, then passed' : 'failed, learned'}</p></header>
<ol class="steps">${steps.map((s, i) => s(i + 1)).join('')}</ol></article>`;
}

export function buildSite(cfg) {
  const a = analyze(cfg);
  const sc = cfg.site;
  const classes = cfg.taxonomy.classes;
  const mock = cfg.llm.provider === 'mock';
  const variants = Object.values(a.variants);
  const chartCfg = { ...cfg.chart, colors: { ...cfg.chart.colors, ...sc.chartColors } };
  const parts = [];

  // 1. headline
  const headline = Object.entries(a.tracks).map(([track, t]) => {
    const comp = new Map(t.comparisons.map((c) => [c.variant, c]));
    const cards = t.variants.map((name) => {
      const v = a.variants[name];
      const c = comp.get(name);
      return `<div class="card" style="--accent:${esc(v.color)}"><div class="card-label">${esc(v.label)}${v.control ? ' <span class="badge muted">control</span>' : ''}</div>
<div class="big">${pct(v.latest?.passRate)}</div>
<div class="muted small">95% CI ${pct(v.latest?.ciLow)}–${pct(v.latest?.ciHigh)} · n=${v.latest?.n ?? 0}×${v.latest?.repeats ?? 0}</div>
${v.control ? '' : `<div class="small">vs control: <strong>${c ? `${pts(c.estimate)} pts` : 'n/a'}</strong> ${c ? `<span class="muted">[${pts(c.low)}, ${pts(c.high)}]</span>` : ''} ${badge(c)}</div>
<div class="small" title="change since its own start minus the control's change over the same period">vs own start, net of control drift: <strong>${v.vsBaselineAdjusted ? `${pts(v.vsBaselineAdjusted.estimate)} pts` : 'n/a'}</strong> ${badge(v.vsBaselineAdjusted)}</div>`}${v.control && v.vsBaseline ? `<div class="small muted">drift since start (no learning): ${pts(v.vsBaseline.estimate)} pts [${pts(v.vsBaseline.low)}, ${pts(v.vsBaseline.high)}]</div>` : ''}</div>`;
    }).join('');
    const title = cfg.charts.find((c) => c.track === track)?.title ?? track;
    return `<section class="track"><h3>${esc(track === 'multipl-e' ? 'MultiPL-E (HumanEval + MBPP, JavaScript)' : `Mutation repair (${cfg.curriculum.name}, contamination-free)`)}</h3>
<div class="cards">${cards}</div><figure class="chart">${renderChart(loadTrackSeries(cfg, track), chartCfg, title)}</figure></section>`;
  }).join('');
  parts.push(`<section id="headline"><h2>Holdout pass rate</h2><p class="lede">Each variant toggles one lever and keeps its own memory. Bands are 95% bootstrap confidence intervals (problems and repeats resampled); "significant" means the CI of the difference excludes zero. The control learns nothing, so its movement is the noise floor.</p>${headline}</section>`);

  // 2. lesson genealogy
  const gen = variants.filter((v) => v.lessons.length).map((v) => {
    const replayIds = new Set(v.replays.map((r) => r.lessonId));
    const rows = v.lessons.map((l) => [`<code>${esc(l.id)}</code>`, esc(l.text), `<code>${esc(l.born.problemId)}</code><br><span class="muted small">${esc(l.born.failureClass)} · ${day(l.born.ts)}</span>`,
      `${l.trainRetrievals} <span class="muted">(${l.usedOn.length} problems)</span>`, `${l.trainPasses}${l.trainRetrievals ? ` <span class="muted">(${pct(l.trainPasses / l.trainRetrievals)})</span>` : ''}`,
      `<span class="badge st-${esc(l.status)}">${esc(l.status)}</span>`, replayIds.has(l.id) ? `<a href="#replay-${slug(v.replays.find((r) => r.lessonId === l.id).id)}" class="replay-link">replay</a>` : '']);
    const never = v.lessons.filter((l) => l.trainRetrievals > 0 && l.trainPasses === 0);
    const conflicts = v.conflicts.length ? `<p><strong>Conflicting pairs:</strong> ${v.conflicts.slice(0, sc.maxConflicts).map((c) => `<code>${esc(c.a)}</code>+<code>${esc(c.b)}</code> failed together ${c.failedTogether}/${c.together}`).join('; ')}</p>` : '';
    return `<details class="variant-block"${v === variants.find((x) => x.lessons.length) ? ' open' : ''}><summary>${esc(v.label)}: ${v.lessons.length} lessons, ${never.length} never helped</summary>
${tableHtml(['id', 'lesson', 'born from failure', 'retrieved', 'passing when retrieved', 'status', ''], rows, 'genealogy')}
${never.length ? `<p><strong>Never helped:</strong> ${never.map((l) => `<code title="${esc(l.text)}">${esc(l.id)}</code>`).join(' ')}</p>` : ''}${conflicts}</details>`;
  }).join('');
  parts.push(`<section id="lessons"><h2>Lesson genealogy</h2><p class="lede">Every lesson, the failure that created it, how often it was retrieved afterwards, and whether those attempts passed.</p>${gen || '<p class="muted">No lessons yet.</p>'}</section>`);

  // 3. replay
  const replays = variants.flatMap((v) => v.replays);
  const options = replays.map((r, i) => `<option value="${i}">${esc(`${r.variant} · ${r.problemId} · ${r.next?.result === 'pass' ? 'fixed' : 'not fixed'}`)}</option>`).join('');
  parts.push(`<section id="replay"><h2>Replay: watch it learn</h2><p class="lede">Attempt fails → the model reflects on the test output → a lesson is stored → the next attempt sees it.</p>
${replays.length ? `<div class="replay-controls" hidden><label>Problem <select id="replay-select">${options}</select></label><span class="stepper"><button type="button" id="prev-step">◀ Back</button><span id="step-label">Step 1 of 4</span><button type="button" id="next-step">Next ▶</button></span><label class="small"><input type="checkbox" id="only-fixed" checked> only ones fixed on the next attempt</label></div>
<div class="replays">${replays.map(replayHtml).join('')}</div>` : '<p class="muted">No attempts with a lesson yet.</p>'}</section>`);

  // 4. gate log
  const gateRows = variants.flatMap((v) => v.gateLog.map((g) => ({ v, g }))).sort((x, y) => (x.g.ts < y.g.ts ? 1 : x.g.ts > y.g.ts ? -1 : x.v.name.localeCompare(y.v.name)));
  parts.push(`<section id="gate"><h2>Gate log</h2><p class="lede">Every decision with the numbers behind it. Accept needs ≥ ${pts(cfg.gate.minImprovement)} pts${cfg.gate.method === 'ci' ? ' and a CI above zero' : ''}; reject if below −${(cfg.gate.tolerance * 100).toFixed(1)} pts${cfg.gate.method === 'ci' ? ' or a CI entirely below zero' : ''}.</p>
${tableHtml(['date', 'variant', 'decision', 'champion', 'candidate', 'difference [95% CI]', 'proposed'], gateRows.map(({ v, g }) => [day(g.ts), esc(v.label), decisionBadge(g.decision), pct(g.champion?.passRate), pct(g.candidate?.passRate),
  g.diff ? `${pts(g.diff.estimate)} <span class="muted">[${pts(g.diff.low)}, ${pts(g.diff.high)}]</span>` : '', g.added && (g.added.lessons || g.added.skills || g.added.promptVersion) ? esc(`${g.added.lessons} lessons, ${g.added.skills} skills${g.added.promptVersion ? `, ${g.added.promptVersion}` : ''}`) : '']), 'gate-log')}</section>`);

  // 5. taxonomy
  const legend = classes.map((c) => `<span class="leg"><i style="background:${sc.classColors[c]}"></i>${esc(c)}</span>`).join('');
  const tax = variants.map((v) => `<div class="tax-cell"><h4>${esc(v.label)}</h4>${taxonomySvg(v, classes, sc.classColors, sc)}</div>`).join('');
  const verdicts = Object.entries(a.tracks).map(([track, t]) => `<p><strong>${esc(track)}:</strong> reduced by self-improvement: ${t.taxonomyChange.filter((x) => x.verdict === 'reduced').map((x) => `<code>${esc(x.class)}</code>`).join(', ') || 'none so far'}; not reduced: ${t.taxonomyChange.filter((x) => x.verdict === 'not-reduced').map((x) => `<code>${esc(x.class)}</code>`).join(', ') || 'n/a'}.</p>`).join('');
  parts.push(`<section id="taxonomy"><h2>Failure taxonomy over time</h2><p class="lede">Share of champion holdout attempts per class, one bar per run (pass included).</p><div class="legend">${legend}</div><div class="tax-grid">${tax}</div>${verdicts}</section>`);

  // 6. cost
  parts.push(`<section id="cost"><h2>Cost</h2>${tableHtml(['variant', 'train requests', 'eval requests', 'tokens in / out', 'solved', 'solved / 100 req', 'requests / solved', 'tokens / solved'],
    variants.map((v) => [esc(v.label), v.cost.trainRequests, v.cost.evalRequests, `${v.cost.tokensIn} / ${v.cost.tokensOut}`, `${v.cost.trainSolved} of ${v.cost.trainProblemsTried}`,
      v.cost.solvedPer100Requests?.toFixed(1) ?? 'n/a', v.cost.requestsPerSolved?.toFixed(1) ?? 'n/a', v.cost.tokensPerSolved?.toFixed(0) ?? 'n/a']))}</section>`);

  const banner = mock ? `<div class="banner" role="note"><strong>Demo data.</strong> These numbers come from the synthetic mock provider (<code>npm run reproduce</code>), not from a language model.</div>` : '';
  const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(sc.title)}</title><meta name="description" content="${esc(sc.description)}">
<style>${CSS}</style></head>
<body><header class="top"><div class="wrap"><h1>${esc(sc.title)}</h1><p class="lede">${esc(sc.description)}</p>
<p class="muted small">${a.generatedFrom.runs} evaluated runs · last run ${day(a.generatedFrom.lastTs)} · model <code>${esc(cfg.llm.model)}</code> · built from <code>state/</code> by <code>site/build.js</code></p>
<nav>${[['headline', 'Results'], ['lessons', 'Lessons'], ['replay', 'Replay'], ['gate', 'Gate log'], ['taxonomy', 'Failures'], ['cost', 'Cost']].map(([id, t]) => `<a href="#${id}">${t}</a>`).join('')}</nav></div></header>
<main class="wrap">${banner}${parts.join('\n')}</main>
<footer class="wrap muted small"><p>Generated from raw logs only. Source, method and caveats: README.md, DECISIONS.md, REPORT.md.</p></footer>
<script>${JS}</script></body></html>
`;
  return { html, analysis: a };
}

const CSS = `
:root{--bg:#fbfbfa;--fg:#16181d;--muted:#5d636f;--card:#fff;--line:#e3e5e8;--code:#f3f4f6;--good:#15803d;--bad:#b91c1c;--warn:#a16207;--link:#1d4ed8;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e8eaed;--muted:#9aa1ad;--card:#171a20;--line:#2a2f38;--code:#1d2129;--good:#4ade80;--bad:#f87171;--warn:#facc15;--link:#93c5fd;color-scheme:dark}}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:0 16px}h1{font-size:1.8rem;margin:.2em 0;letter-spacing:-.01em}h2{font-size:1.35rem;margin:2.2em 0 .3em;padding-top:.4em;border-top:1px solid var(--line)}h3{font-size:1.05rem;margin:1.4em 0 .6em}h4{margin:.2em 0 .5em;font-size:.95rem}
a{color:var(--link)}code{font:.88em ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--code);padding:.08em .3em;border-radius:4px}
.top{background:var(--card);border-bottom:1px solid var(--line);padding:20px 0 8px}nav{display:flex;flex-wrap:wrap;gap:4px 16px;margin-top:8px}nav a{text-decoration:none;font-weight:500}
.lede{color:var(--muted);max-width:72ch;margin:.2em 0 1em}.muted{color:var(--muted)}.small{font-size:.86em}
.banner{margin:16px 0;padding:10px 14px;border:1px solid var(--warn);border-radius:8px;background:color-mix(in srgb,var(--warn) 12%,transparent)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-bottom:12px}
.card{background:var(--card);border:1px solid var(--line);border-top:4px solid var(--accent);border-radius:10px;padding:12px 14px}.card-label{font-weight:600}.big{font-size:2rem;font-weight:700;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.chart{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px;overflow-x:auto;color:var(--fg)}.chart svg{width:100%;height:auto;min-width:520px;display:block}
.badge{display:inline-block;font-size:.75em;font-weight:600;padding:.1em .5em;border-radius:999px;border:1px solid currentColor;white-space:nowrap}.badge.good{color:var(--good)}.badge.bad{color:var(--bad)}.badge.muted{color:var(--muted)}
.dec-accept{color:var(--good)}.dec-reject{color:var(--bad)}.dec-keep{color:var(--warn)}.dec-baseline,.dec-no-change,.dec-no-eval{color:var(--muted)}.st-accepted{color:var(--good)}.st-dropped{color:var(--muted)}.st-pending{color:var(--warn)}
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}table{border-collapse:collapse;width:100%;font-size:.9em}th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{font-weight:600;white-space:nowrap}tbody tr:last-child td{border-bottom:0}td{font-variant-numeric:tabular-nums}
details.variant-block{margin:10px 0}details>summary{cursor:pointer;font-weight:600;padding:6px 0}
.replay{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:12px 0}.replay header h3{margin:0}.replay header p{margin:.1em 0 .6em}
.steps{list-style:none;padding:0;margin:0;display:grid;gap:10px}.step{border-left:3px solid var(--line);padding:2px 0 2px 12px}.step.win{border-left-color:var(--good)}.num{display:inline-grid;place-items:center;width:1.5em;height:1.5em;border-radius:50%;background:var(--code);margin-right:.5em;font-size:.85em}
pre{background:var(--code);border-radius:8px;padding:10px 12px;overflow-x:auto;font:12.5px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;max-height:340px;margin:.4em 0}pre.out{border-left:3px solid var(--bad)}
blockquote{margin:.4em 0;padding:.4em .9em;border-left:3px solid var(--link);background:var(--code);border-radius:0 8px 8px 0;font-style:italic}.tag{display:inline-block;font-size:.78em;background:var(--code);border-radius:4px;padding:0 .4em;margin:0 .3em .3em 0}
.replay-controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px 18px;margin:8px 0}select,button{font:inherit;padding:5px 9px;border-radius:7px;border:1px solid var(--line);background:var(--card);color:var(--fg);max-width:100%}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}
.js .replays .replay{display:none}.js .replays .replay.active{display:block}.js .step.future{display:none}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;margin:8px 0;font-size:.86em}.leg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}
.tax-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}.tax-cell{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 10px}.tax{width:100%;height:auto;display:block}
footer{padding:28px 16px 40px}
@media (max-width:560px){.big{font-size:1.6rem}h1{font-size:1.45rem}th,td{padding:6px 7px}}
`;

const JS = `
(function(){
  var root=document.documentElement;root.classList.add('js');
  var arts=[].slice.call(document.querySelectorAll('.replays .replay'));if(!arts.length)return;
  var ctl=document.querySelector('.replay-controls');ctl.hidden=false;
  var sel=document.getElementById('replay-select'),prev=document.getElementById('prev-step'),next=document.getElementById('next-step'),lab=document.getElementById('step-label'),only=document.getElementById('only-fixed');
  var cur=null,step=1;
  function steps(){return cur?[].slice.call(cur.querySelectorAll('.step')):[]}
  function render(){var s=steps();s.forEach(function(el,i){el.classList.toggle('future',i>=step)});lab.textContent='Step '+step+' of '+s.length;prev.disabled=step<=1;next.disabled=step>=s.length}
  function show(i){arts.forEach(function(a){a.classList.remove('active')});cur=arts[i];if(!cur)return;cur.classList.add('active');step=1;render()}
  function filter(){var first=null;[].slice.call(sel.options).forEach(function(o){var ok=!only.checked||arts[+o.value].dataset.pass==='1';o.hidden=!ok;o.disabled=!ok;if(ok&&first===null)first=+o.value});if(first===null){only.checked=false;return filter()}if(sel.selectedOptions[0]&&sel.selectedOptions[0].disabled||!cur){sel.value=String(first)}show(+sel.value)}
  sel.addEventListener('change',function(){show(+sel.value)});prev.addEventListener('click',function(){if(step>1){step--;render()}});next.addEventListener('click',function(){if(step<steps().length){step++;render()}});only.addEventListener('change',filter);
  function fromHash(){var id=location.hash.slice(1);var i=arts.findIndex(function(a){return a.id===id});if(i>=0){if(arts[i].dataset.pass!=='1')only.checked=false;filter();sel.value=String(i);show(i);step=steps().length;render();document.getElementById('replay').scrollIntoView()}}
  window.addEventListener('hashchange',fromHash);filter();fromHash();
  document.addEventListener('keydown',function(e){if(e.target.tagName==='SELECT'||e.target.tagName==='INPUT')return;var r=document.getElementById('replay').getBoundingClientRect();if(r.top>innerHeight||r.bottom<0)return;if(e.key==='ArrowRight')next.click();if(e.key==='ArrowLeft')prev.click()});
})();
`;

export function writeSite(cfg, out) {
  const { html, analysis } = buildSite(cfg);
  const file = out ?? rootPath(cfg, cfg.site.outFile);
  writeText(file, html);
  return { file, html, analysis };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const argv = process.argv.slice(2);
  const cfg = loadConfig({ dryRun: argv.includes('--dry-run') });
  const out = argv.find((x) => x.startsWith('--out='))?.slice('--out='.length);
  const { file } = writeSite(cfg, out ? path.resolve(out) : undefined);
  console.log(path.relative(process.cwd(), file));
}
