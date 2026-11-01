"""One complete Bolt demo: Bolt Companion (talk, feelings, voice, dance, memory, in the living room) with Bolt's Day Out
(five live minutes in the front yard: canter, jumps, fetch, shoves) as its second tab.

    python merge_bolt_demo.py <companion index.html> <doglife index.html> <out index.html>

Inputs are the two built pages (build_pages.py companion / doglife). The yard page is placed in a
<section id="yardView"> tab: its CSS is scoped to that section (both pages use .hud, .stage, .chips differently), its two
clashing ids (stage, loading) are renamed, and its script starts only when the tab is first opened. Only the visible
tab steps its physics (window.__view), so the two MuJoCo worlds never run at once. The site folder is the union of both
pages' files (Companion's plus the jump brains and hurdles3d.js).
"""
import re
import sys

comp_path, day_path, out_path = sys.argv[1:4]
comp, day = open(comp_path).read(), open(day_path).read()


def scope_css(css, root):
    """Prefix every selector with `root` (nested @media handled; :root/html/body become the root itself)."""
    out, i = [], 0
    while i < len(css):
        j = css.find("{", i)
        if j < 0:
            out.append(css[i:])
            break
        head = css[i:j].strip()
        depth, k = 1, j + 1
        while depth:
            depth += {"{": 1, "}": -1}.get(css[k], 0)
            k += 1
        body = css[j + 1:k - 1]
        if head.startswith("@media") or head.startswith("@supports"):
            out.append(f"{head} {{{scope_css(body, root)}}}\n")
        elif head.startswith("@"):
            out.append(f"{head} {{{body}}}\n")
        else:
            sels = []
            for s in head.split(","):
                s = s.strip()
                if not s:
                    continue
                if re.match(r"^(:root|html|body)\b", s):
                    s = re.sub(r"^(:root(\[[^\]]*\])?(:not\([^)]*\))?|html|body)", root, s)
                    sels.append(s)
                else:
                    sels.append(f"{root} {s}")
            out.append(f"{', '.join(sels)} {{{body}}}\n")
        i = k
    return "".join(out)


# ---------- the yard page, as a tab ----------
d_style = day[day.index("<style>") + 7:day.index("</style>")]
d_body = day[day.index('<div class="wrap">') + len('<div class="wrap">'):day.rindex("</div>", 0, day.index("<script"))]
d_script = day[day.index("<script>\n(() => {"):day.rindex("</script>") + 9]
for a, b in (('id="stage"', 'id="yStage"'), ('id="loading"', 'id="yLoading"')):
    assert d_body.count(a) == 1, a
    d_body = d_body.replace(a, b)
d_body = d_body.replace('<h1>Bolt\'s Day Out</h1>', '<h2 class="ytitle">Bolt\'s day out</h2>')
for a, b in (('$("stage")', '$("yStage")'), ('$("loading")', '$("yLoading")')):
    d_script = d_script.replace(a, b)
# start on first open; step only while the yard tab is the one showing
a = "  resize(); requestAnimationFrame(tick); boot();\n"
assert d_script.count(a) == 1
d_script = d_script.replace(a, "  let yardStarted = false;\n  window.__startYard = () => { if (yardStarted) return; yardStarted = true; resize(); requestAnimationFrame(tick); boot(); };\n")
a = "  function tick(now) {\n"
assert d_script.count(a) == 1
d_script = d_script.replace(a, a + "    if (window.__view !== \"yard\") { last = now; requestAnimationFrame(tick); return; }\n")
a = 'if (e.key === " " && e.target === document.body)'
assert d_script.count(a) == 1
d_script = d_script.replace(a, 'if (e.key === " " && e.target === document.body && window.__view === "yard")')
d_script = d_script.replace("<script>\n(() => {\n  const $ = (id) => document.getElementById(id);",
                            "<script>\n(() => {\n  const $ = (id) => document.getElementById(id);   // the yard tab (Bolt's Day Out)")
# its stylesheet, scoped (BASE_CSS is shared with Companion: only the yard's own rules after it)
cut = d_style.index("  /* Layout: the yard in 3D")
yard_css = scope_css(d_style[cut:], "#yardView")

# ---------- Companion, with the tab bar ----------
c = comp
c = c.replace("<title>Bolt Companion</title>", "<title>Bolt the Robot Dog</title>")
c = c.replace('content="Talk to Bolt, a simulated robot dog', 'content="Talk to Bolt, a simulated robot dog, or watch it live in the yard')
a = "</style>"
c = c.replace(a, yard_css + """
  /* the two tabs: Talk and play (Companion) and Day out (the yard) */
  .tabs { display: flex; gap: 6px; flex-wrap: wrap; }
  .tabs button { font: 600 0.92rem var(--body); padding: 9px 16px; min-height: 42px; border-radius: 999px; border: 1px solid var(--line);
    background: var(--panel); color: var(--ink); cursor: pointer; }
  .tabs button[aria-selected="true"] { background: var(--accent); border-color: var(--accent); color: #fff; }
  .tabs button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  #yardView { display: grid; gap: 18px; }
  #yardView[hidden] { display: none; }
  #talkView { display: grid; gap: 18px; }
  #talkView[hidden] { display: none; }
  #yardView .ytitle { font: 800 1.3rem var(--display); }
</style>""", 1)
a = '<div class="wrap">\n  <div class="top">'
assert c.count(a) == 1, "companion top"
top_end = c.index("</div>", c.index(a) + len(a)) + 6
top = c[c.index(a) + len('<div class="wrap">\n'):top_end]
new_top = top.replace("<h1>Bolt Companion</h1>", "<h1>Bolt the Robot Dog</h1>").replace(
    "<p>Talk to Bolt, a robot dog in a physics simulation. Its mind runs in this page, on your device: nothing you say leaves it.</p>",
    "<p>A robot dog in a physics simulation, with its trained brains moving every leg. Talk and play with it at home, or watch "
    "five live minutes of it in the yard. Everything runs in this page, on your device.</p>")
tabs = ('\n  <div class="tabs" role="tablist" aria-label="Where Bolt is">'
        '<button type="button" role="tab" id="tabTalk" aria-selected="true" aria-controls="talkView">Talk and play</button>'
        '<button type="button" role="tab" id="tabYard" aria-selected="false" aria-controls="yardView">Day out in the yard</button></div>\n')
main_start = top_end
foot_i = c.index("  <footer>", main_start)
talk = c[main_start:foot_i]
c = (c[:c.index(a)] + '<div class="wrap">\n  ' + new_top + tabs + '  <div id="talkView" role="tabpanel" aria-labelledby="tabTalk">' + talk
     + '  </div>\n  <section id="yardView" role="tabpanel" aria-labelledby="tabYard" hidden>' + d_body + "  </section>\n" + c[foot_i:])
# Companion's loop steps only while its tab shows
a = "  function tick(t) {\n    const wall = (t - last) / 1000, dt = Math.min(0.1, wall); last = t;\n"
assert c.count(a) == 1, "companion tick"
c = c.replace(a, "  function tick(t) {\n    if (window.__view === \"yard\") { last = t; requestAnimationFrame(tick); return; }\n"
              "    const wall = (t - last) / 1000, dt = Math.min(0.1, wall); last = t;\n")
# the tab switch, then the yard's script (after Companion's)
switch = """<script>
(() => {
  window.__view = "talk";
  const tabs = { talk: document.getElementById("tabTalk"), yard: document.getElementById("tabYard") };
  const views = { talk: document.getElementById("talkView"), yard: document.getElementById("yardView") };
  function show(v) {
    window.__view = v;
    for (const k of ["talk", "yard"]) { views[k].hidden = k !== v; tabs[k].setAttribute("aria-selected", String(k === v)); }
    if (v === "yard" && window.__startYard) window.__startYard();
    try { location.hash = v === "yard" ? "yard" : ""; } catch (e) {}
  }
  tabs.talk.addEventListener("click", () => show("talk"));
  tabs.yard.addEventListener("click", () => show("yard"));
  window.addEventListener("hashchange", () => show(location.hash === "#yard" ? "yard" : "talk"));
  window.__showView = show;
  if (location.hash === "#yard") requestAnimationFrame(() => show("yard"));
})();
</script>
"""
last_script_end = c.rindex("</script>") + 9
c = c[:last_script_end] + "\n" + d_script.replace("<script src=", "<!-- -->") + "\n" + switch + c[last_script_end:]
assert not re.search(r"\{\{\w+\}\}", c)
assert c.count('id="stage"') == 1 and c.count('id="loading"') == 1 and c.count('id="yStage"') == 1
open(out_path, "w").write(c)
print(f"wrote {out_path}: {len(c) // 1024} KB")
