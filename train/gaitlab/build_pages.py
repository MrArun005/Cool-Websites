"""Assemble Bolt's Castle, the Agility Cup, Bolt at home, Bolt's Agility Run and Bolt's Day Out (doglife) from their templates plus the Gait Lab's 3D code.

    python build_pages.py <scratchpad> [page names, default: castle agility home agilityrun; also doglife, remote]

Bolt Remote (remote): copy remote_template.html to <scratchpad>/remote/template.html; the site folder also needs
live.js doglife.js game.js imitate.js hurdles3d.js remote.js, mujoco/ and data/ (the same files as Bolt's Day Out).
Bolt Run (run): copy run_template.html to <scratchpad>/run/template.html; the site folder needs live.js doglife.js game.js
imitate.js hurdles3d.js run.js, mujoco/ and data/ (the same as Bolt Remote, with run.js in place of remote.js).
"""
import sys, re, os
S = sys.argv[1]
hub = open(f"{S}/hub/index.html").read().split("\n")
css_start = next(i for i, l in enumerate(hub) if l.startswith("<style>")) + 1
css_end = next(i for i, l in enumerate(hub) if l.startswith("</style>"))
base_css = "\n".join(l for l in hub[css_start:css_end] if "/* Layout" not in l and "simulator videos, the training history" not in l)
s0 = next(i for i, l in enumerate(hub) if "// ---------- 3D scene" in l)
s1 = next(i for i, l in enumerate(hub) if "const dog = new THREE.Group(); scene.add(dog);" in l)
scene = "\n".join(hub[s0:s1 + 1])
castle = open(f"{S}/castle/template.html").read()
a = castle.index("     course list and what the announcer said. Same look as the Gait Lab. */") + len("     course list and what the announcer said. Same look as the Gait Lab. */\n")
game_css = castle[a:castle.index("</style>")].rstrip()
# the front yard of Bolt's Day Out (its 3D scenery, Bolt's body and the ball), shared by Bolt Remote ({{YARD_JS}})
here = os.path.dirname(os.path.abspath(__file__))
dl = open(os.path.join(here, "doglife_template.html")).read()
yard = dl[dl.index("  // ---------- daylight: sky, haze, sun ----------"):dl.index("  // ---------- the day ----------")].rstrip()
for name in sys.argv[2:] or ("castle", "agility", "home", "agilityrun"):
    if not os.path.exists(f"{S}/{name}/template.html"):
        continue
    page = open(f"{S}/{name}/template.html").read().replace("{{BASE_CSS}}", base_css).replace("{{GAME_CSS}}", game_css).replace("{{SCENE_JS}}", scene).replace("{{YARD_JS}}", yard)
    assert not re.findall(r"\{\{\w+\}\}", page)
    open(f"{S}/{name}/site/index.html", "w").write(page)
    print(name, len(page))
