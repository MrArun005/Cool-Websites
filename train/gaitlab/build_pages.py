"""Assemble Bolt's Castle, the Agility Cup and Bolt at home from their templates plus the Gait Lab's 3D code."""
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
for name in ("castle", "agility", "home"):
    if not os.path.exists(f"{S}/{name}/template.html"):
        continue
    page = open(f"{S}/{name}/template.html").read().replace("{{BASE_CSS}}", base_css).replace("{{GAME_CSS}}", game_css).replace("{{SCENE_JS}}", scene)
    assert not re.findall(r"\{\{\w+\}\}", page)
    open(f"{S}/{name}/site/index.html", "w").write(page)
    print(name, len(page))
