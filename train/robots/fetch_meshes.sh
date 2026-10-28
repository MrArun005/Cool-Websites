#!/bin/bash
# The visual meshes of the Menagerie A1/Go2 models (19 MB + 28 MB of .obj), kept out of git. Physics does not need
# them: every collision geom is a primitive, and *_nomesh.xml (strip_meshes.py) load without them.
#   bash train/robots/fetch_meshes.sh        # puts assets/ next to a1.xml and go2.xml, so a1.xml / scene.xml load as published
set -e
REV=f3b6013629c7b94f8ee93fd977b161fa8350f9bc   # mujoco_menagerie main, 2026-10-23 (the commit the xml here came from)
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d)
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie "$TMP/m"
cd "$TMP/m" && git checkout -q $REV && git sparse-checkout set unitree_a1 unitree_go2
for r in unitree_a1 unitree_go2; do rm -rf "$HERE/$r/assets"; cp -r "$TMP/m/$r/assets" "$HERE/$r/assets"; done
rm -rf "$TMP"
echo "meshes in $HERE/unitree_a1/assets and $HERE/unitree_go2/assets"
