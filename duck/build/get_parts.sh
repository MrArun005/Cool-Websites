#!/usr/bin/env bash
# Download the Open Duck Mini v2 printable parts and docs into duck/parts/.
# Pinned to the commit this build plan was checked against.
set -euo pipefail

COMMIT=b23317a485b3cec7d8417f352478778b3475173c
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HERE/parts"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git -C "$TMP" init -q
git -C "$TMP" remote add origin https://github.com/apirrone/Open_Duck_Mini
git -C "$TMP" fetch -q --depth 1 origin "$COMMIT"
git -C "$TMP" checkout -q FETCH_HEAD

mkdir -p "$DEST"
cp -r "$TMP/print" "$DEST/"
cp "$TMP/docs/print_guide.md" "$TMP/docs/assembly_guide.md" "$TMP/docs/configure_motors.md" \
   "$TMP/docs/open_duck_mini_v2_wiring_diagram.png" "$TMP/LICENSE" "$DEST/"
echo "Parts and guides are in $DEST ($(ls "$DEST/print"/*.stl | wc -l) STL files)."
