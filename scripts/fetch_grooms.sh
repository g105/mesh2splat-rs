#!/usr/bin/env bash
# Download the hair grooms used by the README and examples/groom.rs into assets/.
#
# The models are Cem Yuksel's (https://www.cemyuksel.com/research/hairmodels),
# free for personal and research use. If you publish anything made from them,
# link to that page; acknowledgements are appreciated. They are not in the
# repository for that reason.
#
# Usage: scripts/fetch_grooms.sh [name ...]   (default: straight wCurly)
set -euo pipefail

BASE="${M2S_HAIR_BASE:-https://www.cemyuksel.com/research/hairmodels}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/assets"
mkdir -p "$DEST"
names=("$@")
[ ${#names[@]} -eq 0 ] && names=(straight wCurly)

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

is_hair() { [ "$(head -c 4 "$1" 2>/dev/null)" = "HAIR" ]; }

for name in "${names[@]}"; do
    out="$DEST/$name.hair"
    if is_hair "$out"; then
        echo "$name.hair: already there"
        continue
    fi
    # The site has offered the models both as plain .hair files and zipped.
    if curl -fsL -o "$tmp/$name.hair" "$BASE/$name.hair" && is_hair "$tmp/$name.hair"; then
        mv "$tmp/$name.hair" "$out"
    elif curl -fsSL -o "$tmp/$name.zip" "$BASE/$name.zip"; then
        unzip -qo "$tmp/$name.zip" -d "$tmp/$name"
        found="$(find "$tmp/$name" -iname '*.hair' | head -n 1)"
        if [ -z "$found" ] || ! is_hair "$found"; then
            echo "$name: the zip holds no .hair file" >&2
            exit 1
        fi
        mv "$found" "$out"
    else
        echo "$name: not found at $BASE/$name.hair or $BASE/$name.zip;" \
             "download it by hand from $BASE into assets/" >&2
        exit 1
    fi
    echo "$name.hair: $(du -h "$out" | cut -f1)"
done
