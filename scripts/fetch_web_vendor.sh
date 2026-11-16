#!/usr/bin/env bash
# Fetch the pinned browser libraries for the viewer into web/vendor/ (not committed), so the demo
# runs without internet. three.js r180 from the jsDelivr npm mirror.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V="$HERE/../web/vendor/three"
VER=0.180.0
BASE="https://cdn.jsdelivr.net/npm/three@${VER}"
files=(
  build/three.module.js
  build/three.core.js
  examples/jsm/controls/OrbitControls.js
  examples/jsm/loaders/GLTFLoader.js
  examples/jsm/utils/BufferGeometryUtils.js
)
for f in "${files[@]}"; do
  mkdir -p "$V/$(dirname "$f")"
  curl -s -f -L -m 120 -o "$V/$f" "$BASE/$f"
  echo "$(wc -c < "$V/$f") $f"
done
echo "three@${VER}" > "$V/VERSION"
