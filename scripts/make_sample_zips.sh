#!/usr/bin/env bash
# Zip the synthetic captures the way a seller uploads a Stray Scanner recording (the folder itself
# inside the zip), for trying the upload page: data/service/samples/<name>.zip
set -eu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$ROOT/data/service/samples"
mkdir -p "$out"
cd "$ROOT/data/synthetic"
for d in */; do
  d="${d%/}"
  [ -f "$d/rgb.mp4" ] || continue
  rm -f "$out/$d.zip"
  python3 -c "import shutil,sys; shutil.make_archive(sys.argv[1], 'zip', '.', sys.argv[2])" "$out/$d" "$d"
  echo "$out/$d.zip $(du -h "$out/$d.zip" | cut -f1)"
done
