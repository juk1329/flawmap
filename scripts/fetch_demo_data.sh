#!/usr/bin/env bash
# Everything the web demo needs on a fresh clone, without the full datasets:
#   1. uCO3D metadata.sqlite (4.9 GB download, 13 GB unpacked) and the public demo videos listed in
#      configs/demo_public.yaml (video, mask, segmented point cloud; ~0.2 GB)
#   2. the synthetic captures the request page offers (rendered here, ~10 min for the small ones,
#      ~5 min more per 166-frame demo face; --no-full skips the two demo faces)
#   3. upload zips of the synthetic captures and three.js for the viewer
#
#   bash scripts/fetch_demo_data.sh [--no-full] [--no-uco3d]
# Needs about 25 GB free on top of the project's 50 GB floor while the metadata unpacks.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate env_isaaclab
export PYTHONPATH=
cd "$ROOT"
full=1; uco=1
for a in "$@"; do
  case "$a" in --no-full) full=0 ;; --no-uco3d) uco=0 ;; *) echo "unknown option $a"; exit 2 ;; esac
done
mkdir -p runs/logs

if [ "$uco" = 1 ]; then
  names=$(python -c "import yaml; print(' '.join(e['name'] for e in yaml.safe_load(open('configs/demo_public.yaml'))['uco3d']))")
  echo "== uCO3D: metadata + $names"
  python -m flawmap.io.download uco3d --sequences $names --kinds rgb_videos mask_videos segmented_point_clouds \
    2>&1 | tee -a runs/logs/download_uco3d_demo.log
fi

render() {   # out-name, then render_capture options
  local out="data/synthetic/$1"; shift
  if [ -f "$out/gt.json" ]; then echo "== $out exists"; return; fi
  echo "== rendering $out"
  python -m flawmap.synth.render_capture "$out" "$@" 2>&1 | tail -1
}
render iphone12_dent_small --face-up back --preset dents --seed 21
render iphone12_back_small --face-up back --preset small-v1 --seed 1
render iphone12_wood_small --face-up back --preset small --background wood --seed 11
render iphone12_paper_small --face-up back --preset small --background paper --seed 12
if [ "$full" = 1 ]; then
  render iphone12_demo_back --face-up back --seed 11
  render iphone12_demo_screen --face-up screen --seed 12
fi

bash scripts/make_sample_zips.sh
bash scripts/fetch_web_vendor.sh
echo "== done. Start the service: flawmap serve --port 8000  (or bash scripts/serve_detached.sh --port 8000)"
