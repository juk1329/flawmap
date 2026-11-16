#!/usr/bin/env bash
# Download the public datasets used by flawmap into data/datasets/ (or $FLAWMAP_DATA_ROOT/datasets).
# Each dataset runs under the disk guard (start only if free - expected >= 80 GB, stop below 60 GB)
# and logs to runs/logs/download_<name>.log.
#
#   bash scripts/download_datasets.sh [omniobject3d] [mobilebrick] [uco3d]      # no argument: all three
#
# Sizes (compressed): OmniObject3D 7 electronics categories ~20 GB, MobileBrick 13.1 GB,
# uCO3D cellular_telephone (video, masks, splats, segmented point clouds) + metadata ~55-60 GB.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate env_isaaclab
export PYTHONPATH=
mkdir -p "$ROOT/runs/logs"
cd "$ROOT"

declare -A EXPECTED=( [omniobject3d]=20 [mobilebrick]=27 [uco3d]=65 )
targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then targets=(omniobject3d mobilebrick uco3d); fi
for t in "${targets[@]}"; do
  log="$ROOT/runs/logs/download_${t}.log"
  echo "=== $t $(date) ===" | tee -a "$log"
  FLAWMAP_DISK_PATH="$ROOT" bash "$HERE/with_disk_guard.sh" "${EXPECTED[$t]}" python -m flawmap.io.download "$t" >>"$log" 2>&1
  echo "=== $t finished rc=$? $(date), $(df -h --output=avail "$ROOT" | tail -1) free ===" | tee -a "$log"
done
