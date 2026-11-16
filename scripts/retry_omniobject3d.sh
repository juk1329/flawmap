#!/usr/bin/env bash
# OmniObject3D's public Google Drive files hit Drive's "too many users downloaded this file"
# quota on 2026-11-15. Retry every hour for up to 12 hours; finished categories are skipped.
# (The OpenXLab route was not used: installing the `openxlab` client would downgrade requests,
# urllib3, setuptools and rich in env_isaaclab.)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate env_isaaclab
export PYTHONPATH=
cd "$ROOT"
log="$ROOT/runs/logs/download_omniobject3d.log"
for round in $(seq 1 12); do
  echo "=== omniobject3d retry round $round $(date) ===" >>"$log"
  FLAWMAP_DISK_PATH="$ROOT" bash "$HERE/with_disk_guard.sh" 20 python -m flawmap.io.download omniobject3d --attempts 1 >>"$log" 2>&1
  n_done=$(ls data/datasets/omniobject3d/videos_processed/.*.done data/datasets/omniobject3d/raw_scans/.*.done 2>/dev/null | wc -l)
  echo "=== round $round: $n_done of 14 archives done, $(df -h --output=avail "$ROOT" | tail -1) free ===" >>"$log"
  if [ "$n_done" -ge 14 ]; then exit 0; fi
  sleep 3600
done
