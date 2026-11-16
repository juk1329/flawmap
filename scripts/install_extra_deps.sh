#!/usr/bin/env bash
# Install the extra Python packages flawmap needs into the existing `env_isaaclab` conda
# environment (Python 3.11). Never creates a new environment.
#
# Usage:  bash scripts/install_extra_deps.sh [--dry-run] [--extras]
#   --extras  also installs the packages for the optional parts (3DGS, learned detector)
set -euo pipefail

source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate env_isaaclab
export PYTHONPATH=

PKGS=(
  "scikit-image"            # ridge filters, morphology, sub-pixel edges
  "plyfile"                 # Gaussian splat / point cloud PLY files
  "gdown"                   # OmniObject3D public Google Drive archives
  "remotezip"               # extract single files from remote zips (uCO3D) with HTTP range requests
)
EXTRAS=(
  "gsplat"                  # 3D Gaussian splatting (compiled with nvcc on first use)
  "segmentation-models-pytorch"
)

DRY=0; WITH_EXTRAS=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --extras) WITH_EXTRAS=1 ;;
  esac
done
if (( WITH_EXTRAS )); then PKGS+=("${EXTRAS[@]}"); fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Constraints keep the pins of the environment intact (numpy 1.26.0, psutil 5.9.8, torch 2.7.0+cu128).
CONSTRAINTS=(-c "$HERE/constraints-isaacsim.txt")

if (( DRY )); then
  pip install --dry-run "${CONSTRAINTS[@]}" "${PKGS[@]}"
  exit 0
fi

pip install "${CONSTRAINTS[@]}" "${PKGS[@]}"
pip check || true
python - <<'PY'
import importlib
for m in ["skimage", "plyfile", "gdown", "remotezip", "numpy", "torch", "cv2", "open3d", "trimesh", "manifold3d"]:
    mod = importlib.import_module(m)
    print(f"{m:12s} {getattr(mod, '__version__', 'ok')}")
PY
