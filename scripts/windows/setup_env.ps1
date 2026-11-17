<#
Windows development environment for flawmap.

The team server runs Ubuntu with the conda env `env_isaaclab` (docs/project_guide.md 2.1). On
Windows there is no such env, so this creates a separate one with the same Python and the same
pinned versions where flawmap uses them:

  1. conda env `flawmap` (Python 3.11), unless it exists
  2. scripts/windows/requirements.txt, constrained by scripts/constraints-isaacsim.txt
  3. flawmap itself in editable mode (`flawmap` command)
  4. env variables: PYTHONUTF8=1 (Korean text files), empty PYTHONPATH,
     FLAWMAP_GUARD_MIN_KB (memory guard floor for service jobs)

  powershell -ExecutionPolicy Bypass -File scripts\windows\setup_env.ps1 [-EnvName flawmap] [-GuardMinKB 4000000]
#>
param(
    [string]$EnvName = "flawmap",
    [int]$GuardMinKB = 4000000
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path

function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" } }

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda not found. Install Miniconda, then open a new PowerShell."
}

if (-not ((conda env list) -match "^\s*$EnvName\s")) {
    Write-Host "== creating conda env $EnvName (Python 3.11)"
    conda create -y -n $EnvName python=3.11 pip; Check "conda create"
}

(& conda "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate $EnvName

Write-Host "== installing packages"
python -m pip install -r "$Root\scripts\windows\requirements.txt" -c "$Root\scripts\constraints-isaacsim.txt"; Check "pip install"
python -m pip install -e "$Root" --no-deps --no-build-isolation; Check "pip install -e"

Write-Host "== env variables"
conda env config vars set -n $EnvName PYTHONUTF8=1 PYTHONPATH= FLAWMAP_GUARD_MIN_KB=$GuardMinKB | Out-Null
conda env config vars list -n $EnvName

Write-Host "== check"
python -c "import numpy, cv2, open3d, warp, trimesh, skimage, fastapi, flawmap; warp.config.quiet = True; warp.init(); print('numpy', numpy.__version__, '| opencv', cv2.__version__, '| open3d', open3d.__version__, '| warp', warp.__version__, '| CUDA', warp.is_cuda_available())"
Check "import check"
Write-Host "== done. Next: conda activate $EnvName; pytest"
