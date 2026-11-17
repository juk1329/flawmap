<#
Windows twin of scripts/fetch_demo_data.sh: what the web demo needs on a fresh clone.

  1. the synthetic captures the request page offers (rendered here; -NoFull skips the two
     166-frame demo faces)
  2. upload zips of them in data/service/samples/
  3. three.js for the viewer in web/vendor/three/ (same files and version as fetch_web_vendor.sh)
  4. only with -Uco3d: the uCO3D metadata (4.9 GB download, 13 GB unpacked, about 25 GB free
     needed meanwhile) and the public demo videos of configs/demo_public.yaml

  powershell -ExecutionPolicy Bypass -File scripts\windows\fetch_demo_data.ps1 [-NoFull] [-Uco3d] [-EnvName flawmap]
#>
param(
    [switch]$NoFull,
    [switch]$Uco3d,
    [string]$EnvName = "flawmap"
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $Root

function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" } }

(& conda "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate $EnvName
New-Item -ItemType Directory -Force "runs\logs" | Out-Null

if ($Uco3d) {
    $names = (python -c "import yaml; print(' '.join(e['name'] for e in yaml.safe_load(open('configs/demo_public.yaml', encoding='utf-8'))['uco3d']))").Trim() -split "\s+"
    Write-Host "== uCO3D: metadata + $($names -join ' ')"
    python -m flawmap.io.download uco3d --sequences @names --kinds rgb_videos mask_videos segmented_point_clouds; Check "uCO3D download"
}

function Render([string]$Name, [string[]]$Opts) {
    $out = "data\synthetic\$Name"
    if (Test-Path "$out\gt.json") { Write-Host "== $out exists"; return }
    Write-Host "== rendering $out"
    python -m flawmap.synth.render_capture $out @Opts; Check "render $Name"
}
Render "iphone12_dent_small" @("--face-up", "back", "--preset", "dents", "--seed", "21")
Render "iphone12_back_small" @("--face-up", "back", "--preset", "small-v1", "--seed", "1")
Render "iphone12_wood_small" @("--face-up", "back", "--preset", "small", "--background", "wood", "--seed", "11")
Render "iphone12_paper_small" @("--face-up", "back", "--preset", "small", "--background", "paper", "--seed", "12")
if (-not $NoFull) {
    Render "iphone12_demo_back" @("--face-up", "back", "--seed", "11")
    Render "iphone12_demo_screen" @("--face-up", "screen", "--seed", "12")
}

# Upload zips: the capture folder itself inside the zip, like a Stray Scanner recording zipped in
# the Files app. Python's zipfile keeps "/" separators (PowerShell 5.1 Compress-Archive does not).
Write-Host "== upload zips"
$samples = "data\service\samples"
New-Item -ItemType Directory -Force $samples | Out-Null
Get-ChildItem "data\synthetic" -Directory | Where-Object { Test-Path "$($_.FullName)\rgb.mp4" } | ForEach-Object {
    $zip = Join-Path $samples $_.Name
    python -c "import shutil, sys; shutil.make_archive(sys.argv[1], 'zip', sys.argv[2], sys.argv[3])" $zip "data\synthetic" $_.Name
    Check "zip $($_.Name)"
    Write-Host ("{0}.zip {1:N1} MB" -f $zip, ((Get-Item "$zip.zip").Length / 1MB))
}

Write-Host "== three.js"
$ver = "0.180.0"
$base = "https://cdn.jsdelivr.net/npm/three@$ver"
$vendor = "web\vendor\three"
$files = @(
    "build/three.module.js",
    "build/three.core.js",
    "examples/jsm/controls/OrbitControls.js",
    "examples/jsm/loaders/GLTFLoader.js",
    "examples/jsm/utils/BufferGeometryUtils.js"
)
foreach ($f in $files) {
    $dst = Join-Path $vendor $f
    New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
    Invoke-WebRequest -UseBasicParsing -TimeoutSec 120 -Uri "$base/$f" -OutFile $dst
    Write-Host ("{0} {1}" -f (Get-Item $dst).Length, $f)
}
[IO.File]::WriteAllText((Join-Path $Root "$vendor\VERSION"), "three@$ver`n")

Write-Host "== done. Start the service: conda activate $EnvName; flawmap serve --port 8000"
