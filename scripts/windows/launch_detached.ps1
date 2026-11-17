<#
Start a long flawmap command that keeps running after the shell (or the Claude session) that started
it ends. Start-Process children were stopped together with the session (2026-11-17); a process
created through WMI (Win32_Process.Create) is not part of that process tree.

  powershell -ExecutionPolicy Bypass -File scripts\windows\launch_detached.ps1 -Name ipad_max -PyArgs "-m flawmap.cli run --device auto ..."

Runs `python -m flawmap.utils.memguard python <args>` in the repository with the flawmap env, and
writes runs\logs\<Name>.log (stdout and stderr) and runs\logs\<Name>.pid.
#>
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$EnvName = "flawmap",
    [int]$GuardMinKB = 4000000,
    [Parameter(Mandatory = $true)][string]$PyArgs     # everything after python, as one string
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
$envDir = Join-Path (Split-Path (Split-Path (Get-Command conda).Source)) "envs\$EnvName"
if (-not (Test-Path "$envDir\python.exe")) { $envDir = "$env:USERPROFILE\miniconda3\envs\$EnvName" }
$py = "$envDir\python.exe"
New-Item -ItemType Directory -Force "$Root\runs\logs" | Out-Null
$logf = "$Root\runs\logs\$Name.log"
$quoted = $PyArgs
$cmd = "cmd.exe /c `"set PYTHONUTF8=1&& set PYTHONPATH=&& set FLAWMAP_GUARD_MIN_KB=$GuardMinKB&& " +
       "set PATH=$envDir;$envDir\Library\bin;$envDir\Scripts;%PATH%&& cd /d `"$Root`" && " +
       "`"$py`" -m flawmap.utils.memguard `"$py`" $quoted > `"$logf`" 2>&1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $Root }
if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create failed: $($r.ReturnValue)" }
Set-Content "$Root\runs\logs\$Name.pid" $r.ProcessId
"started $Name (pid $($r.ProcessId)), log $logf"
