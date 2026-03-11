Param(
    [string]$PythonExe = "D:\code\env\env-torch\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

function Resolve-VsDevCmd {
    $candidates = @(
        "C:\BuildTools\Common7\Tools\VsDevCmd.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            return $c
        }
    }
    $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $installPath) {
            $probe = Join-Path $installPath "Common7\Tools\VsDevCmd.bat"
            if (Test-Path $probe) {
                return $probe
            }
        }
    }
    return $null
}

$vsDev = Resolve-VsDevCmd
if ($null -eq $vsDev) {
    Write-Error "VsDevCmd.bat not found. Install Visual Studio Build Tools 2022 (C++ workload) first."
}

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python not found at $PythonExe"
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$testPy = Join-Path $repo "scripts\test_bkno_cpp.py"

if (-not (Test-Path $testPy)) {
    Write-Error "Test script not found: $testPy"
}

# Run everything in one cmd session so MSVC env vars are active for python build.
$cmd = "`"$vsDev`" -arch=x64 -host_arch=x64 && `"$PythonExe`" `"$testPy`""
cmd.exe /c $cmd
