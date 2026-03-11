Param(
    [string]$PythonExe = "D:\code\env\env-torch\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

function Check-Cmd($name) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        Write-Host ("[MISSING] {0}" -f $name)
        return $false
    }
    Write-Host ("[OK] {0} -> {1}" -f $name, $cmd.Source)
    return $true
}

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

Write-Host "=== BKNO C++ Build Environment Check ==="
Write-Host ("PWD: {0}" -f (Get-Location).Path)

$okPython = Test-Path $PythonExe
if (-not $okPython) {
    Write-Host ("[MISSING] python exe: {0}" -f $PythonExe)
} else {
    Write-Host ("[OK] python exe: {0}" -f $PythonExe)
    & $PythonExe -c "import torch; print('[OK] torch', torch.__version__)"
}

$okCl = Check-Cmd "cl"
$okNinja = Check-Cmd "ninja"
$okCmake = Check-Cmd "cmake"
$vsDevCmd = Resolve-VsDevCmd
if ($null -ne $vsDevCmd) {
    Write-Host ("[OK] VsDevCmd -> {0}" -f $vsDevCmd)
} else {
    Write-Host "[MISSING] VsDevCmd.bat"
}

if ((-not $okCl) -and ($null -eq $vsDevCmd)) {
    Write-Host ""
    Write-Host "MSVC compiler is not in PATH."
    Write-Host "Install Visual Studio Build Tools 2022 with:"
    Write-Host "- Desktop development with C++"
    Write-Host "- MSVC v143 x64/x86 build tools"
    Write-Host "- Windows 10/11 SDK"
}

if (($okCl -or ($null -ne $vsDevCmd)) -and $okPython -and $okNinja) {
    Write-Host ""
    Write-Host "Environment looks ready for BKNO C++ extension build."
} else {
    Write-Host ""
    Write-Host "Environment is not ready yet."
}
