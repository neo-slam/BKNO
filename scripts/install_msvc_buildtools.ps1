Param(
    [string]$InstallerPath = "$env:TEMP\vs_BuildTools.exe"
)

$ErrorActionPreference = "Stop"

Write-Host "Downloading Visual Studio Build Tools installer..."
Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vs_BuildTools.exe" -OutFile $InstallerPath

Write-Host "Launching unattended install (this may take a while)..."
Start-Process -FilePath $InstallerPath -ArgumentList @(
    "--quiet",
    "--wait",
    "--norestart",
    "--nocache",
    "--installPath", "C:\BuildTools",
    "--add", "Microsoft.VisualStudio.Workload.VCTools",
    "--add", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
    "--add", "Microsoft.VisualStudio.Component.Windows10SDK.19041"
) -Wait -NoNewWindow

Write-Host "Done. Open a new terminal and run scripts\run_cpp_test_with_msvc.ps1"

