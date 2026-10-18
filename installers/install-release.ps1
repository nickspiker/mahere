# Mahere's one-line installer for Windows (Nick 2026-10-17: Photon's, one paste and done):
#   powershell -ExecutionPolicy Bypass -c "irm https://brobdingnagian.holdmyoscilloscope.com/mahere/install-release.ps1 | iex"
# The download is checked against the SHA-256 release.sh writes in below (Defender may block running a fresh download, so it is not run to check itself); after that Mahere updates itself.

$ErrorActionPreference = "Stop"
trap {
    Write-Host ""
    Write-Host "ERROR: $_" -ForegroundColor Red
    Write-Host "At: $($_.InvocationInfo.ScriptLineNumber): $($_.InvocationInfo.Line.Trim())" -ForegroundColor Red
    Write-Host ""
    Write-Host "Press Enter to exit..." -ForegroundColor Yellow
    Read-Host
    exit 1
}

$base = "https://brobdingnagian.holdmyoscilloscope.com/mahere"
$expectedHash = "0000000000000000000000000000000000000000000000000000000000000000"

Write-Host "Mahere installer" -ForegroundColor Cyan
Write-Host "================" -ForegroundColor Cyan
Write-Host ""

# x64 Windows runs it natively; ARM64 Windows runs the x64 build under its emulation.
$arch = $env:PROCESSOR_ARCHITECTURE
if ($arch -ne "AMD64" -and $arch -ne "ARM64") {
    Write-Host "Error: unsupported architecture: $arch" -ForegroundColor Red
    exit 1
}
Write-Host "Detected: Windows ($arch)" -ForegroundColor White
Write-Host ""

# Straight to the install folder (Defender often blocks TEMP).
$installDir = "$env:LOCALAPPDATA\Programs\Mahere"
New-Item -ItemType Directory -Path $installDir -Force | Out-Null
$binaryPath = "$installDir\mahere.exe"
$partPath = "$installDir\mahere.exe.part"

Write-Host "Downloading Mahere..." -ForegroundColor Yellow
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest -Uri "$base/mahere-windows-x86_64.exe" -OutFile $partPath -UseBasicParsing

Write-Host "Verifying integrity..." -ForegroundColor Yellow
$actualHash = (Get-FileHash $partPath -Algorithm SHA256).Hash
if ($actualHash -ne $expectedHash) {
    Write-Host "Error: the download is not the published Mahere." -ForegroundColor Red
    Write-Host "  Expected: $expectedHash" -ForegroundColor Red
    Write-Host "  Got:      $actualHash" -ForegroundColor Red
    Remove-Item $partPath -ErrorAction SilentlyContinue
    exit 1
}
Move-Item -Path $partPath -Destination $binaryPath -Force
Write-Host "[OK] Integrity verified, installed to $installDir" -ForegroundColor Green
Write-Host ""

$iconPath = "$installDir\mahere.ico"
try {
    Invoke-WebRequest -Uri "$base/mahere.ico" -OutFile $iconPath -UseBasicParsing
} catch {
    $iconPath = $null
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$installDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$installDir", "User")
    $env:Path += ";$installDir"
    Write-Host "[OK] Added to PATH" -ForegroundColor Green
}

$startMenu = [System.IO.Path]::Combine($env:APPDATA, "Microsoft\Windows\Start Menu\Programs")
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut([System.IO.Path]::Combine($startMenu, "Mahere.lnk"))
$Shortcut.TargetPath = $binaryPath
$Shortcut.Description = "Mahere - offline map for hiking, biking and backroads"
$Shortcut.WorkingDirectory = $installDir
if ($iconPath) { $Shortcut.IconLocation = $iconPath }
$Shortcut.Save()
Write-Host "[OK] Start Menu shortcut created" -ForegroundColor Green
Write-Host ""

Write-Host "Mahere installed." -ForegroundColor Green
Write-Host "Open it from the Start Menu, or run 'mahere' in a new terminal." -ForegroundColor White
Write-Host "Mahere updates itself from here on: Settings > Updates." -ForegroundColor White
Write-Host ""
