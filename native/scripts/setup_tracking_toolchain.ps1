[CmdletBinding()]
param([string]$VisualStudioRoot, [switch]$Offline, [switch]$CompilerOnly)
$ErrorActionPreference = 'Stop'
if (!$IsWindows -or ![Environment]::Is64BitProcess) { throw 'Run in Windows x64 PowerShell 7.' }
$nativeRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (!$VisualStudioRoot) {
  $locator = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
  $VisualStudioRoot = & $locator -latest -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
$VisualStudioRoot = $VisualStudioRoot | Select-Object -First 1
$cmake = Join-Path $VisualStudioRoot 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
if (!(Test-Path -LiteralPath $cmake)) { throw 'Install Visual Studio 2022 C++ tools, Windows SDK and CMake tools first.' }
$lock = Get-Content (Join-Path $nativeRoot 'toolchain.lock.json') -Raw | ConvertFrom-Json
foreach ($asset in $lock.assets) {
  if ($CompilerOnly -and $asset.name -ne 'llvm') { continue }
  $file = Join-Path $nativeRoot $asset.file
  New-Item -ItemType Directory -Force (Split-Path $file) | Out-Null
  if (!(Test-Path -LiteralPath $file)) {
    if ($Offline) { throw "Missing offline asset: $file" }
    Invoke-WebRequest -Uri $asset.url -OutFile "$file.partial"
    $actual = (Get-FileHash -LiteralPath "$file.partial" -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $asset.sha256) { throw "Checksum mismatch: $($asset.name)" }
    Move-Item -LiteralPath "$file.partial" -Destination $file
  }
  if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $asset.sha256) {
    throw "Checksum mismatch: $($asset.name)"
  }
  if ($asset.directory) {
    $directory = Join-Path $nativeRoot $asset.directory
    $marker = Join-Path $directory '.nijitrack-verified-archive'
    if (!(Test-Path -LiteralPath $marker) -or (Get-Content -LiteralPath $marker -Raw).Trim() -ne $asset.sha256) {
      $destination = Join-Path $nativeRoot $asset.destination
      New-Item -ItemType Directory -Force $destination | Out-Null
      Push-Location $destination
      try { & $cmake -E tar xf $file; if ($LASTEXITCODE) { throw "Extraction failed: $($asset.name)" } }
      finally { Pop-Location }
      Set-Content -LiteralPath $marker -Value $asset.sha256 -Encoding ascii
    }
  }
  Write-Output "Verified $($asset.name) $($asset.version)"
}
$configuration = [ordered]@{
  native_root = $nativeRoot
  visual_studio = $VisualStudioRoot
  cmake = $cmake
  ninja = (Join-Path $VisualStudioRoot 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe')
  llvm = (Join-Path $nativeRoot 'vendor/toolchain/clang+llvm-19.1.7-x86_64-pc-windows-msvc')
  halide = (Join-Path $nativeRoot 'vendor/toolchain/Halide-15.0.1-x86-64-windows')
  sdl3 = (Join-Path $nativeRoot 'vendor/toolchain/SDL3-3.4.18')
  bazel = (Join-Path $nativeRoot 'vendor/toolchain/bazel.exe')
  mediapipe = (Join-Path $nativeRoot ($lock.assets | Where-Object name -eq 'mediapipe-source').directory)
}
$configuration | ConvertTo-Json | Set-Content (Join-Path $nativeRoot 'vendor/toolchain/environment.json') -Encoding utf8
if ($CompilerOnly) {
  Write-Output 'Compiler ready. Run scripts/build_native_windows.ps1 to build nijitrack with clang-cl.'
} else {
  Write-Output 'Ready. Run scripts/test_tracking_toolchain.ps1 to validate the native tools without Python.'
}
