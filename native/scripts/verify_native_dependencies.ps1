[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$Directory)
$ErrorActionPreference = 'Stop'
if (!$IsWindows) { throw 'Use bash native/build.sh on Linux/macOS.' }
. (Join-Path $PSScriptRoot 'enter_tracking_environment.ps1')
$directoryPath = (Resolve-Path -LiteralPath $Directory).Path
$report = [Collections.Generic.List[string]]::new()
$forbidden = 'opencv|ffmpeg|ffprobe|^(lib)?(avcodec|avdevice|avfilter|avformat|avutil|swresample|swscale)[-.]|^(lib)?mediapipe[.-]|^(lib)?python[0-9.]*[.-]|^(lib)?Halide[.-]|^(msvcp|vcruntime|concrt|vccorlib)\d.*\.dll$|^api-ms-win-crt-|^ucrtbase\.dll$'
# Scan shipped SDL libraries as well as the executable directory on Unix.
$directories = @($directoryPath)
if (!$IsWindows -and (Split-Path $directoryPath -Leaf) -eq 'bin') {
  $libraryDirectory = Join-Path (Split-Path $directoryPath) 'lib'
  if (Test-Path -LiteralPath $libraryDirectory) { $directories += $libraryDirectory }
}
if ($IsWindows) { $reader = Join-Path $trackingEnvironment.llvm 'bin/llvm-readobj.exe' }
elseif ($IsMacOS) { $reader = (Get-Command otool -ErrorAction Stop).Source }
else { $reader = (Get-Command readelf -ErrorAction Stop).Source }
$inspected = 0
foreach ($scan in $directories) {
  foreach ($file in (Get-ChildItem -LiteralPath $scan -File)) {
    # Check magic bytes: Unix executables have no mandatory extension. Ignore
    # models, static archives, Ninja files and other non-runtime build products.
    $stream = [IO.File]::OpenRead($file.FullName)
    try { $magic = [byte[]]::new(4); $count = $stream.Read($magic,0,4) } finally { $stream.Dispose() }
    if ($count -ne 4) { continue }
    $signature = [Convert]::ToHexString($magic)
    if ($IsWindows) { $binary = $signature.StartsWith('4D5A') }
    elseif ($IsMacOS) { $binary = $signature -in @('CFFAEDFE','CEFAEDFE','FEEDFACF','FEEDFACE','CAFEBABE','BEBAFECA','CAFEBABF','BFBAFECA') }
    else { $binary = $signature -eq '7F454C46' }
    if (!$binary) { continue }
    if ($file.Name -match $forbidden) { throw "Unexpected runtime file: $($file.Name)" }
    if ($IsWindows) {
      $imports = & $reader --coff-imports $file.FullName
      if ($LASTEXITCODE) { throw "Cannot inspect $($file.Name)" }
      $dependencies = @($imports | Select-String '^  Name: ' | ForEach-Object { $_.Line.Trim().Substring(6) })
    } elseif ($IsMacOS) {
      $imports = & $reader -L $file.FullName
      if ($LASTEXITCODE) { throw "Cannot inspect $($file.Name)" }
      $dependencies = @($imports | ForEach-Object { if ($_ -match '^\s+(.+?)\s+\(') { $Matches[1] } })
    } else {
      $imports = & $reader -d $file.FullName
      if ($LASTEXITCODE) { throw "Cannot inspect $($file.Name)" }
      $dependencies = @($imports | ForEach-Object { if ($_ -match '\(NEEDED\).*\[(.+?)\]') { $Matches[1] } })
    }
    foreach ($dependency in $dependencies) {
      if ([IO.Path]::GetFileName($dependency) -match $forbidden) { throw "Forbidden runtime dependency in $($file.Name): $dependency" }
    }
    $report.Add($file.Name+': '+($dependencies -join ', '))
    $inspected++
  }
}
if (!$inspected) { throw "No runtime binaries inspected: $directoryPath" }
$destination = Join-Path $nativeRoot 'results/build-environment/runtime-dependencies.txt'
New-Item -ItemType Directory -Force (Split-Path $destination) | Out-Null
$report | Set-Content -LiteralPath $destination -Encoding utf8
Write-Output "Verified $inspected runtime binaries without OpenCV, FFmpeg, Python, shared MediaPipe/Halide compiler or MSVC CRT: $directoryPath"
