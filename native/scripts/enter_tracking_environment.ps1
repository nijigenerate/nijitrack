# Dot-source this file in PowerShell 7. Changes apply only to the current process.
$ErrorActionPreference = 'Stop'
$nativeRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$trackingEnvironment = Get-Content (Join-Path $nativeRoot 'vendor/toolchain/environment.json') -Raw | ConvertFrom-Json
if (!$IsWindows) {
  $env:Halide_ROOT = $trackingEnvironment.halide
  $env:SDL3_ROOT = $trackingEnvironment.sdl3
  $env:PYTHON_BIN_PATH = Join-Path $nativeRoot 'vendor/toolchain/no-python/python'
  $env:PYTHON3_BIN_PATH = $env:PYTHON_BIN_PATH
  $env:NIJITRACK_PYTHON_GUARD_LOG = Join-Path $nativeRoot 'results/build-environment/python-guard.log'
  $env:PATH = (Split-Path $env:PYTHON_BIN_PATH)+[IO.Path]::PathSeparator+(Split-Path $trackingEnvironment.bazel)+[IO.Path]::PathSeparator+$env:PATH
  $env:CMAKE_PREFIX_PATH = $trackingEnvironment.sdl3+[IO.Path]::PathSeparator+$env:CMAKE_PREFIX_PATH
  return
}
Import-Module (Join-Path $trackingEnvironment.visual_studio 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $trackingEnvironment.visual_studio -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
$env:BAZEL_VS = $trackingEnvironment.visual_studio
$env:BAZEL_VC = Join-Path $trackingEnvironment.visual_studio 'VC'
$env:BAZEL_LLVM = $trackingEnvironment.llvm
$env:BAZEL_SH = Join-Path $env:ProgramFiles 'Git/bin/bash.exe'
$env:Halide_ROOT = $trackingEnvironment.halide
$env:SDL3_ROOT = $trackingEnvironment.sdl3
$guard = Join-Path $nativeRoot 'vendor/toolchain/no-python'
$env:PYTHON_BIN_PATH = Join-Path $guard 'python.exe'
$env:PYTHON3_BIN_PATH = $env:PYTHON_BIN_PATH
$env:NIJITRACK_PYTHON_GUARD_LOG = Join-Path $nativeRoot 'results/build-environment/python-guard.log'
$env:PATH = (("$guard;$($trackingEnvironment.llvm)/bin;$(Split-Path $trackingEnvironment.cmake);$(Split-Path $trackingEnvironment.ninja);$(Split-Path $trackingEnvironment.bazel);$($trackingEnvironment.halide)/bin/Release;$env:PATH" -split ';') | Where-Object { $_ } | Select-Object -Unique) -join ';'
