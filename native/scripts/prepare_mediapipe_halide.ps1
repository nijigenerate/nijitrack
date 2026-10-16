[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
if (!$IsWindows) { throw 'Use bash native/build.sh on Linux/macOS.' }
. (Join-Path $PSScriptRoot 'enter_tracking_environment.ps1')
$guard = Join-Path $nativeRoot 'vendor/toolchain/no-python'
New-Item -ItemType Directory -Force $guard | Out-Null
if (!(Test-Path -LiteralPath $env:PYTHON_BIN_PATH)) {
  if ($IsWindows) {
    & (Join-Path $trackingEnvironment.llvm 'bin/clang-cl.exe') /nologo /EHsc /MD /O2 (Join-Path $nativeRoot 'toolchain/smoke/no_python.cpp') "/Fe:$env:PYTHON_BIN_PATH" "/Fo:$guard/no-python.obj"
  } else {
    & clang++ -std=c++17 -O2 (Join-Path $nativeRoot 'toolchain/smoke/no_python.cpp') -o $env:PYTHON_BIN_PATH
  }
  if ($LASTEXITCODE) { throw 'Cannot build interpreter guard' }
  foreach ($name in @('python3','py')) {
    Copy-Item $env:PYTHON_BIN_PATH (Join-Path $guard ($name+$(if($IsWindows){'.exe'}else{''}))) -Force
  }
}
. (Join-Path $PSScriptRoot 'prepare_tracking_source.ps1')
$utf8 = [Text.UTF8Encoding]::new($false)
function Write-Utf8([string]$path, [string]$body) { [IO.File]::WriteAllText($path, $body.Replace("`r`n","`n"), $utf8) }
$package = Join-Path $source 'mediapipe/tasks/c/nijitrack_tracking'
New-Item -ItemType Directory -Force $package | Out-Null
Copy-Item (Join-Path $nativeRoot 'toolchain/mediapipe/face_hand.BUILD.bazel') (Join-Path $package 'BUILD.bazel') -Force
Copy-Item (Join-Path $nativeRoot 'toolchain/mediapipe/tracking_archive.bzl') $package -Force
$platform = Join-Path $source 'nijitrack_clang'
New-Item -ItemType Directory -Force $platform | Out-Null
Write-Utf8 (Join-Path $platform 'BUILD.bazel') @'
platform(
    name = "windows",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        "@rules_cc//cc/private/toolchain:clang-cl",
    ],
)
'@
$stub = Join-Path $nativeRoot 'build-mediapipe/no-python-repository'
New-Item -ItemType Directory -Force $stub | Out-Null
Write-Utf8 (Join-Path $stub 'WORKSPACE') ''
$swift = Join-Path $nativeRoot 'build-mediapipe/no-swift-repository'
New-Item -ItemType Directory -Force $swift | Out-Null
Write-Utf8 (Join-Path $swift 'WORKSPACE') ''
Write-Utf8 (Join-Path $swift 'BUILD.bazel') ''
Write-Utf8 (Join-Path $stub 'defs.bzl') ('interpreter = "' + $env:PYTHON_BIN_PATH.Replace('\','/') + '"')
Write-Utf8 (Join-Path $stub 'py_version.bzl') @'
TF_PYTHON_VERSION = "3.11"
HERMETIC_PYTHON_VERSION = "3.11"
HERMETIC_PYTHON_VERSION_KIND = ""
USE_PYWRAP_RULES = False
WHEEL_NAME = "disabled"
WHEEL_COLLAB = False
REQUIREMENTS = ""
REQUIREMENTS_WITH_LOCAL_WHEELS = ""
MACOSX_DEPLOYMENT_TARGET = "12.0"
HERMETIC_PYTHON_URL = ""
HERMETIC_PYTHON_SHA256 = ""
HERMETIC_PYTHON_PREFIX = "disabled"
'@
Write-Utf8 (Join-Path $stub 'BUILD.bazel') @'
load("@bazel_tools//tools/python:toolchain.bzl", "py_runtime_pair")
load(":defs.bzl", "interpreter")
package(default_visibility = ["//visibility:public"])
cc_library(name = "python_headers")
alias(name = "headers", actual = ":python_headers")
py_runtime(name = "guard", interpreter_path = interpreter, python_version = "PY3")
py_runtime_pair(name = "pair", py3_runtime = ":guard")
toolchain(name = "py_toolchain", toolchain = ":pair", toolchain_type = "@bazel_tools//tools/python:toolchain_type")
'@
$compilerArgs = if ($IsWindows) { @('-DCMAKE_C_COMPILER=clang-cl','-DCMAKE_CXX_COMPILER=clang-cl') } else { @('-DCMAKE_C_COMPILER=clang','-DCMAKE_CXX_COMPILER=clang++') }
& $trackingEnvironment.cmake -S (Join-Path $nativeRoot 'toolchain/mediapipe/halide') -B (Join-Path $nativeRoot 'build-mediapipe/halide') -G Ninja `
  -DCMAKE_BUILD_TYPE=Release @compilerArgs "-DHalide_ROOT=$env:Halide_ROOT"
if ($LASTEXITCODE) { throw 'Halide configuration failed' }
& $trackingEnvironment.cmake --build (Join-Path $nativeRoot 'build-mediapipe/halide') --parallel 2
if ($LASTEXITCODE) { throw 'Halide generation failed' }
$tensor = Join-Path $source 'mediapipe/calculators/tensor'
$generated = Join-Path $tensor 'nijitrack_halide'
New-Item -ItemType Directory -Force $generated | Out-Null
Copy-Item (Join-Path $nativeRoot 'build-mediapipe/halide/generated/nijitrack_warp.*') $generated -Force
Copy-Item (Join-Path $trackingEnvironment.halide 'include/HalideRuntime.h') $generated -Force
$halideBuild = @'
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "warp",
    srcs = ["nijitrack_warp.lib"],
    hdrs = ["nijitrack_warp.h", "HalideRuntime.h"],
    visibility = ["//mediapipe/calculators/tensor:__pkg__"],
)
'@
if (!$IsWindows) { $halideBuild = $halideBuild.Replace('nijitrack_warp.lib','nijitrack_warp.a') }
Write-Utf8 (Join-Path $generated 'BUILD.bazel') $halideBuild
[ordered]@{source=$source;python_stub=$stub;upstream_version=$asset.version;source_archive_sha256=$asset.sha256;recipe_sha256=(Get-TrackingRecipeHash $nativeRoot)} | ConvertTo-Json | Set-Content (Join-Path $nativeRoot 'build-mediapipe/configuration.json') -Encoding utf8
Write-Output "Prepared C++ source workspace: $source"
