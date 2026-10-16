[CmdletBinding()]
param([switch]$SkipBazel)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter_tracking_environment.ps1')
$build = Join-Path $nativeRoot 'build-toolchain/smoke'
$results = Join-Path $nativeRoot 'results/build-environment'
New-Item -ItemType Directory -Force $results | Out-Null
& $trackingEnvironment.cmake -S (Join-Path $nativeRoot 'toolchain/smoke') -B $build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl `
  "-DHalide_ROOT=$env:Halide_ROOT" "-DSDL3_DIR=$env:SDL3_ROOT/cmake" "-DMediaPipe_ROOT=$($trackingEnvironment.mediapipe)"
if ($LASTEXITCODE) { throw 'Toolchain configuration failed' }
& $trackingEnvironment.cmake --build $build --target no-python --parallel 2
if ($LASTEXITCODE) { throw 'Python guard build failed' }
$guardDirectory = Join-Path $nativeRoot 'vendor/toolchain/no-python'
New-Item -ItemType Directory -Force $guardDirectory | Out-Null
foreach ($name in @('python.exe', 'python3.exe', 'py.exe')) {
  Copy-Item -LiteralPath (Join-Path $build 'no-python.exe') -Destination (Join-Path $guardDirectory $name) -Force
}
if (Test-Path -LiteralPath $env:NIJITRACK_PYTHON_GUARD_LOG) { throw 'A previous build attempted Python; inspect results/build-environment/python-guard.log before retrying.' }
& $trackingEnvironment.cmake --build $build --parallel 2
if ($LASTEXITCODE) { throw 'Halide/SDL build failed' }
$ctest = Join-Path (Split-Path $trackingEnvironment.cmake) 'ctest.exe'
& $ctest --test-dir $build --output-on-failure
if ($LASTEXITCODE) { throw 'Halide/SDL test failed' }
if (!$SkipBazel) {
  # This independent workspace contains no MediaPipe/Python repository rules.
  $bazelRoot = Join-Path $nativeRoot 'build-toolchain/bazel'
  Push-Location (Join-Path $nativeRoot 'toolchain/bazel-probe')
  try {
    & $trackingEnvironment.bazel --nosystem_rc --nohome_rc --noworkspace_rc --batch "--output_user_root=$bazelRoot" `
      run --enable_bzlmod=false --compiler=clang-cl --compilation_mode=opt --jobs=2 `
      --extra_toolchains=@local_config_cc//:cc-toolchain-x64_windows-clang-cl `
      --extra_execution_platforms=//:windows_clang --host_platform=//:windows_clang --platforms=//:windows_clang `
      "--repo_env=BAZEL_LLVM=$env:BAZEL_LLVM" //:compiler_probe
    if ($LASTEXITCODE) { throw 'Bazel clang-cl probe failed' }
  } finally { Pop-Location }
}
if (Test-Path -LiteralPath $env:NIJITRACK_PYTHON_GUARD_LOG) { throw 'Python guard was triggered; validation is not successful.' }
$dumpbin = Join-Path $env:VCToolsInstallDir 'bin/Hostx64/x64/dumpbin.exe'
$imports = & $dumpbin /dependents (Join-Path $build 'tracking-toolchain-smoke.exe')
$imports | Set-Content (Join-Path $results 'smoke-dependencies.txt') -Encoding utf8
if (($imports -join "`n") -match '(?i)opencv|python[0-9]*\.dll|Halide\.dll') { throw 'Unexpected runtime dependency in AOT consumer' }
[ordered]@{
  verified_utc=[DateTime]::UtcNow.ToString('o')
  platform='windows-x86_64'
  compiler=(& clang-cl --version | Select-Object -First 1)
  halide='15.0.1'
  sdl3='3.4.18'
  bazel='7.4.1'
  halide_aot_and_sdl_rendering='passed'
  halide_generator='unmodified pinned MediaPipe rgb_float_generator.cc'
  bazel_clang_cl= $(if ($SkipBazel) { 'not run' } else { 'passed' })
  python='not executed; guard not triggered'
  mediapipe_full_build='not covered by this toolchain smoke test; run build_mediapipe_halide.ps1'
} | ConvertTo-Json | Set-Content (Join-Path $results 'verification.json') -Encoding utf8
Write-Output 'Verified Halide AOT and SDL3. See results/build-environment/verification.json.'
