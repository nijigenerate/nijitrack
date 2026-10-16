[CmdletBinding()]
param([ValidateSet('query','build')][string]$Mode = 'build')
$ErrorActionPreference = 'Stop'
if (!$IsWindows) { throw 'Use bash native/build.sh on Linux/macOS.' }
. (Join-Path $PSScriptRoot 'enter_tracking_environment.ps1')
. (Join-Path $PSScriptRoot 'apply_tracking_patches.ps1')
$config = Get-Content (Join-Path $nativeRoot 'build-mediapipe/configuration.json') -Raw | ConvertFrom-Json
if ($config.recipe_sha256 -ne (Get-TrackingRecipeHash $nativeRoot)) {
  throw 'Tracking source recipe changed. Run prepare_mediapipe_halide.ps1 before building.'
}
& (Join-Path $PSScriptRoot 'patch_tracking_dependencies.ps1')
$root = Join-Path $nativeRoot 'build-mediapipe/bazel'
$javaName = if ($IsWindows) { 'java.exe' } else { 'java' }
$java = Get-ChildItem "$root/install/*/embedded_tools/jdk/bin/$javaName" -File -ErrorAction SilentlyContinue | Select-Object -First 1
if ($java) { $env:JAVA_HOME = Split-Path (Split-Path $java.FullName) }
if ($IsWindows) { $env:PATH = @(
  (Split-Path (Get-Process -Id $PID).Path),
  (Join-Path $nativeRoot 'vendor/toolchain/no-python'),(Join-Path $trackingEnvironment.llvm 'bin'),
  (Split-Path $trackingEnvironment.cmake),(Split-Path $trackingEnvironment.ninja),
  (Join-Path $env:VCToolsInstallDir 'bin/Hostx64/x64'),(Join-Path $env:WindowsSdkVerBinPath 'x64'),
  (Join-Path $env:SystemRoot 'System32'),$env:SystemRoot,
  (Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0'),
  (Join-Path $env:ProgramFiles 'Git/usr/bin'),(Join-Path $env:ProgramFiles 'Git/bin'),
  (Join-Path $env:ProgramFiles 'Git/cmd')
) -join ';' }
$argsList = @('--enable_bzlmod=true', '--experimental_repo_remote_exec', '--compilation_mode=opt', '--jobs=4',
  '--define=MEDIAPIPE_DISABLE_OPENCV=1','--define=MEDIAPIPE_ENABLE_HALIDE=1',
  '--copt=-DNDEBUG','--copt=-DLITERT_DISABLE_NPU',
  "--repo_env=PYTHON_BIN_PATH=$env:PYTHON_BIN_PATH",
  "--repo_env=PATH=$env:PATH",
  "--action_env=PATH=$env:PATH", "--action_env=PYTHON_BIN_PATH=$env:PYTHON_BIN_PATH",
  "--override_repository=python_version_repo=$($config.python_stub)",
  "--override_repository=python=$($config.python_stub)",
  "--override_repository=local_execution_config_python=$($config.python_stub)")
if ($IsWindows) {
  $argsList += @('--compiler=clang-cl',
    '--copt=-D_USE_MATH_DEFINES','--host_copt=-D_USE_MATH_DEFINES',
    "--repo_env=BAZEL_LLVM=$env:BAZEL_LLVM",
    "--override_repository=rules_swift~~non_module_deps~build_bazel_rules_swift_local_config=$nativeRoot/build-mediapipe/no-swift-repository",
    '--features=static_link_msvcrt',
    '--cxxopt=/std:c++20','--host_cxxopt=/std:c++20',
    '--extra_toolchains=@@rules_cc~~cc_configure_extension~local_config_cc//:cc-toolchain-x64_windows-clang-cl',
    '--extra_execution_platforms=//nijitrack_clang:windows','--host_platform=//nijitrack_clang:windows')
} else {
  $argsList += @('--repo_env=CC=clang','--repo_env=CXX=clang++','--cxxopt=-std=c++20','--host_cxxopt=-std=c++20')
}
if ($java) { $argsList += "--repo_env=JAVA_HOME=$env:JAVA_HOME" }
$argsList += '--extra_toolchains=@python//:py_toolchain'
Push-Location $config.source
try {
  # Fetch these sources before analysis can configure LLVM or schedule generators.
  & $trackingEnvironment.bazel --nosystem_rc --nohome_rc --noworkspace_rc --batch "--output_user_root=$root" fetch `
    --enable_bzlmod=true --experimental_repo_remote_exec --repo=@llvm-raw --repo=@XNNPACK --repo=@com_google_protobuf `
    "--repo_env=PATH=$env:PATH" "--override_repository=python_version_repo=$($config.python_stub)" "--override_repository=python=$($config.python_stub)"
  if ($LASTEXITCODE) { throw 'C++ dependency source fetch failed' }
  & (Join-Path $PSScriptRoot 'patch_tracking_dependencies.ps1') -RequireRepositories
  if ($Mode -eq 'query') {
    & $trackingEnvironment.bazel --nosystem_rc --nohome_rc --noworkspace_rc --batch "--output_user_root=$root" query `
      --enable_bzlmod=true --experimental_repo_remote_exec "--repo_env=PATH=$env:PATH" "--override_repository=python_version_repo=$($config.python_stub)" "--override_repository=python=$($config.python_stub)" `
      'deps(//mediapipe/tasks/c/nijitrack_tracking:mediapipe_tracking)'
  } else {
    & $trackingEnvironment.bazel --nosystem_rc --nohome_rc --noworkspace_rc --batch "--output_user_root=$root" build @argsList --output_groups=+linkdeps,+linkopts //mediapipe/tasks/c/nijitrack_tracking:mediapipe_tracking
  }
  if ($LASTEXITCODE) { throw "MediaPipe $Mode failed" }
} finally { Pop-Location }
if ($Mode -eq 'build') {
  $artifacts = Join-Path $config.source 'bazel-bin/mediapipe/tasks/c/nijitrack_tracking'
  $remaining = @(Get-Content (Join-Path $artifacts 'mediapipe_tracking_linkdeps.txt') | Where-Object { $_ })
  foreach ($dependency in $remaining) {
    if ($dependency.TrimStart('@') -ne '//mediapipe/calculators/tensor/nijitrack_halide:warp') {
      throw "Unpackaged static SDK dependency: $dependency"
    }
  }
  $sdk = Join-Path $nativeRoot 'vendor/sdk-halide'
  New-Item -ItemType Directory -Force $sdk | Out-Null
  $archiveName = if ($IsWindows) { 'mediapipe_tracking.lib' } else { 'libmediapipe_tracking.a' }
  Copy-Item (Join-Path $artifacts $archiveName) $sdk -Force
  $warpName = if ($IsWindows) { 'nijitrack_warp.lib' } else { 'nijitrack_warp.a' }
  $installedWarpName = if ($IsWindows) { $warpName } else { 'libnijitrack_warp.a' }
  Copy-Item (Join-Path $nativeRoot "build-mediapipe/halide/generated/$warpName") (Join-Path $sdk $installedWarpName) -Force
  Copy-Item (Join-Path $artifacts 'mediapipe_tracking_linkopts.txt') $sdk -Force
  Copy-Item (Join-Path $trackingEnvironment.mediapipe 'LICENSE') (Join-Path $sdk 'LICENSE.mediapipe') -Force
  Copy-Item (Join-Path $trackingEnvironment.halide 'share/doc/Halide/LICENSE.txt') (Join-Path $sdk 'LICENSE.halide') -Force
  $runtimeDescription = if ($IsWindows) { 'Windows target runtime: static MSVC CRT (/MT).' } else { 'Target runtime: native platform C/C++ system runtime.' }
  @"
MediaPipe $($config.upstream_version)
https://github.com/google-ai-edge/mediapipe
Face/hand C API, static build with OpenCV disabled; no explicit GPU-disable flag.
$runtimeDescription
TensorFlow and transitive sources: pinned by upstream WORKSPACE/MODULE.bazel.
Halide 15.0.1 AOT preprocessing: native/toolchain/mediapipe/halide
Python repository setup and generators replaced with Starlark, C++ and CMake; see native/patches and toolchain.
Pinned downloads: native/toolchain.lock.json and upstream WORKSPACE dependency hashes.
Source archive SHA256: $($config.source_archive_sha256)
Tracking recipe SHA256: $($config.recipe_sha256)
"@ | Set-Content (Join-Path $sdk 'SOURCE.txt') -Encoding utf8
  $external = Get-ChildItem "$root/*/external" -Directory | Select-Object -First 1
  foreach ($repo in (Get-ChildItem -LiteralPath $external.FullName -Directory)) {
    $licenses = @(Get-ChildItem -LiteralPath $repo.FullName -File | Where-Object { $_.Name -match '^(LICENSE|COPYING|NOTICE)([._-].*)?$' })
    if (!$licenses) { continue }
    $destination = Join-Path $sdk ('licenses/'+$repo.Name)
    New-Item -ItemType Directory -Force $destination | Out-Null
    $licenses | Copy-Item -Destination $destination -Force
  }
  Write-Output "Packaged static tracking SDK: $sdk"
}
