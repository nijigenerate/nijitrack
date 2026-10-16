[CmdletBinding()]
param([switch]$Install)
$ErrorActionPreference = 'Stop'
if (!$IsWindows) { throw 'Use bash native/build.sh on Linux/macOS.' }
. (Join-Path $PSScriptRoot 'enter_tracking_environment.ps1')
$build = Join-Path $nativeRoot 'build-sdl-static'
function Clear-ObsoleteMediaRuntime([string]$Directory) {
  if (!$IsWindows) { return } # Cleanup of historical Windows packages only.
  $resolved = [IO.Path]::GetFullPath($Directory)
  $prefix = [IO.Path]::GetFullPath($nativeRoot)+[IO.Path]::DirectorySeparatorChar
  if (!$resolved.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected runtime directory' }
  if (Test-Path -LiteralPath $resolved) {
    Get-ChildItem -LiteralPath $resolved -File | Where-Object {
      $_.Name -match '^(avcodec|avdevice|avfilter|avformat|avutil|swresample|swscale)-\d+\.dll$|^ff(mpeg|probe)\.exe$|^opencv.*\.dll$|^libmediapipe\.dll$|^Halide\.dll$'
    } | Remove-Item -Force
  }
}
Clear-ObsoleteMediaRuntime $build
$compiler = if ($IsWindows) { Join-Path $trackingEnvironment.llvm 'bin/clang-cl.exe' } else { (Get-Command clang++ -ErrorAction Stop).Source }
if (!(Test-Path -LiteralPath $compiler)) { throw 'Run setup_native_toolchain.ps1 first.' }
& $compiler --version
& $trackingEnvironment.cmake -S $nativeRoot -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  "-DCMAKE_CXX_COMPILER=$compiler" -DNIJITRACK_REQUIRE_CLANG=ON -DNIJITRACK_KEYBOARD=ON -DBUILD_TESTING=ON
if ($LASTEXITCODE) { throw 'nijitrack Clang configuration failed' }
& $trackingEnvironment.cmake --build $build --parallel 2
if ($LASTEXITCODE) { throw 'nijitrack Clang build failed' }
$ctest = if ($IsWindows) { Join-Path (Split-Path $trackingEnvironment.cmake) 'ctest.exe' } else { (Get-Command ctest -ErrorAction Stop).Source }
& $ctest --test-dir $build --output-on-failure
if ($LASTEXITCODE) { throw 'nijitrack Clang tests failed' }
& (Join-Path $PSScriptRoot 'verify_native_dependencies.ps1') -Directory $build
if ($Install) {
  $bin = [IO.Path]::GetFullPath((Join-Path $nativeRoot 'install/bin'))
  $expectedBin = [IO.Path]::GetFullPath((Join-Path $nativeRoot 'install'))+[IO.Path]::DirectorySeparatorChar
  if (!$bin.StartsWith($expectedBin,[StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected install path' }
  Clear-ObsoleteMediaRuntime $bin
  foreach ($name in @('LICENSE.opencv','LICENSE.ffmpeg')) {
    $obsoleteLicense = Join-Path $nativeRoot "install/share/nijitrack-cpp/$name"
    if (Test-Path -LiteralPath $obsoleteLicense) { Remove-Item -LiteralPath $obsoleteLicense }
  }
  & $trackingEnvironment.cmake --install $build --prefix (Join-Path $nativeRoot 'install')
  if ($LASTEXITCODE) { throw 'nijitrack Clang install failed' }
  & (Join-Path $PSScriptRoot 'verify_native_dependencies.ps1') -Directory $bin
}
