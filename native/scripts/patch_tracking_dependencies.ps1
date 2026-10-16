[CmdletBinding()]
param([switch]$RequireRepositories)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'apply_tracking_patches.ps1')
$nativeRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$bazelRoot = Join-Path $nativeRoot 'build-mediapipe/bazel'
if (!(Test-Path -LiteralPath $bazelRoot)) {
  if ($RequireRepositories) { throw 'Bazel dependency workspace is missing' }
  return
}
$dependencies = Get-Content (Join-Path $nativeRoot 'patches/dependencies/manifest.json') -Raw | ConvertFrom-Json
$found = [Collections.Generic.HashSet[string]]::new()
foreach ($output in (Get-ChildItem -LiteralPath $bazelRoot -Directory)) {
  foreach ($dependency in $dependencies) {
    $repository = Join-TrackingChild (Join-Path $output.FullName 'external') $dependency.repository
    if (!(Test-Path -LiteralPath $repository)) { continue }
    Invoke-TrackingPatchSeries -Root $repository -Series $dependency.series -AllowApplied
    Copy-TrackingOverlays $repository $dependency.overlays $nativeRoot
    $null = $found.Add($dependency.repository)
  }
}
if ($RequireRepositories) {
  foreach ($dependency in $dependencies) {
    if (!$found.Contains($dependency.repository)) { throw "Expected dependency repository is missing: $($dependency.repository)" }
  }
}
