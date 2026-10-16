[CmdletBinding()]
param([switch]$Dependencies)
$ErrorActionPreference = 'Stop'
$nativeRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $nativeRoot 'scripts/apply_tracking_patches.ps1')
$manifest = Get-Content (Join-Path $nativeRoot 'patches/mediapipe/manifest.json') -Raw | ConvertFrom-Json
$lock = Get-Content (Join-Path $nativeRoot 'toolchain.lock.json') -Raw | ConvertFrom-Json
$asset = $lock.assets | Where-Object name -eq $manifest.upstream_asset
$original = Join-Path $nativeRoot $asset.directory
$testRoot = Join-Path $nativeRoot ('build-patch-tests/' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force $testRoot | Out-Null
function Restore-Inputs {
  foreach ($entry in $manifest.series) {
    foreach ($file in $entry.files) {
      $destination = Join-TrackingChild $testRoot $file.path
      New-Item -ItemType Directory -Force (Split-Path $destination) | Out-Null
      Copy-Item -LiteralPath (Join-TrackingChild $original $file.path) -Destination $destination -Force
    }
  }
}
try {
  Restore-Inputs
  Invoke-TrackingPatchSeries -Root $testRoot -Series $manifest.series
  Invoke-TrackingPatchSeries -Root $testRoot -Series $manifest.series -AllowApplied
  $overlay = $manifest.overlays[0]
  $target = Join-TrackingChild $testRoot $overlay.target
  Copy-Item -LiteralPath (Join-TrackingChild $original $overlay.target) -Destination $target -Force
  Copy-TrackingOverlays $testRoot $manifest.overlays $nativeRoot
  if ((Get-TrackingFileHash $target) -ne (Get-TrackingFileHash (Join-Path $nativeRoot $overlay.source))) { throw 'Overlay mismatch' }
  # Drift in a later input must be rejected before any earlier patch is applied.
  Restore-Inputs
  $last = $manifest.series[-1].files[0]
  Add-Content -LiteralPath (Join-TrackingChild $testRoot $last.path) '# simulated upstream change'
  $rejected = $false
  try { Invoke-TrackingPatchSeries -Root $testRoot -Series $manifest.series } catch {
    if ($_.Exception.Message -notlike 'Unsupported source content:*') { throw }
    $rejected = $true
  }
  if (!$rejected) { throw 'Unexpected source was accepted' }
  $first = $manifest.series[0].files[0]
  if ((Get-TrackingFileHash (Join-TrackingChild $testRoot $first.path)) -ne $first.before) { throw 'Inputs changed before validation completed' }
  $rejected = $false
  try { Copy-TrackingOverlays $testRoot $manifest.overlays $nativeRoot } catch {
    if ($_.Exception.Message -notlike 'Unsupported overlay input:*') { throw }
    $rejected = $true
  }
  if (!$rejected) { throw 'Unexpected overlay input was accepted' }
  if ($Dependencies) {
    $external = Get-ChildItem (Join-Path $nativeRoot 'build-mediapipe/bazel/*/external') -Directory | Select-Object -First 1
    if (!$external) { throw 'Fetch/build the SDK before testing dependency patches' }
    $dependencyEntries = Get-Content (Join-Path $nativeRoot 'patches/dependencies/manifest.json') -Raw | ConvertFrom-Json
    foreach ($dependency in $dependencyEntries) {
      $repository = Join-TrackingChild $external.FullName $dependency.repository
      $fixture = Join-TrackingChild $testRoot $dependency.repository
      foreach ($entry in $dependency.series) {
        foreach ($file in $entry.files) {
          $destination = Join-TrackingChild $fixture $file.path
          New-Item -ItemType Directory -Force (Split-Path $destination) | Out-Null
          Copy-Item -LiteralPath (Join-TrackingChild $repository $file.path) -Destination $destination
          $hash = Get-TrackingFileHash $destination
          if ($hash -eq $file.after) {
            # Recover the pinned preimage in a disposable fixture, never in Bazel's cache.
            $previousCeiling = $env:GIT_CEILING_DIRECTORIES
            try {
              $env:GIT_CEILING_DIRECTORIES = Split-Path $fixture
              & git -c core.autocrlf=false -C $fixture apply --reverse (Join-Path $nativeRoot "patches/$($entry.patch)")
              if ($LASTEXITCODE) { throw 'Dependency reverse-patch fixture failed' }
            } finally { $env:GIT_CEILING_DIRECTORIES = $previousCeiling }
          } elseif ($hash -ne $file.before) { throw "Unknown dependency fixture: $repository" }
        }
      }
      Invoke-TrackingPatchSeries -Root $fixture -Series $dependency.series
      Invoke-TrackingPatchSeries -Root $fixture -Series $dependency.series -AllowApplied
      Copy-TrackingOverlays $fixture $dependency.overlays $nativeRoot
    }
    Write-Output 'LLVM, XNNPACK and Protobuf patches: pinned preimages, forward application and idempotence passed.'
  }
  Write-Output 'Patch application, exact output, idempotence, upstream drift rejection and overlay guards passed.'
} finally {
  $expected = [IO.Path]::GetFullPath((Join-Path $nativeRoot 'build-patch-tests')) + [IO.Path]::DirectorySeparatorChar
  if (!$testRoot.StartsWith($expected, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected test cleanup path' }
  Remove-Item -LiteralPath $testRoot -Recurse -Force
}
