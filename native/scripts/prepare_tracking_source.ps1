# Dot-sourced by prepare_mediapipe_halide.ps1 after the native environment is loaded.
. (Join-Path $PSScriptRoot 'apply_tracking_patches.ps1')
$manifest = Get-Content (Join-Path $nativeRoot 'patches/mediapipe/manifest.json') -Raw | ConvertFrom-Json
$lock = Get-Content (Join-Path $nativeRoot 'toolchain.lock.json') -Raw | ConvertFrom-Json
$assets = @($lock.assets | Where-Object name -eq $manifest.upstream_asset)
if ($assets.Count -ne 1 -or $assets[0].version -ne $manifest.upstream_version) {
  throw 'MediaPipe lock/patch version mismatch. Rebase patches before updating.'
}
$asset = $assets[0]
$archive = Join-TrackingChild $nativeRoot $asset.file
if ((Get-TrackingFileHash $archive) -ne $asset.sha256) { throw 'MediaPipe source archive checksum mismatch' }
$buildRoot = Join-Path $nativeRoot 'build-mediapipe'
$source = Join-Path $buildRoot 'src'
$staging = Join-Path $buildRoot 'source-staging'
function Remove-SourceWorkspace([string]$Path) {
  $full = [IO.Path]::GetFullPath($Path)
  if ($full -notin @([IO.Path]::GetFullPath($source), [IO.Path]::GetFullPath($staging))) { throw "Unexpected cleanup target: $full" }
  if (!(Test-Path -LiteralPath $full)) { return }
  if ((Get-Item -LiteralPath $full -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Workspace root is a link: $full" }
  # PowerShell does not follow directory links without -FollowSymlink. Unlink them first.
  Get-ChildItem -LiteralPath $full -Force -Recurse | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint } | ForEach-Object {
    if ($_.PSIsContainer) { [IO.Directory]::Delete($_.FullName) } else { [IO.File]::Delete($_.FullName) }
  }
  Remove-Item -LiteralPath $full -Recurse -Force
}
Remove-SourceWorkspace $staging
New-Item -ItemType Directory -Force $staging | Out-Null
Push-Location $staging
try {
  & $trackingEnvironment.cmake -E tar xf $archive
  if ($LASTEXITCODE) { throw 'MediaPipe source extraction failed' }
} finally { Pop-Location }
$fresh = Join-Path $staging ('mediapipe-' + $asset.version)
$nativeAssets = Get-Content (Join-Path $nativeRoot 'assets.lock.json') -Raw | ConvertFrom-Json
if ($nativeAssets.sdk.source_sha -ne $asset.version) { throw 'Native C API headers and MediaPipe source versions differ. Update/review assets.lock.json and the headers.' }
foreach ($header in $nativeAssets.files.PSObject.Properties | Where-Object Name -like 'vendor/include/*') {
  $relative = $header.Name.Substring('vendor/include/'.Length)
  if ((Get-TrackingFileHash (Join-TrackingChild $fresh $relative)) -ne $header.Value -or
      (Get-TrackingFileHash (Join-TrackingChild $nativeRoot $header.Name)) -ne $header.Value) {
    throw "Native C API header differs from pinned upstream: $relative"
  }
}
Invoke-TrackingPatchSeries -Root $fresh -Series $manifest.series
Copy-TrackingOverlays $fresh $manifest.overlays $nativeRoot
# Replace only after all upstream patches and overlay guards pass.
$configuration = Join-Path $buildRoot 'configuration.json'
if (Test-Path -LiteralPath $configuration) { Remove-Item -LiteralPath $configuration -Force }
Remove-SourceWorkspace $source
Move-Item -LiteralPath $fresh -Destination $source
Remove-SourceWorkspace $staging
