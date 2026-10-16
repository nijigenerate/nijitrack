# Shared by source preparation and dependency preparation. No source rewriting here.
$ErrorActionPreference = 'Stop'
function Get-TrackingFileHash([string]$Path) {
  (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Join-TrackingChild([string]$Root, [string]$Relative) {
  $base = [IO.Path]::GetFullPath($Root) + [IO.Path]::DirectorySeparatorChar
  $path = [IO.Path]::GetFullPath((Join-Path $Root $Relative))
  $comparison = if ($IsWindows) { [StringComparison]::OrdinalIgnoreCase } else { [StringComparison]::Ordinal }
  if (!$path.StartsWith($base, $comparison)) { throw "Path escapes root: $Relative" }
  return $path
}
function Invoke-TrackingPatchSeries {
  param([string]$Root, $Series, [switch]$AllowApplied)
  $patchRoot = Join-Path $PSScriptRoot '../patches'
  # Validate every input before making any edits. An upstream update must be reviewed.
  $pending = @()
  foreach ($entry in $Series) {
    $states = @($entry.files | ForEach-Object {
      $path = Join-TrackingChild $Root $_.path
      $hash = Get-TrackingFileHash $path
      if ($hash -eq $_.before) { 'original' }
      elseif ($AllowApplied -and $hash -eq $_.after) { 'applied' }
      else { throw "Unsupported source content: $path. Review/rebase the patch series; no automatic replacement is allowed." }
    } | Select-Object -Unique)
    if ($states.Count -ne 1) { throw "Partially applied patch: $($entry.patch)" }
    if ($states[0] -eq 'original') {
      $patch = Join-TrackingChild $patchRoot $entry.patch
      Invoke-TrackingGitApply $Root $patch -Check
      $pending += $entry
    }
  }
  foreach ($entry in $pending) {
    $patch = Join-TrackingChild $patchRoot $entry.patch
    Invoke-TrackingGitApply $Root $patch
    foreach ($file in $entry.files) {
      if ((Get-TrackingFileHash (Join-TrackingChild $Root $file.path)) -ne $file.after) {
        throw "Patched output mismatch: $($file.path)"
      }
    }
  }
}
function Invoke-TrackingGitApply([string]$Root, [string]$Patch, [switch]$Check) {
  # Do not discover nijitrack's enclosing repository: git apply would silently skip
  # paths relative to this ignored build directory. These sources need no .git.
  $previousCeiling = $env:GIT_CEILING_DIRECTORIES
  try {
    $env:GIT_CEILING_DIRECTORIES = Split-Path ([IO.Path]::GetFullPath($Root))
    $applyOption = if ($Check) { '--check' } else { '--whitespace=nowarn' }
    & git -c core.autocrlf=false -C $Root apply $applyOption $Patch
    if ($LASTEXITCODE) { throw "Patch application/check failed: $Patch" }
  } finally { $env:GIT_CEILING_DIRECTORIES = $previousCeiling }
}
function Copy-TrackingOverlays([string]$Root, $Overlays, [string]$NativeRoot) {
  foreach ($overlay in $Overlays) {
    $target = Join-TrackingChild $Root $overlay.target
    if ($overlay.before -and (Get-TrackingFileHash $target) -ne $overlay.before) { throw "Unsupported overlay input: $target" }
    New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
    Copy-Item -LiteralPath (Join-TrackingChild $NativeRoot $overlay.source) -Destination $target -Force
  }
}
function Get-TrackingRecipeHash([string]$NativeRoot) {
  $files = @(
    Get-ChildItem -LiteralPath (Join-Path $NativeRoot 'patches') -File -Recurse
    Get-ChildItem -LiteralPath (Join-Path $NativeRoot 'toolchain/mediapipe') -File -Recurse
    foreach ($name in @('prepare_mediapipe_halide.ps1','prepare_tracking_source.ps1','apply_tracking_patches.ps1','patch_tracking_dependencies.ps1','build_mediapipe_halide.ps1')) {
      Get-Item -LiteralPath (Join-Path $NativeRoot "scripts/$name")
    }
    Get-Item -LiteralPath (Join-Path $NativeRoot 'toolchain.lock.json')
    Get-Item -LiteralPath (Join-Path $NativeRoot 'assets.lock.json')
  )
  $fingerprint = ($files | Sort-Object FullName | ForEach-Object {
    $_.FullName.Substring($NativeRoot.Length).Replace('\','/') + ':' + (Get-TrackingFileHash $_.FullName)
  }) -join "`n"
  [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($fingerprint))).ToLowerInvariant()
}
