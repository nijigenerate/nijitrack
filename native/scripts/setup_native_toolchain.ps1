[CmdletBinding()]
param([switch]$Offline)
$ErrorActionPreference = 'Stop'
if (!$IsWindows) { throw 'Use bash native/build.sh setup on Linux/macOS.' }
& (Join-Path $PSScriptRoot 'setup_tracking_toolchain.ps1') -Offline:$Offline
