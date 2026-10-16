# Compatibility entry point. All operating systems use build_native.ps1.
[CmdletBinding()]
param([switch]$Install)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'build_native.ps1') -Install:$Install
