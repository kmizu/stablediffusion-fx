param(
  [Parameter(Mandatory=$true)][string]$Model,
  [Parameter(Mandatory=$true)][string]$OutputHome
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$packagedJar = Join-Path $repoRoot 'target/stablediffusion-nui.jar'
$javaExe = 'C:/Program Files/Java/jdk-21/bin/java.exe'
$pythonExe = Join-Path $repoRoot 'python/.venv-smoke/Scripts/python.exe'
$env:PYTHONPATH = Join-Path $repoRoot 'python'
$before = (Get-FileHash -LiteralPath $packagedJar -Algorithm SHA256).Hash
foreach ($check in @(
  @{ project='engine'; main='sdfx.engine.RealVramServiceCheck'; home='service' },
  @{ project='ui'; main='sdfx.ui.RealVramViewCheck'; home='view' }
)) {
  $testClasses = Join-Path $repoRoot ($check.project + '/target/scala-3.3.6/test-classes')
  & $javaExe '-cp' "$testClasses;$packagedJar" $check.main (Join-Path $OutputHome $check.home) $pythonExe $Model
  if ($LASTEXITCODE -ne 0) { throw "Packaged VRAM check failed: $($check.main)" }
}
$after = (Get-FileHash -LiteralPath $packagedJar -Algorithm SHA256).Hash
if ($before -ne $after) { throw 'Assembled jar changed during native acceptance' }
Write-Output "PASS packaged VRAM service and JavaFX view; unchanged jar SHA256 $after"
