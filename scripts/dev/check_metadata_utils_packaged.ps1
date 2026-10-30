param(
  [Parameter(Mandatory=$true)][string]$PackageRoot,
  [Parameter(Mandatory=$true)][string]$Python,
  [Parameter(Mandatory=$true)][string]$StagedModel,
  [Parameter(Mandatory=$true)][string]$StagedLora,
  [Parameter(Mandatory=$true)][string]$OutputHome,
  [ValidateSet('sd15','sdxl','dry-run','metadata-only')][string]$ModelFamily='dry-run',
  [switch]$WorkerDiagnostics
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$package = (Resolve-Path -LiteralPath $PackageRoot).Path
$jar = Join-Path $package 'app/stablediffusion-nui.jar'
$modules = @(git -C $repo ls-files -- python/sdfx | Where-Object { $_ -match '\.py$' })
if (-not ($modules -contains 'python/sdfx/metadata_utils.py')) { throw 'New module must be included in package input.' }
foreach ($module in $modules) {
  if ((Get-FileHash -LiteralPath (Join-Path $repo $module)).Hash -ne
      (Get-FileHash -LiteralPath (Join-Path $package $module)).Hash) { throw "Packaged module differs: $module" }
}
$before = (Get-FileHash -LiteralPath $jar -Algorithm SHA256).Hash
if ($before -ne (Get-FileHash -LiteralPath (Join-Path $repo 'target/stablediffusion-nui.jar')).Hash) {
  throw 'Extracted jar differs from fresh assembly.'
}
$env:PYTHONPATH = Join-Path $package 'python'
if ($WorkerDiagnostics) {
  $startup = Join-Path $repo 'scripts/dev/metadata_worker_diagnostics'
  $env:PYTHONPATH = "$startup;$env:PYTHONPATH"
  [IO.Directory]::CreateDirectory([IO.Path]::GetFullPath($OutputHome)) | Out-Null
  $env:SDFX_METADATA_DIAGNOSTIC_FILE = Join-Path ([IO.Path]::GetFullPath($OutputHome)) 'metadata-worker-phases.log'
}
$classes = Join-Path $repo 'ui/target/scala-3.3.6/test-classes'
$arguments = @('-cp', "$classes;$jar", 'sdfx.ui.RealMetadataUtilsCheck', $OutputHome, $Python, $StagedModel, $StagedLora)
if ($ModelFamily -ne 'dry-run') { $arguments += $ModelFamily }
& 'C:/Program Files/Java/jdk-21/bin/java.exe' @arguments
if ($LASTEXITCODE -ne 0) { throw "Packaged Metadata Utils acceptance failed: $LASTEXITCODE" }
if ((Get-FileHash -LiteralPath $jar).Hash -ne $before) { throw 'Jar changed during acceptance.' }
Write-Output "PASS extracted Metadata Utils native acceptance; jar SHA256 $before"
