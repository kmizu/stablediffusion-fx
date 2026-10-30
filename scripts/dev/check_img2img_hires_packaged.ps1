param(
  [Parameter(Mandatory=$true)][string]$PackageRoot,
  [Parameter(Mandatory=$true)][string]$Python,
  [Parameter(Mandatory=$true)][string]$Model,
  [Parameter(Mandatory=$true)][string]$Lora,
  [Parameter(Mandatory=$true)][string]$OutputHome,
  [switch]$Sdxl
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$package = (Resolve-Path -LiteralPath $PackageRoot).Path
$jar = Join-Path $package 'app/stablediffusion-nui.jar'
$modules = @(git -C $repo ls-files -- python/sdfx | Where-Object { $_ -match '\.py$' })
if (-not ($modules -contains 'python/sdfx/img2img_hires_fix.py')) { throw 'Add new runtime module to the index before packaging.' }
foreach ($module in $modules) {
  if ((Get-FileHash -LiteralPath (Join-Path $repo $module)).Hash -ne
      (Get-FileHash -LiteralPath (Join-Path $package $module)).Hash) { throw "Packaged module differs: $module" }
}
$before = (Get-FileHash -LiteralPath $jar -Algorithm SHA256).Hash
if ($before -ne (Get-FileHash -LiteralPath (Join-Path $repo 'target/stablediffusion-nui.jar') -Algorithm SHA256).Hash) {
  throw 'Extracted jar differs from the fresh assembly.'
}
$env:PYTHONPATH = Join-Path $package 'python'
$classes = Join-Path $repo 'ui/target/scala-3.3.6/test-classes'
$arguments = @('-cp', "$classes;$jar", 'sdfx.ui.RealImg2ImgHiresGenerationCheck', $OutputHome, $Python, $Model, $Lora)
if ($Sdxl) { $arguments += 'sdxl' }
# Only the harness comes from test classes; all product Scala/Python is extracted.
& 'C:/Program Files/Java/jdk-21/bin/java.exe' @arguments
if ($LASTEXITCODE -ne 0) { throw "Packaged img2img Hires acceptance failed: $LASTEXITCODE" }
if ((Get-FileHash -LiteralPath $jar -Algorithm SHA256).Hash -ne $before) { throw 'Jar changed during acceptance.' }
Write-Output "PASS extracted img2img Hires native checks; jar SHA256 $before"
