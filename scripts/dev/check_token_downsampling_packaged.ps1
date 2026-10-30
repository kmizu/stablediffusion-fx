param(
  [Parameter(Mandatory=$true)][string]$PackageRoot,
  [Parameter(Mandatory=$true)][string]$Model,
  [Parameter(Mandatory=$true)][string]$OutputHome
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$package = (Resolve-Path -LiteralPath $PackageRoot).Path
$jar = Join-Path $package 'app/stablediffusion-nui.jar'
$modules = @(git -C $repo ls-files -- python/sdfx | Where-Object { $_ -match '\.py$' })
if (-not ($modules -contains 'python/sdfx/token_downsampling.py')) { throw 'Stage the new module before packaging.' }
foreach ($module in $modules) {
  if ((Get-FileHash -LiteralPath (Join-Path $repo $module)).Hash -ne
      (Get-FileHash -LiteralPath (Join-Path $package $module)).Hash) { throw "Packaged module differs: $module" }
}
$before = (Get-FileHash -LiteralPath $jar -Algorithm SHA256).Hash
$env:PYTHONPATH = Join-Path $package 'python'
$classes = Join-Path $repo 'ui/target/scala-3.3.6/test-classes'
$python = Join-Path $repo 'python/.venv-smoke/Scripts/python.exe'
# jpackage omits java.exe. SDNUI.exe/bundled-runtime startup is separately checked.
& 'C:/Program Files/Java/jdk-21/bin/java.exe' '-cp' "$classes;$jar" 'sdfx.ui.RealTokenDownsamplingCheck' $OutputHome $python $Model
if ($LASTEXITCODE -ne 0) { throw "Packaged native acceptance failed: $LASTEXITCODE" }
if ((Get-FileHash -LiteralPath $jar -Algorithm SHA256).Hash -ne $before) { throw 'Jar changed during acceptance.' }
Write-Output "PASS extracted Token Downsampling native checks; jar SHA256 $before"
