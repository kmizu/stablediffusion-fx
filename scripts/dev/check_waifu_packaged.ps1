param(
  [Parameter(Mandatory=$true)][string]$Model,
  [Parameter(Mandatory=$true)][string]$InputImage,
  [Parameter(Mandatory=$true)][string]$OutputHome,
  [string]$PackageRoot = ''
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$jar = Join-Path $repoRoot 'target/stablediffusion-nui.jar'
$pythonRoot = Join-Path $repoRoot 'python'
$java = 'C:/Program Files/Java/jdk-21/bin/java.exe'
if ($PackageRoot) {
  $package = (Resolve-Path -LiteralPath $PackageRoot).Path
  $jar = Join-Path $package 'app/stablediffusion-nui.jar'
  $pythonRoot = Join-Path $package 'python'
  $packagedJava = Join-Path $package 'runtime/bin/java.exe'
  if (Test-Path -LiteralPath $packagedJava) { $java = $packagedJava }
  else { Write-Output 'Package runtime has no java launcher; native harness uses JDK 21 with packaged jar/Python. SDNUI.exe runtime acceptance is checked separately.' }
  $modules = @(git -C $repoRoot ls-files -- python/sdfx | Where-Object { $_ -match '\.py$' })
  foreach ($module in $modules) {
    $source = Join-Path $repoRoot $module
    $installed = Join-Path $package $module
    if ((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $installed).Hash) {
      throw "Packaged engine source differs: $module"
    }
  }
}
$manifest = Join-Path $pythonRoot 'sdfx/waifu2x_assets.json'
$canonicalManifest = Join-Path $repoRoot 'python/sdfx/waifu2x_assets.json'
if ((Get-FileHash -LiteralPath $manifest).Hash -ne (Get-FileHash -LiteralPath $canonicalManifest).Hash) {
  throw 'Packaged Waifu2x manifest differs from verified source'
}
$assets = @(Get-Content -Raw -LiteralPath $manifest | ConvertFrom-Json)
if ($assets.Count -ne 9) { throw 'Packaged model manifest must contain all nine models' }
$before = (Get-FileHash -LiteralPath $jar -Algorithm SHA256).Hash
$env:PYTHONPATH = $pythonRoot
$env:SDFX_WAIFU2X_CACHE = Join-Path $repoRoot 'outputs/waifu2x-models'
$testClasses = Join-Path $repoRoot 'ui/target/scala-3.3.6/test-classes'
$pythonExe = Join-Path $repoRoot 'python/.venv-smoke/Scripts/python.exe'
& $java '-cp' "$testClasses;$jar" 'sdfx.ui.RealWaifuCheck' $OutputHome $pythonExe $Model $InputImage
if ($LASTEXITCODE -ne 0) { throw "Packaged Waifu2x acceptance failed: $LASTEXITCODE" }
if ((Get-FileHash -LiteralPath $jar -Algorithm SHA256).Hash -ne $before) { throw 'Jar changed during acceptance' }
Write-Output "PASS packaged Waifu2x: nine-model manifest, native families/cancellation/recovery, jar SHA256 $before"
