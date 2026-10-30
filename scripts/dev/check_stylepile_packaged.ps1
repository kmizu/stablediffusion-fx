param(
  [Parameter(Mandatory=$true)][string]$Model,
  [Parameter(Mandatory=$true)][string]$OutputHome,
  [string]$JavaExe = 'C:/Program Files/Java/jdk-21/bin/java.exe'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$packagedJar = Join-Path $repoRoot 'target/stablediffusion-nui.jar'
$testClasses = Join-Path $repoRoot 'ui/target/scala-3.3.6/test-classes'
$pythonExe = Join-Path $repoRoot 'python/.venv-smoke/Scripts/python.exe'
foreach ($required in @($JavaExe, $packagedJar, $pythonExe, (Join-Path $testClasses 'sdfx/ui/RealStylePileCheck.class'))) {
  if (-not (Test-Path -LiteralPath $required)) { throw "Missing acceptance prerequisite: $required" }
}
if (Test-Path -LiteralPath (Join-Path $testClasses 'stylepile')) {
  throw 'Test resource directory would shadow the packaged StylePile catalog'
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($packagedJar)
try {
  $previews = @($archive.Entries | Where-Object { $_.FullName -match '^stylepile/StylePile/.*\.jpg$' })
  if ($previews.Count -ne 512) { throw "Expected 512 packaged source previews, got $($previews.Count)" }
  foreach ($entry in @('stylepile/catalog.json', 'stylepile/manifest.json', 'stylepile/LICENSE', 'sdfx/plugins/prompt/StylePileCatalog$.class')) {
    if ($null -eq $archive.GetEntry($entry)) { throw "Packaged entry missing: $entry" }
  }
  $sourceManifest = Get-Content -Raw -LiteralPath (Join-Path $repoRoot 'plugins/src/main/resources/stylepile/manifest.json') | ConvertFrom-Json
  $hasher = [System.Security.Cryptography.SHA256]::Create()
  try {
    $verified = 0
    foreach ($item in $sourceManifest.files) {
      if ($item.path -eq 'scripts/StylePile.py') { continue }
      $entry = $archive.GetEntry('stylepile/' + $item.path)
      if ($null -eq $entry) { throw "Missing packaged source file: $($item.path)" }
      $stream = $entry.Open()
      try { $digest = [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
      finally { $stream.Dispose() }
      if ($digest -ne $item.sha256) { throw "Packaged source SHA256 mismatch: $($item.path)" }
      $verified++
    }
    if ($verified -ne 517) { throw "Wrong packaged source verification count: $verified" }
  } finally { $hasher.Dispose() }

} finally { $archive.Dispose() }
$before = (Get-FileHash -LiteralPath $packagedJar -Algorithm SHA256).Hash
$env:PYTHONPATH = Join-Path $repoRoot 'python'
Push-Location $repoRoot
try {
  # Only the test harness and assembled jar are on the classpath: main classes/resources
  # must come from the jar, never plugins/target/classes or source resource directories.
  & $JavaExe '-cp' "$testClasses;$packagedJar" 'sdfx.ui.RealStylePileCheck' $OutputHome $pythonExe $Model
  if ($LASTEXITCODE -ne 0) { throw "Packaged native acceptance failed: exit $LASTEXITCODE" }
} finally { Pop-Location }
$after = (Get-FileHash -LiteralPath $packagedJar -Algorithm SHA256).Hash
if ($before -ne $after) { throw 'Assembled jar changed during native acceptance' }
Write-Output "PASS packaged StylePile: 517 source hashes, 512 previews, native gallery/manual RGB, unchanged jar SHA256 $after"
