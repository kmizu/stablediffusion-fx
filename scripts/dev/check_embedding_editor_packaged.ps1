param(
  [Parameter(Mandatory=$true)][string]$PackageRoot,
  [Parameter(Mandatory=$true)][string]$Python,
  [Parameter(Mandatory=$true)][string]$Checkpoint,
  [ValidateSet("gpu", "model_offload", "sequential_offload")][string]$VramMode = "model_offload",
  [Parameter(Mandatory=$true)][string]$OutputHome
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$package = (Resolve-Path -LiteralPath $PackageRoot).Path
$jar = Join-Path $package 'app/stablediffusion-nui.jar'
$launcher = Join-Path $package 'SDNUI.exe'
if (-not (Test-Path -LiteralPath $launcher)) { throw 'Extracted application launcher is missing.' }
$modules = @(git -C $repo ls-files -- python/sdfx | Where-Object { $_ -match '\.py$' })
if ($LASTEXITCODE -ne 0) { throw 'git ls-files failed' }
foreach ($required in @('python/sdfx/embedding_editor_generation.py', 'python/sdfx/embedding_editor_service.py', 'python/sdfx/embedding_editor_preview.py', 'python/sdfx/embedding_merge_pipeline.py', 'python/sdfx/photomaker_adapters.py')) {
  if (-not ($modules -contains $required)) { throw "New module is not indexed for packaging: $required" }
}
foreach ($module in $modules) {
  if ((Get-FileHash -LiteralPath (Join-Path $repo $module) -Algorithm SHA256).Hash -ne
      (Get-FileHash -LiteralPath (Join-Path $package $module) -Algorithm SHA256).Hash) { throw "Packaged module differs: $module" }
}
$before = (Get-FileHash -LiteralPath $jar -Algorithm SHA256).Hash
if ($before -ne (Get-FileHash -LiteralPath (Join-Path $repo 'target/stablediffusion-nui.jar') -Algorithm SHA256).Hash) {
  throw 'Extracted jar differs from fresh assembly.'
}
$output = [IO.Path]::GetFullPath($OutputHome)
[IO.Directory]::CreateDirectory($output) | Out-Null
$reportPath = Join-Path $output 'package-run-complete.txt'
if (Test-Path -LiteralPath $reportPath) { throw 'Use a fresh output directory for packaged acceptance.' }
$classes = Join-Path $repo 'ui/target/scala-3.3.6/test-classes'
$config = Join-Path $package 'app/SDNUI.cfg'
Add-Type -Path (Join-Path $PSScriptRoot 'EditorOwnedJob.cs')
$ownedJob = $null
$configBytes = [IO.File]::ReadAllBytes($config)
$configHash = (Get-FileHash -LiteralPath $config -Algorithm SHA256).Hash
[IO.File]::WriteAllBytes((Join-Path $output 'original-SDNUI.cfg'), $configBytes)
try {
  $text = [Text.Encoding]::UTF8.GetString($configBytes)
  if (-not $text.Contains('app.mainclass=sdfx.ui.Launcher')) { throw 'Unexpected packaged main class.' }
  $text = $text.Replace('app.mainclass=sdfx.ui.Launcher', 'app.mainclass=sdfx.ui.RealEmbeddingEditorProtectedGpuCheck')
  $text = $text.Replace('app.classpath=$APPDIR\stablediffusion-nui.jar', 'app.classpath=$APPDIR\stablediffusion-nui.jar;' + $classes)
  [IO.File]::WriteAllText($config, $text, [Text.UTF8Encoding]::new($false))
  $arguments = @($output, $Python, $Checkpoint, $VramMode, (Join-Path $package 'python'))
  $ownedJob = [EditorOwnedJob]::new($launcher, $arguments,
    (Join-Path $output 'launcher-stdout.log'), (Join-Path $output 'launcher-stderr.log'))
  Write-Output "Owned packaged launcher PID $($ownedJob.Pid); descendants assigned before launch"
  if (-not $ownedJob.Wait(600000)) {
    $ownedJob.TerminateAndWait(30000)
    throw 'Owned packaged acceptance timed out; its private job has exited.'
  }
  $exitCode = $ownedJob.ExitCode
  if ($ownedJob.Active -ne 0) {
    $ownedJob.TerminateAndWait(30000)
    throw 'Packaged launcher exited while an owned descendant was still running.'
  }
  if ($exitCode -ne 0) { throw "Packaged Embedding Editor acceptance failed: $exitCode" }
  $receipts = @(Get-ChildItem -LiteralPath $output -Filter 'receipt.json' -File -Recurse)
  if ($receipts.Count -ne 1) { throw "Expected one editor receipt, found $($receipts.Count)." }
  $report = Get-Content -LiteralPath $receipts[0].FullName -Raw | ConvertFrom-Json
  if (-not $report.inversion_saved_matches_preview) { throw 'Inversion override preview/save parity is incomplete.' }
  if (-not $report.preview_source_unchanged) { throw 'Preview changed the source embedding.' }
  if (-not $report.save_consent_enforced -or -not $report.saved_reload_matches_preview -or -not $report.independent_owner_collision_refused) { throw 'Native save/reload acceptance is incomplete.' }
  [IO.File]::WriteAllText($reportPath, $receipts[0].FullName, [Text.UTF8Encoding]::new($false))
} finally {
  try {
    if ($null -ne $ownedJob) {
      try { if ($ownedJob.Active -ne 0) { $ownedJob.TerminateAndWait(30000) } }
      finally { $ownedJob.Dispose() }
    }
  } finally {
    [IO.File]::WriteAllBytes($config, $configBytes)
    if ((Get-FileHash -LiteralPath $config -Algorithm SHA256).Hash -ne $configHash) { throw 'Packaged launcher config restoration failed.' }
  }
}
if ((Get-FileHash -LiteralPath $jar -Algorithm SHA256).Hash -ne $before) { throw 'Jar changed during acceptance.' }
Write-Output "PASS extracted Embedding Editor native acceptance; jar SHA256 $before"
