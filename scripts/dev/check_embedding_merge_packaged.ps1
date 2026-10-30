param(
  [Parameter(Mandatory=$true)][string]$PackageRoot,
  [Parameter(Mandatory=$true)][string]$Python,
  [Parameter(Mandatory=$true)][string]$ModelDirectory,
  [Parameter(Mandatory=$true)][string]$ModelName,
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
foreach ($required in @('python/sdfx/embedding_merge.py', 'python/sdfx/embedding_merge_pipeline.py', 'python/sdfx/photomaker_adapters.py')) {
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
$reportPath = Join-Path $output 'native-report.json'
if (Test-Path -LiteralPath $reportPath) { throw 'Use a fresh output directory for packaged acceptance.' }
$classes = Join-Path $repo 'ui/target/scala-3.3.6/test-classes'
$config = Join-Path $package 'app/SDNUI.cfg'
$configBytes = [IO.File]::ReadAllBytes($config)
$configHash = (Get-FileHash -LiteralPath $config -Algorithm SHA256).Hash
[IO.File]::WriteAllBytes((Join-Path $output 'original-SDNUI.cfg'), $configBytes)
try {
  $text = [Text.Encoding]::UTF8.GetString($configBytes)
  if (-not $text.Contains('app.mainclass=sdfx.ui.Launcher')) { throw 'Unexpected packaged main class.' }
  $text = $text.Replace('app.mainclass=sdfx.ui.Launcher', 'app.mainclass=sdfx.ui.RealEmbeddingMergeCheck')
  $text = $text.Replace('app.classpath=$APPDIR\stablediffusion-nui.jar', 'app.classpath=$APPDIR\stablediffusion-nui.jar;' + $classes)
  [IO.File]::WriteAllText($config, $text, [Text.UTF8Encoding]::new($false))
  $arguments = @($Python, $ModelDirectory, $ModelName, $output, (Join-Path $output 'mainwindow.png')) |
    ForEach-Object { '"' + $_ + '"' }
  $process = Start-Process -FilePath $launcher -ArgumentList $arguments -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $output 'launcher-stdout.log') -RedirectStandardError (Join-Path $output 'launcher-stderr.log')
  Write-Output "Owned packaged launcher PID $($process.Id)"
  $deadline = [DateTime]::UtcNow.AddMinutes(10)
  while (-not $process.WaitForExit(1000)) {
    if ([DateTime]::UtcNow -ge $deadline) {
      $process.Kill()
      throw 'Owned packaged acceptance process timed out.'
    }
  }
  $process.WaitForExit()
  if ($process.ExitCode -ne 0) { throw "Packaged Embedding Merge acceptance failed: $($process.ExitCode)" }
  if (-not (Test-Path -LiteralPath $reportPath)) { throw 'Packaged harness produced no completion report.' }
  $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
  if ($report.status -ne 'passed-pending-visual-inspection') { throw "Packaged acceptance report failed: $($report.status)" }
} finally {
  [IO.File]::WriteAllBytes($config, $configBytes)
  if ((Get-FileHash -LiteralPath $config -Algorithm SHA256).Hash -ne $configHash) { throw 'Packaged launcher config restoration failed.' }
}
if ((Get-FileHash -LiteralPath $jar -Algorithm SHA256).Hash -ne $before) { throw 'Jar changed during acceptance.' }
Write-Output "PASS extracted Embedding Merge native acceptance; jar SHA256 $before"
