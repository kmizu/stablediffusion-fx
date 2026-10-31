param(
  [Parameter(Mandatory=$true)][string]$PackageRoot,
  [Parameter(Mandatory=$true)][string]$Python,
  [Parameter(Mandatory=$true)][string]$Model,
  [Parameter(Mandatory=$true)][string]$OutputHome,
  [Parameter(Mandatory=$true)][int]$Size,
  [string]$Lora = '',
  [string]$LoraTagOptions = '',
  [string]$Embedding = '',
  [ValidateSet('gpu','model_offload','sequential_offload')][string]$VramMode = 'gpu',
  [string]$ModesHome = '',
  [string]$LogDirectory = ''
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($LoraTagOptions -and -not $Lora) { throw 'LoraTagOptions requires Lora.' }
if ($Embedding -and (-not $Lora -or -not $LoraTagOptions)) { throw 'Embedding acceptance requires Lora and LoraTagOptions.' }
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$package = (Resolve-Path -LiteralPath $PackageRoot).Path
$jar = Join-Path $package 'app/stablediffusion-nui.jar'
$launcher = Join-Path $package 'SDNUI.exe'
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) { throw 'Extracted package launcher is missing.' }
$modules = @(git -C $repo ls-files -- python/sdfx | Where-Object { $_ -match '\.py$' })
foreach ($required in @('python/sdfx/model_keyword.py', 'python/sdfx/model_keyword_conditioning.py', 'python/sdfx/dynamic_lora.py', 'python/sdfx/data/model_keyword/model-keyword.txt', 'python/sdfx/data/model_keyword/lora-keyword.txt')) {
  if (-not (git -C $repo ls-files --error-unmatch -- $required 2>$null)) { throw "Required source is not staged: $required" }
  if ((Get-FileHash -LiteralPath (Join-Path $repo $required)).Hash -ne (Get-FileHash -LiteralPath (Join-Path $package $required)).Hash) { throw "Packaged file differs: $required" }
}
foreach ($module in $modules) {
  if ((Get-FileHash -LiteralPath (Join-Path $repo $module)).Hash -ne (Get-FileHash -LiteralPath (Join-Path $package $module)).Hash) { throw "Packaged module differs: $module" }
}
$before = (Get-FileHash -LiteralPath $jar -Algorithm SHA256).Hash
if ($before -ne (Get-FileHash -LiteralPath (Join-Path $repo 'target/stablediffusion-nui.jar') -Algorithm SHA256).Hash) { throw 'Extracted jar differs from the fresh assembly.' }
if (-not $LogDirectory) { $LogDirectory = "$OutputHome.logs" }
if (Test-Path -LiteralPath $LogDirectory) { throw "Log directory already exists: $LogDirectory" }
New-Item -ItemType Directory -Path $LogDirectory | Out-Null
$logDir = (Resolve-Path -LiteralPath $LogDirectory).Path
$run = [ordered]@{
  started_utc = [DateTime]::UtcNow.ToString('o')
  package_root = $package
  jar_sha256 = $before
  python = (Resolve-Path -LiteralPath $Python).Path
  model = (Resolve-Path -LiteralPath $Model).Path
  model_sha256 = (Get-FileHash -LiteralPath $Model -Algorithm SHA256).Hash
  lora = if ($Lora) { (Resolve-Path -LiteralPath $Lora).Path } else { $null }
  lora_sha256 = if ($Lora) { (Get-FileHash -LiteralPath $Lora -Algorithm SHA256).Hash } else { $null }
  lora_tag_options = $LoraTagOptions
  embedding = if ($Embedding) { (Resolve-Path -LiteralPath $Embedding).Path } else { $null }
  embedding_sha256 = if ($Embedding) { (Get-FileHash -LiteralPath $Embedding -Algorithm SHA256).Hash } else { $null }
  size = $Size
  vram_mode = $VramMode
  generation_home = [IO.Path]::GetFullPath($OutputHome)
  modes_home = if ($ModesHome) { [IO.Path]::GetFullPath($ModesHome) } else { $null }
  git_head = (git -C $repo rev-parse HEAD)
}
$run | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logDir 'run.json') -Encoding utf8
git -C $repo status --short | Set-Content -LiteralPath (Join-Path $logDir 'git-status.txt') -Encoding utf8
$env:PYTHONPATH = Join-Path $package 'python'
$classes = Join-Path $repo 'ui/target/scala-3.3.6/test-classes'
Get-ChildItem -LiteralPath (Join-Path $classes 'sdfx/ui') -Filter 'RealModelKeyword*Check*.class' |
  ForEach-Object { '{0}  {1}' -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash, $_.Name } |
  Set-Content -LiteralPath (Join-Path $logDir 'harness-sha256.txt') -Encoding utf8
$arguments = @('-cp', "$classes;$jar", 'sdfx.ui.RealModelKeywordGenerationCheck', $OutputHome, $Python, $Model, $Size.ToString())
if ($Lora) { $arguments += $Lora }
if ($LoraTagOptions) { $arguments += $LoraTagOptions }
if ($Embedding) { $arguments += $Embedding }
if ($VramMode -ne 'gpu') { $arguments += "--vram-mode=$VramMode" }
$config = Join-Path $package 'app/SDNUI.cfg'
$configBytes = [IO.File]::ReadAllBytes($config)
$configHash = (Get-FileHash -LiteralPath $config -Algorithm SHA256).Hash
[IO.File]::WriteAllBytes((Join-Path $logDir 'original-SDNUI.cfg'), $configBytes)
function Run-PackagedHarness([string]$className, [string[]]$harnessArgs, [string]$label) {
  $text = [Text.Encoding]::UTF8.GetString($configBytes)
  if (-not $text.Contains('app.mainclass=sdfx.ui.Launcher')) { throw 'Unexpected packaged main class.' }
  $text = $text.Replace('app.mainclass=sdfx.ui.Launcher', "app.mainclass=$className")
  $text = $text.Replace('app.classpath=$APPDIR\stablediffusion-nui.jar', 'app.classpath=$APPDIR\stablediffusion-nui.jar;' + $classes)
  [IO.File]::WriteAllText($config, $text, [Text.UTF8Encoding]::new($false))
  $quoted = @($harnessArgs | ForEach-Object { '"' + $_ + '"' })
  $process = Start-Process -FilePath $launcher -ArgumentList $quoted -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logDir "$label-stdout.log") -RedirectStandardError (Join-Path $logDir "$label-stderr.log")
  Write-Output "Owned packaged launcher PID $($process.Id)"
  $deadline = [DateTime]::UtcNow.AddMinutes(12)
  while (-not $process.WaitForExit(1000)) {
    if ([DateTime]::UtcNow -ge $deadline) { $process.Kill(); throw "Packaged $label acceptance timed out." }
  }
  $process.WaitForExit()
  if ($process.ExitCode -ne 0) { throw "Packaged $label acceptance failed: $($process.ExitCode)" }
}
try {
  Run-PackagedHarness 'sdfx.ui.RealModelKeywordGenerationCheck' $arguments[3..($arguments.Length - 1)] 'generation'
  if ($ModesHome) {
    Run-PackagedHarness 'sdfx.ui.RealModelKeywordModesCheck' @($ModesHome, $Python, $Model, $Size.ToString()) 'modes'
  }
} finally {
  [IO.File]::WriteAllBytes($config, $configBytes)
  if ((Get-FileHash -LiteralPath $config -Algorithm SHA256).Hash -ne $configHash) { throw 'Packaged launcher config restoration failed.' }
}
if ((Get-FileHash -LiteralPath $jar -Algorithm SHA256).Hash -ne $before) { throw 'Jar changed during acceptance.' }
Write-Output "PASS extracted Model Keyword native checks; jar SHA256 $before; logs $logDir"
