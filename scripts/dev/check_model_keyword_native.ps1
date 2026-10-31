param(
  [Parameter(Mandatory=$true)][string]$Python,
  [Parameter(Mandatory=$true)][string]$Model,
  [Parameter(Mandatory=$true)][string]$OutputHome,
  [Parameter(Mandatory=$true)][int]$Size,
  [string]$Lora = '',
  [string]$LoraTagOptions = '',
  [string]$Embedding = '',
  [ValidateSet('gpu','model_offload','sequential_offload')][string]$VramMode = 'gpu',
  [string]$TracePythonPath = '',
  [switch]$CheckLoraReenable,
  [string]$ModesHome = '',
  [string]$LogDirectory = ''
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($LoraTagOptions -and -not $Lora) { throw 'LoraTagOptions requires Lora.' }
if ($Embedding -and (-not $Lora -or -not $LoraTagOptions)) { throw 'Embedding acceptance requires Lora and LoraTagOptions.' }
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$jar = Join-Path $repo 'target/stablediffusion-nui.jar'
$classes = Join-Path $repo 'ui/target/scala-3.3.6/test-classes'
if (-not $LogDirectory) { $LogDirectory = "$OutputHome.logs" }
if (Test-Path -LiteralPath $LogDirectory) { throw "Log directory already exists: $LogDirectory" }
New-Item -ItemType Directory -Path $LogDirectory | Out-Null
$logDir = (Resolve-Path -LiteralPath $LogDirectory).Path
$run = [ordered]@{
  started_utc = [DateTime]::UtcNow.ToString('o')
  repo = $repo
  jar_sha256 = (Get-FileHash -LiteralPath $jar -Algorithm SHA256).Hash
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
  diagnostic_trace = [bool]$TracePythonPath
  check_lora_reenable = [bool]$CheckLoraReenable
  generation_home = [IO.Path]::GetFullPath($OutputHome)
  modes_home = if ($ModesHome) { [IO.Path]::GetFullPath($ModesHome) } else { $null }
  git_head = (git -C $repo rev-parse HEAD)
}
$run | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logDir 'run.json') -Encoding utf8
git -C $repo status --short | Set-Content -LiteralPath (Join-Path $logDir 'git-status.txt') -Encoding utf8
Get-ChildItem -LiteralPath (Join-Path $repo 'python/sdfx') -Recurse -File -Filter '*.py' |
  ForEach-Object { '{0}  {1}' -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash, $_.FullName.Substring($repo.Length + 1) } |
  Set-Content -LiteralPath (Join-Path $logDir 'python-sha256.txt') -Encoding utf8
Get-ChildItem -LiteralPath (Join-Path $classes 'sdfx/ui') -Filter 'RealModelKeyword*Check*.class' |
  ForEach-Object { '{0}  {1}' -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash, $_.Name } |
  Set-Content -LiteralPath (Join-Path $logDir 'harness-sha256.txt') -Encoding utf8
$env:PYTHONPATH = Join-Path $repo 'python'
if ($TracePythonPath) {
  $env:PYTHONPATH = "$(Resolve-Path -LiteralPath $TracePythonPath);$env:PYTHONPATH"
  $env:SDFX_KEYWORD_OFFLOAD_TRACE = Join-Path $logDir 'offload-parameter-trace.jsonl'
}
$arguments = @('-cp', "$classes;$jar", 'sdfx.ui.RealModelKeywordGenerationCheck', $OutputHome, $Python, $Model, $Size.ToString())
if ($Lora) { $arguments += $Lora }
if ($LoraTagOptions) { $arguments += $LoraTagOptions }
if ($Embedding) { $arguments += $Embedding }
if ($VramMode -ne 'gpu') { $arguments += "--vram-mode=$VramMode" }
if ($CheckLoraReenable) { $arguments += '--check-lora-reenable' }
& 'C:/Program Files/Java/jdk-21/bin/java.exe' @arguments 2>&1 | Tee-Object -FilePath (Join-Path $logDir 'generation.log')
if ($LASTEXITCODE -ne 0) { throw "Native Model Keyword acceptance failed: $LASTEXITCODE" }
if ($ModesHome) {
  & 'C:/Program Files/Java/jdk-21/bin/java.exe' -cp "$classes;$jar" 'sdfx.ui.RealModelKeywordModesCheck' $ModesHome $Python $Model $Size 2>&1 | Tee-Object -FilePath (Join-Path $logDir 'modes.log')
  if ($LASTEXITCODE -ne 0) { throw "Native Model Keyword mode acceptance failed: $LASTEXITCODE" }
}
if ((Get-FileHash -LiteralPath $jar -Algorithm SHA256).Hash -ne $run.jar_sha256) { throw 'Jar changed during acceptance.' }
Write-Output "PASS source Model Keyword native checks; logs $logDir"
