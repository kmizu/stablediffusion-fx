param(
  [Parameter(Mandatory=$true)][string]$Python,
  [Parameter(Mandatory=$true)][string]$PhotoMakerSource,
  [Parameter(Mandatory=$true)][string]$Target
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath $PhotoMakerSource).Path
$python = (Resolve-Path -LiteralPath $Python).Path
if ((git -C $source rev-parse HEAD) -ne '060b4fcb10b76a4554edf565d6106b7e36c968f0') {
  throw 'PhotoMaker checkout is not the researched source revision.'
}
if (git -C $source status --porcelain) { throw 'PhotoMaker checkout has local changes.' }
if (Test-Path -LiteralPath $Target) { throw "Refusing to overwrite target: $Target" }
New-Item -ItemType Directory -Path $Target | Out-Null
$target = (Resolve-Path -LiteralPath $Target).Path
$priorPythonPath = $env:PYTHONPATH
$vsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vsRoot = & $vsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsRoot) { throw 'Visual Studio C++ Build Tools not found.' }
$vsDev = Join-Path $vsRoot 'Common7/Tools/VsDevCmd.bat'
$oldBuildEnv = @{}
$allowedBuildVars = @('PATH', 'INCLUDE', 'LIB', 'LIBPATH', 'VCToolsInstallDir', 'WindowsSdkDir', 'WindowsSDKVersion', 'UniversalCRTSdkDir', 'UCRTVersion')
$command = 'call "' + $vsDev + '" -arch=x64 -host_arch=x64 >nul && set'
foreach ($line in (& $env:ComSpec /d /c $command)) {
  $parts = $line -split '=', 2
  if ($parts.Count -eq 2 -and $allowedBuildVars -contains $parts[0]) {
    $oldBuildEnv[$parts[0]] = [Environment]::GetEnvironmentVariable($parts[0], 'Process')
    [Environment]::SetEnvironmentVariable($parts[0], $parts[1], 'Process')
  }
}
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw 'Visual C++ compiler environment could not be initialized.' }
try {
  # pip's source build needs numpy and Cython during metadata preparation.
  # The target is separate: do not modify the shared smoke environment.
  & $python -m pip install --target $target 'numpy==2.2.5' 'Cython>=3,<4' wheel setuptools
  if ($LASTEXITCODE -ne 0) { throw 'Could not prepare isolated build dependencies.' }
  $env:PYTHONPATH = $target
  $release = Invoke-RestMethod 'https://pypi.org/pypi/insightface/0.7.3/json'
  $artifact = @($release.urls | Where-Object { $_.filename -eq 'insightface-0.7.3.tar.gz' })
  $expected = 'f191f719612ebb37018f41936814500544cd0f86e6fcd676c023f354c668ddf7'
  if ($artifact.Count -ne 1 -or $artifact[0].digests.sha256 -ne $expected) { throw 'Unexpected insightface source metadata.' }
  $archive = Join-Path $target 'insightface-0.7.3.tar.gz'
  Invoke-WebRequest -Uri $artifact[0].url -OutFile $archive
  if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw 'Insightface source hash mismatch.' }
  & $python -m pip install --target $target --no-build-isolation --no-deps $archive
  if ($LASTEXITCODE -ne 0) { throw 'Could not build isolated insightface 0.7.3.' }
  # These are required by upstream insightface 0.7.3; pip installs their
  # dependencies into the target, not the shared environment.
  & $python -m pip install --target $target 'numpy==2.2.5' scikit-learn albumentations easydict prettytable
  if ($LASTEXITCODE -ne 0) { throw 'Could not install isolated runtime dependencies.' }
  $env:PYTHONPATH = "$target;$source"
  & $python -c 'import photomaker, insightface; print("PhotoMaker and insightface imported:", photomaker.__file__, insightface.__file__)'
  if ($LASTEXITCODE -ne 0) { throw 'Genuine PhotoMaker package import failed.' }
  & $python -m pip list --path $target --format=json | Set-Content -LiteralPath (Join-Path $target 'installed-packages.json') -Encoding utf8
  Write-Output "PASS isolated PhotoMaker import. For the worker, prepend $target and $source to PYTHONPATH."
} finally {
  $env:PYTHONPATH = $priorPythonPath
  foreach ($name in $oldBuildEnv.Keys) { [Environment]::SetEnvironmentVariable($name, $oldBuildEnv[$name], 'Process') }
}
