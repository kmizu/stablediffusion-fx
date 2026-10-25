# smoke: Scala step, then Python tests. Exits non-zero on failure.
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
sbt -batch "engine/runMain sdfx.engine.tools.Smoke"
if ($LASTEXITCODE -ne 0) { Write-Error "sbt step failed"; exit 1 }
Set-Location (Join-Path $root 'python')
$py = Join-Path $root 'python/.venv-test/Scripts/python.exe'
$tmp = $null
if (-not (Test-Path $py)) {
  $tmp = Join-Path ([IO.Path]::GetTempPath()) ("sdfx-pytest-" + [guid]::NewGuid().ToString('N'))
  python -m venv $tmp
  $py = Join-Path $tmp 'Scripts/python.exe'
  & $py -m pip install -q pytest pillow numpy
}
& $py -m pytest -q
$code = $LASTEXITCODE
if ($tmp) { Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue }
exit $code
