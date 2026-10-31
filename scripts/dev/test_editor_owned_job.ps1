# Light Windows process-isolation regression; all spawned processes belong to this test.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Add-Type -Path (Join-Path $PSScriptRoot 'EditorOwnedJob.cs')
$psExe = (Get-Process -Id $PID).Path
$prefix = Join-Path $root ('outputs/editor-owned-job-' + [guid]::NewGuid().ToString('N'))
$childScript = $prefix + '.ps1'
@'
param([string]$ChildFile)
$child = Start-Process -FilePath (Get-Process -Id $PID).Path -ArgumentList '-NoProfile','-Command','Start-Sleep -Seconds 60' -WindowStyle Hidden -PassThru
[IO.File]::WriteAllText($ChildFile, [string]$child.Id)
Start-Sleep -Seconds 60
'@ | Set-Content -Encoding utf8 $childScript
$sentinel = Start-Process -FilePath $psExe -ArgumentList '-NoProfile','-Command','Start-Sleep -Seconds 60' -WindowStyle Hidden -PassThru
$owned = $null
try {
  $owned = [EditorOwnedJob]::new($psExe, @('-NoProfile','-File',$childScript,($prefix+'.pid')), ($prefix+'.out'), ($prefix+'.err'))
  $deadline = [DateTime]::UtcNow.AddSeconds(15)
  while (-not (Test-Path -LiteralPath ($prefix+'.pid')) -and [DateTime]::UtcNow -lt $deadline) { Start-Sleep -Milliseconds 100 }
  if (-not (Test-Path -LiteralPath ($prefix+'.pid'))) { throw 'owned child did not start' }
  $childPid = [int]([IO.File]::ReadAllText($prefix+'.pid'))
  if ($owned.Active -lt 2) { throw 'owned child was not assigned to private job' }
  $owned.TerminateAndWait(10000)
  if ($owned.Active -ne 0 -or (Get-Process -Id $childPid -ErrorAction SilentlyContinue)) { throw 'owned process remains' }
  if (-not (Get-Process -Id $sentinel.Id -ErrorAction SilentlyContinue)) { throw 'unrelated process stopped' }
  Write-Output "PASS owned parent $($owned.Pid) and child $childPid exited; unrelated sentinel $($sentinel.Id) survived"
} finally {
  if ($null -ne $owned) { try { if ($owned.Active -ne 0) { $owned.TerminateAndWait(10000) } } finally { $owned.Dispose() } }
  if (-not $sentinel.HasExited) { $sentinel.Kill(); $sentinel.WaitForExit() }
}
