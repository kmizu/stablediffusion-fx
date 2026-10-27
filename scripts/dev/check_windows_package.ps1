[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Zip,
    [Parameter(Mandatory)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$ExpectedSha256,
    [Parameter(Mandatory)][string]$Python,
    [Parameter(Mandatory)][string]$ArtifactRoot
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$zipPath = [IO.Path]::GetFullPath($Zip)
$pythonPath = [IO.Path]::GetFullPath($Python)
foreach ($file in @($zipPath, $pythonPath)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Required file missing: $file" }
}
$actual = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $ExpectedSha256.ToLowerInvariant()) { throw 'ZIP SHA-256 does not match the supplied release hash.' }

# A new child only; never remove, replace or traverse another extracted build.
$root = [IO.Path]::GetFullPath($ArtifactRoot)
[IO.Directory]::CreateDirectory($root) | Out-Null
$run = Join-Path $root ('package check ' + [Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($run) | Out-Null
Expand-Archive -LiteralPath $zipPath -DestinationPath $run
$packageHome = Join-Path $run 'SDFX'
$exe = Join-Path $packageHome 'SDFX.exe'
foreach ($required in @($exe, (Join-Path $packageHome 'runtime/bin/server/jvm.dll'),
                         (Join-Path $packageHome 'app/stablediffusion-fx.jar'), (Join-Path $packageHome 'python/sdfx/__main__.py'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Package component missing: $required" }
}
$weights = @(Get-ChildItem -LiteralPath $packageHome -File -Recurse | Where-Object {
    $_.Extension.ToLowerInvariant() -in @('.safetensors', '.ckpt', '.onnx', '.pt', '.pth', '.bin')
})
if ($weights.Count -ne 0) { throw "Unexpected potential model weights in package: $($weights[0].FullName)" }

# The extracted app tree is this run's isolated home. Only the explicit Python
# path is stored; no user settings, model or venv are copied into the package.
$config = Join-Path $packageHome 'config'
[IO.Directory]::CreateDirectory($config) | Out-Null
$settings = @{ python_exe = $pythonPath; output_dir = 'outputs' } | ConvertTo-Json -Compress
[IO.File]::WriteAllText((Join-Path $config 'settings.json'), $settings, [Text.UTF8Encoding]::new($false))
$screenshot = Join-Path $run 'dry run.png'
$stdout = Join-Path $run 'launcher.stdout.log'
$stderr = Join-Path $run 'launcher.stderr.log'
$oldPath = $env:PATH
$oldJavaHome = $env:JAVA_HOME
try {
    # Do not use a system Java, even if an unrelated JDK is on the user's PATH.
    $parts = @($oldPath -split ';' | Where-Object {
        $_ -and $_ -notmatch '(?i)(?:^|[\\/])(jdk[^\\/]*|jre[^\\/]*|java[^\\/]*)(?:[\\/]|$)' -and
        -not (Test-Path -LiteralPath (Join-Path $_ 'java.exe') -PathType Leaf)
    })
    $env:PATH = $parts -join ';'
    Remove-Item Env:JAVA_HOME -ErrorAction SilentlyContinue
    # Windows paths cannot contain double quotes; explicit quoting preserves spaces.
    $args = @('--dry-run', '--autogen', '--home', ('"' + $packageHome + '"'),
              '--screenshot', ('"' + $screenshot + '"'), '--exit-after', '15')
    $process = Start-Process -FilePath $exe -ArgumentList $args -WorkingDirectory $run -PassThru `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
} finally {
    $env:PATH = $oldPath
    if ($null -eq $oldJavaHome) { Remove-Item Env:JAVA_HOME -ErrorAction SilentlyContinue }
    else { $env:JAVA_HOME = $oldJavaHome }
}

if (-not $process.WaitForExit(30000)) {
    [pscustomobject]@{ status = 'still-running'; pid = $process.Id; artifacts = $run; screenshot = $screenshot; zip_sha256 = $actual } |
        ConvertTo-Json -Compress
    return
}
$process.Refresh()
if ($process.ExitCode -ne 0) { throw "Packaged launcher exited $($process.ExitCode); logs retained at $run" }
if (-not (Test-Path -LiteralPath $screenshot -PathType Leaf)) { throw "Screenshot missing; logs retained at $run" }
$outputs = Join-Path $packageHome 'outputs'
$png = @(if (Test-Path -LiteralPath $outputs -PathType Container) { Get-ChildItem -LiteralPath $outputs -File -Recurse -Filter '*.png' })
if ($png.Count -eq 0) { throw "Dry-run autogen produced no PNG; logs retained at $run" }
[pscustomobject]@{ status = 'passed'; pid = $process.Id; artifacts = $run; screenshot = $screenshot;
    generated_png_count = $png.Count; zip_sha256 = $actual; note = 'dry-run only; inspect screenshot and logs' } |
    ConvertTo-Json -Compress
