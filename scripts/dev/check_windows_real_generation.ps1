[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$PackageHome,
    [Parameter(Mandatory)][string]$Python,
    [Parameter(Mandatory)][string]$Checkpoint,
    [Parameter(Mandatory)][string]$ArtifactRoot
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$package = [IO.Path]::GetFullPath($PackageHome)
$pythonPath = [IO.Path]::GetFullPath($Python)
$checkpointPath = [IO.Path]::GetFullPath($Checkpoint)
$exe = Join-Path $package 'SDNUI.exe'
foreach ($file in @($exe, $pythonPath, $checkpointPath, (Join-Path $package 'app/stablediffusion-nui.jar'), (Join-Path $package 'runtime/bin/server/jvm.dll'))) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Required file missing: $file" }
}
# Caller explicitly supplies a disposable extracted package; never follow config links.
foreach ($path in @($package, (Join-Path $package 'config'))) {
    if ((Test-Path -LiteralPath $path) -and ((Get-Item -LiteralPath $path).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Refusing linked package/config directory: $path"
    }
}
$runId = [Guid]::NewGuid().ToString('N')
$run = Join-Path ([IO.Path]::GetFullPath($ArtifactRoot)) ('real-package-' + $runId)
[IO.Directory]::CreateDirectory($run) | Out-Null
$outputName = 'real-check-' + $runId
$outputs = Join-Path $package $outputName
if (Test-Path -LiteralPath $outputs) { throw 'Unique output directory already exists.' }
[IO.Directory]::CreateDirectory($outputs) | Out-Null
$config = Join-Path $package 'config'
[IO.Directory]::CreateDirectory($config) | Out-Null
foreach ($name in @('settings.json', 'ui-state.json')) {
    $existing = Join-Path $config $name
    if (Test-Path -LiteralPath $existing) {
        if ((Get-Item -LiteralPath $existing).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refusing linked config: $existing" }
        Copy-Item -LiteralPath $existing -Destination (Join-Path $run ('before-' + $name))
    }
}
$prompt = 'blue hair, detailed portrait, packaged NUI check ' + $runId
$settings = @{
    python_exe = $pythonPath; hf_offline = $true; output_dir = $outputName
    model_dirs = @{ checkpoints = @((Split-Path -Parent $checkpointPath)) }
    last_model = $checkpointPath
} | ConvertTo-Json -Depth 8
$ui = @{
    selected_tab = 'txt2img'
    params = @{ txt2img = @{
        prompt = $prompt; negative_prompt = 'blurry, low quality'
        width = 256; height = 256; steps = 2; seed = 7303
        batch_size = 1; n_iter = 1; cfg_scale = 6.5
        sampler_name = 'Euler'; scheduler = 'Automatic'
    }}
} | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText((Join-Path $config 'settings.json'), $settings, [Text.UTF8Encoding]::new($false))
[IO.File]::WriteAllText((Join-Path $config 'ui-state.json'), $ui, [Text.UTF8Encoding]::new($false))
Copy-Item -LiteralPath (Join-Path $config 'settings.json') -Destination (Join-Path $run 'requested-settings.json')
Copy-Item -LiteralPath (Join-Path $config 'ui-state.json') -Destination (Join-Path $run 'requested-ui-state.json')
$screenshot = Join-Path $run 'real-generation.png'
$stdout = Join-Path $run 'launcher.stdout.log'
$stderr = Join-Path $run 'launcher.stderr.log'
$oldPath = $env:PATH
$oldJavaHome = $env:JAVA_HOME
try {
    $parts = @($oldPath -split ';' | Where-Object {
        $_ -and $_ -notmatch '(?i)(?:^|[\\/])(jdk[^\\/]*|jre[^\\/]*|java[^\\/]*)(?:[\\/]|$)' -and
        -not (Test-Path -LiteralPath (Join-Path $_ 'java.exe') -PathType Leaf)
    })
    $env:PATH = $parts -join ';'
    Remove-Item Env:JAVA_HOME -ErrorAction SilentlyContinue
    $launchArgs = @('--autogen', '--home', ('"' + $package + '"'), '--screenshot', ('"' + $screenshot + '"'), '--exit-after', '90')
    $process = Start-Process -FilePath $exe -ArgumentList $launchArgs -WorkingDirectory $run -PassThru `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
} finally {
    $env:PATH = $oldPath
    if ($null -eq $oldJavaHome) { Remove-Item Env:JAVA_HOME -ErrorAction SilentlyContinue }
    else { $env:JAVA_HOME = $oldJavaHome }
}
$context = [ordered]@{ pid = $process.Id; artifacts = $run; package_home = $package; outputs = $outputs; screenshot = $screenshot }
[IO.File]::WriteAllText((Join-Path $run 'process.json'), ($context | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
$finished = $false
for ($attempt = 1; $attempt -le 8; $attempt++) {
    if ($process.WaitForExit(15000)) { $finished = $true; break }
    Write-Host "Real package check running: PID $($process.Id), $($attempt * 15)s, artifacts $run"
}
if (-not $finished) {
    $context['status'] = 'still-running'
    $context | ConvertTo-Json -Compress
    return
}
$process.Refresh()
if ($process.ExitCode -ne 0) { throw "Packaged launcher exited $($process.ExitCode); artifacts $run" }
if (-not (Test-Path -LiteralPath $screenshot -PathType Leaf)) { throw "Screenshot missing; artifacts $run" }
# Explicit Python/Pillow verifies a unique output tree, never earlier dry-run images.
$verify = @'
import hashlib, json, pathlib, re, sys
from PIL import Image
out, screenshot, checkpoint, prompt, report, jar = sys.argv[1:]
files = sorted(pathlib.Path(out).rglob('*.png'))
assert len(files) == 1, f'Expected exactly one new generated PNG, found {len(files)}'
p = files[0]
with Image.open(p) as image:
    image.load()
    assert image.size == (256, 256), image.size
    metadata = image.info.get('parameters', '')
    assert metadata.splitlines()[0] == prompt, metadata
    for key, value in [('Seed', '7303'), ('Steps', '2'), ('Size', '256x256')]:
        assert re.search(r'(?:^|,\s*)' + key + r':\s*' + re.escape(value) + r'(?=,|$)', metadata, re.M), metadata
    model = re.search(r'(?:^|,\s*)Model:\s*([^,\n]+)', metadata, re.M)
    assert model, metadata
    assert pathlib.Path(checkpoint).stem.lower() in model.group(1).lower(), (checkpoint, model.group(1))
with Image.open(screenshot) as shot:
    shot.load()
    assert shot.width > 0 and shot.height > 0
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
result = dict(status='passed', png=str(p), png_sha256=sha(p), width=256, height=256,
              parameters=metadata, screenshot=screenshot, screenshot_sha256=sha(screenshot),
              checkpoint=checkpoint, jar_sha256=sha(jar), note='Actual packaged generation; visual inspection remains required')
pathlib.Path(report).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
'@
$verifier = Join-Path $run 'verify_generation.py'
[IO.File]::WriteAllText($verifier, $verify, [Text.UTF8Encoding]::new($false))
& $pythonPath $verifier $outputs $screenshot $checkpointPath $prompt (Join-Path $run 'real-generation-report.json') (Join-Path $package 'app/stablediffusion-nui.jar')
if ($LASTEXITCODE -ne 0) { throw "PNG/metadata verification failed; artifacts $run" }
