[CmdletBinding()]
param(
    [string]$JdkHome = $env:JAVA_HOME,
    [string]$Jar = (Join-Path $PSScriptRoot '../target/stablediffusion-nui.jar'),
    [ValidatePattern('^\d{1,3}(\.\d{1,3}){1,2}$')]
    [string]$Version = '0.02',
    [string]$OutputRoot = ''
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $OutputRoot) { $OutputRoot = Join-Path $PSScriptRoot "../dist/windows-v$Version" }
if (-not $JdkHome) { throw 'Set JAVA_HOME or pass -JdkHome to a Windows x64 JDK 21.' }
$jpackage = Join-Path ([IO.Path]::GetFullPath($JdkHome)) 'bin/jpackage.exe'
$java = Join-Path ([IO.Path]::GetFullPath($JdkHome)) 'bin/java.exe'
$jarPath = [IO.Path]::GetFullPath($Jar)
foreach ($file in @($jpackage, $java, $jarPath)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Required file missing: $file" }
}
$start = [Diagnostics.ProcessStartInfo]::new()
$start.FileName = $java
$start.Arguments = '-XshowSettings:properties -version'
$start.UseShellExecute = $false
$start.RedirectStandardError = $true
$start.RedirectStandardOutput = $true
$process = [Diagnostics.Process]::Start($start)
$versionInfo = $process.StandardError.ReadToEnd() + $process.StandardOutput.ReadToEnd()
$process.WaitForExit()
if ($process.ExitCode -ne 0 -or $versionInfo -notmatch 'version "21[.+"]') { throw 'A JDK 21 installation is required.' }
if ($versionInfo -notmatch 'os.arch\s*=\s*amd64') { throw 'A Windows x64 JDK is required.' }

# A unique child prevents overwriting an earlier release or arbitrary user files.
$root = [IO.Path]::GetFullPath($OutputRoot)
[IO.Directory]::CreateDirectory($root) | Out-Null
$run = Join-Path $root ('build-' + [Guid]::NewGuid().ToString('N'))
if (-not ([IO.Path]::GetFullPath($run)).StartsWith($root.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid output path.' }
[IO.Directory]::CreateDirectory($run) | Out-Null
$inputDir = Join-Path $run 'input'
$imageDest = Join-Path $run 'image'
[IO.Directory]::CreateDirectory($inputDir) | Out-Null
[IO.Directory]::CreateDirectory($imageDest) | Out-Null
Copy-Item -LiteralPath $jarPath -Destination (Join-Path $inputDir 'stablediffusion-nui.jar')

& $jpackage --type app-image --name SDNUI --description "Stable Diffusion NUI" --app-version $Version --dest $imageDest `
    --input $inputDir --main-jar stablediffusion-nui.jar --main-class sdfx.ui.Launcher `
    --java-options '-Dsdfx.home=$APPDIR\..' --java-options '-Dprism.allowhidpi=true' --java-options '-Xmx2g' --java-options '-Dfile.encoding=UTF-8'
if ($LASTEXITCODE -ne 0) { throw "jpackage failed (exit $LASTEXITCODE); staging retained at $run" }
$image = Join-Path $imageDest 'SDNUI'
foreach ($required in @('SDNUI.exe', 'app/stablediffusion-nui.jar', 'runtime/legal')) {
    if (-not (Test-Path -LiteralPath (Join-Path $image $required))) { throw "Missing packaged component: $required" }
}

# Only tracked, explicitly allowed source and licence files are copied: no models, environments,
# credentials, caches, settings, or output files from the developer's checkout.
$tracked = @(git -C $repo ls-files -- python/sdfx models licenses)
if ($LASTEXITCODE -ne 0) { throw 'git ls-files failed; a Git checkout is required.' }
$payload = @($tracked | Where-Object { $_ -match '^python/sdfx/.+\.py$|^python/sdfx/.+/LICENSE$|^models/.+/README\.txt$|^licenses/[A-Za-z0-9_.-]+\.txt$' })
if (-not ($payload -contains 'python/sdfx/__main__.py')) { throw 'Python source files missing from Git index.' }
foreach ($relative in $payload) {
    $source = Join-Path $repo $relative
    $item = Get-Item -LiteralPath $source
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refusing linked source: $relative" }
    $destination = Join-Path $image $relative
    [IO.Directory]::CreateDirectory((Split-Path -Parent $destination)) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
}
foreach ($relative in @('python/requirements.txt', 'python/requirements-torch.txt', 'python/requirements-dev.txt', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'DISTRIBUTION_NOTICES.md')) {
    $source = Join-Path $repo $relative
    $destination = Join-Path $image $relative
    [IO.Directory]::CreateDirectory((Split-Path -Parent $destination)) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
}
Copy-Item -LiteralPath (Join-Path $repo 'docs/WINDOWS_RELEASE.md') -Destination (Join-Path $image 'WINDOWS_RELEASE.md')
$zipName = "SDNUI-v$Version-windows-x64.zip"
$zip = Join-Path $run $zipName
Compress-Archive -LiteralPath $image -DestinationPath $zip -CompressionLevel Optimal
$hash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $run 'SHA256SUMS.txt'), "$hash  $zipName`n", [Text.UTF8Encoding]::new($false))
Write-Host "Package: $zip"
Write-Host "SHA-256: $hash"
Write-Host "Staging retained: $run"
