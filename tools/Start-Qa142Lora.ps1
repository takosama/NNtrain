[CmdletBinding()]
param(
    [switch]$Start,
    [switch]$DryRun,
    [switch]$Resume,
    [string]$RunDirectory,
    [ValidateRange(0, 1048576)][int]$ContextLength = 0,
    [ValidateRange(1, 1000)][int]$Epochs = 2,
    [string]$ModelPath
)
$ErrorActionPreference = 'Stop'
if ($Start -and $DryRun) { throw 'Choose -Start or -DryRun.' }
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $ModelPath) { $ModelPath = Join-Path $repo 'models\Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf' }
$ModelPath = [IO.Path]::GetFullPath($ModelPath)
if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) { throw "Model missing: $ModelPath" }
if ($Resume -and -not $RunDirectory) { throw '-Resume requires the original -RunDirectory.' }
if (-not $RunDirectory) {
    $RunDirectory = Join-Path $repo ('checkpoints\qa142-exact-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0,8))
}
$RunDirectory = [IO.Path]::GetFullPath($RunDirectory)
$config = Join-Path $RunDirectory 'train.json'
$adapter = Join-Path $RunDirectory 'adapter.bin'
$prepared = Join-Path $RunDirectory 'preparation.json'
$dll = Join-Path $repo 'NNtrain.Cli\bin\Release\net10.0\NNtrain.Cli.dll'
$source = Join-Path $repo 'data\rintya\rintya_qa_142.jsonl'

# CPU build only. No ExecutionPolicy or network/security settings are changed.
& dotnet build (Join-Path $repo 'NNtrain.Cli\NNtrain.Cli.csproj') -c Release --no-restore -v minimal
if ($LASTEXITCODE -ne 0) { throw 'CLI build failed.' }
if (Test-Path -LiteralPath $RunDirectory) {
    if (-not (Test-Path -LiteralPath $config -PathType Leaf) -or -not (Test-Path -LiteralPath $prepared -PathType Leaf)) {
        throw 'Existing directory is not a prepared QA142 run. Choose a new path.'
    }
    if ($Resume) {
        if (-not (Test-Path -LiteralPath $adapter -PathType Leaf)) { throw '-Resume requires optimizer checkpoint adapter.bin; exported GGUF is insufficient.' }
    } elseif (Test-Path -LiteralPath $adapter) {
        throw 'Checkpoint already exists. Use -Resume explicitly or a new run directory.'
    }
} else {
    if ($Resume) { throw 'Resume directory missing.' }
    & dotnet $dll qwen-lora-prepare --model $ModelPath --data $source --output $RunDirectory --context $ContextLength --epochs $Epochs
    if ($LASTEXITCODE -ne 0) { throw 'CPU preparation failed; training was not started.' }
}

# Prevent duplicate runs/dry-runs in the same directory. The lock is scoped to
# this invocation and released normally; no learning process is stopped.
$lockPath = Join-Path $RunDirectory 'launcher.lock'
$runLock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
try {
    $manifest = Get-Content -LiteralPath $prepared -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.Status -ne 'prepared-cpu-only') { throw 'Preparation manifest is incomplete.' }
    if ($PSBoundParameters.ContainsKey('Epochs') -and $Epochs -ne $manifest.Epochs) {
        throw 'Epochs differs from this saved preparation. Use a new run directory for new settings.'
    }
    if ($PSBoundParameters.ContainsKey('ContextLength') -and $ContextLength -ne 0 -and $ContextLength -ne $manifest.ContextLength) {
        throw 'ContextLength differs from this saved preparation. It was not modified.'
    }
    if ([IO.Path]::GetFullPath($manifest.Model) -ne $ModelPath) { throw 'Prepared model path differs. Create a new preparation instead of changing the saved run.' }
    if ((Get-FileHash -LiteralPath (Join-Path $RunDirectory 'training.jsonl') -Algorithm SHA256).Hash -ne $manifest.TrainingSha256) {
        throw 'Prepared training dataset changed. Refusing an accidental resume/new-run mix.'
    }
    $trainArgs = @('qwen-lora','--model',$ModelPath,'--config',$config)
    if ($Resume) { $trainArgs += '--resume' }
    & dotnet $dll @trainArgs --dry-run 2>&1 | Tee-Object -FilePath (Join-Path $RunDirectory 'dry-run.log')
    if ($LASTEXITCODE -ne 0) { throw 'CPU dry-run failed; no GPU work was started.' }
    Write-Host "Run directory: $RunDirectory"
    Write-Host 'Prepared snapshot uses exact precision and preserves all examples. See preparation.json for actual lengths and proposed settings.'
    if (-not $Start) {
        Write-Host 'CPU preparation/validation only. To start this prepared run:'
        $resumeText = if ($Resume) { ' -Resume' } else { '' }
        Write-Host ('.\tools\Start-Qa142Lora.ps1 -Start -RunDirectory "' + $RunDirectory + '"' + $resumeText)
        return
    }
    $active = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^(NNtrain\.Cli|SpeedProbe)\.exe$' -or
        ($_.Name -eq 'dotnet.exe' -and $_.CommandLine -match '(NNtrain\.Cli\.dll|SpeedProbe\.dll)')
    }
    if ($active) { throw 'Another NNtrain CLI/probe is running. Finish it before starting; it was not stopped.' }
    Write-Host 'Starting the explicitly requested training. Ctrl+C interrupts it; resume later from the saved adapter.bin.'
    & dotnet $dll @trainArgs 2>&1 | Tee-Object -FilePath (Join-Path $RunDirectory 'training-console.log') -Append
    if ($LASTEXITCODE -ne 0) { throw 'Training failed/interrupted. Preserve the logs and resume only from a saved checkpoint.' }
} finally {
    $runLock.Dispose()
}
