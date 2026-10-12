<#
.SYNOPSIS
Plan or run a sequential Gemma 3 Q8_0 CPU/OpenCL benchmark.
.DESCRIPTION
The default invocation prints a plan and does not load the model or use a GPU.
Add -Run only when a benchmark window has been coordinated. Each case launches
the same NNtrain.Cli binary with a fresh process, fixed prompt, and greedy
sampling. Raw stdout/stderr and parsed timing results are saved per case.
#>
[CmdletBinding()]
param(
    [string] $ModelPath = (Join-Path $PSScriptRoot '..\..\models\huggingface\ggml-org\gemma-3-27b-it-GGUF\gemma-3-27b-it-Q8_0.gguf'),
    [string] $CliPath = (Join-Path $PSScriptRoot '..\..\NNtrain.Cli\bin\Release\net10.0\NNtrain.Cli.exe'),
    [string] $OutputDirectory = (Join-Path $PSScriptRoot '..\..\benchmark-results'),
    [string] $RuntimeLabel = 'current',
    [string[]] $GpuLayers = @('0', '10', '20'),
    [string[]] $Threads = @('8'),
    [string[]] $Contexts = @('1024', '4096'),
    [ValidateRange(1, 20)] [int] $Repetitions = 2,
    [ValidateRange(1, 256)] [int] $MaxNewTokens = 48,
    [ValidateRange(2, 256)] [int] $MinGeneratedTokens = 32,
    [ValidateRange(60, 3600)] [int] $TimeoutSeconds = 900,
    [int] $Seed = 20261012,
    [string] $Prompt = 'List the integers from 1 through 100, separated by commas. Do not stop before 100. Do not add any explanation.',
    [switch] $Run
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Convert-IntegerList([string[]] $Values, [string] $Name) {
    $numbers = [Collections.Generic.List[int]]::new()
    foreach ($entry in $Values) {
        foreach ($field in $entry.Split(',')) {
            $number = 0
            if (-not [int]::TryParse($field.Trim(), [ref]$number)) {
                throw "Invalid integer in ${Name}: '$field'"
            }
            $numbers.Add($number)
        }
    }
    return $numbers.ToArray()
}

[int[]] $gpuLayerValues = @(Convert-IntegerList $GpuLayers 'GpuLayers')
[int[]] $threadValues = @(Convert-IntegerList $Threads 'Threads')
[int[]] $contextValues = @(Convert-IntegerList $Contexts 'Contexts')
if ($gpuLayerValues.Count -eq 0 -or $threadValues.Count -eq 0 -or $contextValues.Count -eq 0) {
    throw 'GpuLayers, Threads, and Contexts must each contain at least one value.'
}
if (@($gpuLayerValues | Where-Object { $_ -lt 0 -or $_ -gt 63 }).Count -gt 0) {
    throw 'GpuLayers must be between 0 and 63 for this Gemma 3 model.'
}
if (@($threadValues | Where-Object { $_ -lt 1 }).Count -gt 0) {
    throw 'Threads must be positive.'
}
if (@($contextValues | Where-Object { $_ -lt 1 }).Count -gt 0) {
    throw 'Contexts must be positive.'
}
if ($MinGeneratedTokens -gt $MaxNewTokens) {
    throw 'MinGeneratedTokens cannot exceed MaxNewTokens.'
}

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$modelFull = [IO.Path]::GetFullPath($ModelPath)
$cliFull = [IO.Path]::GetFullPath($CliPath)
$outputRoot = [IO.Path]::GetFullPath($OutputDirectory)
$plan = @(
    foreach ($rep in 1..$Repetitions) {
        foreach ($context in ($contextValues | Sort-Object -Unique)) {
            foreach ($layers in ($gpuLayerValues | Sort-Object -Unique)) {
                foreach ($threadCount in ($threadValues | Sort-Object -Unique)) {
                    [pscustomobject]@{
                        Repetition = $rep
                        Context = $context
                        GpuLayers = $layers
                        Threads = $threadCount
                        Backend = if ($layers -eq 0) { 'cpu' } else { 'opencl' }
                    }
                }
            }
        }
    }
)

# A fixed shuffle keeps settings from always running in speed order while
# preserving a reproducible case sequence for a given plan and seed.
$random = [Random]::new($Seed)
for ($i = $plan.Count - 1; $i -gt 0; $i--) {
    $j = $random.Next($i + 1)
    $swap = $plan[$i]
    $plan[$i] = $plan[$j]
    $plan[$j] = $swap
}

Write-Host "Gemma benchmark plan: $($plan.Count) sequential runs; model=$modelFull"
Write-Host "CLI=$cliFull; label=$RuntimeLabel; prompt mode=short/fixed; seed=$Seed"
Write-Host "Configured contexts differ; the prompt text is identical and short in every case."
$plan | Format-Table Repetition, Context, GpuLayers, Threads, Backend -AutoSize
if (-not $Run) {
    Write-Host 'Plan only. Add -Run after scheduling a GPU benchmark window.'
    return
}

if (-not (Test-Path -LiteralPath $modelFull -PathType Leaf)) { throw "Model missing: $modelFull" }
if (-not (Test-Path -LiteralPath $cliFull -PathType Leaf)) { throw "CLI missing: $cliFull" }
$nativePath = Join-Path (Split-Path -Parent $cliFull) 'gemma-opencl\ggml-opencl.dll'
if (($gpuLayerValues | Where-Object { $_ -gt 0 }).Count -gt 0 -and
    -not (Test-Path -LiteralPath $nativePath -PathType Leaf)) {
    throw "OpenCL native DLL missing beside CLI: $nativePath"
}

$expectedModelSha256 = '794d14393b9a15de7f08da16dc35c503edb3eb48dd58fa5beb37b7cf2a188379'
Write-Host 'Hashing the model once before timing runs...'
$modelSha256 = (Get-FileHash -LiteralPath $modelFull -Algorithm SHA256).Hash.ToLowerInvariant()
if ($modelSha256 -ne $expectedModelSha256) {
    throw "Model SHA-256 mismatch: $modelSha256"
}
$cliSha256 = (Get-FileHash -LiteralPath $cliFull -Algorithm SHA256).Hash.ToLowerInvariant()
$nativeSha256 = if (Test-Path -LiteralPath $nativePath -PathType Leaf) {
    (Get-FileHash -LiteralPath $nativePath -Algorithm SHA256).Hash.ToLowerInvariant()
} else { $null }

$runId = 'gemma3-opencl-sweep-' + (Get-Date -AsUTC -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$runDirectory = Join-Path $outputRoot $runId
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
$manifest = [ordered]@{
    RunId = $runId
    StartedUtc = (Get-Date).ToUniversalTime().ToString('o')
    Model = $modelFull
    ModelSha256 = $modelSha256
    Cli = $cliFull
    CliSha256 = $cliSha256
    OpenClDll = if (Test-Path -LiteralPath $nativePath -PathType Leaf) { $nativePath } else { $null }
    OpenClDllSha256 = $nativeSha256
    RuntimeLabel = $RuntimeLabel
    Prompt = $Prompt
    MaxNewTokens = $MaxNewTokens
    MinGeneratedTokens = $MinGeneratedTokens
    TimeoutSeconds = $TimeoutSeconds
    Seed = $Seed
    Platform = 'Intel'
    Device = 'B580 (first matching OpenCL device)'
    Cases = $plan
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runDirectory 'manifest.json') -Encoding utf8NoBOM

$records = [Collections.Generic.List[object]]::new()
$timingPattern = 'timing: load=([0-9.]+) s, prompt=(\d+) tokens, prefill/first-token=([0-9.]+) ms, decode=([0-9.]+|n/a)(?: tok/s)?, generated=(\d+), generation=([0-9.]+) s'
for ($index = 0; $index -lt $plan.Count; $index++) {
    $case = $plan[$index]
    $caseId = '{0:D3}-ctx{1}-ngl{2}-t{3}-r{4}' -f ($index + 1), $case.Context, $case.GpuLayers, $case.Threads, $case.Repetition
    $stdoutPath = Join-Path $runDirectory "$caseId.stdout.log"
    $stderrPath = Join-Path $runDirectory "$caseId.stderr.log"
    Write-Host "[$($index + 1)/$($plan.Count)] $caseId"

    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $cliFull
    $startInfo.WorkingDirectory = $repoRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.StandardOutputEncoding = [Text.Encoding]::UTF8
    $startInfo.StandardErrorEncoding = [Text.Encoding]::UTF8
    $startInfo.Environment['GGML_OPENCL_PLATFORM'] = 'Intel'
    $startInfo.Environment['GGML_OPENCL_DEVICE'] = 'B580'
    foreach ($argument in @(
        'gemma-gguf', '--model', $modelFull, '--prompt', $Prompt,
        '--context', [string]$case.Context,
        '--gpu-layers', [string]$case.GpuLayers,
        '--backend', $case.Backend,
        '--threads', [string]$case.Threads,
        '--max-new-tokens', [string]$MaxNewTokens,
        '--temperature', '0', '--top-p', '1', '--top-k', '1', '--no-stream'
    )) { [void]$startInfo.ArgumentList.Add($argument) }

    $startedUtc = (Get-Date).ToUniversalTime()
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    $timedOut = $false
    $exitCode = -1
    $startedProcess = $false
    $stdout = ''
    $stderr = ''
    try {
        [void]$process.Start()
        $startedProcess = $true
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            $timedOut = $true
            $process.Kill($true)
        }
        $process.WaitForExit()
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        if (-not $timedOut) { $exitCode = $process.ExitCode }
    }
    catch {
        $stderr += "`nHarness error: $($_.Exception.Message)"
        if ($startedProcess -and -not $process.HasExited) {
            try { $process.Kill($true) } catch { }
        }
    }
    finally {
        $watch.Stop()
        $process.Dispose()
        $stdout | Set-Content -LiteralPath $stdoutPath -Encoding utf8NoBOM
        $stderr | Set-Content -LiteralPath $stderrPath -Encoding utf8NoBOM
    }

    $matches = [regex]::Matches($stderr, $timingPattern)
    $timing = if ($matches.Count -gt 0) { $matches[$matches.Count - 1] } else { $null }
    $generated = if ($timing) { [int]$timing.Groups[5].Value } else { 0 }
    $status = if ($timedOut) { 'timeout' }
        elseif ($exitCode -ne 0) { 'error' }
        elseif (-not $timing) { 'missing_timing' }
        elseif ($generated -lt $MinGeneratedTokens) { 'short_output' }
        elseif ($timing.Groups[4].Value -eq 'n/a') { 'missing_decode_rate' }
        else { 'ok' }
    $splitMatches = [regex]::Matches($stderr, 'sched_reserve: graph splits = (\d+)')
    $graphSplits = if ($splitMatches.Count -gt 0) {
        [int]$splitMatches[$splitMatches.Count - 1].Groups[1].Value
    } else { $null }
    $record = [pscustomobject]@{
        CaseId = $caseId
        Status = $status
        Repetition = $case.Repetition
        Context = $case.Context
        GpuLayers = $case.GpuLayers
        Threads = $case.Threads
        Backend = $case.Backend
        StartedUtc = $startedUtc.ToString('o')
        WallSeconds = [Math]::Round($watch.Elapsed.TotalSeconds, 3)
        ExitCode = $exitCode
        PromptTokens = if ($timing) { [int]$timing.Groups[2].Value } else { $null }
        GeneratedTokens = $generated
        LoadSeconds = if ($timing) { [double]::Parse($timing.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture) } else { $null }
        FirstTokenMilliseconds = if ($timing) { [double]::Parse($timing.Groups[3].Value, [Globalization.CultureInfo]::InvariantCulture) } else { $null }
        DecodeTokensPerSecond = if ($timing -and $timing.Groups[4].Value -ne 'n/a') {
            [double]::Parse($timing.Groups[4].Value, [Globalization.CultureInfo]::InvariantCulture)
        } else { $null }
        GenerationSeconds = if ($timing) { [double]::Parse($timing.Groups[6].Value, [Globalization.CultureInfo]::InvariantCulture) } else { $null }
        GraphSplits = $graphSplits
        StdoutLog = [IO.Path]::GetFileName($stdoutPath)
        StderrLog = [IO.Path]::GetFileName($stderrPath)
    }
    $records.Add($record)
    $records | Export-Csv -LiteralPath (Join-Path $runDirectory 'runs.csv') -NoTypeInformation -Encoding utf8NoBOM
    Write-Host "  $status; generated=$generated; decode=$($record.DecodeTokensPerSecond) tok/s; first=$($record.FirstTokenMilliseconds) ms"
}

function Get-Median([double[]] $Values) {
    if ($Values.Count -eq 0) { return $null }
    $sorted = @($Values | Sort-Object)
    $mid = [int][Math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2 -eq 1) { return [double]$sorted[$mid] }
    return ([double]$sorted[$mid - 1] + [double]$sorted[$mid]) / 2
}

$summary = @(
    $records | Group-Object Context, GpuLayers, Threads | ForEach-Object {
        $group = @($_.Group)
        $ok = @($group | Where-Object { $_.Status -eq 'ok' })
        $rates = [double[]]@($ok | ForEach-Object { $_.DecodeTokensPerSecond })
        $firsts = [double[]]@($ok | ForEach-Object { $_.FirstTokenMilliseconds })
        $mean = if ($rates.Count) { ($rates | Measure-Object -Average).Average } else { $null }
        $stdev = if ($rates.Count -gt 1) {
            $sumSquares = 0.0
            foreach ($rate in $rates) { $sumSquares += [Math]::Pow($rate - $mean, 2) }
            [Math]::Sqrt($sumSquares / ($rates.Count - 1))
        } else { $null }
        [pscustomobject]@{
            Context = $group[0].Context
            GpuLayers = $group[0].GpuLayers
            Threads = $group[0].Threads
            Samples = $group.Count
            ValidSamples = $ok.Count
            MedianDecodeTokensPerSecond = Get-Median $rates
            MeanDecodeTokensPerSecond = $mean
            SampleStdevDecodeTokensPerSecond = $stdev
            MedianFirstTokenMilliseconds = Get-Median $firsts
            PromptTokens = if ($ok.Count) { $ok[0].PromptTokens } else { $null }
            GraphSplits = if ($ok.Count) { $ok[0].GraphSplits } else { $null }
        }
    }
)
$summary | Sort-Object Context, GpuLayers, Threads |
    Export-Csv -LiteralPath (Join-Path $runDirectory 'summary.csv') -NoTypeInformation -Encoding utf8NoBOM
$summary | Sort-Object Context, GpuLayers, Threads |
    Format-Table Context, GpuLayers, Threads, ValidSamples, MedianDecodeTokensPerSecond, MedianFirstTokenMilliseconds -AutoSize
Write-Host "Results: $runDirectory"
