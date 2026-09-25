param(
    [string]$SourceDirectory = $PSScriptRoot,
    [string]$OutputPath,
    [switch]$Markdown
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $SourceDirectory 'arc-generation-final-summary-20260924.json'
}
$OutputPath = [IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $OutputPath) { throw 'Use a new summary output path.' }

function Assert-Condition([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

function Get-ComparableValue($Value) {
    if ($Value -is [pscustomobject]) {
        $ordered = [ordered]@{}
        foreach ($property in @($Value.PSObject.Properties | Sort-Object Name)) {
            $ordered[$property.Name] = $property.Value
        }
        return ConvertTo-Json -InputObject $ordered -Depth 30 -Compress
    }
    return ConvertTo-Json -InputObject $Value -Depth 30 -Compress
}

function Get-Median([double[]]$Values) {
    if ($Values.Count -eq 0) { return $null }
    $ordered = @($Values | Sort-Object)
    $middle = [int][Math]::Floor($ordered.Count / 2)
    if ($ordered.Count % 2) { return [double]$ordered[$middle] }
    return ([double]$ordered[$middle - 1] + [double]$ordered[$middle]) / 2
}

function Get-Sum([double[]]$Values) {
    if ($Values.Count -eq 0) { return 0.0 }
    return [double]($Values | Measure-Object -Sum).Sum
}

function Get-SequenceKey([object[]]$Tokens) { return $Tokens -join ',' }

function Get-LaneStatistics([object[]]$Samples) {
    $indices = @($Samples | ForEach-Object { $_.LaneSnapshot.PSObject.Properties.Name } | Sort-Object -Unique)
    foreach ($index in $indices) {
        $present = @($Samples | Where-Object { $null -ne $_.LaneSnapshot.PSObject.Properties[$index] })
        $snapshots = @($present | ForEach-Object { $_.LaneSnapshot.PSObject.Properties[$index].Value })
        $deltas = @($present | ForEach-Object { $_.LaneDelta.PSObject.Properties[$index].Value })
        [pscustomobject]@{
            DeviceIndex = [int]$index
            Samples = $present.Count
            SessionCumulativePeakBackendMiB = ($snapshots | Measure-Object PeakAllocatedBytes -Maximum).Maximum / 1MB
            MeanRetainedBackendMiB = ($snapshots | Measure-Object AllocatedBytes -Average).Average / 1MB
            MeanAllocatedBytesDelta = ($deltas | Measure-Object AllocatedBytes -Average).Average
            MaximumAbsoluteAllocatedBytesDelta = (@($deltas | ForEach-Object { [Math]::Abs($_.AllocatedBytes) }) | Measure-Object -Maximum).Maximum
            MeanKernelLaunchCount = ($deltas | Measure-Object KernelLaunchCount -Average).Average
            MeanKernelEventMs = ($deltas | Measure-Object KernelMilliseconds -Average).Average
            MeanHostToDeviceBytes = ($deltas | Measure-Object H2DBytes -Average).Average
            MeanDeviceToHostBytes = ($deltas | Measure-Object D2HBytes -Average).Average
        }
    }
}

function Get-RunStatistics([object[]]$Samples, [int]$NewTokens, [int]$PromptLength) {
    if ($Samples.Count -eq 0) { return $null }
    $durations = @($Samples | ForEach-Object { [double]$_.TotalMs })
    $totalMs = Get-Sum $durations
    $medianMs = Get-Median $durations
    $firstTokens = @($Samples | Where-Object { $null -ne $_.FirstTokenMs } | ForEach-Object { [double]$_.FirstTokenMs })
    $decodeTimes = @($Samples | Where-Object { $null -ne $_.DecodeMs } | ForEach-Object { [double]$_.DecodeMs })
    $hasDecode = $NewTokens -gt 1 -and $decodeTimes.Count -eq $Samples.Count -and (Get-Sum $decodeTimes) -gt 0
    $sequenceKeys = @($Samples | ForEach-Object { Get-SequenceKey $_.GeneratedTokenIds })
    [pscustomobject]@{
        Samples = $Samples.Count
        GeneratedTokens = [long]$NewTokens * $Samples.Count
        TotalMs = $totalMs
        MeanMs = $totalMs / $Samples.Count
        MedianMs = $medianMs
        MinimumMs = ($durations | Measure-Object -Minimum).Minimum
        MaximumMs = ($durations | Measure-Object -Maximum).Maximum
        AggregateTokensPerSecond = $NewTokens * $Samples.Count * 1000.0 / $totalMs
        MedianBasedTokensPerSecond = $NewTokens * 1000.0 / $medianMs
        FirstTokenMedianMs = if ($firstTokens.Count -eq $Samples.Count) { Get-Median $firstTokens } else { $null }
        FirstTokenMeanMs = if ($firstTokens.Count -eq $Samples.Count) { (Get-Sum $firstTokens) / $Samples.Count } else { $null }
        DecodeMedianMs = if ($hasDecode) { Get-Median $decodeTimes } else { $null }
        AggregateDecodeTokensPerSecond = if ($hasDecode) { ($NewTokens - 1) * $Samples.Count * 1000.0 / (Get-Sum $decodeTimes) } else { $null }
        DeterministicAcrossRuns = @($sequenceKeys | Sort-Object -Unique).Count -eq 1
        DistinctGeneratedTokenCounts = @($Samples | ForEach-Object {
            @($_.GeneratedTokenIds | Select-Object -Skip $PromptLength | Sort-Object -Unique).Count
        })
        Lanes = @(Get-LaneStatistics $Samples)
    }
}

function Compare-GeneratedOutput($Left, $Right, [string]$LeftName, [string]$RightName, [int]$PromptLength) {
    $leftTokens = @($Left.GeneratedTokenIds)
    $rightTokens = @($Right.GeneratedTokenIds)
    $differenceCount = 0
    $generatedDifferenceCount = 0
    $firstDifference = $null
    $length = [Math]::Max($leftTokens.Count, $rightTokens.Count)
    for ($index = 0; $index -lt $length; ++$index) {
        $leftToken = if ($index -lt $leftTokens.Count) { $leftTokens[$index] } else { $null }
        $rightToken = if ($index -lt $rightTokens.Count) { $rightTokens[$index] } else { $null }
        if ($leftToken -ne $rightToken) {
            ++$differenceCount
            if ($index -ge $PromptLength) { ++$generatedDifferenceCount }
            if ($null -eq $firstDifference) {
                $firstDifference = [pscustomobject]@{
                    FullSequenceIndexZeroBased = $index
                    NewTokenIndexOneBased = if ($index -ge $PromptLength) { $index - $PromptLength + 1 } else { $null }
                    LeftToken = $leftToken
                    RightToken = $rightToken
                }
            }
        }
    }
    [pscustomobject]@{
        Left = $LeftName
        Right = $RightName
        LeftFullTokenCount = $leftTokens.Count
        RightFullTokenCount = $rightTokens.Count
        DifferentFullSequencePositions = $differenceCount
        DifferentGeneratedTokenPositions = $generatedDifferenceCount
        FirstDifference = $firstDifference
        LeftDistinctGeneratedTokens = @($leftTokens | Select-Object -Skip $PromptLength | Sort-Object -Unique).Count
        RightDistinctGeneratedTokens = @($rightTokens | Select-Object -Skip $PromptLength | Sort-Object -Unique).Count
    }
}

function Get-ProfileSummary($Document) {
    $samples = @($Document.Result.Samples)
    foreach ($device in $Document.Result.Devices) {
        $index = [string]$device.DeviceIndex
        $entries = @($samples | ForEach-Object { $_.Profile.PSObject.Properties[$index].Value } | Where-Object { $null -ne $_ })
        $kernels = @($entries | Where-Object { $_.Kind -eq 'gpu-kernel' })
        $groups = @($kernels | Group-Object Detail | ForEach-Object {
            [pscustomobject]@{
                Detail = $_.Name
                MeanGpuEventMs = ($_.Group | Measure-Object Milliseconds -Sum).Sum / $samples.Count
                MeanCalls = ($_.Group | Measure-Object Count -Sum).Sum / $samples.Count
            }
        } | Sort-Object MeanGpuEventMs -Descending)
        [pscustomobject]@{
            DeviceIndex = $device.DeviceIndex
            DeviceName = $device.Name
            MeanGpuKernelEventMs = ($kernels | Measure-Object Milliseconds -Sum).Sum / $samples.Count
            MeanKernelCalls = ($kernels | Measure-Object Count -Sum).Sum / $samples.Count
            KernelGroups = $groups
        }
    }
}

$roles = @('a1', 'b', 'a2', 'tp', 'auto', 'profile')
$documents = [ordered]@{}
$sources = [ordered]@{}
$optimizationFlags = @('InferenceKvCache', 'InferenceGemv', 'InferencePackedEmbedding', 'InferenceSmallRowNorm', 'InferenceFusedGemv')
foreach ($role in $roles) {
    $path = [IO.Path]::GetFullPath((Join-Path $SourceDirectory "arc-generation-final-$role-20260924.json"))
    Assert-Condition (Test-Path -LiteralPath $path -PathType Leaf) "Missing final artifact: $path"
    $data = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    $documents[$role] = $data
    $sources[$role] = [pscustomobject]@{ Path = $path; Sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash }
    $measured = @($data.Result.Samples)
    $warmup = @($data.Result.WarmupSamples)
    Assert-Condition ($data.SchemaVersion -eq 1) "$role has an unsupported schema."
    Assert-Condition ($measured.Count -gt 0 -and $measured.Count -eq $data.MeasuredRuns) "$role measured-run count mismatch."
    Assert-Condition ($warmup.Count -eq $data.WarmupRuns) "$role warmup-run count mismatch."
    Assert-Condition ($data.MaxNewTokens -eq 200 -and $data.Sampling.TopK -eq 40 -and $data.Sampling.Temperature -eq 1 -and $null -eq $data.Sampling.StopTokenId) "$role is not the planned 200-token, topK40, temperature1, no-EOS-stop workload."
    foreach ($set in @(@{ Name = 'measured'; Samples = $measured }, @{ Name = 'warmup'; Samples = $warmup })) {
        for ($run = 0; $run -lt $set.Samples.Count; ++$run) {
            $sample = $set.Samples[$run]
            Assert-Condition ($sample.Run -eq $run + 1) "$role $($set.Name) run numbering mismatch."
            Assert-Condition ([double]::IsFinite($sample.TotalMs) -and $sample.TotalMs -gt 0) "$role contains invalid elapsed time."
            Assert-Condition ($sample.GeneratedTokenIds.Count -eq $data.PromptTokenIds.Count + $data.MaxNewTokens) "$role generated-token count mismatch."
            $prefix = @($sample.GeneratedTokenIds | Select-Object -First $data.PromptTokenIds.Count)
            Assert-Condition ((Get-SequenceKey $prefix) -ceq (Get-SequenceKey $data.PromptTokenIds)) "$role changed the prompt IDs."
        }
    }
}

$reference = $documents['a1']
$sameWorkFields = @('ConfigurationSha256', 'BinarySha256', 'Seed', 'Shape', 'Precision', 'PromptTokenIds', 'Sampling', 'MaxNewTokens', 'ModelSource', 'CheckpointSha256')
foreach ($role in @('b', 'a2', 'tp', 'auto', 'profile')) {
    $data = $documents[$role]
    foreach ($field in $sameWorkFields) {
        Assert-Condition ((Get-ComparableValue $reference.PSObject.Properties[$field].Value) -ceq
            (Get-ComparableValue $data.PSObject.Properties[$field].Value)) "$role does not match a1: $field"
    }
    Assert-Condition ($reference.Result.ParameterCount -eq $data.Result.ParameterCount) "$role parameter-count mismatch."
    foreach ($property in $reference.ArcOptions.PSObject.Properties) {
        if ($property.Name -in $optimizationFlags -or $property.Name -eq 'DetailedProfiling') { continue }
        Assert-Condition ((Get-ComparableValue $property.Value) -ceq
            (Get-ComparableValue $data.ArcOptions.PSObject.Properties[$property.Name].Value)) "$role changed unrelated Arc option $($property.Name)."
    }
}
foreach ($role in @('a1', 'a2')) {
    foreach ($flag in $optimizationFlags) {
        Assert-Condition ($documents[$role].ArcOptions.PSObject.Properties[$flag].Value -eq $false) "$role baseline optimization $flag must be disabled."
    }
    Assert-Condition ($documents[$role].WarmupRuns -eq $documents['b'].WarmupRuns -and
        $documents[$role].MeasuredRuns -eq $documents['b'].MeasuredRuns) "$role and b must use the same warmup/measured run counts."
}
foreach ($role in @('tp', 'auto', 'profile')) {
    foreach ($flag in $optimizationFlags) {
        Assert-Condition ($documents[$role].ArcOptions.PSObject.Properties[$flag].Value -eq
            $documents['b'].ArcOptions.PSObject.Properties[$flag].Value) "$role differs from selected option $flag."
    }
}
foreach ($role in @('a1', 'b', 'a2', 'profile')) {
    Assert-Condition ($documents[$role].Result.SelectedMode -eq 'single') "$role must measure single-GPU generation."
}
Assert-Condition ($documents['tp'].RequestedMode -ieq 'tensorParallel' -and $documents['tp'].Result.SelectedMode -eq 'tensorParallel') 'tp did not use forced tensor parallelism.'
Assert-Condition ($documents['auto'].RequestedMode -ieq 'auto') 'auto did not request automatic selection.'
Assert-Condition ($documents['profile'].ArcOptions.DetailedProfiling -eq $true) 'profile must enable detailed profiling.'
foreach ($role in @('a1', 'b', 'a2', 'tp', 'auto')) {
    Assert-Condition ($documents[$role].ArcOptions.DetailedProfiling -eq $false) "$role timing includes detailed profiling."
}

$newTokens = [int]$reference.MaxNewTokens
$promptLength = [int]$reference.PromptTokenIds.Count
$statistics = [ordered]@{}
$artifacts = [ordered]@{}
foreach ($role in $roles) {
    $data = $documents[$role]
    $statistics[$role] = Get-RunStatistics @($data.Result.Samples) $newTokens $promptLength
    $artifacts[$role] = [pscustomobject]@{
        Source = $sources[$role]
        RequestedMode = $data.RequestedMode
        SelectedMode = $data.Result.SelectedMode
        Devices = $data.Result.Devices
        ArcOptions = $data.ArcOptions
        StreamingCallback = $data.Result.StreamingCallback
        ReportedDeterministicAcrossMeasuredRuns = $data.Result.DeterministicAcrossMeasuredRuns
        Measured = $statistics[$role]
        WarmupExcludedFromComparison = Get-RunStatistics @($data.Result.WarmupSamples) $newTokens $promptLength
        WarmupMatchesFirstMeasured = @($data.Result.WarmupSamples | Where-Object {
            (Get-SequenceKey $_.GeneratedTokenIds) -cne (Get-SequenceKey $data.Result.Samples[0].GeneratedTokenIds)
        }).Count -eq 0
    }
}
$pooledSamples = @($documents['a1'].Result.Samples) + @($documents['a2'].Result.Samples)
$baseline = Get-RunStatistics $pooledSamples $newTokens $promptLength
$selected = $statistics['b']
$comparisons = @()
foreach ($pair in @(@('a1', 'b'), @('a2', 'b'), @('a1', 'a2'), @('b', 'tp'), @('b', 'auto'))) {
    foreach ($left in $documents[$pair[0]].Result.Samples) {
        foreach ($right in $documents[$pair[1]].Result.Samples) {
            $comparisons += Compare-GeneratedOutput $left $right "$($pair[0])/run-$($left.Run)" "$($pair[1])/run-$($right.Run)" $promptLength
        }
    }
}
$primaryOutputComparison = Compare-GeneratedOutput $documents['a1'].Result.Samples[0] $documents['b'].Result.Samples[0] 'a1/run-1' 'b/run-1' $promptLength
$summary = [pscustomobject]@{
    SchemaVersion = 1
    CreatedUtc = [DateTimeOffset]::UtcNow.ToString('O')
    ConfigurationSha256 = $reference.ConfigurationSha256
    BinarySha256 = $reference.BinarySha256
    ModelSource = $reference.ModelSource
    CheckpointSha256 = $reference.CheckpointSha256
    Seed = $reference.Seed
    Shape = $reference.Shape
    Precision = $reference.Precision
    Sampling = $reference.Sampling
    PromptTokenIds = $reference.PromptTokenIds
    MaxNewTokens = $newTokens
    Verification = [pscustomobject]@{
        SameConfigurationPromptSamplingShapePrecisionSeedAndBinaries = $true
        BaselinesDisableAllFiveGenerationOptions = $true
        SelectedOptionsMatchTpAutoAndProfile = $true
        WarmupsExcludedFromTimingComparison = $true
        GenerationLengthAndPromptValidatedForEveryRun = $true
    }
    Artifacts = $artifacts
    PooledBaselineA1A2 = $baseline
    SelectedSingle = $selected
    Improvement = [pscustomobject]@{
        AggregateThroughputFactor = $selected.AggregateTokensPerSecond / $baseline.AggregateTokensPerSecond
        AggregateThroughputPercent = ($selected.AggregateTokensPerSecond / $baseline.AggregateTokensPerSecond - 1) * 100
        MeanGenerationTimeReductionPercent = (1 - $selected.MeanMs / $baseline.MeanMs) * 100
        MedianThroughputFactor = $baseline.MedianMs / $selected.MedianMs
        MedianThroughputPercent = ($baseline.MedianMs / $selected.MedianMs - 1) * 100
        BaselineA2VersusA1AggregateThroughputDriftPercent = ($statistics['a2'].AggregateTokensPerSecond / $statistics['a1'].AggregateTokensPerSecond - 1) * 100
        ForcedTpVersusSelectedSingleAggregateThroughputFactor = $statistics['tp'].AggregateTokensPerSecond / $selected.AggregateTokensPerSecond
    }
    Auto = [pscustomobject]@{ SelectedMode = $documents['auto'].Result.SelectedMode; Selection = $documents['auto'].Result.RouteSelection }
    PrimaryOutputComparison = $primaryOutputComparison
    OutputComparisonsAllRunPairs = $comparisons
    DetailedProfileLanes = @(Get-ProfileSummary $documents['profile'])
    Notes = 'The baseline pools only measured a1+a2 runs; selected uses only measured b runs. Main throughput is total generated tokens divided by total measured wall time. Median-based throughput is reported separately. Warmups and the detailed profile are not included in the A/B/A speed calculation. Prefill is measured at the first streaming callback; decode excludes first token and final cleanup. Backend peaks are cumulative native allocation peaks for each process/session, including warmup and auto calibration, not driver VRAM. Output differences are reported without requiring equality: repeated deterministic sampling within one route does not imply identical sampled continuations between numerically different routes. Distinct token counts exclude the prompt. Synthetic weights do not establish trained-checkpoint quality. Kernel event durations overlap wall/host times and must not be added to them.'
}

[void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($OutputPath))
$stream = [IO.File]::Open($OutputPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
$writer = [IO.StreamWriter]::new($stream, [Text.UTF8Encoding]::new($false))
try { $writer.Write(($summary | ConvertTo-Json -Depth 40)) } finally { $writer.Dispose() }
Write-Output "Summary saved: $OutputPath"
if ($Markdown) {
    Write-Output '| Route | Runs | Mean ms | Median ms | Aggregate tok/s | First token median ms | Decode tok/s |'
    Write-Output '|---|---:|---:|---:|---:|---:|---:|'
    foreach ($row in @(@{ Name = 'Baseline a1+a2'; Stats = $baseline }, @{ Name = 'Selected single'; Stats = $selected },
        @{ Name = 'Forced TP'; Stats = $statistics['tp'] }, @{ Name = 'Auto'; Stats = $statistics['auto'] })) {
        $stats = $row.Stats
        Write-Output ('| {0} | {1} | {2:F2} | {3:F2} | {4:F2} | {5:F2} | {6:F2} |' -f
            $row.Name, $stats.Samples, $stats.MeanMs, $stats.MedianMs, $stats.AggregateTokensPerSecond,
            $stats.FirstTokenMedianMs, $stats.AggregateDecodeTokensPerSecond)
    }
    Write-Output ('A/B first-run continuation differences: {0}/{1}; first difference: {2}; distinct generated tokens: {3} / {4}.' -f
        $primaryOutputComparison.DifferentGeneratedTokenPositions, $newTokens,
        (ConvertTo-Json -InputObject $primaryOutputComparison.FirstDifference -Compress),
        $primaryOutputComparison.LeftDistinctGeneratedTokens, $primaryOutputComparison.RightDistinctGeneratedTokens)
}
