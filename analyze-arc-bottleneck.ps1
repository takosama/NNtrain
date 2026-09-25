param(
    [Parameter(Mandatory = $true)][string]$On,
    [Parameter(Mandatory = $true)][string]$Off,
    [Parameter(Mandatory = $true)][string]$OnRepeat
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Read-Probe([string]$path) {
    $resolved = (Resolve-Path -LiteralPath $path).Path
    $probe = Get-Content -LiteralPath $resolved -Raw | ConvertFrom-Json
    if (@($probe.Results).Count -ne 1 -or @($probe.Results[0].Samples).Count -lt 2) {
        throw "Expected one device result and at least two measured updates: $resolved"
    }
    foreach ($sample in $probe.Results[0].Samples) {
        if (!$sample.FiniteGradient) { throw "Non-finite gradient at step $($sample.Step): $resolved" }
        $timeline = $sample.Timeline
        if ($null -eq $timeline) { throw "Missing exclusive timeline: $resolved" }
        $partition = $timeline.Partition
        if ($timeline.MissingEvents -ne 0 -or $timeline.OpaqueAllocationCopies -ne 0 -or
            [Math]::Abs($partition.CoverageFraction - 1.0) -gt 1e-9 -or
            $partition.GpuOverlapMs -gt 0.001 -or $partition.UnattributedHostMs -gt 0.001) {
            throw "Timeline quality gate failed for step $($sample.Step): $resolved"
        }
    }
    return $probe
}

function Mean-Category($probe, [string]$name) {
    $values = foreach ($sample in $probe.Results[0].Samples) {
        $matches = @($sample.Timeline.Partition.WallCategories | Where-Object Name -EQ $name)
        if ($matches.Count -eq 0) { 0.0 } else { [double]$matches[0].Milliseconds }
    }
    return [double](($values | Measure-Object -Average).Average)
}

function Mean-CategoryPrefix($probe, [string]$prefix) {
    $values = foreach ($sample in $probe.Results[0].Samples) {
        [double](($sample.Timeline.Partition.WallCategories |
            Where-Object { $_.Name.StartsWith($prefix, [StringComparison]::Ordinal) } |
            Measure-Object Milliseconds -Sum).Sum)
    }
    return [double](($values | Measure-Object -Average).Average)
}

function Mean-Operation($probe, [string]$name) {
    $values = foreach ($sample in $probe.Results[0].Samples) {
        $matches = @($sample.Timeline.Partition.WallOperations | Where-Object Name -EQ $name)
        if ($matches.Count -eq 0) { 0.0 } else { [double]$matches[0].Milliseconds }
    }
    return [double](($values | Measure-Object -Average).Average)
}

$a = Read-Probe $On
$b = Read-Probe $Off
$a2 = Read-Probe $OnRepeat

foreach ($candidate in @($b, $a2)) {
    if ($candidate.ConfigurationSha256 -ne $a.ConfigurationSha256 -or
        $candidate.Seed -ne $a.Seed -or
        $candidate.WarmupSteps -ne $a.WarmupSteps -or
        $candidate.MeasuredSteps -ne $a.MeasuredSteps -or
        $candidate.Results[0].DeviceName -ne $a.Results[0].DeviceName -or
        $candidate.Results[0].DriverVersion -ne $a.Results[0].DriverVersion -or
        ($candidate.Shape | ConvertTo-Json -Compress) -ne ($a.Shape | ConvertTo-Json -Compress) -or
        ($candidate.Precision | ConvertTo-Json -Compress) -ne ($a.Precision | ConvertTo-Json -Compress) -or
        ($candidate.Optimizer | ConvertTo-Json -Compress) -ne ($a.Optimizer | ConvertTo-Json -Compress)) {
        throw 'Configuration, seed, shape, precision, or measurement-count mismatch.'
    }
    foreach ($part in $a.BinarySha256.PSObject.Properties.Name) {
        if ($candidate.BinarySha256.$part -ne $a.BinarySha256.$part) {
            throw "Binary mismatch: $part"
        }
    }
    for ($i = 0; $i -lt $a.Results[0].Samples.Count; $i++) {
        $reference = $a.Results[0].Samples[$i]
        $current = $candidate.Results[0].Samples[$i]
        if ($reference.Step -ne $current.Step -or
            $reference.LaneDelta.H2DBytes -ne $current.LaneDelta.H2DBytes -or
            $reference.LaneDelta.D2HBytes -ne $current.LaneDelta.D2HBytes) {
            throw "Step or transfer-volume mismatch at measured sample $i."
        }
    }
}

$differences = @($a.ArcOptions.PSObject.Properties.Name | Where-Object { $a.ArcOptions.$_ -ne $b.ArcOptions.$_ })
if ($differences.Count -ne 1 -or $differences[0] -ne 'BlockIoAttention' -or
    !$a.ArcOptions.BlockIoAttention -or $b.ArcOptions.BlockIoAttention -or
    ($a2.ArcOptions | ConvertTo-Json -Compress) -ne ($a.ArcOptions | ConvertTo-Json -Compress)) {
    throw 'Expected only BlockIoAttention on/off/on to change.'
}

$rows = @(
    @{ Label = 'ON'; Probe = $a; Kernel = 'backward/attention_dkv_block_slm_causal' },
    @{ Label = 'OFF'; Probe = $b; Kernel = 'backward/attention_dkv_d32_k32_q32_causal' },
    @{ Label = 'ON-repeat'; Probe = $a2; Kernel = 'backward/attention_dkv_block_slm_causal' }
) | ForEach-Object {
    $probe = $_.Probe
    [PSCustomObject]@{
        Mode = $_.Label
        StepP50Ms = [Math]::Round([double]$probe.Results[0].StepP50Ms, 2)
        GpuBusyMs = [Math]::Round((Mean-CategoryPrefix $probe 'GPU/'), 2)
        GpuIdleMs = [Math]::Round((Mean-CategoryPrefix $probe 'GPU-idle/'), 2)
        AttentionMs = [Math]::Round((Mean-Category $probe 'GPU/attention'), 2)
        DkvMs = [Math]::Round((Mean-Operation $probe $_.Kernel), 2)
        GemmMs = [Math]::Round((Mean-Category $probe 'GPU/GEMM'), 2)
        PackMs = [Math]::Round((Mean-Category $probe 'GPU/pack-decode-publication'), 2)
    }
}

Write-Output "Validated config SHA-256: $($a.ConfigurationSha256)"
Write-Output 'Binary SHA-256 values match; BlockIoAttention is the sole option difference; all timeline quality gates pass.'
$rows | Format-Table -AutoSize
$onP50 = ($rows[0].StepP50Ms + $rows[2].StepP50Ms) / 2
$onDkv = ($rows[0].DkvMs + $rows[2].DkvMs) / 2
Write-Output ('OFF minus mean ON: wall {0:N2} ms/update; dK/dV {1:N2} ms/update.' -f
    ($rows[1].StepP50Ms - $onP50), ($rows[1].DkvMs - $onDkv))
