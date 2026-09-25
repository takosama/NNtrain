param([string]$OutputPath = 'benchmark-results/arc-overall-final-summary-20260924.json')
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $OutputPath) { throw 'Use a new output path.' }
$runs = @{}
foreach ($label in @('a1','b','a2','profile')) {
    $runs[$label] = Get-Content -LiteralPath "benchmark-results/arc-overall-final-$label-20260924.json" -Raw | ConvertFrom-Json
}
function Median($Values) {
    $ordered = @($Values | Sort-Object)
    if ($ordered.Count % 2) { return $ordered[[int][Math]::Floor($ordered.Count / 2)] }
    return ($ordered[$ordered.Count / 2 - 1] + $ordered[$ordered.Count / 2]) / 2
}
$referenceHash = $runs.a1.BinarySha256 | ConvertTo-Json -Compress
foreach ($entry in $runs.Values) {
    if (($entry.BinarySha256 | ConvertTo-Json -Compress) -ne $referenceHash) { throw 'Binary mismatch.' }
    if ($entry.ConfigurationSha256 -ne $runs.a1.ConfigurationSha256) { throw 'Configuration mismatch.' }
    foreach ($sample in $entry.Results[0].Samples) {
        if (-not $sample.FiniteGradient) { throw 'Non-finite gradient.' }
        foreach ($laneIndex in @('0','1')) {
            if ($sample.LaneDelta.$laneIndex.AllocatedBytes -ne 0) { throw 'Retained allocation changed.' }
        }
    }
}
$control = @($runs.a1.Results[0].Samples) + @($runs.a2.Results[0].Samples)
$selected = @($runs.b.Results[0].Samples)
$controlMedian = Median $control.TotalMs
$selectedMedian = Median $selected.TotalMs
$comparisons = for ($i = 0; $i -lt $selected.Count; $i++) {
    $a = $runs.a1.Results[0].Samples[$i]
    $b = $selected[$i]
    foreach ($laneIndex in @('0','1')) {
        if ($a.LaneDelta.$laneIndex.H2DBytes -ne $b.LaneDelta.$laneIndex.H2DBytes -or
            $a.LaneDelta.$laneIndex.D2HBytes -ne $b.LaneDelta.$laneIndex.D2HBytes) { throw 'Transfer volume changed.' }
    }
    [pscustomobject]@{Step=$b.Step; ControlLoss=$a.Loss; SelectedLoss=$b.Loss;
        LossDifference=$b.Loss-$a.Loss; ControlGradientNorm=$a.GradientNorm;
        SelectedGradientNorm=$b.GradientNorm;
        GradientNormRelativeDifference=($b.GradientNorm-$a.GradientNorm)/$a.GradientNorm}
}
$profile = $runs.profile.Results[0]
$baselineProfile = Get-Content -LiteralPath 'benchmark-results/arc-overall-baseline-profile-20260924.json' -Raw | ConvertFrom-Json
$lanes = foreach ($laneIndex in @('0','1')) {
    $samples = @($profile.Samples)
    $events = @($samples | ForEach-Object { $_.Profile.$laneIndex } | Where-Object Kind -eq 'gpu-kernel')
    $oldEvents = @($baselineProfile.Results[0].Samples | ForEach-Object { $_.Profile.$laneIndex } | Where-Object Kind -eq 'gpu-kernel')
    $timeline = @($samples | ForEach-Object { $_.Timelines.$laneIndex })
    [pscustomobject]@{DeviceIndex=$laneIndex;
        ControlPeakBackendMiB=($control | ForEach-Object { $_.LaneSnapshot.$laneIndex.PeakAllocatedBytes } | Measure-Object -Maximum).Maximum / 1MB;
        SelectedPeakBackendMiB=($selected | ForEach-Object { $_.LaneSnapshot.$laneIndex.PeakAllocatedBytes } | Measure-Object -Maximum).Maximum / 1MB;
        H2DBytesPerUpdate=$selected[0].LaneDelta.$laneIndex.H2DBytes;
        D2HBytesPerUpdate=$selected[0].LaneDelta.$laneIndex.D2HBytes;
        BaselineAttentionKernelMs=($oldEvents | Where-Object Detail -match '/attention_' | Measure-Object Milliseconds -Sum).Sum/$baselineProfile.Results[0].Samples.Count;
        SelectedAttentionKernelMs=($events | Where-Object Detail -match '/attention_' | Measure-Object Milliseconds -Sum).Sum/$samples.Count;
        BaselineNonAttentionKernelMs=($oldEvents | Where-Object Detail -notmatch '/attention_' | Measure-Object Milliseconds -Sum).Sum/$baselineProfile.Results[0].Samples.Count;
        SelectedNonAttentionKernelMs=($events | Where-Object Detail -notmatch '/attention_' | Measure-Object Milliseconds -Sum).Sum/$samples.Count;
        Timeline=@($timeline | ForEach-Object { [pscustomobject]@{WallMs=$_.WallMs; CoverageFraction=$_.Partition.CoverageFraction;
            DeviceEvents=$_.DeviceEvents; MissingEvents=$_.MissingEvents; OpaqueAllocationCopies=$_.OpaqueAllocationCopies; TracePath=$_.TracePath} });
        WallCategories=@($timeline | ForEach-Object { $_.Partition.WallCategories } | Group-Object Name | ForEach-Object {
            [pscustomobject]@{Name=$_.Name; MeanMs=($_.Group | Measure-Object Milliseconds -Sum).Sum/$samples.Count}
        } | Sort-Object MeanMs -Descending);
        Kernels=@($events | Group-Object Detail | ForEach-Object {
            [pscustomobject]@{Name=$_.Name; MeanMs=($_.Group | Measure-Object Milliseconds -Sum).Sum/$samples.Count;
                Calls=($_.Group | Measure-Object Count -Sum).Sum/$samples.Count}
        } | Sort-Object MeanMs -Descending)}
}
$summary = [pscustomobject]@{
    ConfigurationSha256=$runs.a1.ConfigurationSha256; BinarySha256=$runs.a1.BinarySha256;
    Shape=$runs.a1.Shape; Precision=$runs.a1.Precision;
    Runs=@('a1','b','a2' | ForEach-Object { [pscustomobject]@{Label=$_;
        StepP50Ms=$runs[$_].Results[0].StepP50Ms; TokensPerSecond=$runs[$_].Results[0].TokensPerSecond;
        Samples=$runs[$_].Results[0].Samples.TotalMs} });
    ControlPooledP50Ms=$controlMedian; SelectedP50Ms=$selectedMedian;
    ControlTokensPerSecond=262144000/$controlMedian; SelectedTokensPerSecond=262144000/$selectedMedian;
    ThroughputGainPercent=100*($controlMedian/$selectedMedian-1);
    SavedMillisecondsPerUpdate=$controlMedian-$selectedMedian;
    Comparisons=@($comparisons); AllGradientsFinite=$true;
    LiveAllocationDeltaBytes=0; TransferVolumesEqual=$true;
    ProfileStepP50Ms=$profile.StepP50Ms; Lanes=@($lanes)
}
$summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $OutputPath -Encoding utf8
$summary | Select-Object ControlPooledP50Ms,SelectedP50Ms,ControlTokensPerSecond,SelectedTokensPerSecond,ThroughputGainPercent,SavedMillisecondsPerUpdate | Format-List
$comparisons | Format-Table -AutoSize
$lanes | Select-Object DeviceIndex,ControlPeakBackendMiB,SelectedPeakBackendMiB,BaselineAttentionKernelMs,SelectedAttentionKernelMs,BaselineNonAttentionKernelMs,SelectedNonAttentionKernelMs | Format-Table -AutoSize
