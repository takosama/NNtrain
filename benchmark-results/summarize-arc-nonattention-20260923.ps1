param([Parameter(Mandatory=$true)][string]$ProfilePath,
      [Parameter(Mandatory=$true)][string]$OutputPath)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $OutputPath) { throw 'Use a new output path.' }
$data = Get-Content -LiteralPath $ProfilePath -Raw | ConvertFrom-Json
$run = $data.Results[0]
$samples = @($run.Samples)
$lanes = foreach ($index in $run.DeviceIndices) {
    $key = [string]$index
    $profiles = @($samples | ForEach-Object { $_.Profile.$key })
    $categories = @($samples | ForEach-Object { $_.Timelines.$key.Partition.WallCategories })
    $kernels = @($profiles | Where-Object { $_.Kind -eq 'gpu-kernel' })
    $nonAttention = @($kernels | Where-Object { $_.Detail -notmatch '/attention_' })
    $groups = @($nonAttention | Group-Object Detail | ForEach-Object {
        [pscustomobject]@{Detail=$_.Name;
            MeanMs=($_.Group | Measure-Object Milliseconds -Sum).Sum/$samples.Count;
            MeanCalls=($_.Group | Measure-Object Count -Sum).Sum/$samples.Count}
    } | Sort-Object MeanMs -Descending)
    [pscustomobject]@{DeviceIndex=$index;
        MeanWallMs=($samples | Measure-Object TotalMs -Average).Average;
        AttentionKernelMs=($kernels | Where-Object { $_.Detail -match '/attention_' } | Measure-Object Milliseconds -Sum).Sum/$samples.Count;
        NonAttentionKernelMs=($nonAttention | Measure-Object Milliseconds -Sum).Sum/$samples.Count;
        ExclusiveWallCategories=@($categories | Group-Object Name | ForEach-Object {
            [pscustomobject]@{Name=$_.Name; MeanMs=($_.Group | Measure-Object Milliseconds -Sum).Sum/$samples.Count}
        } | Sort-Object MeanMs -Descending);
        NonAttentionKernels=$groups;
        NonKernelProfile=@($profiles | Where-Object { $_.Kind -ne 'gpu-kernel' } | Group-Object Kind,Detail | ForEach-Object {
            [pscustomobject]@{Kind=$_.Group[0].Kind; Detail=$_.Group[0].Detail;
                MeanMs=($_.Group | Measure-Object Milliseconds -Sum).Sum/$samples.Count;
                MeanCalls=($_.Group | Measure-Object Count -Sum).Sum/$samples.Count}
        } | Sort-Object MeanMs -Descending)}
}
[pscustomobject]@{Source=(Resolve-Path -LiteralPath $ProfilePath).Path;
    ConfigurationSha256=$data.ConfigurationSha256; BinarySha256=$data.BinarySha256;
    Shape=$data.Shape; Precision=$data.Precision; Samples=$samples.Count;
    StepP50Ms=$run.StepP50Ms; TokensPerSecond=$run.TokensPerSecond; Lanes=@($lanes)
} | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $OutputPath -Encoding utf8
$lanes[0].NonAttentionKernels | Select-Object -First 30 | Format-Table -AutoSize
