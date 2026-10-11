param([string]$OutputDir = (Join-Path $PSScriptRoot 'rerun'))
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$buildRoot = Join-Path $PSScriptRoot 'staged-build'
if (-not (Test-Path -LiteralPath $buildRoot -PathType Container)) {
    throw "Run prepare_staged_build.py first: $buildRoot"
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
if ((Test-Path -LiteralPath $OutputDir) -and
    (Get-ChildItem -LiteralPath $OutputDir -Force | Select-Object -First 1)) {
    throw "Choose a fresh output directory: $OutputDir"
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
Set-Location $repoRoot

foreach ($project in @('NNtrain.Gui/NNtrain.Gui.csproj', 'NNtrain.Benchmarks/NNtrain.Benchmarks.csproj',
    'tools/iq2s-packed-decode-bench/Iq2sPackedDecodeBench.csproj')) {
    dotnet build (Join-Path $buildRoot $project) -c Release --nologo -v minimal
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
dotnet test (Join-Path $buildRoot 'NNtrain.Core.Tests/NNtrain.Core.Tests.csproj') -c Release --nologo -v minimal `
    --filter 'FullyQualifiedName~Qwen35Iq2PackedDecodeTests|FullyQualifiedName~Qwen35Iq2PackedPrefillTests|FullyQualifiedName~Qwen35Iq2PackedRoutingTests' `
    --logger 'trx;LogFileName=packed-iq2-release.trx' --results-directory (Join-Path $OutputDir 'tests')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$bench = Join-Path $buildRoot 'NNtrain.Benchmarks/bin/Release/net10.0/NNtrain.Benchmarks.dll'
$model = Join-Path $repoRoot 'models/Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf'
foreach ($mode in @('greedy', 'sampling', 'lora')) {
    foreach ($variant in @('original', 'packed')) {
        $extra = @()
        if ($mode -eq 'sampling') { $extra = @('--temperature','0.6','--top-p','0.95','--top-k','20','--seed','5429') }
        if ($mode -eq 'lora') { $extra = @('--adapter', (Join-Path $repoRoot 'models/lora_rintya_qa142_Qwen3.8-27B-Uncensored-noMTP-IQ2_M_2epoch.gguf')) }
        dotnet $bench --qwen35-generation-probe --model $model `
            --output (Join-Path $OutputDir "final-$mode-$variant.json") `
            --devices 0,1 --tokens 128 --runs 4 --logits 16 --profile off `
            --options (Join-Path $PSScriptRoot "$variant-options.json") @extra
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
dotnet (Join-Path $buildRoot 'tools/iq2s-packed-decode-bench/bin/Release/net10.0/NNtrain.Iq2sPackedDecodeBench.dll') `
    --model $model --output (Join-Path $OutputDir 'micro-release') --repo $buildRoot `
    --samples 9 --iterations 16 --profile 0 --hash-model
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
