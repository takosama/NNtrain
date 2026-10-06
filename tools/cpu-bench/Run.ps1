param([string]$OutputDirectory, [switch]$IncludeProfile)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$runId = [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0,8)
if (!$OutputDirectory) { $OutputDirectory = Join-Path $repoRoot "benchmark-results/cpu-$runId" }
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Choose a new output directory; existing results are never overwritten.' }
$scratch = Join-Path ([IO.Path]::GetTempPath()) "nntrain-cpu-$runId"
New-Item -ItemType Directory -Path $scratch,$OutputDirectory | Out-Null
$baseline = Join-Path $scratch 'baseline'
New-Item -ItemType Directory -Path $baseline | Out-Null

# Snapshot the original changed sources. Other Core source paths are read only.
# git ls-files excludes the added CPU helper and all untracked user files.
$sources = & git -C $repoRoot ls-files 'NNtrain.Core/*.cs'
if ($LASTEXITCODE) { throw 'git ls-files failed' }
$items = foreach ($source in $sources) {
    $path = Join-Path $repoRoot $source
    if ($source -in @('NNtrain.Core/Tensors/Tensor.Qwen.cs','NNtrain.Core/Tensors/TensorStorage.cs')) {
        $path = Join-Path $baseline ([IO.Path]::GetFileName($source))
        & git -C $repoRoot show "HEAD:$source" | Set-Content -LiteralPath $path -Encoding utf8
        if ($LASTEXITCODE) { throw "git show failed: $source" }
    }
    '<Compile Include="' + [Security.SecurityElement]::Escape($path) + '" />'
}
$refs = foreach ($name in @('NNtrain.Runtime','NNtrain.Cuda','NNtrain.Arc')) {
    '<ProjectReference Include="' + [Security.SecurityElement]::Escape((Join-Path $repoRoot "$name/$name.csproj")) + '" />'
}
$core = '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable><AllowUnsafeBlocks>true</AllowUnsafeBlocks><EnableDefaultCompileItems>false</EnableDefaultCompileItems><AssemblyName>NNtrain.Core</AssemblyName></PropertyGroup><ItemGroup>' + ($items -join "`n") + '</ItemGroup><ItemGroup>' + ($refs -join "`n") + '</ItemGroup></Project>'
$core | Set-Content (Join-Path $baseline 'OriginalCore.csproj') -Encoding utf8
$harness = '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable><EnableDefaultCompileItems>false</EnableDefaultCompileItems></PropertyGroup><ItemGroup><ProjectReference Include="OriginalCore.csproj"/><Compile Include="' + [Security.SecurityElement]::Escape((Join-Path $PSScriptRoot 'Program.cs')) + '"/></ItemGroup></Project>'
$harness | Set-Content (Join-Path $baseline 'OriginalBench.csproj') -Encoding utf8
$previousTiering = $env:DOTNET_TieredCompilation
try {
    $env:DOTNET_TieredCompilation = '0'
    foreach ($variant in @('original','current')) {
        $project = if ($variant -eq 'original') { Join-Path $baseline 'OriginalBench.csproj' } else { Join-Path $PSScriptRoot 'CpuBench.csproj' }
        $artifacts = Join-Path $scratch "$variant-out"
        & dotnet build $project -c Release --artifacts-path $artifacts -m:1 -p:UseSharedCompilation=false "-p:RestoreConfigFile=$(Join-Path $PSScriptRoot 'NuGet.Config')" -v:q
        if ($LASTEXITCODE) { throw "$variant build failed" }
        $name = if ($variant -eq 'original') { 'OriginalBench' } else { 'CpuBench' }
        $dll = Join-Path $artifacts "bin/$name/release/$name.dll"
        & dotnet $dll --verify > (Join-Path $OutputDirectory "$variant-verify.txt")
        if ($LASTEXITCODE) { throw "$variant verification failed" }
        & dotnet $dll --matrix > (Join-Path $OutputDirectory "$variant-matrix.json.txt")
        if ($LASTEXITCODE) { throw "$variant matrix benchmark failed" }
        if ($IncludeProfile) {
            & dotnet $dll --profile > (Join-Path $OutputDirectory "$variant-profile.json.txt")
            if ($LASTEXITCODE) { throw "$variant profile failed" }
        }
    }
} finally { $env:DOTNET_TieredCompilation = $previousTiering }
Write-Output "CPU results: $OutputDirectory; isolated builds: $scratch"
