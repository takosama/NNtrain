$ErrorActionPreference = 'Stop'

$cudaRoot = [Environment]::GetEnvironmentVariable('CUDA_PATH', 'Machine')
if (-not $cudaRoot) {
    $cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0'
}
$nvcc = Join-Path $cudaRoot 'bin\nvcc.exe'
if (-not (Test-Path -LiteralPath $nvcc)) {
    throw "nvcc was not found at $nvcc"
}

$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$visualStudio = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
$toolsRoot = Join-Path $visualStudio 'VC\Tools\MSVC'
$compiler = Get-ChildItem -LiteralPath $toolsRoot -Directory |
    Sort-Object Name |
    Select-Object -Last 1
$ccbin = Join-Path $compiler.FullName 'bin\Hostx64\x64'
$source = Join-Path $PSScriptRoot 'NNtrain.Core\native\flash_attention.cu'
$output = Join-Path $PSScriptRoot `
    'NNtrain.Core\runtimes\win-x64\native\NNtrain.CudaKernels.dll'

& $nvcc -O3 --use_fast_math -lineinfo -arch=sm_86 `
    -allow-unsupported-compiler -ccbin $ccbin -shared $source -o $output
if ($LASTEXITCODE -ne 0) {
    throw "nvcc failed with exit code $LASTEXITCODE"
}
Write-Host "Built $output"
