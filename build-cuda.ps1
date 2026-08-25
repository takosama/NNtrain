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
$sources = @(
    (Join-Path $PSScriptRoot 'NNtrain.Core\native\flash_attention.cu'),
    (Join-Path $PSScriptRoot 'NNtrain.Core\native\cuda_runtime_bridge.cu'),
    (Join-Path $PSScriptRoot 'NNtrain.Core\native\tensor_kernels.cu')
)
$output = Join-Path $PSScriptRoot `
    'NNtrain.Core\runtimes\win-x64\native\NNtrain.CudaKernels.dll'

& $nvcc -O3 --use_fast_math -lineinfo `
    -gencode 'arch=compute_80,code=sm_80' `
    -gencode 'arch=compute_86,code=sm_86' `
    -gencode 'arch=compute_89,code=sm_89' `
    -gencode 'arch=compute_90,code=sm_90' `
    -gencode 'arch=compute_90,code=compute_90' `
    -allow-unsupported-compiler -ccbin $ccbin -shared $sources -o $output
if ($LASTEXITCODE -ne 0) {
    throw "nvcc failed with exit code $LASTEXITCODE"
}
Write-Host "Built $output"
