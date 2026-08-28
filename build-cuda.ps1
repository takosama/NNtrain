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
$nativeProjectRoot = Join-Path $PSScriptRoot 'NNtrain.Cuda'
$nativeSourceRoot = Join-Path $nativeProjectRoot 'native'
$nativeRuntimeRoot = Join-Path $nativeProjectRoot `
    'runtimes\win-x64\native'
$sources = @(
    (Join-Path $nativeSourceRoot 'bfp8_embedding.cu'),
    (Join-Path $nativeSourceRoot 'classification_accuracy.cu'),
    (Join-Path $nativeSourceRoot 'flash_attention.cu'),
    (Join-Path $nativeSourceRoot 'gradient_collectives.cu'),
    (Join-Path $nativeSourceRoot 'nekomuon_stats.cu'),
    (Join-Path $nativeSourceRoot 'cuda_runtime_bridge.cu'),
    (Join-Path $nativeSourceRoot 'public_ops.cu'),
    (Join-Path $nativeSourceRoot 'pure_bfloat16_gradients.cu'),
    (Join-Path $nativeSourceRoot 'tensor_kernels.cu')
)
foreach ($source in $sources) {
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "CUDA source owned by NNtrain.Cuda was not found: $source"
    }
}
New-Item -ItemType Directory -Force -Path $nativeRuntimeRoot | Out-Null
$output = Join-Path $nativeRuntimeRoot 'NNtrain.CudaKernels.dll'

& $nvcc -O3 --use_fast_math -lineinfo -std=c++17 -diag-suppress=177 `
    -Xcompiler '/wd4819' `
    -gencode 'arch=compute_80,code=sm_80' `
    -gencode 'arch=compute_86,code=sm_86' `
    -gencode 'arch=compute_89,code=sm_89' `
    -gencode 'arch=compute_90,code=sm_90' `
    -gencode 'arch=compute_90,code=compute_90' `
    -allow-unsupported-compiler -ccbin $ccbin -shared $sources -o $output
if ($LASTEXITCODE -ne 0) {
    throw "nvcc failed with exit code $LASTEXITCODE"
}

$dumpbin = Join-Path $ccbin 'dumpbin.exe'
$exportTable = & $dumpbin /nologo /exports $output
if ($LASTEXITCODE -ne 0) {
    throw "dumpbin failed with exit code $LASTEXITCODE"
}
$requiredGatewayExports = @(
    'nntrain_abi_version',
    'nntrain_tensor_accumulate_scalar',
    'nntrain_tensor_linear_encode_bfp8_relu',
    'nntrain_tensor_embedding_backward_reduced',
    'nntrain_tensor_embedding_positions_backward_reduced',
    'nntrain_tensor_topk_float',
    'nntrain_tensor_topk_bf16',
    'nntrain_gradient_record_ready_external',
    'nntrain_cuda_stream_begin_capture',
    'nntrain_cuda_stream_end_capture',
    'nntrain_cuda_graph_instantiate',
    'nntrain_cuda_graph_launch',
    'nntrain_cuda_graph_destroy',
    'nntrain_cuda_graph_exec_destroy',
    'nntrain_cuda_graph_dropout_mask',
    'nntrain_cuda_graph_counter_set',
    'nntrain_cuda_graph_counter_advance',
    'nntrain_cuda_graph_dropout_forward_float',
    'nntrain_cuda_graph_dropout_forward_bf16',
    'nntrain_cuda_graph_add_dropout_forward_float',
    'nntrain_cuda_graph_add_dropout_forward_bf16',
    'nntrain_cuda_graph_dropout_backward_float',
    'nntrain_cuda_graph_add_dropout_backward_float'
    'nntrain_cuda_graph_residual_dropout_layer_norm_forward'
    'nntrain_cuda_graph_residual_dropout_layer_norm_forward_bf16'
    'nntrain_cuda_graph_residual_dropout_layer_norm_backward'
    'nntrain_cuda_graph_residual_dropout_layer_norm_backward_bf16'
    'nntrain_cuda_graph_residual_dropout_layer_norm_backward_bf16_branch_gradient'
    'nntrain_cuda_graph_residual_dropout_layer_norm_backward_bf16_io_gradient'
    'nntrain_tensor_embedding_backward_reduced_bf16_gradient'
    'nntrain_tensor_embedding_positions_backward_reduced_bf16_gradient'
    'nntrain_tensor_dropout_backward_bf16_gradient'
    'nntrain_tensor_add_dropout_backward_bf16_gradient'
    'nntrain_tensor_linear_bias_backward_bf16_gradient'
    'nntrain_tensor_bf16_gradient_squared_sum'
    'nntrain_tensor_bf16_gradient_scale'
    'nntrain_cuda_graph_dropout_backward_bf16_gradient'
    'nntrain_cuda_graph_add_dropout_backward_bf16_gradient'
    'nntrain_classification_correct_f32'
    'nntrain_classification_correct_bf16'
    'nntrain_classification_correct_bfp8'
    'nntrain_public_binary_float'
    'nntrain_public_binary_bf16'
    'nntrain_public_binary_backward_float'
    'nntrain_public_binary_backward_bf16'
    'nntrain_public_binary_backward_bf16_gradient'
    'nntrain_public_unary_float'
    'nntrain_public_unary_bf16'
    'nntrain_public_unary_backward_float'
    'nntrain_public_unary_backward_bf16'
    'nntrain_public_unary_backward_bf16_gradient'
    'nntrain_public_reduce_float'
    'nntrain_public_reduce_bf16'
    'nntrain_public_reduce_backward_float'
    'nntrain_public_reduce_backward_bf16'
    'nntrain_public_reduce_backward_bf16_gradient'
    'nntrain_public_forget_scan_float'
    'nntrain_public_forget_scan_bf16'
    'nntrain_public_forget_scan_backward'
    'nntrain_public_forget_scan_backward_bf16_gradient'
    'nntrain_public_hyena_float'
    'nntrain_public_hyena_bf16'
    'nntrain_public_hyena_parallel_float'
    'nntrain_public_hyena_parallel_bf16'
    'nntrain_public_hyena_backward_float'
    'nntrain_public_hyena_backward_bf16'
    'nntrain_public_hyena_backward_parallel_float'
    'nntrain_public_hyena_backward_parallel_bf16'
    'nntrain_public_hyena_backward_bf16_gradient'
    'nntrain_public_shape_accumulate_bf16_gradient'
    'nntrain_public_transpose_bf16'
    'nntrain_public_transpose_backward_bf16_gradient'
    'nntrain_public_broadcast_add_float'
    'nntrain_public_broadcast_add_bf16'
    'nntrain_public_broadcast_add_backward'
    'nntrain_public_broadcast_add_backward_bf16_gradient'
    'nntrain_public_causal_mask_float'
    'nntrain_public_causal_mask_bf16'
    'nntrain_public_causal_mask_backward'
    'nntrain_public_causal_mask_backward_bf16_gradient'
    'nntrain_public_softmax_float'
    'nntrain_public_softmax_bf16'
    'nntrain_public_softmax_backward'
    'nntrain_public_softmax_backward_bf16_gradient'
)
foreach ($requiredExport in $requiredGatewayExports) {
    if (-not ($exportTable -match "\b$([Regex]::Escape($requiredExport))\b")) {
        throw "Native payload is missing required export $requiredExport"
    }
}
Write-Host "Built $output"
