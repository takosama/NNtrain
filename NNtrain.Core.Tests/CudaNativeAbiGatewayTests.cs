using System.Runtime.InteropServices;
using System.Reflection;
using System.Text;
using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Interop;
using Xunit;

public sealed class CudaNativeAbiGatewayTests
{
    private static readonly (string EntryPoint, string Library)[]
        BlasGatewayImports =
    [
        ("cublasCreate_v2", CudaNativeGateway.CublasLibraryName),
        ("cublasDestroy_v2", CudaNativeGateway.CublasLibraryName),
        ("cublasSetStream_v2", CudaNativeGateway.CublasLibraryName),
        ("cublasGemmEx", CudaNativeGateway.CublasLibraryName),
        ("cublasGemmStridedBatchedEx", CudaNativeGateway.CublasLibraryName),
        ("cublasSgeam", CudaNativeGateway.CublasLibraryName),
        ("cublasLtCreate", CudaNativeGateway.CublasLtLibraryName),
        ("cublasLtDestroy", CudaNativeGateway.CublasLtLibraryName),
        ("cublasLtMatmulDescCreate", CudaNativeGateway.CublasLtLibraryName),
        ("cublasLtMatmulDescDestroy", CudaNativeGateway.CublasLtLibraryName),
        ("cublasLtMatmulDescSetAttribute",
            CudaNativeGateway.CublasLtLibraryName),
        ("cublasLtMatrixLayoutCreate", CudaNativeGateway.CublasLtLibraryName),
        ("cublasLtMatrixLayoutDestroy",
            CudaNativeGateway.CublasLtLibraryName),
        ("cublasLtMatmulPreferenceCreate",
            CudaNativeGateway.CublasLtLibraryName),
        ("cublasLtMatmulPreferenceDestroy",
            CudaNativeGateway.CublasLtLibraryName),
        ("cublasLtMatmulPreferenceSetAttribute",
            CudaNativeGateway.CublasLtLibraryName),
        ("cublasLtMatmulAlgoGetHeuristic",
            CudaNativeGateway.CublasLtLibraryName),
        ("cublasLtMatmulAlgoCapGetAttribute",
            CudaNativeGateway.CublasLtLibraryName),
        ("cublasLtMatmul", CudaNativeGateway.CublasLtLibraryName),
    ];

    private static readonly string[] CoreWrapperGatewayExports =
    [
        "nntrain_tensor_add_float",
        "nntrain_tensor_add_bf16",
        "nntrain_tensor_add_backward",
        "nntrain_tensor_embedding_float",
        "nntrain_tensor_embedding_bf16",
        "nntrain_tensor_embedding_backward",
        "nntrain_tensor_embedding_positions_float",
        "nntrain_tensor_embedding_positions_bf16",
        "nntrain_tensor_embedding_positions_backward",
        "nntrain_tensor_dropout_float",
        "nntrain_tensor_dropout_bf16",
        "nntrain_tensor_dropout_backward",
        "nntrain_tensor_add_dropout_float",
        "nntrain_tensor_add_dropout_bf16",
        "nntrain_tensor_add_dropout_backward",
        "nntrain_tensor_linear_bias_float",
        "nntrain_tensor_linear_bias_bf16",
        "nntrain_tensor_linear_mask_float",
        "nntrain_tensor_linear_encode_bf16",
        "nntrain_tensor_linear_encode_bfp8_relu",
        "nntrain_tensor_linear_mask_bf16_gradient",
        "nntrain_tensor_linear_bias_backward_float",
        "nntrain_tensor_linear_bias_backward_bf16",
        "nntrain_tensor_scale",
        "nntrain_tensor_accumulate",
        "nntrain_tensor_copy",
        "nntrain_tensor_encode_bf16",
        "nntrain_tensor_decode_bf16",
        "nntrain_tensor_softmax_probabilities",
        "nntrain_tensor_cross_entropy_probabilities_backward",
        "nntrain_tensor_squared_sum",
        "nntrain_tensor_cross_entropy_float",
        "nntrain_tensor_cross_entropy_bf16",
        "nntrain_tensor_cross_entropy_backward_float",
        "nntrain_tensor_cross_entropy_backward_bf16",
        "nntrain_tensor_cross_entropy_backward_bf16_output",
        "nntrain_optimizer_adamw",
        "nntrain_optimizer_adamw_bfp8_moments",
        "nntrain_optimizer_adamw_bfp8_apply",
        "nntrain_optimizer_adamw_bf16_state",
        "nntrain_optimizer_adamw_publish",
        "nntrain_optimizer_adamw_bf16_state_publish",
        "nntrain_optimizer_adamw_pure_bf16",
        "nntrain_optimizer_publish_bf16",
        "nntrain_optimizer_gather_stats",
        "nntrain_optimizer_adamw_multi_tensor",
        "nntrain_optimizer_accumulate_finite_status",
        "nntrain_optimizer_neko_moments",
        "nntrain_optimizer_neko_initialize",
        "nntrain_optimizer_neko_initialize_corrected",
        "nntrain_optimizer_neko_initialize_bf16_corrected",
        "nntrain_optimizer_neko_update_device_control",
        "nntrain_optimizer_neko_initialize_device_stats",
        "nntrain_optimizer_neko_initialize_bf16_device_stats",
        "nntrain_optimizer_neko_interpolate",
        "nntrain_optimizer_neko_transpose_back",
        "nntrain_optimizer_neko_apply",
        "nntrain_optimizer_neko_apply_bf16",
        "nntrain_optimizer_neko_combine",
        "nntrain_optimizer_neko_combine_batched",
        "nntrain_optimizer_symmetric_gram",
        "nntrain_optimizer_symmetric_gram_bf16_operands",
        "nntrain_optimizer_newton_schulz",
        "nntrain_optimizer_newton_schulz_bf16_operands",
        "nntrain_gradient_comm_create",
        "nntrain_gradient_event_create",
        "nntrain_gradient_pack_bf16",
        "nntrain_gradient_record_ready",
        "nntrain_gradient_record_ready_external",
        "nntrain_gradient_exchange_bf16",
        "nntrain_gradient_host_pipeline_create",
        "nntrain_gradient_host_pipeline_exchange_bf16",
        "nntrain_gradient_host_pipeline_destroy",
        "nntrain_gradient_unpack_float",
        "nntrain_gradient_comm_synchronize",
        "nntrain_gradient_event_destroy",
        "nntrain_gradient_comm_destroy",
    ];

    private static readonly string[] Bfp8GradientExports =
    [
        "nntrain_bfp8_gradient_quantize",
        "nntrain_bfp8_gradient_quantize_accumulate",
        "nntrain_bfp8_gradient_squared_sum",
        "nntrain_bfp8_gradient_scale",
        "nntrain_bfp8_gradient_reduce",
        "nntrain_bfp8_gradient_broadcast",
    ];

    private static readonly string[] TrainingKernelExports =
    [
        "nntrain_layer_norm_forward",
        "nntrain_layer_norm_forward_bf16",
        "nntrain_layer_norm_backward",
        "nntrain_layer_norm_backward_bf16",
        "nntrain_residual_dropout_layer_norm_forward",
        "nntrain_residual_dropout_layer_norm_forward_bf16",
        "nntrain_residual_dropout_layer_norm_backward",
        "nntrain_residual_dropout_layer_norm_backward_bf16",
        "nntrain_residual_dropout_layer_norm_backward_bf16_branch_gradient",
        "nntrain_residual_dropout_layer_norm_backward_bf16_io_gradient",
        "nntrain_flash_attention_forward",
        "nntrain_flash_attention_backward",
        "nntrain_flash_attention_forward_bf16",
        "nntrain_flash_attention_backward_bf16",
        "nntrain_flash_attention_forward_bf16_tensor_core",
        "nntrain_flash_attention_forward_bf16_tensor_core_sync",
        "nntrain_flash_attention_backward_bf16_tensor_core",
        "nntrain_flash_attention_backward_bf16_tensor_core_parallel_dkv",
        "nntrain_flash_attention_backward_bf16_tensor_core_bf16_gradient",
        "nntrain_flash_attention_backward_bf16_tensor_core_bf16_io_gradient",
        "nntrain_flash_attention_backward_bf16_tensor_core_bf16_io_gradient_sync",
        "nntrain_flash_attention_incremental_bf16",
        "nntrain_flash_attention_prefill_cache_bf16",
        "nntrain_forget_forward",
        "nntrain_forget_backward",
        "nntrain_forget_memory_forward_bf16_tensor_core",
        "nntrain_nekomuon_moments_stats_compact",
        "nntrain_nekomuon_moments_stats_compact_finite",
        "nntrain_nekomuon_moments_stats_bf16_compact",
        "nntrain_nekomuon_moments_stats_bf16_compact_finite",
        "nntrain_tensor_accumulate_scalar",
        "nntrain_tensor_embedding_backward_reduced",
        "nntrain_tensor_embedding_positions_backward_reduced",
        "nntrain_tensor_topk_float",
        "nntrain_tensor_topk_bf16",
        "nntrain_cuda_stream_begin_capture",
        "nntrain_cuda_stream_end_capture",
        "nntrain_cuda_graph_instantiate",
        "nntrain_cuda_graph_launch",
        "nntrain_cuda_graph_destroy",
        "nntrain_cuda_graph_exec_destroy",
        "nntrain_cuda_graph_dropout_mask",
        "nntrain_cuda_graph_counter_set",
        "nntrain_cuda_graph_counter_advance",
        "nntrain_cuda_graph_dropout_forward_float",
        "nntrain_cuda_graph_dropout_forward_bf16",
        "nntrain_cuda_graph_add_dropout_forward_float",
        "nntrain_cuda_graph_add_dropout_forward_bf16",
        "nntrain_cuda_graph_dropout_backward_float",
        "nntrain_cuda_graph_add_dropout_backward_float",
        "nntrain_cuda_graph_residual_dropout_layer_norm_forward",
        "nntrain_cuda_graph_residual_dropout_layer_norm_forward_bf16",
        "nntrain_cuda_graph_residual_dropout_layer_norm_backward",
        "nntrain_cuda_graph_residual_dropout_layer_norm_backward_bf16",
        "nntrain_cuda_graph_residual_dropout_layer_norm_backward_bf16_branch_gradient",
        "nntrain_cuda_graph_residual_dropout_layer_norm_backward_bf16_io_gradient",
        "nntrain_tensor_embedding_backward_reduced_bf16_gradient",
        "nntrain_tensor_embedding_positions_backward_reduced_bf16_gradient",
        "nntrain_tensor_dropout_backward_bf16_gradient",
        "nntrain_tensor_add_dropout_backward_bf16_gradient",
        "nntrain_tensor_linear_bias_backward_bf16_gradient",
        "nntrain_tensor_bf16_gradient_squared_sum",
        "nntrain_tensor_bf16_gradient_scale",
        "nntrain_cuda_graph_dropout_backward_bf16_gradient",
        "nntrain_cuda_graph_add_dropout_backward_bf16_gradient",
        "nntrain_classification_correct_f32",
        "nntrain_classification_correct_bf16",
        "nntrain_classification_correct_bfp8",
    ];

    private static readonly string[] RequiredExports =
    [
        "nntrain_abi_version",
        "nntrain_last_error",
        "nntrain_error_snapshot",
        "nntrain_capability_bitmap",
        "nntrain_cuda_device_count",
        "nntrain_cuda_device_name",
        "nntrain_cuda_set_device",
        "nntrain_cuda_use_external_stream",
        "nntrain_cuda_synchronize",
        "nntrain_cuda_mem_info",
        "nntrain_cuda_malloc",
        "nntrain_cuda_free",
        "nntrain_cuda_memset",
        "nntrain_cuda_memset_async",
        "nntrain_cuda_copy_h2d",
        "nntrain_cuda_copy_h2d_async",
        "nntrain_cuda_copy_d2h",
        "nntrain_cuda_copy_d2h_async",
        "nntrain_cuda_host_alloc",
        "nntrain_cuda_host_free",
        "nntrain_cuda_stream_create",
        "nntrain_cuda_stream_destroy",
        "nntrain_cuda_stream_synchronize",
        "nntrain_cuda_event_create",
        "nntrain_cuda_event_destroy",
        "nntrain_cuda_event_record",
        "nntrain_cuda_event_query",
        "nntrain_cuda_copy_d2h_async_record",
        "nntrain_cuda_copy_h2d_async_record",
        "nntrain_cuda_event_synchronize",
        "nntrain_cuda_copy_d2d",
        "nntrain_cuda_copy_d2d_async",
        "nntrain_cuda_can_access_peer",
        "nntrain_cuda_error_string",
        "nntrain_cuda_bfp8_quantize_f32",
        "nntrain_cuda_bfp8_dequantize_f32",
        "nntrain_cuda_bfp8_dequantize_bf16",
        "nntrain_cuda_bfp8_quantize_bf16",
        "nntrain_cuda_bfp8_requantize_i32",
        "nntrain_cuda_bfp8_transpose_i8",
        "nntrain_bfp8_embedding_forward",
        "nntrain_bfp8_embedding_positions_forward",
        .. Bfp8GradientExports,
        .. TrainingKernelExports,
        "nntrain_optimizer_accumulate_finite_status",
        .. CoreWrapperGatewayExports,
    ];

    [Fact]
    public void PackedAbiVersionRoundTrips()
    {
        var version = new CudaAbiVersion(7, 19);

        Assert.Equal(version, CudaAbiVersion.FromPacked(version.Packed));
        Assert.Equal("7.19", version.ToString());
    }

    [Fact]
    public void TrainingKernelPInvokesAreOwnedByVersionedGateway()
    {
        Assembly coreAssembly = typeof(Tensor).Assembly;
        string[] coreTypeNames =
        [
            "NNtrain.CudaLayerNorm",
            "NNtrain.CudaFlashAttention",
            "NNtrain.CudaForgetMemoryNative",
            "NNtrain.CudaForgetMemoryTensorCore",
            "NNtrain.CudaBfp8GradientNative",
            "NNtrain.CudaNekoMuon",
            "NNtrain.CudaTensorNative",
            "NNtrain.CudaOptimizerNative",
            "NNtrain.CudaGradientBuckets",
            "NNtrain.CudaBlas",
            "NNtrain.CudaBlasLt",
            "NNtrain.CudaBlasLtInt8",
        ];
        foreach (string typeName in coreTypeNames)
        {
            Type coreType = coreAssembly.GetType(
                typeName,
                throwOnError: true)!;
            Assert.DoesNotContain(
                coreType.GetMethods(
                    BindingFlags.Static |
                    BindingFlags.Public |
                    BindingFlags.NonPublic),
                method => method.GetCustomAttribute<DllImportAttribute>()
                    is not null);
        }

        HashSet<string> gatewayImports = GetNestedTypes(
                typeof(CudaNativeGateway))
            .SelectMany(type => type.GetMethods(
                BindingFlags.Static |
                BindingFlags.Public |
                BindingFlags.NonPublic))
            .Select(method => method.GetCustomAttribute<DllImportAttribute>())
            .Where(attribute => attribute is not null)
            .Select(attribute => attribute!.EntryPoint)
            .Where(entryPoint => !string.IsNullOrWhiteSpace(entryPoint))
            .Cast<string>()
            .ToHashSet(StringComparer.Ordinal);

        foreach (string export in TrainingKernelExports)
            Assert.Contains(export, gatewayImports);
        foreach (string export in Bfp8GradientExports)
            Assert.Contains(export, gatewayImports);
        foreach (string export in CoreWrapperGatewayExports)
            Assert.Contains(export, gatewayImports);
    }

    [Fact]
    public void EveryCudaPayloadImportIsOwnedByTheVersionedGatewayAssembly()
    {
        Assembly coreAssembly = typeof(Tensor).Assembly;
        foreach (Type type in coreAssembly.GetTypes())
        {
            foreach (MethodInfo method in type.GetMethods(
                         BindingFlags.Static |
                         BindingFlags.Instance |
                         BindingFlags.Public |
                         BindingFlags.NonPublic))
            {
                DllImportAttribute? import =
                    method.GetCustomAttribute<DllImportAttribute>();
                Assert.False(
                    string.Equals(
                        import?.Value,
                        CudaNativeGateway.LibraryName,
                        StringComparison.OrdinalIgnoreCase),
                    $"Core type {type.FullName}.{method.Name} directly " +
                    $"imports {CudaNativeGateway.LibraryName}.");
            }
        }

        Assembly cudaAssembly = typeof(CudaNativeGateway).Assembly;
        foreach (Type type in cudaAssembly.GetTypes())
        {
            foreach (MethodInfo method in type.GetMethods(
                         BindingFlags.Static |
                         BindingFlags.Instance |
                         BindingFlags.Public |
                         BindingFlags.NonPublic))
            {
                DllImportAttribute? import =
                    method.GetCustomAttribute<DllImportAttribute>();
                if (!string.Equals(
                        import?.Value,
                        CudaNativeGateway.LibraryName,
                        StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                Assert.True(
                    type == typeof(CudaNativeGateway) ||
                        type.DeclaringType == typeof(CudaNativeGateway),
                    $"CUDA payload import escaped the versioned gateway: " +
                    $"{type.FullName}.{method.Name}.");
            }
        }
    }

    [Fact]
    public void CudaBlasPInvokesAreOwnedOnlyByVersionedGateway()
    {
        Assembly coreAssembly = typeof(Tensor).Assembly;
        foreach (Type type in coreAssembly.GetTypes())
        {
            foreach (MethodInfo method in type.GetMethods(
                         BindingFlags.Static |
                         BindingFlags.Instance |
                         BindingFlags.Public |
                         BindingFlags.NonPublic))
            {
                DllImportAttribute? import =
                    method.GetCustomAttribute<DllImportAttribute>();
                Assert.False(
                    import?.Value.Contains(
                        "cublas",
                        StringComparison.OrdinalIgnoreCase) == true,
                    $"Core type {type.FullName}.{method.Name} directly " +
                    $"imports {import?.Value}.");
            }
        }

        Dictionary<string, string> gatewayImports = GetNestedTypes(
                typeof(CudaNativeGateway))
            .SelectMany(type => type.GetMethods(
                BindingFlags.Static |
                BindingFlags.Public |
                BindingFlags.NonPublic))
            .Select(method => method.GetCustomAttribute<DllImportAttribute>())
            .Where(attribute => attribute?.EntryPoint is not null)
            .ToDictionary(
                attribute => attribute!.EntryPoint!,
                attribute => attribute!.Value,
                StringComparer.Ordinal);

        foreach ((string entryPoint, string library) in BlasGatewayImports)
        {
            Assert.True(
                gatewayImports.TryGetValue(entryPoint, out string? actual),
                $"Missing CUDA BLAS gateway import: {entryPoint}");
            Assert.Equal(library, actual);
        }
    }

    [Fact]
    public void CudaBlasVendorGatewayCreatesBothHandleKinds()
    {
        if (CudaNativeGateway.DeviceCount(out int deviceCount) != 0 ||
            deviceCount == 0)
        {
            return;
        }

        Assert.Equal(0, CudaNativeGateway.SetDevice(0));
        Assert.Equal(0, CudaNativeGateway.CublasCreate(out nint blas));
        try
        {
            Assert.NotEqual(nint.Zero, blas);
        }
        finally
        {
            if (blas != nint.Zero)
                Assert.Equal(0, CudaNativeGateway.CublasDestroy(blas));
        }

        Assert.Equal(0, CudaNativeGateway.CublasLtCreate(out nint blasLt));
        try
        {
            Assert.NotEqual(nint.Zero, blasLt);
        }
        finally
        {
            if (blasLt != nint.Zero)
                Assert.Equal(0, CudaNativeGateway.CublasLtDestroy(blasLt));
        }
    }

    [Fact]
    public void TrainingKernelOperationsDeclareMinimumCapabilities()
    {
        Assert.Equal(
            CudaKernelFeature.FusedLayerNorm,
            CudaNativeGateway.RequiredFeatures(
                CudaNativeOperation.LayerNormForward));
        Assert.Equal(
            CudaKernelFeature.FlashAttention |
                CudaKernelFeature.BFloat16 |
                CudaKernelFeature.TensorCores,
            CudaNativeGateway.RequiredFeatures(
                CudaNativeOperation
                    .FlashAttentionBackwardBFloat16TensorCoreParallelDkv));
        Assert.Equal(
            CudaKernelFeature.ForgetMemory |
                CudaKernelFeature.BFloat16 |
                CudaKernelFeature.TensorCores,
            CudaNativeGateway.RequiredFeatures(
                CudaNativeOperation
                    .ForgetMemoryForwardBFloat16TensorCore));
        Assert.Equal(
            CudaAbiVersion.Bfp8EmbeddingMinor,
            CudaAbiVersion.TrainingKernelGatewayMinor);
        Assert.Equal(
            CudaKernelFeature.Bfp8Quantization,
            CudaNativeGateway.RequiredFeatures(
                CudaNativeOperation.Bfp8GradientScale));
        Assert.Equal(
            CudaKernelFeature.BlockReducedMuon,
            CudaNativeGateway.RequiredFeatures(
                CudaNativeOperation.NekoMuonMomentsStatsCompactFinite));
        Assert.Equal(7, CudaAbiVersion.NekoMuonFiniteStatusMinor);
        Assert.Equal(8, CudaAbiVersion.CudaOutputGradientSeedMinor);
        Assert.Equal(9, CudaAbiVersion.ReducedEmbeddingBackwardMinor);
        Assert.Equal(10, CudaAbiVersion.TensorTopKMinor);
        Assert.Equal(11, CudaAbiVersion.CudaGraphMinor);
        Assert.Equal(12, CudaAbiVersion.CudaGraphDropoutMinor);
        Assert.Equal(15, CudaAbiVersion.PureBFloat16GradientMinor);
        Assert.Equal(16, CudaAbiVersion.ClassificationAccuracyMinor);
        Assert.Equal(17, CudaAbiVersion.GraphFusedLayerNormMinor);
        Assert.Equal(18, CudaAbiVersion.PureBFloat16OptimizerMinor);
        Assert.Equal(19, CudaAbiVersion.ExternalGradientReadyEventMinor);
        Assert.Equal(
            CudaKernelFeature.None,
            CudaNativeGateway.RequiredFeatures(
                CudaNativeOperation.TensorAccumulateScalar));
        Assert.Equal(
            CudaKernelFeature.None,
            CudaNativeGateway.RequiredFeatures(
                CudaNativeOperation.EmbeddingPositionsBackwardReduced));
        Assert.Equal(
            CudaKernelFeature.None,
            CudaNativeGateway.RequiredFeatures(
                CudaNativeOperation.TensorTopK));
        Assert.Equal(
            CudaKernelFeature.CudaGraphs,
            CudaNativeGateway.RequiredFeatures(
                CudaNativeOperation.GraphLaunch));
        Assert.Equal(
            CudaKernelFeature.CudaGraphs,
            CudaNativeGateway.RequiredFeatures(
                CudaNativeOperation.GraphRngStep));
        Assert.Equal(
            CudaKernelFeature.CudaGraphs,
            CudaNativeGateway.RequiredFeatures(
                CudaNativeOperation.GraphDropoutForward));
        Assert.Equal(
            CudaKernelFeature.CudaGraphs,
            CudaNativeGateway.RequiredFeatures(
                CudaNativeOperation.GraphAddDropoutBackward));
        Assert.Equal(
            CudaKernelFeature.None,
            CudaNativeGateway.RequiredFeatures(
                CudaNativeOperation.ClassificationCorrectCount));
        Assert.Equal(
            CudaKernelFeature.BFloat16,
            CudaNativeGateway.RequiredFeatures(
                CudaNativeOperation.TensorPrimitiveBFloat16));
        Assert.Equal(
            CudaKernelFeature.Bfp8Quantization,
            CudaNativeGateway.RequiredFeatures(
                CudaNativeOperation.OptimizerBfp8));
        Assert.Equal(
            CudaKernelFeature.BlockReducedMuon |
                CudaKernelFeature.BFloat16 |
                CudaKernelFeature.TensorCores,
            CudaNativeGateway.RequiredFeatures(
                CudaNativeOperation.OptimizerNekoMuonBFloat16));
        Assert.Equal(
            CudaKernelFeature.AsynchronousGradientReduction |
                CudaKernelFeature.BFloat16,
            CudaNativeGateway.RequiredFeatures(
                CudaNativeOperation.GradientCollectiveBFloat16));
    }

    [Fact]
    public void NativePayloadRetainsRuntimeExportsAndAddsVersionedGatewayExports()
    {
        string path = Path.Combine(
            AppContext.BaseDirectory,
            CudaNativeGateway.LibraryName);
        Assert.True(File.Exists(path), $"Native payload not found at {path}.");

        nint library = NativeLibrary.Load(path);
        try
        {
            foreach (string export in RequiredExports)
            {
                Assert.True(
                    NativeLibrary.TryGetExport(library, export, out _),
                    $"Missing native export: {export}");
            }
        }
        finally
        {
            NativeLibrary.Free(library);
        }
    }

    [Fact]
    public void GatewayValidatesAbiAndConvertsDeviceCapabilities()
    {
        CudaNativeGateway.EnsureCompatibleAbi();

        Assert.Equal(1, CudaNativeGateway.AbiVersion.Major);
        Assert.True(
            CudaNativeGateway.AbiVersion.Minor >=
                CudaAbiVersion.ExternalGradientReadyEventMinor);
        Assert.Equal(
            0,
            CudaNativeGateway.DeviceCount(out int deviceCount));
        Assert.True(deviceCount > 0);
        Assert.Equal(
            0,
            CudaNativeGateway.KernelCapabilities(
                0,
                out CudaKernelCapabilities capabilities));
        Assert.True(capabilities.ComputeCapabilityMajor >= 8);
        Assert.True(capabilities.Supports(CudaKernelFeature.TensorCores));
        Assert.True(capabilities.Supports(CudaKernelFeature.BFloat16));
        Assert.True(capabilities.Supports(CudaKernelFeature.FlashAttention));
        Assert.True(capabilities.Supports(CudaKernelFeature.Bfp8Quantization));
        Assert.True(capabilities.Supports(CudaKernelFeature.Int8TensorCores));
        Assert.True(capabilities.Supports(CudaKernelFeature.CudaGraphs));
    }

    [Fact]
    public void StreamAwareMemoryOperationsExecuteInSubmissionOrder()
    {
        const int elementCount = 4;
        const int byteCount = elementCount * sizeof(int);
        nuint bytes = byteCount;
        nint inputHost = 0;
        nint outputHost = 0;
        nint sourceDevice = 0;
        nint destinationDevice = 0;
        nint stream = 0;
        try
        {
            Assert.Equal(
                0,
                CudaNativeGateway.HostAllocate(bytes, out inputHost));
            Assert.Equal(
                0,
                CudaNativeGateway.HostAllocate(
                    checked(bytes * 2), out outputHost));
            Assert.Equal(
                0,
                CudaNativeGateway.Allocate(
                    0, bytes, out sourceDevice));
            Assert.Equal(
                0,
                CudaNativeGateway.Allocate(
                    0, bytes, out destinationDevice));
            Assert.Equal(
                0,
                CudaNativeGateway.StreamCreate(0, out stream));

            for (int index = 0; index < elementCount; index++)
            {
                Marshal.WriteInt32(
                    inputHost,
                    index * sizeof(int),
                    1000 + index);
            }

            Assert.Equal(
                0,
                CudaNativeGateway.CopyHostToDeviceAsync(
                    0, sourceDevice, inputHost, bytes, stream));
            Assert.Equal(
                0,
                CudaNativeGateway.MemsetAsync(
                    0, destinationDevice, 0, bytes, stream));
            Assert.Equal(
                0,
                CudaNativeGateway.CopyDeviceToHostAsync(
                    0, outputHost, destinationDevice, bytes, stream));
            Assert.Equal(
                0,
                CudaNativeGateway.CopyDeviceToDeviceAsync(
                    0,
                    destinationDevice,
                    0,
                    sourceDevice,
                    bytes,
                    stream));
            Assert.Equal(
                0,
                CudaNativeGateway.CopyDeviceToHostAsync(
                    0,
                    outputHost + byteCount,
                    destinationDevice,
                    bytes,
                    stream));
            Assert.Equal(
                0,
                CudaNativeGateway.StreamSynchronize(0, stream));

            for (int index = 0; index < elementCount; index++)
            {
                Assert.Equal(
                    0,
                    Marshal.ReadInt32(
                        outputHost,
                        index * sizeof(int)));
                Assert.Equal(
                    1000 + index,
                    Marshal.ReadInt32(
                        outputHost,
                        byteCount + index * sizeof(int)));
            }
        }
        finally
        {
            if (stream != 0)
            {
                _ = CudaNativeGateway.StreamSynchronize(0, stream);
                _ = CudaNativeGateway.StreamDestroy(0, stream);
            }
            if (destinationDevice != 0)
                _ = CudaNativeGateway.Free(0, destinationDevice);
            if (sourceDevice != 0)
                _ = CudaNativeGateway.Free(0, sourceDevice);
            if (outputHost != 0)
                _ = CudaNativeGateway.HostFree(outputHost);
            if (inputHost != 0)
                _ = CudaNativeGateway.HostFree(inputHost);
        }
    }

    [Fact]
    public void FailureSnapshotIsImmutableAndCopiedIntoCompatibleException()
    {
        int status = CudaNativeGateway.DeviceName(
            0,
            new StringBuilder(),
            capacity: 0);
        Assert.NotEqual(0, status);

        NativeCudaException exception = Assert.Throws<NativeCudaException>(
            () => NativeCudaRuntime.Check(status, "ABI probe"));

        Assert.Equal(status, exception.Status);
        Assert.Equal(
            $"ABI probe failed with CUDA error {status}: " +
                CudaNativeGateway.ErrorString(status),
            exception.Message);
        Assert.True(exception.NativeError.HasValue);
        CudaNativeErrorInfo nativeError = exception.NativeError.Value;
        Assert.Equal(CudaNativeGateway.AbiVersion, nativeError.AbiVersion);
        Assert.True(nativeError.Sequence > 0);
        Assert.Equal(status, nativeError.Status);
        Assert.Equal(0, nativeError.DeviceIndex);
        Assert.Equal(CudaNativeOperation.DeviceName, nativeError.Operation);

        Assert.True(CudaNativeGateway.TryGetLastError(out var latest));
        Assert.Equal(nativeError.Sequence, latest.Sequence);
        Assert.Equal(nativeError.Status, latest.Status);
        Assert.Equal(nativeError.DeviceIndex, latest.DeviceIndex);
        Assert.Equal(nativeError.Operation, latest.Operation);
    }

    [Fact]
    public void MigratedCoreWrapperFailureKeepsDeviceAndOperationContext()
    {
        int status = CudaNativeGateway.GradientPackBFloat16(
            device: 0,
            source: 0,
            destination: 0,
            destinationOffset: 0,
            length: 0,
            computeStream: 0);
        Assert.NotEqual(0, status);

        NativeCudaException exception = Assert.Throws<NativeCudaException>(
            () => NativeCudaRuntime.Check(
                status,
                "migrated gradient gateway probe"));

        Assert.True(exception.NativeError.HasValue);
        CudaNativeErrorInfo error = exception.NativeError.Value;
        Assert.Equal(CudaNativeGateway.AbiVersion, error.AbiVersion);
        Assert.Equal(status, error.Status);
        Assert.Equal(0, error.DeviceIndex);
        Assert.Equal(
            CudaNativeOperation.GradientCollectiveBFloat16,
            error.Operation);
    }

    private static IEnumerable<Type> GetNestedTypes(Type root)
    {
        foreach (Type nested in root.GetNestedTypes(
                     BindingFlags.Public | BindingFlags.NonPublic))
        {
            yield return nested;
            foreach (Type descendant in GetNestedTypes(nested))
                yield return descendant;
        }
    }
}
