namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// Computes LayerNorm(residual + dropout(branch)) as one CUDA operation.
    /// The CUDA backward also fuses the LayerNorm input gradient with the
    /// residual and regenerated-dropout-mask gradients.
    /// </summary>
    public Tensor AddDropoutLayerNormLastDim(
        Tensor branch,
        Tensor gamma,
        Tensor beta,
        float probability,
        Random? random = null,
        float eps = 1e-5f)
    {
        ArgumentNullException.ThrowIfNull(branch);
        ArgumentNullException.ThrowIfNull(gamma);
        ArgumentNullException.ThrowIfNull(beta);
        if (!float.IsFinite(probability)
            || probability < 0f
            || probability >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(probability), probability,
                "Dropout probability must be finite and in [0, 1).");
        }
        if (eps <= 0f || !float.IsFinite(eps))
            throw new ArgumentOutOfRangeException(nameof(eps));
        if (!_shape.AsSpan().SequenceEqual(branch._shape))
        {
            throw new ArgumentException(
                "Residual and dropout branch must have identical shapes.",
                nameof(branch));
        }
        gamma.CheckRank(1);
        beta.CheckRank(1);
        int columns = _shape[^1];
        int rows = Numel / columns;
        if (gamma._shape[0] != columns || beta._shape[0] != columns)
        {
            throw new ArgumentException(
                $"LayerNorm parameters must have shape [{columns}].");
        }

        bool bfloat16Cuda = ExecutionDevice == TensorDevice.Cuda
            && DType == TensorDType.BFloat16
            && branch.DType == TensorDType.BFloat16
            && gamma.DType == TensorDType.BFloat16
            && beta.DType == TensorDType.BFloat16;
        bool float32Cuda = ExecutionDevice == TensorDevice.Cuda
            && DType == TensorDType.Float32
            && branch.DType == TensorDType.Float32
            && gamma.DType == TensorDType.Float32
            && beta.DType == TensorDType.Float32;
        if (!bfloat16Cuda && !float32Cuda)
        {
            return probability == 0f
                ? AddLayerNormLastDim(branch, gamma, beta, eps)
                : AddDropout(
                    branch,
                    probability,
                    random,
                    directBFloat16BranchGradient: true)
                    .LayerNormLastDim(gamma, beta, eps);
        }

        random ??= Random.Shared;
        uint seed = probability == 0f ? 0u : NextDropoutSeed(random);
        float dropoutScale = probability == 0f
            ? 1f
            : 1f / (1f - probability);
        uint dropThreshold = probability == 0f
            ? 0u
            : (uint)(probability * (uint.MaxValue + 1d));

        if (ExecutionDevice == TensorDevice.Cuda)
        {
            if (bfloat16Cuda)
            {
                TensorCudaKernels.BFloat16LayerNormResidentContext? context =
                    CudaOperationProfiler.IsEnabled
                        ? CudaOperationProfiler.Measure(
                            "forward.residual_dropout_layer_norm",
                            () => TensorCudaKernels
                                .TryResidualDropoutLayerNormForwardBFloat16Resident(
                                    this, branch, gamma, beta, rows, columns,
                                    seed, dropThreshold, dropoutScale, eps))
                        : TensorCudaKernels
                            .TryResidualDropoutLayerNormForwardBFloat16Resident(
                                this, branch, gamma, beta, rows, columns,
                                seed, dropThreshold, dropoutScale, eps);
                if (context is not null)
                {
                    Tensor result = FromCudaResult(
                        context.Output,
                        CudaDeviceIndex,
                        _shape,
                        [this, branch, gamma, beta],
                        TensorDType.BFloat16);
                    ConfigureBFloat16Backward(
                        result, branch, gamma, beta, context, rows, columns,
                        seed, dropThreshold, dropoutScale);
                    return result;
                }
            }
            else if (float32Cuda)
            {
                TensorCudaKernels.LayerNormResidentContext? context =
                    CudaOperationProfiler.IsEnabled
                        ? CudaOperationProfiler.Measure(
                            "forward.residual_dropout_layer_norm",
                            () => TensorCudaKernels
                                .TryResidualDropoutLayerNormForwardResident(
                                    this, branch, gamma, beta, rows, columns,
                                    seed, dropThreshold, dropoutScale, eps))
                        : TensorCudaKernels
                            .TryResidualDropoutLayerNormForwardResident(
                                this, branch, gamma, beta, rows, columns,
                                seed, dropThreshold, dropoutScale, eps);
                if (context is not null)
                {
                    Tensor result = FromCudaResult(
                        context.Output,
                        CudaDeviceIndex,
                        _shape,
                        [this, branch, gamma, beta]);
                    ConfigureFloat32Backward(
                        result, branch, gamma, beta, context, rows, columns,
                        seed, dropThreshold, dropoutScale);
                    return result;
                }
            }
        }

        return probability == 0f
            ? AddLayerNormLastDim(branch, gamma, beta, eps)
            : AddDropout(
                branch,
                probability,
                random,
                directBFloat16BranchGradient: true)
                .LayerNormLastDim(gamma, beta, eps);
    }

    private void ConfigureBFloat16Backward(
        Tensor result,
        Tensor branch,
        Tensor gamma,
        Tensor beta,
        TensorCudaKernels.BFloat16LayerNormResidentContext context,
        int rows,
        int columns,
        uint seed,
        uint dropThreshold,
        float dropoutScale)
    {
        if (!AutogradContext.IsRecordingEnabled)
        {
            if (!CudaInferenceScope.TrackResource(context))
                context.Dispose();
            return;
        }
        int deviceIndex = CudaDeviceIndex;
        AutogradLease<TensorCudaKernels.BFloat16LayerNormResidentContext>
            lease = AutogradLease<TensorCudaKernels
                .BFloat16LayerNormResidentContext>.Own(
                context,
                AutogradLeaseMetadata.CudaOwned(
                    deviceIndex,
                    TensorDType.BFloat16,
                    DataVersion),
                static saved => saved.Dispose());
        result.Node.SetBackward(lease, savedContext =>
        {
            void Backward() => TensorCudaKernels
                .ResidualDropoutLayerNormBackwardBFloat16Resident(
                    this, branch, gamma, beta, result, savedContext,
                    rows, columns, ReferenceEquals(this, branch),
                    seed, dropThreshold, dropoutScale);
            if (CudaOperationProfiler.IsEnabled)
                CudaOperationProfiler.Measure(
                    "backward.residual_dropout_layer_norm", Backward);
            else
                Backward();
        });
    }

    private void ConfigureFloat32Backward(
        Tensor result,
        Tensor branch,
        Tensor gamma,
        Tensor beta,
        TensorCudaKernels.LayerNormResidentContext context,
        int rows,
        int columns,
        uint seed,
        uint dropThreshold,
        float dropoutScale)
    {
        if (!AutogradContext.IsRecordingEnabled)
        {
            if (!CudaInferenceScope.TrackResource(context))
                context.Dispose();
            return;
        }
        int deviceIndex = CudaDeviceIndex;
        AutogradLease<TensorCudaKernels.LayerNormResidentContext> lease =
            AutogradLease<TensorCudaKernels.LayerNormResidentContext>.Own(
                context,
                AutogradLeaseMetadata.CudaOwned(
                    deviceIndex,
                    TensorDType.Float32,
                    DataVersion),
                static saved => saved.Dispose());
        result.Node.SetBackward(lease, savedContext =>
        {
            void Backward() => TensorCudaKernels
                .ResidualDropoutLayerNormBackwardResident(
                    this, branch, gamma, beta, result, savedContext,
                    rows, columns, ReferenceEquals(this, branch),
                    seed, dropThreshold, dropoutScale);
            if (CudaOperationProfiler.IsEnabled)
                CudaOperationProfiler.Measure(
                    "backward.residual_dropout_layer_norm", Backward);
            else
                Backward();
        });
    }
}
