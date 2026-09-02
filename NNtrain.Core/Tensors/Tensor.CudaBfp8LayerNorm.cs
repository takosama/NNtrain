using System.Runtime.ExceptionServices;
using NNtrain.Cuda.Interop;

namespace NNtrain;

internal static partial class TensorCudaKernels
{
    internal static Bfp8LayerNormResidentContext
        LayerNormForwardBfp8Resident(
            Tensor input,
            Tensor gamma,
            Tensor beta,
            Bfp8QuantizationDescriptor outputDescriptor,
            int rows,
            int columns,
            float epsilon)
        => LayerNormForwardBfp8ResidentCore(
            input,
            branch: null,
            gamma,
            beta,
            outputDescriptor,
            rows,
            columns,
            seed: 0,
            dropThreshold: 0,
            dropoutScale: 1f,
            epsilon);

    internal static Bfp8LayerNormResidentContext
        ResidualDropoutLayerNormForwardBfp8Resident(
            Tensor residual,
            Tensor branch,
            Tensor gamma,
            Tensor beta,
            Bfp8QuantizationDescriptor outputDescriptor,
            int rows,
            int columns,
            uint seed,
            uint dropThreshold,
            float dropoutScale,
            float epsilon,
            CudaGraphDropoutToken? graphToken = null)
    {
        ArgumentNullException.ThrowIfNull(branch);
        return LayerNormForwardBfp8ResidentCore(
            residual,
            branch,
            gamma,
            beta,
            outputDescriptor,
            rows,
            columns,
            seed,
            dropThreshold,
            dropoutScale,
            epsilon,
            graphToken);
    }

    private static Bfp8LayerNormResidentContext
        LayerNormForwardBfp8ResidentCore(
            Tensor input,
            Tensor? branch,
            Tensor gamma,
            Tensor beta,
            Bfp8QuantizationDescriptor outputDescriptor,
            int rows,
            int columns,
            uint seed,
            uint dropThreshold,
            float dropoutScale,
            float epsilon,
            CudaGraphDropoutToken? graphToken = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(gamma);
        ArgumentNullException.ThrowIfNull(beta);
        ArgumentNullException.ThrowIfNull(outputDescriptor);
        RequireBfp8LayerNormOperand(input, nameof(input));
        RequireBfp8LayerNormOperand(gamma, nameof(gamma));
        RequireBfp8LayerNormOperand(beta, nameof(beta));
        if (branch is not null)
            RequireBfp8LayerNormOperand(branch, nameof(branch));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);
        if (checked(rows * columns) != input.Numel)
        {
            throw new ArgumentException(
                "LayerNorm rows and columns must cover the input tensor.");
        }

        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<ushort>? decodedInput = null;
        NativeCudaBuffer<ushort>? decodedBranch = null;
        NativeCudaBuffer<ushort>? decodedGamma = null;
        NativeCudaBuffer<ushort>? decodedBeta = null;
        NativeCudaBuffer<ushort>? decodedOutput = null;
        NativeCudaBuffer<float>? means = null;
        NativeCudaBuffer<float>? inverses = null;
        CudaBfp8OwnedBuffers? encodedOutput = null;
        try
        {
            if (branch is not null
                && CanUseDirectBfp8FusedLayerNorm(
                    input, branch, gamma, beta,
                    outputDescriptor, columns))
            {
                means = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
                inverses = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
                encodedOutput = CudaBfp8OwnedBuffers.Allocate(
                    accelerator, input.Numel, outputDescriptor);
                CudaLayerNorm.FusedForwardBfp8Block128x512(
                    accelerator,
                    input.EnsureCudaBfp8Buffer(deviceIndex),
                    branch.EnsureCudaBfp8Buffer(deviceIndex),
                    gamma.EnsureCudaBfp8Buffer(deviceIndex),
                    beta.EnsureCudaBfp8Buffer(deviceIndex),
                    new CudaBfp8BufferView(
                        encodedOutput.Payload,
                        encodedOutput.Scales,
                        outputDescriptor),
                    means,
                    inverses,
                    rows,
                    columns,
                    seed,
                    dropThreshold,
                    dropoutScale,
                    epsilon,
                    graphToken);
                var directContext = new Bfp8LayerNormResidentContext(
                    means,
                    inverses,
                    encodedOutput,
                    accelerator,
                    fused: true);
                means = null;
                inverses = null;
                encodedOutput = null;
                return directContext;
            }

            decodedInput = Tensor.RentCudaBFloat16Buffer(
                deviceIndex, input.Numel);
            if (branch is not null)
            {
                decodedBranch = Tensor.RentCudaBFloat16Buffer(
                    deviceIndex, branch.Numel);
            }
            decodedGamma = Tensor.RentCudaBFloat16Buffer(
                deviceIndex, columns);
            decodedBeta = Tensor.RentCudaBFloat16Buffer(
                deviceIndex, columns);
            decodedOutput = Tensor.RentCudaBFloat16Buffer(
                deviceIndex, input.Numel);
            means = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
            inverses = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
            encodedOutput = CudaBfp8OwnedBuffers.Allocate(
                accelerator, input.Numel, outputDescriptor);

            nint stream = accelerator.DefaultStream;
            DecodeLayerNormOperand(
                input, decodedInput, deviceIndex, stream);
            if (branch is not null)
            {
                DecodeLayerNormOperand(
                    branch,
                    decodedBranch
                        ?? throw new InvalidOperationException(
                            "The fused LayerNorm branch decode is missing."),
                    deviceIndex,
                    stream);
            }
            DecodeLayerNormOperand(
                gamma, decodedGamma, deviceIndex, stream);
            DecodeLayerNormOperand(
                beta, decodedBeta, deviceIndex, stream);

            bool succeeded = branch is null
                ? CudaLayerNorm.TryForwardBFloat16(
                    accelerator,
                    decodedInput,
                    decodedGamma,
                    decodedBeta,
                    decodedOutput,
                    means,
                    inverses,
                    rows,
                    columns,
                    epsilon)
                : graphToken is { } token
                    ? CudaLayerNorm.TryFusedForwardBFloat16Graph(
                        accelerator,
                        decodedInput,
                        decodedBranch
                            ?? throw new InvalidOperationException(
                                "The fused LayerNorm branch decode is missing."),
                        decodedGamma,
                        decodedBeta,
                        decodedOutput,
                        means,
                        inverses,
                        rows,
                        columns,
                        token,
                        dropThreshold,
                        dropoutScale,
                        epsilon)
                    : CudaLayerNorm.TryFusedForwardBFloat16(
                        accelerator,
                        decodedInput,
                        decodedBranch
                            ?? throw new InvalidOperationException(
                                "The fused LayerNorm branch decode is missing."),
                        decodedGamma,
                        decodedBeta,
                        decodedOutput,
                        means,
                        inverses,
                        rows,
                        columns,
                        seed,
                        dropThreshold,
                        dropoutScale,
                        epsilon);
            if (!succeeded)
            {
                throw new PlatformNotSupportedException(
                    "BFP8 LayerNorm requires the resident CUDA BF16 " +
                    "warp/block reduction backend; CPU fallback is forbidden.");
            }

            CudaBfp8Native.QuantizeBFloat16(
                deviceIndex,
                decodedOutput,
                encodedOutput.Payload,
                encodedOutput.Scales,
                outputDescriptor,
                stream);

            // Only row statistics are needed after forward. Re-decode the
            // authoritative BFP8 parents when this node runs backward rather
            // than retaining several full-size BF16 activation copies for
            // every Transformer layer.
            Tensor.ReturnCudaBFloat16Buffer(accelerator, decodedOutput);
            decodedOutput = null;
            Tensor.ReturnCudaBFloat16Buffer(accelerator, decodedBeta);
            decodedBeta = null;
            Tensor.ReturnCudaBFloat16Buffer(accelerator, decodedGamma);
            decodedGamma = null;
            if (decodedBranch is not null)
            {
                Tensor.ReturnCudaBFloat16Buffer(accelerator, decodedBranch);
                decodedBranch = null;
            }
            Tensor.ReturnCudaBFloat16Buffer(accelerator, decodedInput);
            decodedInput = null;

            var context = new Bfp8LayerNormResidentContext(
                means,
                inverses,
                encodedOutput,
                accelerator,
                fused: branch is not null);
            means = null;
            inverses = null;
            encodedOutput = null;
            return context;
        }
        catch (Exception failure)
        {
            List<Exception>? cleanupFailures = null;
            if (encodedOutput is not null)
            {
                TryCleanupBfp8LayerNorm(
                    encodedOutput.Dispose, ref cleanupFailures);
            }
            TryReturnBfp8LayerNormFloat(
                accelerator, inverses, ref cleanupFailures);
            TryReturnBfp8LayerNormFloat(
                accelerator, means, ref cleanupFailures);
            TryReturnBfp8LayerNormBFloat16(
                accelerator, decodedOutput, ref cleanupFailures);
            TryReturnBfp8LayerNormBFloat16(
                accelerator, decodedBeta, ref cleanupFailures);
            TryReturnBfp8LayerNormBFloat16(
                accelerator, decodedGamma, ref cleanupFailures);
            TryReturnBfp8LayerNormBFloat16(
                accelerator, decodedBranch, ref cleanupFailures);
            TryReturnBfp8LayerNormBFloat16(
                accelerator, decodedInput, ref cleanupFailures);
            ThrowBfp8LayerNormFailure(
                "BFP8 LayerNorm forward and rollback failed.",
                failure,
                cleanupFailures);
            throw;
        }
    }

    internal static void LayerNormBackwardBfp8Resident(
        Tensor input,
        Tensor gamma,
        Tensor beta,
        Tensor output,
        Bfp8LayerNormResidentContext context,
        int rows,
        int columns)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (context.Fused)
        {
            throw new InvalidOperationException(
                "A fused BFP8 LayerNorm context cannot run plain backward.");
        }
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<ushort>? decodedInput = null;
        NativeCudaBuffer<ushort>? decodedGamma = null;
        try
        {
            decodedInput = Tensor.RentCudaBFloat16Buffer(
                deviceIndex, input.Numel);
            decodedGamma = Tensor.RentCudaBFloat16Buffer(
                deviceIndex, columns);
            nint stream = accelerator.DefaultStream;
            DecodeLayerNormOperand(
                input, decodedInput, deviceIndex, stream);
            DecodeLayerNormOperand(
                gamma, decodedGamma, deviceIndex, stream);
            CudaLayerNorm.BackwardBFloat16(
                accelerator,
                decodedInput,
                decodedGamma,
                context.Means,
                context.Inverses,
                output.EnsureCudaGradientBuffer(deviceIndex),
                input.EnsureCudaGradientBuffer(deviceIndex),
                gamma.EnsureCudaGradientBuffer(deviceIndex),
                beta.EnsureCudaGradientBuffer(deviceIndex),
                rows,
                columns);
            input.MarkCudaGradientMutated(deviceIndex);
            gamma.MarkCudaGradientMutated(deviceIndex);
            beta.MarkCudaGradientMutated(deviceIndex);
        }
        finally
        {
            if (decodedGamma is not null)
            {
                Tensor.ReturnCudaBFloat16Buffer(
                    accelerator, decodedGamma);
            }
            if (decodedInput is not null)
            {
                Tensor.ReturnCudaBFloat16Buffer(
                    accelerator, decodedInput);
            }
        }
    }

    internal static void ResidualDropoutLayerNormBackwardBfp8Resident(
        Tensor residual,
        Tensor branch,
        Tensor gamma,
        Tensor beta,
        Tensor output,
        Bfp8LayerNormResidentContext context,
        int rows,
        int columns,
        bool sameParent,
        uint seed,
        uint dropThreshold,
        float dropoutScale,
        CudaGraphDropoutToken? graphToken = null)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (!context.Fused)
        {
            throw new InvalidOperationException(
                "A plain BFP8 LayerNorm context cannot run fused backward.");
        }
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<float> residualGradient =
            residual.EnsureCudaGradientBuffer(deviceIndex);
        NativeCudaBuffer<float> branchGradient = sameParent
            ? residualGradient
            : branch.EnsureCudaGradientBuffer(deviceIndex);
        NativeCudaBuffer<ushort>? decodedResidual = null;
        NativeCudaBuffer<ushort>? decodedBranch = null;
        NativeCudaBuffer<ushort>? decodedGamma = null;
        try
        {
            if (CanUseDirectBfp8FusedLayerNorm(
                    residual,
                    branch,
                    gamma,
                    beta,
                    output.Bfp8Quantization,
                    columns))
            {
                CudaLayerNorm.FusedBackwardBfp8Block128x512(
                    accelerator,
                    residual.EnsureCudaBfp8Buffer(deviceIndex),
                    branch.EnsureCudaBfp8Buffer(deviceIndex),
                    gamma.EnsureCudaBfp8Buffer(deviceIndex),
                    context.Means,
                    context.Inverses,
                    output.EnsureCudaGradientBuffer(deviceIndex),
                    residualGradient,
                    branchGradient,
                    gamma.EnsureCudaGradientBuffer(deviceIndex),
                    beta.EnsureCudaGradientBuffer(deviceIndex),
                    rows,
                    columns,
                    sameParent,
                    seed,
                    dropThreshold,
                    dropoutScale,
                    graphToken);
                residual.MarkCudaGradientMutated(deviceIndex);
                if (!sameParent)
                    branch.MarkCudaGradientMutated(deviceIndex);
                gamma.MarkCudaGradientMutated(deviceIndex);
                beta.MarkCudaGradientMutated(deviceIndex);
                return;
            }

            decodedResidual = Tensor.RentCudaBFloat16Buffer(
                deviceIndex, residual.Numel);
            decodedBranch = Tensor.RentCudaBFloat16Buffer(
                deviceIndex, branch.Numel);
            decodedGamma = Tensor.RentCudaBFloat16Buffer(
                deviceIndex, columns);
            nint stream = accelerator.DefaultStream;
            DecodeLayerNormOperand(
                residual, decodedResidual, deviceIndex, stream);
            DecodeLayerNormOperand(
                branch, decodedBranch, deviceIndex, stream);
            DecodeLayerNormOperand(
                gamma, decodedGamma, deviceIndex, stream);
            if (graphToken is { } token)
            {
                CudaLayerNorm.FusedBackwardBFloat16Graph(
                    accelerator,
                    decodedResidual,
                    decodedBranch,
                    decodedGamma,
                    context.Means,
                    context.Inverses,
                    output.EnsureCudaGradientBuffer(deviceIndex),
                    residualGradient,
                    branchGradient,
                    gamma.EnsureCudaGradientBuffer(deviceIndex),
                    beta.EnsureCudaGradientBuffer(deviceIndex),
                    rows,
                    columns,
                    sameParent,
                    token,
                    dropThreshold,
                    dropoutScale);
            }
            else
            {
                CudaLayerNorm.FusedBackwardBFloat16(
                    accelerator,
                    decodedResidual,
                    decodedBranch,
                    decodedGamma,
                    context.Means,
                    context.Inverses,
                    output.EnsureCudaGradientBuffer(deviceIndex),
                    residualGradient,
                    branchGradient,
                    gamma.EnsureCudaGradientBuffer(deviceIndex),
                    beta.EnsureCudaGradientBuffer(deviceIndex),
                    rows,
                    columns,
                    sameParent,
                    seed,
                    dropThreshold,
                    dropoutScale);
            }
            residual.MarkCudaGradientMutated(deviceIndex);
            if (!sameParent)
                branch.MarkCudaGradientMutated(deviceIndex);
            gamma.MarkCudaGradientMutated(deviceIndex);
            beta.MarkCudaGradientMutated(deviceIndex);
        }
        finally
        {
            if (decodedGamma is not null)
            {
                Tensor.ReturnCudaBFloat16Buffer(
                    accelerator, decodedGamma);
            }
            if (decodedBranch is not null)
            {
                Tensor.ReturnCudaBFloat16Buffer(
                    accelerator, decodedBranch);
            }
            if (decodedResidual is not null)
            {
                Tensor.ReturnCudaBFloat16Buffer(
                    accelerator, decodedResidual);
            }
        }
    }

    private static void RequireBfp8LayerNormOperand(
        Tensor tensor,
        string parameterName)
    {
        if (tensor.DType != TensorDType.Bfp8)
        {
            throw new ArgumentException(
                "The resident BFP8 LayerNorm path requires BFP8 operands.",
                parameterName);
        }
    }

    private static bool CanUseDirectBfp8FusedLayerNorm(
        Tensor residual,
        Tensor branch,
        Tensor gamma,
        Tensor beta,
        Bfp8QuantizationDescriptor? outputDescriptor,
        int columns)
    {
        int blockSize;
        if (columns == 512
            && !CudaDispatchPolicy.Current
                .DisableDirectBfp8LayerNormBlock32x512
            && IsBlockSize(outputDescriptor, residual.Numel, 32))
        {
            if (CudaNativeGateway.AbiVersion.Minor
                < CudaAbiVersion.DirectBfp8LayerNormBlock32x512Minor)
            {
                return false;
            }
            blockSize = 32;
        }
        else if (columns == 512)
        {
            blockSize = 128;
        }
        else if (columns == 384
            && CudaNativeGateway.AbiVersion.Minor
                >= CudaAbiVersion.DirectBfp8LayerNormBlock32x384Minor)
        {
            blockSize = 32;
        }
        else
        {
            return false;
        }

        return IsBlockSize(
                residual.Bfp8Quantization, residual.Numel, blockSize)
            && IsBlockSize(
                branch.Bfp8Quantization, branch.Numel, blockSize)
            && IsBlockSize(
                gamma.Bfp8Quantization, gamma.Numel, blockSize)
            && IsBlockSize(
                beta.Bfp8Quantization, beta.Numel, blockSize)
            && IsBlockSize(outputDescriptor, residual.Numel, blockSize);
    }

    private static bool IsBlockSize(
        Bfp8QuantizationDescriptor? descriptor,
        int length,
        int blockSize)
        => descriptor is not null
            && descriptor.Granularity == Bfp8ScaleGranularity.Block
            && descriptor.GetEffectiveBlockSize(length) == blockSize;

    private static void DecodeLayerNormOperand(
        Tensor tensor,
        NativeCudaBuffer<ushort> destination,
        int deviceIndex,
        nint stream)
    {
        CudaBfp8BufferView view = tensor.EnsureCudaBfp8Buffer(deviceIndex);
        CudaBfp8Native.DequantizeBFloat16(
            deviceIndex,
            view.Payload,
            view.Scales,
            destination,
            view.Descriptor,
            stream);
    }

    internal sealed class Bfp8LayerNormResidentContext : IDisposable
    {
        private readonly NativeCudaDevice _accelerator;
        private CudaBfp8OwnedBuffers? _encodedOutput;
        private int _disposed;

        internal Bfp8LayerNormResidentContext(
            NativeCudaBuffer<float> means,
            NativeCudaBuffer<float> inverses,
            CudaBfp8OwnedBuffers encodedOutput,
            NativeCudaDevice accelerator,
            bool fused)
        {
            Means = means;
            Inverses = inverses;
            _encodedOutput = encodedOutput;
            _accelerator = accelerator;
            Fused = fused;
        }

        internal NativeCudaBuffer<float> Means { get; }
        internal NativeCudaBuffer<float> Inverses { get; }
        internal bool Fused { get; }

        internal CudaBfp8OwnedBuffers DetachEncodedOutput()
        {
            ObjectDisposedException.ThrowIf(
                Volatile.Read(ref _disposed) != 0,
                this);
            return Interlocked.Exchange(ref _encodedOutput, null)
                ?? throw new InvalidOperationException(
                    "The BFP8 LayerNorm output was already detached.");
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;

            List<Exception>? failures = null;
            CudaBfp8OwnedBuffers? encodedOutput =
                Interlocked.Exchange(ref _encodedOutput, null);
            if (encodedOutput is not null)
            {
                TryCleanupBfp8LayerNorm(
                    encodedOutput.Dispose, ref failures);
            }
            TryReturnBfp8LayerNormFloat(
                _accelerator, Inverses, ref failures);
            TryReturnBfp8LayerNormFloat(
                _accelerator, Means, ref failures);
            GC.SuppressFinalize(this);

            if (failures is [Exception failure])
                ExceptionDispatchInfo.Capture(failure).Throw();
            if (failures is { Count: > 1 })
            {
                throw new AggregateException(
                    "BFP8 LayerNorm saved-context cleanup failed.",
                    failures);
            }
        }
    }

    private static void TryReturnBfp8LayerNormBFloat16(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort>? buffer,
        ref List<Exception>? failures)
    {
        if (buffer is null)
            return;
        TryCleanupBfp8LayerNorm(
            () => Tensor.ReturnCudaBFloat16Buffer(accelerator, buffer),
            ref failures);
    }

    private static void TryReturnBfp8LayerNormFloat(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float>? buffer,
        ref List<Exception>? failures)
    {
        if (buffer is null)
            return;
        TryCleanupBfp8LayerNorm(
            () => Tensor.ReturnCudaFloatBuffer(accelerator, buffer),
            ref failures);
    }

    private static void TryCleanupBfp8LayerNorm(
        Action cleanup,
        ref List<Exception>? failures)
    {
        try
        {
            cleanup();
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
    }

    private static void ThrowBfp8LayerNormFailure(
        string message,
        Exception failure,
        List<Exception>? cleanupFailures)
    {
        if (cleanupFailures is null)
            ExceptionDispatchInfo.Capture(failure).Throw();
        cleanupFailures.Insert(0, failure);
        throw new AggregateException(message, cleanupFailures);
    }
}

public partial class Tensor
{
    private Tensor LayerNormLastDimBfp8Cuda(
        Tensor gamma,
        Tensor beta,
        int rows,
        int columns,
        float epsilon)
    {
        int deviceIndex = CudaDeviceIndex;
        Bfp8QuantizationDescriptor outputDescriptor =
            SelectBfp8ResultDescriptor(this, gamma, beta);
        TensorCudaKernels.Bfp8LayerNormResidentContext context =
            CudaOperationProfiler.IsEnabled
                ? CudaOperationProfiler.Measure(
                    "forward.layer_norm",
                    () => TensorCudaKernels.LayerNormForwardBfp8Resident(
                        this,
                        gamma,
                        beta,
                        outputDescriptor,
                        rows,
                        columns,
                        epsilon))
                : TensorCudaKernels.LayerNormForwardBfp8Resident(
                    this,
                    gamma,
                    beta,
                    outputDescriptor,
                    rows,
                    columns,
                    epsilon);
        Tensor result;
        try
        {
            using CudaBfp8OwnedBuffers output =
                context.DetachEncodedOutput();
            result = FromCudaBfp8Result(
                output,
                deviceIndex,
                _shape,
                [this, gamma, beta]);
        }
        catch (Exception conversionFailure)
        {
            DisposeBfp8LayerNormContextAfterConversionFailure(
                context,
                conversionFailure,
                "BFP8 LayerNorm result construction failed.");
            throw;
        }

        if (AutogradContext.IsRecordingEnabled)
        {
            AutogradLease<TensorCudaKernels.Bfp8LayerNormResidentContext>
                lease = AutogradLease<TensorCudaKernels
                    .Bfp8LayerNormResidentContext>.Own(
                        context,
                        AutogradLeaseMetadata.CudaOwned(
                            deviceIndex,
                            TensorDType.Bfp8,
                            DataVersion),
                        static saved => saved.Dispose());
            result.Node.SetBackward(lease, savedContext =>
            {
                void Backward() => TensorCudaKernels
                    .LayerNormBackwardBfp8Resident(
                        this,
                        gamma,
                        beta,
                        result,
                        savedContext,
                        rows,
                        columns);
                if (CudaOperationProfiler.IsEnabled)
                    CudaOperationProfiler.Measure(
                        "backward.layer_norm", Backward);
                else
                    Backward();
            });
        }
        else if (!CudaInferenceScope.TrackResource(context))
        {
            context.Dispose();
        }
        return result;
    }

    private Tensor AddDropoutLayerNormLastDimBfp8Cuda(
        Tensor branch,
        Tensor gamma,
        Tensor beta,
        int rows,
        int columns,
        uint seed,
        uint dropThreshold,
        float dropoutScale,
        float epsilon,
        CudaGraphDropoutToken? graphToken = null)
    {
        int deviceIndex = CudaDeviceIndex;
        Bfp8QuantizationDescriptor outputDescriptor =
            SelectBfp8ResultDescriptor(this, branch, gamma, beta);
        TensorCudaKernels.Bfp8LayerNormResidentContext context =
            CudaOperationProfiler.IsEnabled
                ? CudaOperationProfiler.Measure(
                    "forward.residual_dropout_layer_norm",
                    () => TensorCudaKernels
                        .ResidualDropoutLayerNormForwardBfp8Resident(
                            this,
                            branch,
                            gamma,
                            beta,
                            outputDescriptor,
                            rows,
                            columns,
                            seed,
                            dropThreshold,
                            dropoutScale,
                            epsilon,
                            graphToken))
                : TensorCudaKernels
                    .ResidualDropoutLayerNormForwardBfp8Resident(
                        this,
                        branch,
                        gamma,
                        beta,
                        outputDescriptor,
                        rows,
                        columns,
                        seed,
                        dropThreshold,
                        dropoutScale,
                        epsilon,
                        graphToken);
        Tensor result;
        try
        {
            using CudaBfp8OwnedBuffers output =
                context.DetachEncodedOutput();
            result = FromCudaBfp8Result(
                output,
                deviceIndex,
                _shape,
                [this, branch, gamma, beta]);
        }
        catch (Exception conversionFailure)
        {
            DisposeBfp8LayerNormContextAfterConversionFailure(
                context,
                conversionFailure,
                "BFP8 residual/dropout/LayerNorm result construction failed.");
            throw;
        }

        if (AutogradContext.IsRecordingEnabled)
        {
            AutogradLease<TensorCudaKernels.Bfp8LayerNormResidentContext>
                lease = AutogradLease<TensorCudaKernels
                    .Bfp8LayerNormResidentContext>.Own(
                        context,
                        AutogradLeaseMetadata.CudaOwned(
                            deviceIndex,
                            TensorDType.Bfp8,
                            DataVersion),
                        static saved => saved.Dispose());
            result.Node.SetBackward(lease, savedContext =>
            {
                void Backward() => TensorCudaKernels
                    .ResidualDropoutLayerNormBackwardBfp8Resident(
                        this,
                        branch,
                        gamma,
                        beta,
                        result,
                        savedContext,
                        rows,
                        columns,
                        ReferenceEquals(this, branch),
                        seed,
                        dropThreshold,
                        dropoutScale,
                        graphToken);
                if (CudaOperationProfiler.IsEnabled)
                {
                    CudaOperationProfiler.Measure(
                        "backward.residual_dropout_layer_norm", Backward);
                }
                else
                {
                    Backward();
                }
            });
        }
        else if (!CudaInferenceScope.TrackResource(context))
        {
            context.Dispose();
        }
        return result;
    }

    private static void DisposeBfp8LayerNormContextAfterConversionFailure(
        IDisposable context,
        Exception conversionFailure,
        string message)
    {
        try
        {
            context.Dispose();
        }
        catch (Exception cleanupFailure)
        {
            throw new AggregateException(
                message,
                conversionFailure,
                cleanupFailure);
        }
        ExceptionDispatchInfo.Capture(conversionFailure).Throw();
    }
}
