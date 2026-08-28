
using NNtrain.Runtime.Execution;

namespace NNtrain;

internal static partial class TensorCudaKernels
{
    private static bool UsesDirectBFloat16Gradients
        => TensorExecutionContext.UsesBFloat16GradientStorage;

    internal static BFloat16AttentionResidentContext
        AttentionForwardBFloat16Resident(
            Tensor projected,
            int batch,
            int sequence,
            int modelWidth,
            int numHeads,
            bool causal)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var output = Tensor.RentCudaBFloat16Buffer(
            deviceIndex,
            checked(batch * sequence * modelWidth));
        var softmaxLogSumExp = Tensor.RentCudaFloatBuffer(
            deviceIndex,
            checked(batch * numHeads * sequence));
        bool succeeded = CudaFlashAttention.TryForwardBFloat16(
            accelerator,
            projected.EnsureCudaBFloat16Buffer(deviceIndex),
            output,
            softmaxLogSumExp,
            batch,
            sequence,
            modelWidth,
            numHeads,
            causal,
            out bool tensorCore);
        if (!succeeded)
        {
            Tensor.ReturnCudaBFloat16Buffer(accelerator, output);
            Tensor.ReturnCudaFloatBuffer(accelerator, softmaxLogSumExp);
            throw new PlatformNotSupportedException(
                "Physical BF16 attention requires NNtrain.CudaKernels with " +
                "the BF16 FlashAttention entry points.");
        }
        NativeCudaBuffer<float>? rowDelta = tensorCore
            ? Tensor.RentCudaFloatBuffer(
                deviceIndex,
                checked(batch * numHeads * sequence))
            : null;
        return new BFloat16AttentionResidentContext(
            output,
            softmaxLogSumExp,
            rowDelta,
            accelerator,
            tensorCore);
    }

    internal static void AttentionBackwardBFloat16Resident(
        Tensor projected,
        Tensor output,
        BFloat16AttentionResidentContext context,
        int batch,
        int sequence,
        int modelWidth,
        int numHeads,
        bool causal)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        bool directBFloat16Gradient = UsesDirectBFloat16Gradients
            && context.TensorCore
            && !projected.HasGradientBuffer
            && !CudaDispatchPolicy.Current
                .DisableDirectAttentionBFloat16Gradient;
        NativeCudaBuffer<ushort>? bfloat16Gradient = directBFloat16Gradient
            ? Tensor.RentCudaBFloat16Buffer(deviceIndex, projected.Numel)
            : null;
        NativeCudaBuffer<ushort>? outputBFloat16Gradient = null;
        bool useBFloat16InputGradient = directBFloat16Gradient
            && output.TryGetCudaBFloat16GradientBuffer(
                deviceIndex,
                out outputBFloat16Gradient);
        try
        {
            CudaFlashAttention.BackwardBFloat16(
                accelerator,
                projected.EnsureCudaBFloat16Buffer(deviceIndex),
                context.Output,
                useBFloat16InputGradient
                    ? null
                    : output.EnsureCudaGradientBuffer(deviceIndex),
                useBFloat16InputGradient
                    ? outputBFloat16Gradient
                    : null,
                context.SoftmaxLogSumExp,
                context.RowDelta,
                directBFloat16Gradient
                    ? null
                    : projected.EnsureCudaGradientBuffer(deviceIndex),
                bfloat16Gradient,
                batch,
                sequence,
                modelWidth,
                numHeads,
                causal,
                context.TensorCore);
            if (bfloat16Gradient is not null)
            {
                projected.AdoptCudaBFloat16GradientBuffer(
                    bfloat16Gradient,
                    deviceIndex);
                bfloat16Gradient = null;
            }
            else
            {
                projected.MarkCudaGradientMutated(deviceIndex);
            }
        }
        finally
        {
            if (bfloat16Gradient is not null)
            {
                Tensor.ReturnCudaBFloat16Buffer(
                    accelerator,
                    bfloat16Gradient);
            }
        }
    }

    internal sealed class BFloat16AttentionResidentContext(
        NativeCudaBuffer<ushort> output,
        NativeCudaBuffer<float> softmaxLogSumExp,
        NativeCudaBuffer<float>? rowDelta,
        NativeCudaDevice accelerator,
        bool tensorCore) : IDisposable
    {
        private int _disposed;
        internal NativeCudaBuffer<ushort> Output { get; } = output;
        internal NativeCudaBuffer<float> SoftmaxLogSumExp { get; }
            = softmaxLogSumExp;
        internal NativeCudaBuffer<float>? RowDelta { get; }
            = rowDelta;
        internal bool TensorCore { get; } = tensorCore;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            var releases = new List<Action>
            {
                () => Tensor.ReturnCudaFloatBuffer(
                    accelerator,
                    SoftmaxLogSumExp),
            };
            if (RowDelta is not null)
            {
                releases.Add(() => Tensor.ReturnCudaFloatBuffer(
                    accelerator,
                    RowDelta));
            }
            try
            {
                CudaResourceCleanup.RunAll(
                    "CUDA BF16 attention context cleanup failed.",
                    releases);
            }
            finally
            {
                GC.SuppressFinalize(this);
            }
        }
    }

    internal static NativeCudaBuffer<ushort>
        CopyForwardBFloat16Resident(Tensor input)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var output = Tensor.RentCudaBFloat16Buffer(deviceIndex, input.Numel);
        input.EnsureCudaBFloat16Buffer(deviceIndex).View.CopyTo(output.View);
        return output;
    }

    internal static NativeCudaBuffer<ushort>
        CopyRangeForwardBFloat16Resident(Tensor input, int offset, int length)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        var output = Tensor.RentCudaBFloat16Buffer(deviceIndex, length);
        input.EnsureCudaBFloat16Buffer(deviceIndex).View
            .SubView(offset, length).CopyTo(output.View);
        return output;
    }

    internal static NativeCudaBuffer<ushort>
        AddForwardBFloat16Resident(Tensor left, Tensor right)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var output = Tensor.RentCudaBFloat16Buffer(deviceIndex, left.Numel);
        CudaTensorNative.Add(
            deviceIndex,
            left.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
            right.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
            output.NativePtr,
            left.Numel,
            bfloat16: true);
        return output;
    }

    internal static BFloat16EmbeddingResidentContext
        EmbeddingForwardBFloat16Resident(
            Tensor table,
            int[] indices,
            int width)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var indicesBuffer = Tensor.RentCudaIntBuffer(deviceIndex, indices);
        var output = Tensor.RentCudaBFloat16Buffer(
            deviceIndex,
            checked(indices.Length * width));
        CudaTensorNative.Embedding(
            deviceIndex,
            table.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
            indicesBuffer.NativePtr,
            output.NativePtr,
            checked(indices.Length * width),
            width,
            bfloat16: true);
        return new BFloat16EmbeddingResidentContext(
            output,
            indicesBuffer,
            accelerator);
    }

    internal static void EmbeddingBackwardBFloat16Resident(
        Tensor output,
        Tensor table,
        BFloat16EmbeddingResidentContext context,
        int width)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        if (UsesDirectBFloat16Gradients)
        {
            using CudaBFloat16GradientSource outputGradientSource =
                CudaBFloat16GradientSource.Acquire(output, deviceIndex);
            using var targets =
                new CudaPureBFloat16GradientTargetSet(deviceIndex);
            CudaPureBFloat16GradientTarget tableGradientTarget =
                targets.Get(table);
            tableGradientTarget.EnsureZeroInitialized();
            CudaEmbeddingBackwardDispatcher.BackwardBFloat16Gradient(
                deviceIndex,
                context.Indices.NativePtr,
                outputGradientSource.Buffer.NativePtr,
                tableGradientTarget.Buffer.NativePtr,
                output.Numel,
                width);
            targets.CommitAll();
            return;
        }

        var outputGradient = output.EnsureCudaGradientBuffer(deviceIndex);
        var tableGradient = table.EnsureCudaGradientBuffer(deviceIndex);
        CudaEmbeddingBackwardDispatcher.Backward(
            deviceIndex,
            context.Indices.NativePtr,
            outputGradient.NativePtr,
            tableGradient.NativePtr,
            output.Numel,
            width);
        table.MarkCudaGradientMutated(deviceIndex);
    }

    internal sealed class BFloat16EmbeddingResidentContext(
        NativeCudaBuffer<ushort> output,
        NativeCudaBuffer<int> indices,
        NativeCudaDevice accelerator) : IDisposable
    {
        private int _disposed;
        internal NativeCudaBuffer<ushort> Output { get; } = output;
        internal NativeCudaBuffer<int> Indices { get; } = indices;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            Tensor.ReturnCudaIntBuffer(accelerator, Indices);
            GC.SuppressFinalize(this);
        }
    }

    internal static BFloat16EmbeddingPositionsResidentContext
        EmbeddingWithPositionsForwardBFloat16Resident(
            Tensor tokenTable,
            Tensor positionTable,
            int[] indices,
            int sequenceLength,
            int width)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var indicesBuffer = Tensor.RentCudaIntBuffer(deviceIndex, indices);
        var output = Tensor.RentCudaBFloat16Buffer(
            deviceIndex,
            checked(indices.Length * width));
        CudaTensorNative.EmbeddingPositions(
            deviceIndex,
            tokenTable.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
            positionTable.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
            indicesBuffer.NativePtr,
            output.NativePtr,
            checked(indices.Length * width),
            sequenceLength,
            width,
            bfloat16: true);
        return new BFloat16EmbeddingPositionsResidentContext(
            output,
            indicesBuffer,
            accelerator);
    }

    internal static void EmbeddingWithPositionsBackwardBFloat16Resident(
        Tensor output,
        Tensor tokenTable,
        Tensor positionTable,
        BFloat16EmbeddingPositionsResidentContext context,
        int sequenceLength,
        int width)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        if (UsesDirectBFloat16Gradients)
        {
            bool sameTable = ReferenceEquals(tokenTable, positionTable);
            using CudaBFloat16GradientSource outputGradientSource =
                CudaBFloat16GradientSource.Acquire(output, deviceIndex);
            using var targets =
                new CudaPureBFloat16GradientTargetSet(deviceIndex);
            CudaPureBFloat16GradientTarget tokenGradientTarget =
                targets.Get(tokenTable);
            CudaPureBFloat16GradientTarget positionGradientTarget =
                targets.Get(positionTable);
            tokenGradientTarget.EnsureZeroInitialized();
            if (!sameTable)
                positionGradientTarget.EnsureZeroInitialized();
            CudaEmbeddingBackwardDispatcher
                .BackwardWithPositionsBFloat16Gradient(
                    deviceIndex,
                    context.Indices.NativePtr,
                    outputGradientSource.Buffer.NativePtr,
                    tokenGradientTarget.Buffer.NativePtr,
                    positionGradientTarget.Buffer.NativePtr,
                    output.Numel,
                    sequenceLength,
                    width);
            targets.CommitAll();
            return;
        }

        var outputGradient = output.EnsureCudaGradientBuffer(deviceIndex);
        var tokenGradient = tokenTable.EnsureCudaGradientBuffer(deviceIndex);
        var positionGradient = positionTable.EnsureCudaGradientBuffer(deviceIndex);
        CudaEmbeddingBackwardDispatcher.BackwardWithPositions(
            deviceIndex,
            context.Indices.NativePtr,
            outputGradient.NativePtr,
            tokenGradient.NativePtr,
            positionGradient.NativePtr,
            output.Numel,
            sequenceLength,
            width);
        tokenTable.MarkCudaGradientMutated(deviceIndex);
        positionTable.MarkCudaGradientMutated(deviceIndex);
    }

    internal sealed class BFloat16EmbeddingPositionsResidentContext(
        NativeCudaBuffer<ushort> output,
        NativeCudaBuffer<int> indices,
        NativeCudaDevice accelerator) : IDisposable
    {
        private int _disposed;
        internal NativeCudaBuffer<ushort> Output { get; } = output;
        internal NativeCudaBuffer<int> Indices { get; } = indices;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            Tensor.ReturnCudaIntBuffer(accelerator, Indices);
            GC.SuppressFinalize(this);
        }
    }

    internal static NativeCudaBuffer<ushort>
        DropoutForwardBFloat16Resident(
            Tensor input,
            uint seed,
            uint dropThreshold,
            float scale)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var output = Tensor.RentCudaBFloat16Buffer(deviceIndex, input.Numel);
        CudaTensorNative.Dropout(
            deviceIndex,
            input.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
            output.NativePtr,
            input.Numel,
            seed,
            dropThreshold,
            scale,
            bfloat16: true);
        return output;
    }

    internal static NativeCudaBuffer<ushort>
        AddDropoutForwardBFloat16Resident(
            Tensor residual,
            Tensor branch,
            uint seed,
            uint dropThreshold,
            float scale)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var output = Tensor.RentCudaBFloat16Buffer(deviceIndex, residual.Numel);
        CudaTensorNative.AddDropout(
            deviceIndex,
            residual.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
            branch.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
            output.NativePtr,
            residual.Numel,
            seed,
            dropThreshold,
            scale,
            bfloat16: true);
        return output;
    }

    internal static BFloat16LayerNormResidentContext
        LayerNormForwardBFloat16Resident(
            Tensor input,
            Tensor gamma,
            Tensor beta,
            int rows,
            int columns,
            float epsilon)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var output = Tensor.RentCudaBFloat16Buffer(deviceIndex, input.Numel);
        var means = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
        var inverses = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
        bool native = CudaLayerNorm.TryForwardBFloat16(
            accelerator,
            input.EnsureCudaBFloat16Buffer(deviceIndex),
            gamma.EnsureCudaBFloat16Buffer(deviceIndex),
            beta.EnsureCudaBFloat16Buffer(deviceIndex),
            output,
            means,
            inverses,
            rows,
            columns,
            epsilon);
        if (!native)
        {
            Tensor.ReturnCudaBFloat16Buffer(accelerator, output);
            Tensor.ReturnCudaFloatBuffer(accelerator, means);
            Tensor.ReturnCudaFloatBuffer(accelerator, inverses);
            throw new PlatformNotSupportedException(
                "BF16 CUDA LayerNorm requires the native reduction kernel.");
        }
        return new BFloat16LayerNormResidentContext(
            output,
            means,
            inverses,
            accelerator,
            native);
    }

    internal static BFloat16LayerNormResidentContext?
        TryResidualDropoutLayerNormForwardBFloat16Resident(
            Tensor residual,
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
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var output = Tensor.RentCudaBFloat16Buffer(deviceIndex, residual.Numel);
        var means = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
        var inverses = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
        bool succeeded = graphToken is { } token
            ? CudaLayerNorm.TryFusedForwardBFloat16Graph(
                accelerator,
                residual.EnsureCudaBFloat16Buffer(deviceIndex),
                branch.EnsureCudaBFloat16Buffer(deviceIndex),
                gamma.EnsureCudaBFloat16Buffer(deviceIndex),
                beta.EnsureCudaBFloat16Buffer(deviceIndex),
                output,
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
                residual.EnsureCudaBFloat16Buffer(deviceIndex),
                branch.EnsureCudaBFloat16Buffer(deviceIndex),
                gamma.EnsureCudaBFloat16Buffer(deviceIndex),
                beta.EnsureCudaBFloat16Buffer(deviceIndex),
                output,
                means,
                inverses,
                rows,
                columns,
                seed,
                dropThreshold,
                dropoutScale,
                epsilon);
        if (succeeded)
        {
            return new BFloat16LayerNormResidentContext(
                output, means, inverses, accelerator, native: true);
        }
        Tensor.ReturnCudaBFloat16Buffer(accelerator, output);
        Tensor.ReturnCudaFloatBuffer(accelerator, means);
        Tensor.ReturnCudaFloatBuffer(accelerator, inverses);
        return null;
    }

    internal static CrossEntropyResidentContext
        CrossEntropyForwardBFloat16Resident(
            Tensor logits,
            int[] labels,
            int rows,
            int columns,
            int ignoreIndex,
            int validRows,
            float labelSmoothing)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var labelsBuffer = Tensor.RentCudaIntBuffer(deviceIndex, labels);
        var maxima = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
        var inverseSums = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
        var rowLosses = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
        var loss = Tensor.RentCudaFloatBuffer(deviceIndex, 1);
        CudaTensorNative.CrossEntropy(
            deviceIndex,
            logits.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
            labelsBuffer.NativePtr,
            maxima.NativePtr,
            inverseSums.NativePtr,
            rowLosses.NativePtr,
            loss.NativePtr,
            rows,
            columns,
            ignoreIndex,
            validRows,
            labelSmoothing,
            bfloat16: true);
        return new CrossEntropyResidentContext(
            loss,
            maxima,
            inverseSums,
            rowLosses,
            labelsBuffer,
            accelerator);
    }

    internal static void CrossEntropyBackwardBFloat16Resident(
        Tensor logits,
        Tensor loss,
        CrossEntropyResidentContext context,
        int columns,
        int ignoreIndex,
        int validRows,
        float labelSmoothing)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        if (logits.AllowInPlaceBFloat16Gradient
            && AutogradEngine.IsReleasingGraph)
        {
            NativeCudaBuffer<ushort> inPlace =
                logits.EnsureCudaBFloat16Buffer(deviceIndex);
            CudaTensorNative.CrossEntropyBackwardBFloat16Output(
                deviceIndex,
                inPlace.NativePtr,
                context.Maxima.NativePtr,
                context.InverseSums.NativePtr,
                context.Labels.NativePtr,
                inPlace.NativePtr,
                loss.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                logits.Numel,
                columns,
                ignoreIndex,
                validRows,
                labelSmoothing);
            logits.MarkCudaBFloat16DataAsGradientInPlace(
                inPlace,
                deviceIndex);
            return;
        }
        if (UsesDirectBFloat16Gradients)
        {
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            NativeCudaBuffer<ushort> encodedGradient =
                Tensor.RentCudaBFloat16Buffer(deviceIndex, logits.Numel);
            try
            {
                CudaTensorNative.CrossEntropyBackwardBFloat16Output(
                    deviceIndex,
                    logits.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                    context.Maxima.NativePtr,
                    context.InverseSums.NativePtr,
                    context.Labels.NativePtr,
                    encodedGradient.NativePtr,
                    loss.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    logits.Numel,
                    columns,
                    ignoreIndex,
                    validRows,
                    labelSmoothing);
                logits.AdoptCudaBFloat16GradientBuffer(
                    encodedGradient,
                    deviceIndex);
            }
            catch
            {
                Tensor.ReturnCudaBFloat16Buffer(accelerator, encodedGradient);
                throw;
            }
        }
        else
        {
            NativeCudaBuffer<float> gradient =
                logits.EnsureCudaGradientBuffer(deviceIndex);
            CudaTensorNative.CrossEntropyBackward(
                deviceIndex,
                logits.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                context.Maxima.NativePtr,
                context.InverseSums.NativePtr,
                context.Labels.NativePtr,
                gradient.NativePtr,
                loss.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                logits.Numel,
                columns,
                ignoreIndex,
                validRows,
                labelSmoothing,
                bfloat16: true);
            logits.MarkCudaGradientMutated(deviceIndex);
        }
    }

    internal static void LayerNormBackwardBFloat16Resident(
        Tensor input,
        Tensor gamma,
        Tensor beta,
        Tensor output,
        BFloat16LayerNormResidentContext context,
        int rows,
        int columns)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        if (!context.Native)
        {
            throw new InvalidOperationException(
                "BF16 CUDA LayerNorm context was not produced by native CUDA.");
        }
        if (UsesDirectBFloat16Gradients)
        {
            NativeCudaBuffer<float>? decodedOutput = null;
            NativeCudaBuffer<float>? inputContribution = null;
            NativeCudaBuffer<float>? gammaContribution = null;
            NativeCudaBuffer<float>? betaContribution = null;
            try
            {
                using CudaBFloat16GradientSource outputGradientSource =
                    CudaBFloat16GradientSource.Acquire(
                        output,
                        deviceIndex);
                using var targets =
                    new CudaPureBFloat16GradientTargetSet(deviceIndex);
                CudaPureBFloat16GradientTarget inputGradientTarget =
                    targets.Get(input);
                CudaPureBFloat16GradientTarget gammaGradientTarget =
                    targets.Get(gamma);
                CudaPureBFloat16GradientTarget betaGradientTarget =
                    targets.Get(beta);

                decodedOutput = Tensor.RentCudaFloatBuffer(
                    deviceIndex,
                    output.Numel);
                inputContribution = Tensor.RentCudaFloatBuffer(
                    deviceIndex,
                    input.Numel);
                gammaContribution = Tensor.RentCudaFloatBuffer(
                    deviceIndex,
                    gamma.Numel);
                betaContribution = Tensor.RentCudaFloatBuffer(
                    deviceIndex,
                    beta.Numel);
                inputContribution.MemSetToZero();
                gammaContribution.MemSetToZero();
                betaContribution.MemSetToZero();
                CudaTensorNative.DecodeBFloat16(
                    deviceIndex,
                    outputGradientSource.Buffer.NativePtr,
                    decodedOutput.NativePtr,
                    output.Numel);
                CudaLayerNorm.BackwardBFloat16(
                    accelerator,
                    input.EnsureCudaBFloat16Buffer(deviceIndex),
                    gamma.EnsureCudaBFloat16Buffer(deviceIndex),
                    context.Means,
                    context.Inverses,
                    decodedOutput,
                    inputContribution,
                    gammaContribution,
                    betaContribution,
                    rows,
                    columns);
                inputGradientTarget.AccumulateFloat32(
                    inputContribution,
                    input.Numel);
                gammaGradientTarget.AccumulateFloat32(
                    gammaContribution,
                    gamma.Numel);
                betaGradientTarget.AccumulateFloat32(
                    betaContribution,
                    beta.Numel);
                targets.CommitAll();
            }
            finally
            {
                if (betaContribution is not null)
                    Tensor.ReturnCudaFloatBuffer(accelerator, betaContribution);
                if (gammaContribution is not null)
                    Tensor.ReturnCudaFloatBuffer(accelerator, gammaContribution);
                if (inputContribution is not null)
                    Tensor.ReturnCudaFloatBuffer(accelerator, inputContribution);
                if (decodedOutput is not null)
                    Tensor.ReturnCudaFloatBuffer(accelerator, decodedOutput);
            }
            return;
        }

        var outputGradient = output.EnsureCudaGradientBuffer(deviceIndex);
        var inputGradient = input.EnsureCudaGradientBuffer(deviceIndex);
        var gammaGradient = gamma.EnsureCudaGradientBuffer(deviceIndex);
        var betaGradient = beta.EnsureCudaGradientBuffer(deviceIndex);
        CudaLayerNorm.BackwardBFloat16(
            accelerator,
            input.EnsureCudaBFloat16Buffer(deviceIndex),
            gamma.EnsureCudaBFloat16Buffer(deviceIndex),
            context.Means,
            context.Inverses,
            outputGradient,
            inputGradient,
            gammaGradient,
            betaGradient,
            rows,
            columns);
        input.MarkCudaGradientMutated(deviceIndex);
        gamma.MarkCudaGradientMutated(deviceIndex);
        beta.MarkCudaGradientMutated(deviceIndex);
    }

    internal static void ResidualDropoutLayerNormBackwardBFloat16Resident(
        Tensor residual,
        Tensor branch,
        Tensor gamma,
        Tensor beta,
        Tensor output,
        BFloat16LayerNormResidentContext context,
        int rows,
        int columns,
        bool sameParent,
        uint seed,
        uint dropThreshold,
        float dropoutScale,
        CudaGraphDropoutToken? graphToken = null)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        if (UsesDirectBFloat16Gradients)
        {
            NativeCudaBuffer<float>? decodedOutput = null;
            NativeCudaBuffer<float>? residualContribution = null;
            NativeCudaBuffer<float>? branchContribution = null;
            NativeCudaBuffer<float>? gammaContribution = null;
            NativeCudaBuffer<float>? betaContribution = null;
            try
            {
                using CudaBFloat16GradientSource outputGradientSource =
                    CudaBFloat16GradientSource.Acquire(
                        output,
                        deviceIndex);
                using var targets =
                    new CudaPureBFloat16GradientTargetSet(deviceIndex);
                CudaPureBFloat16GradientTarget residualGradientTarget =
                    targets.Get(residual);
                CudaPureBFloat16GradientTarget branchGradientTarget =
                    targets.Get(branch);
                CudaPureBFloat16GradientTarget gammaGradientTarget =
                    targets.Get(gamma);
                CudaPureBFloat16GradientTarget betaGradientTarget =
                    targets.Get(beta);

                decodedOutput = Tensor.RentCudaFloatBuffer(
                    deviceIndex,
                    output.Numel);
                residualContribution = Tensor.RentCudaFloatBuffer(
                    deviceIndex,
                    residual.Numel);
                branchContribution = sameParent
                    ? residualContribution
                    : Tensor.RentCudaFloatBuffer(
                        deviceIndex,
                        branch.Numel);
                gammaContribution = Tensor.RentCudaFloatBuffer(
                    deviceIndex,
                    gamma.Numel);
                betaContribution = Tensor.RentCudaFloatBuffer(
                    deviceIndex,
                    beta.Numel);
                residualContribution.MemSetToZero();
                if (!sameParent)
                    branchContribution.MemSetToZero();
                gammaContribution.MemSetToZero();
                betaContribution.MemSetToZero();
                CudaTensorNative.DecodeBFloat16(
                    deviceIndex,
                    outputGradientSource.Buffer.NativePtr,
                    decodedOutput.NativePtr,
                    output.Numel);
                if (graphToken is { } token)
                {
                    CudaLayerNorm.FusedBackwardBFloat16Graph(
                        accelerator,
                        residual.EnsureCudaBFloat16Buffer(deviceIndex),
                        branch.EnsureCudaBFloat16Buffer(deviceIndex),
                        gamma.EnsureCudaBFloat16Buffer(deviceIndex),
                        context.Means,
                        context.Inverses,
                        decodedOutput,
                        residualContribution,
                        branchContribution,
                        gammaContribution,
                        betaContribution,
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
                        residual.EnsureCudaBFloat16Buffer(deviceIndex),
                        branch.EnsureCudaBFloat16Buffer(deviceIndex),
                        gamma.EnsureCudaBFloat16Buffer(deviceIndex),
                        context.Means,
                        context.Inverses,
                        decodedOutput,
                        residualContribution,
                        branchContribution,
                        gammaContribution,
                        betaContribution,
                        rows,
                        columns,
                        sameParent,
                        seed,
                        dropThreshold,
                        dropoutScale);
                }
                residualGradientTarget.AccumulateFloat32(
                    residualContribution,
                    residual.Numel);
                if (!sameParent)
                {
                    branchGradientTarget.AccumulateFloat32(
                        branchContribution,
                        branch.Numel);
                }
                gammaGradientTarget.AccumulateFloat32(
                    gammaContribution,
                    gamma.Numel);
                betaGradientTarget.AccumulateFloat32(
                    betaContribution,
                    beta.Numel);
                targets.CommitAll();
            }
            finally
            {
                if (betaContribution is not null)
                    Tensor.ReturnCudaFloatBuffer(accelerator, betaContribution);
                if (gammaContribution is not null)
                    Tensor.ReturnCudaFloatBuffer(accelerator, gammaContribution);
                if (!sameParent && branchContribution is not null)
                {
                    Tensor.ReturnCudaFloatBuffer(
                        accelerator,
                        branchContribution);
                }
                if (residualContribution is not null)
                {
                    Tensor.ReturnCudaFloatBuffer(
                        accelerator,
                        residualContribution);
                }
                if (decodedOutput is not null)
                    Tensor.ReturnCudaFloatBuffer(accelerator, decodedOutput);
            }
            return;
        }

        bool directBranchGradient = UsesDirectBFloat16Gradients
            && !sameParent
            && !branch.HasGradientBuffer
            && !CudaDispatchPolicy.Current
                .DisableDirectLayerNormBFloat16BranchGradient;
        NativeCudaBuffer<ushort>? branchGradientBFloat16 =
            directBranchGradient
                ? Tensor.RentCudaBFloat16Buffer(deviceIndex, branch.Numel)
                : null;
        NativeCudaBuffer<ushort>? outputGradientBFloat16 = null;
        bool useBFloat16OutputGradient = directBranchGradient
            && output.TryGetCudaBFloat16GradientBuffer(
                deviceIndex,
                out outputGradientBFloat16);
        try
        {
            if (branchGradientBFloat16 is not null)
            {
                NativeCudaBuffer<float>? outputGradient =
                    useBFloat16OutputGradient
                        ? null
                        : output.EnsureCudaGradientBuffer(deviceIndex);
                NativeCudaBuffer<ushort>? encodedOutputGradient =
                    useBFloat16OutputGradient
                        ? outputGradientBFloat16
                        : null;
                if (graphToken is { } token)
                {
                    CudaLayerNorm.FusedBackwardBFloat16DirectBranchGraph(
                        accelerator,
                        residual.EnsureCudaBFloat16Buffer(deviceIndex),
                        branch.EnsureCudaBFloat16Buffer(deviceIndex),
                        gamma.EnsureCudaBFloat16Buffer(deviceIndex),
                        context.Means,
                        context.Inverses,
                        outputGradient,
                        encodedOutputGradient,
                        residual.EnsureCudaGradientBuffer(deviceIndex),
                        branchGradientBFloat16,
                        gamma.EnsureCudaGradientBuffer(deviceIndex),
                        beta.EnsureCudaGradientBuffer(deviceIndex),
                        rows,
                        columns,
                        token,
                        dropThreshold,
                        dropoutScale);
                }
                else
                {
                    CudaLayerNorm.FusedBackwardBFloat16DirectBranch(
                        accelerator,
                        residual.EnsureCudaBFloat16Buffer(deviceIndex),
                        branch.EnsureCudaBFloat16Buffer(deviceIndex),
                        gamma.EnsureCudaBFloat16Buffer(deviceIndex),
                        context.Means,
                        context.Inverses,
                        outputGradient,
                        encodedOutputGradient,
                        residual.EnsureCudaGradientBuffer(deviceIndex),
                        branchGradientBFloat16,
                        gamma.EnsureCudaGradientBuffer(deviceIndex),
                        beta.EnsureCudaGradientBuffer(deviceIndex),
                        rows,
                        columns,
                        seed,
                        dropThreshold,
                        dropoutScale);
                }
                branch.AdoptCudaBFloat16GradientBuffer(
                    branchGradientBFloat16,
                    deviceIndex);
                branchGradientBFloat16 = null;
            }
            else
            {
                NativeCudaBuffer<float> residualGradient =
                    residual.EnsureCudaGradientBuffer(deviceIndex);
                NativeCudaBuffer<float> branchGradient = sameParent
                    ? residualGradient
                    : branch.EnsureCudaGradientBuffer(deviceIndex);
                if (graphToken is { } token)
                {
                    CudaLayerNorm.FusedBackwardBFloat16Graph(
                        accelerator,
                        residual.EnsureCudaBFloat16Buffer(deviceIndex),
                        branch.EnsureCudaBFloat16Buffer(deviceIndex),
                        gamma.EnsureCudaBFloat16Buffer(deviceIndex),
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
                        residual.EnsureCudaBFloat16Buffer(deviceIndex),
                        branch.EnsureCudaBFloat16Buffer(deviceIndex),
                        gamma.EnsureCudaBFloat16Buffer(deviceIndex),
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
            }
        }
        finally
        {
            if (branchGradientBFloat16 is not null)
            {
                Tensor.ReturnCudaBFloat16Buffer(
                    accelerator,
                    branchGradientBFloat16);
            }
        }
        residual.MarkCudaGradientMutated(deviceIndex);
        if (!sameParent && !directBranchGradient)
            branch.MarkCudaGradientMutated(deviceIndex);
        gamma.MarkCudaGradientMutated(deviceIndex);
        beta.MarkCudaGradientMutated(deviceIndex);
    }

    internal sealed class BFloat16LayerNormResidentContext(
        NativeCudaBuffer<ushort> output,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        NativeCudaDevice accelerator,
        bool native) : IDisposable
    {
        private int _disposed;
        internal NativeCudaBuffer<ushort> Output { get; } = output;
        internal NativeCudaBuffer<float> Means { get; } = means;
        internal NativeCudaBuffer<float> Inverses { get; } = inverses;
        internal bool Native { get; } = native;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            try
            {
                CudaResourceCleanup.RunAll(
                    "CUDA BF16 LayerNorm context cleanup failed.",
                    () => Tensor.ReturnCudaFloatBuffer(accelerator, Means),
                    () => Tensor.ReturnCudaFloatBuffer(accelerator, Inverses));
            }
            finally
            {
                GC.SuppressFinalize(this);
            }
        }
    }

    internal static NativeCudaBuffer<ushort>
        LinearForwardBFloat16Resident(
            Tensor input,
            Tensor weight,
            Tensor bias,
            int rows,
            int inputWidth,
            int outputWidth,
            bool applyRelu)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var inputBuffer = input.EnsureCudaBFloat16Buffer(deviceIndex);
        var weightBuffer = weight.EnsureCudaBFloat16Buffer(deviceIndex);
        var biasBuffer = bias.EnsureCudaBFloat16Buffer(deviceIndex);
        var output = Tensor.RentCudaBFloat16Buffer(
            deviceIndex,
            checked(rows * outputWidth));
        if (CudaBlasLt.TryLinearForwardBFloat16(
            accelerator,
            deviceIndex,
            inputBuffer,
            weightBuffer,
            biasBuffer,
            output,
            rows,
            inputWidth,
            outputWidth,
            applyRelu))
        {
            return output;
        }
        CudaBlas.LinearForwardBFloat16(
            accelerator,
            deviceIndex,
            inputBuffer,
            weightBuffer,
            output,
            rows,
            inputWidth,
            outputWidth);
        CudaTensorNative.LinearBias(
            deviceIndex,
            output.NativePtr,
            biasBuffer.NativePtr,
            checked(rows * outputWidth),
            outputWidth,
            applyRelu,
            bfloat16: true);
        return output;
    }

    internal static void LinearBackwardBFloat16Resident(
        Tensor input,
        Tensor weight,
        Tensor bias,
        Tensor output,
        int rows,
        int inputWidth,
        int outputWidth,
        bool applyRelu)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<ushort>? encodedGradient = null;
        bool hasBFloat16Gradient = output.TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out encodedGradient);
        bool borrowedEncodedGradient = !applyRelu && hasBFloat16Gradient;
        if (!borrowedEncodedGradient)
        {
            NativeCudaBuffer<ushort>? sourceBFloat16Gradient =
                hasBFloat16Gradient ? encodedGradient : null;
            encodedGradient = Tensor.RentCudaBFloat16Buffer(
                deviceIndex,
                checked(rows * outputWidth));
            void EncodeGradient()
            {
                int length = checked(rows * outputWidth);
                if (sourceBFloat16Gradient is not null && applyRelu)
                {
                    CudaTensorNative.LinearMaskBFloat16Gradient(
                        deviceIndex,
                        sourceBFloat16Gradient.NativePtr,
                        output.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                        encodedGradient.NativePtr,
                        length);
                }
                else
                {
                    NativeCudaBuffer<float> outputGradient =
                        output.EnsureCudaGradientBuffer(deviceIndex);
                    CudaTensorNative.LinearEncodeBFloat16(
                        deviceIndex,
                        outputGradient.NativePtr,
                        output.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                        encodedGradient.NativePtr,
                        length,
                        applyRelu);
                }
            }
            if (CudaOperationProfiler.IsEnabled)
            {
                CudaOperationProfiler.Measure(
                    $"backward.linear_bfloat16_encode[{inputWidth}->{outputWidth}]",
                    EncodeGradient);
            }
            else
            {
                EncodeGradient();
            }
        }

        if (UsesDirectBFloat16Gradients)
        {
            try
            {
                using var targets =
                    new CudaPureBFloat16GradientTargetSet(deviceIndex);
                CudaPureBFloat16GradientTarget inputGradient =
                    targets.Get(input);
                CudaPureBFloat16GradientTarget weightGradientTarget =
                    targets.Get(weight);
                CudaPureBFloat16GradientTarget biasGradientTarget =
                    targets.Get(bias);

                CudaBlas.LinearBackwardInputBFloat16Accumulate(
                    accelerator,
                    deviceIndex,
                    encodedGradient!,
                    weight.EnsureCudaBFloat16Buffer(deviceIndex),
                    inputGradient.Buffer,
                    rows,
                    inputWidth,
                    outputWidth,
                    inputGradient.HasValue);
                inputGradient.MarkFullContributionWritten();

                CudaBlas.LinearBackwardWeightBFloat16Direct(
                    accelerator,
                    deviceIndex,
                    input.EnsureCudaBFloat16Buffer(deviceIndex),
                    encodedGradient!,
                    weightGradientTarget.Buffer,
                    rows,
                    inputWidth,
                    outputWidth,
                    weightGradientTarget.HasValue);
                weightGradientTarget.MarkFullContributionWritten();

                // Rows reduce in FP32 inside the CUDA kernel, then the single
                // completed contribution is accumulated directly into the
                // authoritative BF16 bias gradient.
                biasGradientTarget.EnsureZeroInitialized();
                CudaPureBFloat16GradientNative.LinearBiasBackward(
                    deviceIndex,
                    encodedGradient!.NativePtr,
                    biasGradientTarget.Buffer.NativePtr,
                    rows,
                    outputWidth,
                    accelerator.DefaultStream);

                targets.CommitAll();
            }
            finally
            {
                if (!borrowedEncodedGradient)
                {
                    Tensor.ReturnCudaBFloat16Buffer(
                        accelerator,
                        encodedGradient!);
                }
            }
            return;
        }

        bool directInputGradient = UsesDirectBFloat16Gradients
            && !input.HasGradientBuffer
            && !CudaDispatchPolicy.Current
                .DisableDirectLinearBFloat16Gradient;
        NativeCudaBuffer<ushort>? directInputGradientBuffer =
            directInputGradient
                ? Tensor.RentCudaBFloat16Buffer(
                    deviceIndex,
                    checked(rows * inputWidth))
                : null;
        var weightGradient = weight.EnsureCudaGradientBuffer(deviceIndex);
        var biasGradient = bias.EnsureCudaGradientBuffer(deviceIndex);
        try
        {
            if (directInputGradientBuffer is not null)
            {
                CudaBlas.LinearBackwardInputBFloat16Direct(
                    accelerator,
                    deviceIndex,
                    encodedGradient!,
                    weight.EnsureCudaBFloat16Buffer(deviceIndex),
                    directInputGradientBuffer,
                    rows,
                    inputWidth,
                    outputWidth);
            }
            else
            {
                CudaBlas.LinearBackwardInputBFloat16(
                    accelerator,
                    deviceIndex,
                    encodedGradient!,
                    weight.EnsureCudaBFloat16Buffer(deviceIndex),
                    input.EnsureCudaGradientBuffer(deviceIndex),
                    rows,
                    inputWidth,
                    outputWidth);
            }
            CudaBlas.LinearBackwardWeightBFloat16(
                accelerator,
                deviceIndex,
                input.EnsureCudaBFloat16Buffer(deviceIndex),
                encodedGradient!,
                weightGradient,
                rows,
                inputWidth,
                outputWidth);
            CudaTensorNative.LinearBiasBackward(
                deviceIndex,
                encodedGradient!.NativePtr,
                biasGradient.NativePtr,
                rows,
                outputWidth,
                bfloat16: true);
            if (directInputGradientBuffer is not null)
            {
                input.AdoptCudaBFloat16GradientBuffer(
                    directInputGradientBuffer,
                    deviceIndex);
                directInputGradientBuffer = null;
            }
            else
            {
                input.MarkCudaGradientMutated(deviceIndex);
            }
        }
        finally
        {
            if (directInputGradientBuffer is not null)
            {
                Tensor.ReturnCudaBFloat16Buffer(
                    accelerator,
                    directInputGradientBuffer);
            }
            if (!borrowedEncodedGradient)
                Tensor.ReturnCudaBFloat16Buffer(accelerator, encodedGradient!);
        }
        weight.MarkCudaGradientMutated(deviceIndex);
        bias.MarkCudaGradientMutated(deviceIndex);
    }
}
