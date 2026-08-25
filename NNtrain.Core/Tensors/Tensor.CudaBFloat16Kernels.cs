
namespace NNtrain;

internal static partial class TensorCudaKernels
{
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
        CudaFlashAttention.BackwardBFloat16(
            accelerator,
            projected.EnsureCudaBFloat16Buffer(deviceIndex),
            context.Output,
            output.EnsureCudaGradientBuffer(deviceIndex),
            context.SoftmaxLogSumExp,
            context.RowDelta,
            projected.EnsureCudaGradientBuffer(deviceIndex),
            batch,
            sequence,
            modelWidth,
            numHeads,
            causal,
            context.TensorCore);
        projected.MarkCudaGradientMutated(deviceIndex);
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
            Tensor.ReturnCudaFloatBuffer(accelerator, SoftmaxLogSumExp);
            if (RowDelta is not null)
                Tensor.ReturnCudaFloatBuffer(accelerator, RowDelta);
            GC.SuppressFinalize(this);
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
        var outputGradient = output.EnsureCudaGradientBuffer(deviceIndex);
        var tableGradient = table.EnsureCudaGradientBuffer(deviceIndex);
        CudaTensorNative.EmbeddingBackward(
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
        var outputGradient = output.EnsureCudaGradientBuffer(deviceIndex);
        var tokenGradient = tokenTable.EnsureCudaGradientBuffer(deviceIndex);
        var positionGradient = positionTable.EnsureCudaGradientBuffer(deviceIndex);
        CudaTensorNative.EmbeddingPositionsBackward(
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
            float epsilon)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var output = Tensor.RentCudaBFloat16Buffer(deviceIndex, residual.Numel);
        var means = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
        var inverses = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
        bool succeeded = CudaLayerNorm.TryFusedForwardBFloat16(
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
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
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
                encodedGradient, deviceIndex);
        }
        catch
        {
            Tensor.ReturnCudaBFloat16Buffer(accelerator, encodedGradient);
            throw;
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
        var outputGradient = output.EnsureCudaGradientBuffer(deviceIndex);
        var inputGradient = input.EnsureCudaGradientBuffer(deviceIndex);
        var gammaGradient = gamma.EnsureCudaGradientBuffer(deviceIndex);
        var betaGradient = beta.EnsureCudaGradientBuffer(deviceIndex);
        if (!context.Native)
            throw new InvalidOperationException(
                "BF16 CUDA LayerNorm context was not produced by native CUDA.");
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
        float dropoutScale)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        CudaLayerNorm.FusedBackwardBFloat16(
            accelerator,
            residual.EnsureCudaBFloat16Buffer(deviceIndex),
            branch.EnsureCudaBFloat16Buffer(deviceIndex),
            gamma.EnsureCudaBFloat16Buffer(deviceIndex),
            context.Means,
            context.Inverses,
            output.EnsureCudaGradientBuffer(deviceIndex),
            residual.EnsureCudaGradientBuffer(deviceIndex),
            sameParent
                ? residual.EnsureCudaGradientBuffer(deviceIndex)
                : branch.EnsureCudaGradientBuffer(deviceIndex),
            gamma.EnsureCudaGradientBuffer(deviceIndex),
            beta.EnsureCudaGradientBuffer(deviceIndex),
            rows,
            columns,
            sameParent,
            seed,
            dropThreshold,
            dropoutScale);
        residual.MarkCudaGradientMutated(deviceIndex);
        if (!sameParent)
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
            Tensor.ReturnCudaFloatBuffer(accelerator, Means);
            Tensor.ReturnCudaFloatBuffer(accelerator, Inverses);
            GC.SuppressFinalize(this);
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
        bool borrowedEncodedGradient = !applyRelu
            && output.TryGetCudaBFloat16GradientBuffer(
                deviceIndex,
                out encodedGradient);
        if (!borrowedEncodedGradient)
        {
            NativeCudaBuffer<float> outputGradient =
                output.EnsureCudaGradientBuffer(deviceIndex);
            encodedGradient = Tensor.RentCudaBFloat16Buffer(
                deviceIndex,
                checked(rows * outputWidth));
            void EncodeGradient()
            {
                int length = checked(rows * outputWidth);
                CudaTensorNative.LinearEncodeBFloat16(
                    deviceIndex,
                    outputGradient.NativePtr,
                    output.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                    encodedGradient.NativePtr,
                    length,
                    applyRelu);
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

        var inputGradient = input.EnsureCudaGradientBuffer(deviceIndex);
        var weightGradient = weight.EnsureCudaGradientBuffer(deviceIndex);
        var biasGradient = bias.EnsureCudaGradientBuffer(deviceIndex);
        CudaBlas.LinearBackwardInputBFloat16(
            accelerator,
            deviceIndex,
            encodedGradient!,
            weight.EnsureCudaBFloat16Buffer(deviceIndex),
            inputGradient,
            rows,
            inputWidth,
            outputWidth);
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
        if (!borrowedEncodedGradient)
            Tensor.ReturnCudaBFloat16Buffer(accelerator, encodedGradient!);
        input.MarkCudaGradientMutated(deviceIndex);
        weight.MarkCudaGradientMutated(deviceIndex);
        bias.MarkCudaGradientMutated(deviceIndex);
    }
}
