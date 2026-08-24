using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

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
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var output = Tensor.RentCudaBFloat16Buffer(
            deviceIndex,
            checked(batch * sequence * modelWidth));
        bool succeeded = CudaFlashAttention.TryForwardBFloat16(
            accelerator,
            projected.EnsureCudaBFloat16Buffer(deviceIndex),
            output,
            batch,
            sequence,
            modelWidth,
            numHeads,
            causal);
        if (!succeeded)
        {
            Tensor.ReturnCudaBFloat16Buffer(accelerator, output);
            throw new PlatformNotSupportedException(
                "Physical BF16 attention requires NNtrain.CudaKernels with " +
                "the BF16 FlashAttention entry points.");
        }
        return new BFloat16AttentionResidentContext(output);
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
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        CudaFlashAttention.BackwardBFloat16(
            accelerator,
            projected.EnsureCudaBFloat16Buffer(deviceIndex),
            context.Output,
            output.EnsureCudaGradientBuffer(deviceIndex),
            projected.EnsureCudaGradientBuffer(deviceIndex),
            batch,
            sequence,
            modelWidth,
            numHeads,
            causal);
        projected.MarkCudaGradientMutated(deviceIndex);
    }

    internal sealed class BFloat16AttentionResidentContext(
        MemoryBuffer1D<ushort, Stride1D.Dense> output) : IDisposable
    {
        private int _disposed;
        internal MemoryBuffer1D<ushort, Stride1D.Dense> Output { get; } = output;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            GC.SuppressFinalize(this);
        }
    }

    internal static MemoryBuffer1D<ushort, Stride1D.Dense>
        CopyForwardBFloat16Resident(Tensor input)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var output = Tensor.RentCudaBFloat16Buffer(deviceIndex, input.Numel);
        input.EnsureCudaBFloat16Buffer(deviceIndex).View.CopyTo(output.View);
        return output;
    }

    internal static MemoryBuffer1D<ushort, Stride1D.Dense>
        CopyRangeForwardBFloat16Resident(Tensor input, int offset, int length)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        var output = Tensor.RentCudaBFloat16Buffer(deviceIndex, length);
        input.EnsureCudaBFloat16Buffer(deviceIndex).View
            .SubView(offset, length).CopyTo(output.View);
        return output;
    }

    internal static MemoryBuffer1D<ushort, Stride1D.Dense>
        AddForwardBFloat16Resident(Tensor left, Tensor right)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var output = Tensor.RentCudaBFloat16Buffer(deviceIndex, left.Numel);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<ushort>, ArrayView<ushort>, ArrayView<ushort>>(
                AddForwardBFloat16Kernel);
        kernel(left.Numel,
            left.EnsureCudaBFloat16Buffer(deviceIndex).View,
            right.EnsureCudaBFloat16Buffer(deviceIndex).View,
            output.View);
        return output;
    }

    internal static BFloat16EmbeddingResidentContext
        EmbeddingForwardBFloat16Resident(
            Tensor table,
            int[] indices,
            int width)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var indicesBuffer = Tensor.RentCudaIntBuffer(deviceIndex, indices);
        var output = Tensor.RentCudaBFloat16Buffer(
            deviceIndex,
            checked(indices.Length * width));
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<ushort>, ArrayView<int>, ArrayView<ushort>, int>(
                EmbeddingForwardBFloat16Kernel);
        kernel(
            checked(indices.Length * width),
            table.EnsureCudaBFloat16Buffer(deviceIndex).View,
            indicesBuffer.View,
            output.View,
            width);
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
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var outputGradient = output.EnsureCudaGradientBuffer(deviceIndex);
        var tableGradient = table.EnsureCudaGradientBuffer(deviceIndex);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<float>, ArrayView<float>, int>(
                EmbeddingBackwardKernel);
        kernel(
            output.Numel,
            context.Indices.View,
            outputGradient.View,
            tableGradient.View,
            width);
        table.MarkCudaGradientMutated(deviceIndex);
    }

    internal sealed class BFloat16EmbeddingResidentContext(
        MemoryBuffer1D<ushort, Stride1D.Dense> output,
        MemoryBuffer1D<int, Stride1D.Dense> indices,
        CudaAccelerator accelerator) : IDisposable
    {
        private int _disposed;
        internal MemoryBuffer1D<ushort, Stride1D.Dense> Output { get; } = output;
        internal MemoryBuffer1D<int, Stride1D.Dense> Indices { get; } = indices;

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
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var indicesBuffer = Tensor.RentCudaIntBuffer(deviceIndex, indices);
        var output = Tensor.RentCudaBFloat16Buffer(
            deviceIndex,
            checked(indices.Length * width));
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<ushort>, ArrayView<ushort>, ArrayView<int>,
            ArrayView<ushort>, int, int>(EmbeddingPositionsForwardBFloat16Kernel);
        kernel(checked(indices.Length * width),
            tokenTable.EnsureCudaBFloat16Buffer(deviceIndex).View,
            positionTable.EnsureCudaBFloat16Buffer(deviceIndex).View,
            indicesBuffer.View,
            output.View,
            sequenceLength,
            width);
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
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var outputGradient = output.EnsureCudaGradientBuffer(deviceIndex);
        var tokenGradient = tokenTable.EnsureCudaGradientBuffer(deviceIndex);
        var positionGradient = positionTable.EnsureCudaGradientBuffer(deviceIndex);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, int, int>(EmbeddingPositionsBackwardKernel);
        kernel(output.Numel,
            context.Indices.View,
            outputGradient.View,
            tokenGradient.View,
            positionGradient.View,
            sequenceLength,
            width);
        tokenTable.MarkCudaGradientMutated(deviceIndex);
        positionTable.MarkCudaGradientMutated(deviceIndex);
    }

    internal sealed class BFloat16EmbeddingPositionsResidentContext(
        MemoryBuffer1D<ushort, Stride1D.Dense> output,
        MemoryBuffer1D<int, Stride1D.Dense> indices,
        CudaAccelerator accelerator) : IDisposable
    {
        private int _disposed;
        internal MemoryBuffer1D<ushort, Stride1D.Dense> Output { get; } = output;
        internal MemoryBuffer1D<int, Stride1D.Dense> Indices { get; } = indices;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            Tensor.ReturnCudaIntBuffer(accelerator, Indices);
            GC.SuppressFinalize(this);
        }
    }

    internal static MemoryBuffer1D<ushort, Stride1D.Dense>
        DropoutForwardBFloat16Resident(
            Tensor input,
            uint seed,
            uint dropThreshold,
            float scale)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var output = Tensor.RentCudaBFloat16Buffer(deviceIndex, input.Numel);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<ushort>, ArrayView<ushort>, uint, uint, float>(
                DropoutForwardBFloat16Kernel);
        kernel(input.Numel,
            input.EnsureCudaBFloat16Buffer(deviceIndex).View,
            output.View,
            seed,
            dropThreshold,
            scale);
        return output;
    }

    internal static MemoryBuffer1D<ushort, Stride1D.Dense>
        AddDropoutForwardBFloat16Resident(
            Tensor residual,
            Tensor branch,
            uint seed,
            uint dropThreshold,
            float scale)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var output = Tensor.RentCudaBFloat16Buffer(deviceIndex, residual.Numel);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<ushort>, ArrayView<ushort>, ArrayView<ushort>,
            uint, uint, float>(AddDropoutForwardBFloat16Kernel);
        kernel(residual.Numel,
            residual.EnsureCudaBFloat16Buffer(deviceIndex).View,
            branch.EnsureCudaBFloat16Buffer(deviceIndex).View,
            output.View,
            seed,
            dropThreshold,
            scale);
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
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var output = Tensor.RentCudaBFloat16Buffer(deviceIndex, input.Numel);
        var normalized = Tensor.RentCudaFloatBuffer(deviceIndex, input.Numel);
        var inverses = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<ushort>, ArrayView<ushort>, ArrayView<ushort>,
            ArrayView<ushort>, ArrayView<float>, ArrayView<float>, int, float>(
                LayerNormForwardBFloat16Kernel);
        kernel(rows,
            input.EnsureCudaBFloat16Buffer(deviceIndex).View,
            gamma.EnsureCudaBFloat16Buffer(deviceIndex).View,
            beta.EnsureCudaBFloat16Buffer(deviceIndex).View,
            output.View,
            normalized.View,
            inverses.View,
            columns,
            epsilon);
        return new BFloat16LayerNormResidentContext(
            output,
            normalized,
            inverses,
            accelerator);
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
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var labelsBuffer = Tensor.RentCudaIntBuffer(deviceIndex, labels);
        const int lanes = 32;
        var partialMaxima = Tensor.RentCudaFloatBuffer(
            deviceIndex,
            checked(rows * lanes));
        var partialSums = Tensor.RentCudaFloatBuffer(
            deviceIndex,
            checked(rows * lanes));
        var maxima = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
        var inverseSums = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
        var loss = Tensor.RentCudaFloatBuffer(deviceIndex, 1);
        loss.MemSetToZero();
        var statsKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<ushort>, ArrayView<float>, ArrayView<float>,
            int, int>(CrossEntropyPartialStatsBFloat16Kernel);
        statsKernel(checked(rows * lanes),
            logits.EnsureCudaBFloat16Buffer(deviceIndex).View,
            partialMaxima.View,
            partialSums.View,
            columns,
            lanes);
        var reduceKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(
                CrossEntropyReduceStatsKernel);
        reduceKernel(rows,
            partialMaxima.View,
            partialSums.View,
            maxima.View,
            lanes);
        var exponentialKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<ushort>, ArrayView<float>, ArrayView<float>,
            int, int>(CrossEntropyPartialExponentialBFloat16Kernel);
        exponentialKernel(checked(rows * lanes),
            logits.EnsureCudaBFloat16Buffer(deviceIndex).View,
            maxima.View,
            partialMaxima.View,
            columns,
            lanes);
        var finalizeKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<ushort>, ArrayView<int>, ArrayView<float>,
            ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, int, int, int, int, float>(
                CrossEntropyFinalizeBFloat16Kernel);
        finalizeKernel(rows,
            logits.EnsureCudaBFloat16Buffer(deviceIndex).View,
            labelsBuffer.View,
            partialMaxima.View,
            partialSums.View,
            maxima.View,
            inverseSums.View,
            loss.View,
            columns,
            lanes,
            ignoreIndex,
            validRows,
            labelSmoothing);
        Tensor.ReturnCudaFloatBuffer(accelerator, partialMaxima);
        Tensor.ReturnCudaFloatBuffer(accelerator, partialSums);
        return new CrossEntropyResidentContext(
            loss,
            maxima,
            inverseSums,
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
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<ushort>, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<float>, ArrayView<float>, int, int,
            int, float>(CrossEntropyBackwardBFloat16Kernel);
        kernel(logits.Numel,
            logits.EnsureCudaBFloat16Buffer(deviceIndex).View,
            context.Maxima.View,
            context.InverseSums.View,
            context.Labels.View,
            logits.EnsureCudaGradientBuffer(deviceIndex).View,
            loss.EnsureCudaGradientBuffer(deviceIndex).View,
            columns,
            ignoreIndex,
            validRows,
            labelSmoothing);
        logits.MarkCudaGradientMutated(deviceIndex);
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
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var outputGradient = output.EnsureCudaGradientBuffer(deviceIndex);
        var inputGradient = input.EnsureCudaGradientBuffer(deviceIndex);
        var gammaGradient = gamma.EnsureCudaGradientBuffer(deviceIndex);
        var betaGradient = beta.EnsureCudaGradientBuffer(deviceIndex);
        var inputKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<ushort>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, ArrayView<float>, int>(
                LayerNormBackwardInputBFloat16Kernel);
        inputKernel(rows,
            gamma.EnsureCudaBFloat16Buffer(deviceIndex).View,
            context.Normalized.View,
            context.Inverses.View,
            outputGradient.View,
            inputGradient.View,
            columns);
        var parameterKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, int, int>(LayerNormBackwardParameterKernel);
        parameterKernel(columns,
            context.Normalized.View,
            outputGradient.View,
            gammaGradient.View,
            betaGradient.View,
            rows,
            columns);
        input.MarkCudaGradientMutated(deviceIndex);
        gamma.MarkCudaGradientMutated(deviceIndex);
        beta.MarkCudaGradientMutated(deviceIndex);
    }

    internal sealed class BFloat16LayerNormResidentContext(
        MemoryBuffer1D<ushort, Stride1D.Dense> output,
        MemoryBuffer1D<float, Stride1D.Dense> normalized,
        MemoryBuffer1D<float, Stride1D.Dense> inverses,
        CudaAccelerator accelerator) : IDisposable
    {
        private int _disposed;
        internal MemoryBuffer1D<ushort, Stride1D.Dense> Output { get; } = output;
        internal MemoryBuffer1D<float, Stride1D.Dense> Normalized { get; } = normalized;
        internal MemoryBuffer1D<float, Stride1D.Dense> Inverses { get; } = inverses;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            Tensor.ReturnCudaFloatBuffer(accelerator, Normalized);
            Tensor.ReturnCudaFloatBuffer(accelerator, Inverses);
            GC.SuppressFinalize(this);
        }
    }

    internal static MemoryBuffer1D<ushort, Stride1D.Dense>
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
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var inputBuffer = input.EnsureCudaBFloat16Buffer(deviceIndex);
        var weightBuffer = weight.EnsureCudaBFloat16Buffer(deviceIndex);
        var biasBuffer = bias.EnsureCudaBFloat16Buffer(deviceIndex);
        var output = Tensor.RentCudaBFloat16Buffer(
            deviceIndex,
            checked(rows * outputWidth));
        CudaBlas.LinearForwardBFloat16(
            accelerator,
            deviceIndex,
            inputBuffer,
            weightBuffer,
            output,
            rows,
            inputWidth,
            outputWidth);
        var biasKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<ushort>, ArrayView<ushort>, int, int>(
                BFloat16LinearBiasKernel);
        biasKernel(
            checked(rows * outputWidth),
            output.View,
            biasBuffer.View,
            outputWidth,
            applyRelu ? 1 : 0);
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
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var outputGradient = output.EnsureCudaGradientBuffer(deviceIndex);
        var encodedGradient = Tensor.RentCudaBFloat16Buffer(
            deviceIndex,
            checked(rows * outputWidth));
        var encodeKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<ushort>, ArrayView<ushort>, int>(
                EncodeLinearGradientBFloat16Kernel);
        encodeKernel(
            checked(rows * outputWidth),
            outputGradient.View,
            encodedGradient.View,
            output.EnsureCudaBFloat16Buffer(deviceIndex).View,
            applyRelu ? 1 : 0);

        var inputGradient = input.EnsureCudaGradientBuffer(deviceIndex);
        var weightGradient = weight.EnsureCudaGradientBuffer(deviceIndex);
        var biasGradient = bias.EnsureCudaGradientBuffer(deviceIndex);
        CudaBlas.LinearBackwardInputBFloat16(
            accelerator,
            deviceIndex,
            encodedGradient,
            weight.EnsureCudaBFloat16Buffer(deviceIndex),
            inputGradient,
            rows,
            inputWidth,
            outputWidth);
        CudaBlas.LinearBackwardWeightBFloat16(
            accelerator,
            deviceIndex,
            input.EnsureCudaBFloat16Buffer(deviceIndex),
            encodedGradient,
            weightGradient,
            rows,
            inputWidth,
            outputWidth);
        var biasKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<ushort>, ArrayView<float>, int, int>(
                BFloat16LinearBackwardBiasKernel);
        biasKernel(
            outputWidth,
            encodedGradient.View,
            biasGradient.View,
            rows,
            outputWidth);
        Tensor.ReturnCudaBFloat16Buffer(accelerator, encodedGradient);
        input.MarkCudaGradientMutated(deviceIndex);
        weight.MarkCudaGradientMutated(deviceIndex);
        bias.MarkCudaGradientMutated(deviceIndex);
    }

    private static void BFloat16LinearBiasKernel(
        Index1D index,
        ArrayView<ushort> output,
        ArrayView<ushort> bias,
        int outputWidth,
        int applyRelu)
    {
        int i = index;
        float value = DecodeBFloat16(output[i])
            + DecodeBFloat16(bias[i % outputWidth]);
        if (applyRelu != 0 && value < 0f)
            value = 0f;
        output[i] = EncodeBFloat16(value);
    }

    private static void AddForwardBFloat16Kernel(
        Index1D index,
        ArrayView<ushort> left,
        ArrayView<ushort> right,
        ArrayView<ushort> output)
    {
        int i = index;
        output[i] = EncodeBFloat16(
            DecodeBFloat16(left[i]) + DecodeBFloat16(right[i]));
    }

    private static void EmbeddingPositionsForwardBFloat16Kernel(
        Index1D index,
        ArrayView<ushort> tokens,
        ArrayView<ushort> positions,
        ArrayView<int> indices,
        ArrayView<ushort> output,
        int sequenceLength,
        int width)
    {
        int i = index;
        int position = i / width;
        int column = i - position * width;
        int token = indices[position];
        output[i] = EncodeBFloat16(
            DecodeBFloat16(tokens[token * width + column])
            + DecodeBFloat16(
                positions[(position % sequenceLength) * width + column]));
    }

    private static void EmbeddingForwardBFloat16Kernel(
        Index1D index,
        ArrayView<ushort> table,
        ArrayView<int> indices,
        ArrayView<ushort> output,
        int width)
    {
        int linear = index;
        int position = linear / width;
        int column = linear - position * width;
        output[linear] = table[indices[position] * width + column];
    }

    private static void DropoutForwardBFloat16Kernel(
        Index1D index,
        ArrayView<ushort> input,
        ArrayView<ushort> output,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        int i = index;
        output[i] = EncodeBFloat16(
            DecodeBFloat16(input[i])
            * DropoutMultiplier(seed, i, dropThreshold, scale));
    }

    private static void AddDropoutForwardBFloat16Kernel(
        Index1D index,
        ArrayView<ushort> residual,
        ArrayView<ushort> branch,
        ArrayView<ushort> output,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        int i = index;
        output[i] = EncodeBFloat16(
            DecodeBFloat16(residual[i])
            + DecodeBFloat16(branch[i])
                * DropoutMultiplier(seed, i, dropThreshold, scale));
    }

    private static void LayerNormForwardBFloat16Kernel(
        Index1D rowIndex,
        ArrayView<ushort> input,
        ArrayView<ushort> gamma,
        ArrayView<ushort> beta,
        ArrayView<ushort> output,
        ArrayView<float> normalized,
        ArrayView<float> inverses,
        int columns,
        float epsilon)
    {
        int row = rowIndex;
        int offset = row * columns;
        float mean = 0f;
        for (int column = 0; column < columns; column++)
            mean += DecodeBFloat16(input[offset + column]);
        mean /= columns;
        float variance = 0f;
        for (int column = 0; column < columns; column++)
        {
            float difference = DecodeBFloat16(input[offset + column]) - mean;
            variance += difference * difference;
        }
        float inverse = 1f / XMath.Sqrt(variance / columns + epsilon);
        inverses[row] = inverse;
        for (int column = 0; column < columns; column++)
        {
            float value = (DecodeBFloat16(input[offset + column]) - mean)
                * inverse;
            normalized[offset + column] = value;
            output[offset + column] = EncodeBFloat16(
                value * DecodeBFloat16(gamma[column])
                + DecodeBFloat16(beta[column]));
        }
    }

    private static void LayerNormBackwardInputBFloat16Kernel(
        Index1D rowIndex,
        ArrayView<ushort> gamma,
        ArrayView<float> normalized,
        ArrayView<float> inverses,
        ArrayView<float> outputGradient,
        ArrayView<float> inputGradient,
        int columns)
    {
        int row = rowIndex;
        int offset = row * columns;
        float sum = 0f;
        float normalizedSum = 0f;
        for (int column = 0; column < columns; column++)
        {
            float dxhat = outputGradient[offset + column]
                * DecodeBFloat16(gamma[column]);
            sum += dxhat;
            normalizedSum += dxhat * normalized[offset + column];
        }
        float scale = inverses[row] / columns;
        for (int column = 0; column < columns; column++)
        {
            float dxhat = outputGradient[offset + column]
                * DecodeBFloat16(gamma[column]);
            inputGradient[offset + column] += scale
                * (columns * dxhat - sum
                    - normalized[offset + column] * normalizedSum);
        }
    }

    private static void CrossEntropyPartialStatsBFloat16Kernel(
        Index1D workIndex,
        ArrayView<ushort> logits,
        ArrayView<float> partialMaxima,
        ArrayView<float> partialLogitSums,
        int columns,
        int lanes)
    {
        int work = workIndex;
        int lane = work % lanes;
        int row = work / lanes;
        int offset = row * columns;
        float maximum = float.NegativeInfinity;
        float sum = 0f;
        for (int column = lane; column < columns; column += lanes)
        {
            float value = DecodeBFloat16(logits[offset + column]);
            maximum = XMath.Max(maximum, value);
            sum += value;
        }
        partialMaxima[work] = maximum;
        partialLogitSums[work] = sum;
    }

    private static void CrossEntropyPartialExponentialBFloat16Kernel(
        Index1D workIndex,
        ArrayView<ushort> logits,
        ArrayView<float> maxima,
        ArrayView<float> partialExponentialSums,
        int columns,
        int lanes)
    {
        int work = workIndex;
        int lane = work % lanes;
        int row = work / lanes;
        int offset = row * columns;
        float sum = 0f;
        for (int column = lane; column < columns; column += lanes)
            sum += XMath.Exp(DecodeBFloat16(logits[offset + column]) - maxima[row]);
        partialExponentialSums[work] = sum;
    }

    private static void CrossEntropyFinalizeBFloat16Kernel(
        Index1D rowIndex,
        ArrayView<ushort> logits,
        ArrayView<int> labels,
        ArrayView<float> partialExponentialSums,
        ArrayView<float> partialLogitSums,
        ArrayView<float> maxima,
        ArrayView<float> inverseSums,
        ArrayView<float> loss,
        int columns,
        int lanes,
        int ignoreIndex,
        int validRows,
        float labelSmoothing)
    {
        int row = rowIndex;
        int partialOffset = row * lanes;
        float exponentialSum = 0f;
        float logitSum = 0f;
        for (int lane = 0; lane < lanes; lane++)
        {
            exponentialSum += partialExponentialSums[partialOffset + lane];
            logitSum += partialLogitSums[partialOffset + lane];
        }
        inverseSums[row] = 1f / exponentialSum;
        int label = labels[row];
        if (label == ignoreIndex)
            return;
        float normalizer = maxima[row] + XMath.Log(exponentialSum);
        float negativeLogLikelihood = normalizer
            - DecodeBFloat16(logits[row * columns + label]);
        float uniformLoss = normalizer - logitSum / columns;
        float rowLoss = (1f - labelSmoothing) * negativeLogLikelihood
            + labelSmoothing * uniformLoss;
        Atomic.Add(ref loss[0], rowLoss / validRows);
    }

    private static void CrossEntropyBackwardBFloat16Kernel(
        Index1D index,
        ArrayView<ushort> logits,
        ArrayView<float> maxima,
        ArrayView<float> inverseSums,
        ArrayView<int> labels,
        ArrayView<float> gradient,
        ArrayView<float> upstreamGradient,
        int columns,
        int ignoreIndex,
        int validRows,
        float labelSmoothing)
    {
        int linear = index;
        int row = linear / columns;
        int column = linear - row * columns;
        int label = labels[row];
        if (label == ignoreIndex)
            return;
        float probability = XMath.Exp(
            DecodeBFloat16(logits[linear]) - maxima[row]) * inverseSums[row];
        float target = labelSmoothing / columns;
        if (column == label)
            target += 1f - labelSmoothing;
        gradient[linear] += upstreamGradient[0]
            / validRows * (probability - target);
    }

    private static void EncodeLinearGradientBFloat16Kernel(
        Index1D index,
        ArrayView<float> gradient,
        ArrayView<ushort> encoded,
        ArrayView<ushort> output,
        int applyRelu)
    {
        int i = index;
        float value = gradient[i];
        if (applyRelu != 0 && DecodeBFloat16(output[i]) <= 0f)
            value = 0f;
        encoded[i] = EncodeBFloat16(value);
    }

    private static void BFloat16LinearBackwardBiasKernel(
        Index1D columnIndex,
        ArrayView<ushort> outputGradient,
        ArrayView<float> biasGradient,
        int rows,
        int outputWidth)
    {
        int column = columnIndex;
        float sum = 0f;
        for (int row = 0; row < rows; row++)
            sum += DecodeBFloat16(outputGradient[row * outputWidth + column]);
        biasGradient[column] += sum;
    }

    private static float DecodeBFloat16(ushort value)
        => Interop.IntAsFloat((uint)value << 16);

    private static ushort EncodeBFloat16(float value)
    {
        uint bits = Interop.FloatAsInt(value);
        uint roundingBias = 0x7FFFu + ((bits >> 16) & 1u);
        return (ushort)((bits + roundingBias) >> 16);
    }
}
