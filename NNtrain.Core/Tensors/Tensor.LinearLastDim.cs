using System.Buffers;

namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// Applies a dense projection to the last dimension without creating a
    /// flattened view tensor. All leading dimensions are preserved.
    /// </summary>
    internal Tensor LinearLastDim(
        Tensor weight,
        Tensor bias,
        bool applyRelu)
    {
        ArgumentNullException.ThrowIfNull(weight);
        ArgumentNullException.ThrowIfNull(bias);
        if (Rank < 2)
        {
            throw new InvalidOperationException(
                "LinearLastDim requires an input with rank 2 or greater.");
        }
        weight.CheckRank(2);
        bias.CheckRank(1);

        int inputWidth = _shape[^1];
        int outputWidth = weight._shape[0];
        if (weight._shape[1] != inputWidth
            || bias._shape[0] != outputWidth)
        {
            throw new ArgumentException(
                "LinearLastDim requires input [..., input], weight " +
                "[output, input], and bias [output].");
        }

        int rows = Numel / inputWidth;
        int outputLength = checked(rows * outputWidth);
        if (ExecutionDevice == TensorDevice.Cuda)
        {
            float[] inputValues = GetPhysicalFloat32ComputeCache();
            float[] outputValues = TensorCudaKernels.LinearForward(
                inputValues,
                weight,
                bias,
                rows,
                inputWidth,
                outputWidth,
                applyRelu,
                DType == TensorDType.BFloat16
                    && weight.DType == TensorDType.BFloat16
                    && bias.DType == TensorDType.BFloat16);
            float[] weightValues = weight.GetPhysicalFloat32ComputeCache();
            int[] cudaOutputShape = (int[])_shape.Clone();
            cudaOutputShape[^1] = outputWidth;
            var cudaResult = new Tensor(
                outputValues,
                cudaOutputShape,
                [this, weight, bias]);
            if (AutogradContext.IsRecordingEnabled)
            {
                float[] storedOutputValues =
                    cudaResult.GetPhysicalFloat32ComputeCache();
                cudaResult.Node.BackwardAction = () =>
                    TensorCudaKernels.LinearBackward(
                        inputValues,
                        weightValues,
                        storedOutputValues,
                        cudaResult._grad,
                        EnsureGradientBuffer(),
                        weight.EnsureGradientBuffer(),
                        bias.EnsureGradientBuffer(),
                        rows,
                        inputWidth,
                        outputWidth,
                        applyRelu);
            }
            return cudaResult;
        }
        // The managed Float16 fallback widens each activation only once for
        // this node. The native F16C path below instead keeps the physical
        // Half payload in cache and widens eight values per instruction.
        bool useOutputVectorization = weight.DType == TensorDType.Float32
            && CanUseTransposedRightKernel(inputWidth, outputWidth);
        float[]? transposedWeight = useOutputVectorization
            ? weight.GetTransposedData2D()
            : null;
        Half[]? nativeInput = null;
        Half[]? nativeWeight = null;
        Half[]? nativeBias = null;
        bool hasNativeFloat16Operands = !useOutputVectorization
            && SimdEnabled
            && IsFloat16NativeAccelerated
            && _data.TryGetFloat16Buffer(out nativeInput)
            && weight._data.TryGetFloat16Buffer(out nativeWeight)
            && bias._data.TryGetFloat16Buffer(out nativeBias);
        bool useNativeFloat16Forward = hasNativeFloat16Operands;
        float[]? output = useNativeFloat16Forward
            ? null
            : new float[outputLength];
        Half[]? nativeOutput = useNativeFloat16Forward
            ? new Half[outputLength]
            : null;
        float[]? float16WeightCache = !useNativeFloat16Forward
            && weight.DType == TensorDType.Float16
            ? weight.GetPhysicalFloat32ComputeCache()
            : null;
        float[]? float16BiasCache = !useNativeFloat16Forward
            && bias.DType == TensorDType.Float16
            ? bias.GetPhysicalFloat32ComputeCache()
            : null;
        float[]? decodedForwardInput = null;
        if (!useNativeFloat16Forward
            && DType == TensorDType.Float16
            && _data.TryGetFloat16Buffer(out Half[] forwardInputHalf))
        {
            decodedForwardInput = ArrayPool<float>.Shared.Rent(Numel);
            TensorStorageCodec.DecodeFloat16(
                forwardInputHalf,
                decodedForwardInput.AsSpan(0, Numel));
        }

        if (useNativeFloat16Forward)
        {
            try
            {
                int chunkCount = Math.Min(
                    rows,
                    Math.Max(1, EffectiveMaxDegreeOfParallelism * 4));
                int rowsPerChunk = (rows + chunkCount - 1) / chunkCount;
                RunParallel(0, chunkCount, chunk =>
                {
                    int rowStart = chunk * rowsPerChunk;
                    int rowCount = Math.Min(rowsPerChunk, rows - rowStart);
                    if (rowCount > 0)
                    {
                        TensorFloat16Native.LinearForwardRows(
                            nativeInput!,
                            nativeWeight!,
                            nativeBias!,
                            nativeOutput!,
                            rowStart,
                            rowCount,
                            inputWidth,
                            outputWidth,
                            applyRelu);
                    }
                });
            }
            catch (Exception exception) when (IsNativeDispatchFailure(exception))
            {
                TensorFloat16Native.DisableAfterFailure();
                useNativeFloat16Forward = false;
                output = new float[outputLength];
                nativeOutput = null;
                float16WeightCache = weight.DType == TensorDType.Float16
                    ? weight.GetPhysicalFloat32ComputeCache()
                    : null;
                float16BiasCache = bias.DType == TensorDType.Float16
                    ? bias.GetPhysicalFloat32ComputeCache()
                    : null;
                if (DType == TensorDType.Float16
                    && _data.TryGetFloat16Buffer(out Half[] fallbackInputHalf))
                {
                    decodedForwardInput = ArrayPool<float>.Shared.Rent(Numel);
                    TensorStorageCodec.DecodeFloat16(
                        fallbackInputHalf,
                        decodedForwardInput.AsSpan(0, Numel));
                }
            }
        }

        void ForwardRow(int row)
        {
            int inputOffset = row * inputWidth;
            int outputOffset = row * outputWidth;
            if (DType == TensorDType.BFloat16
                && weight.DType == TensorDType.BFloat16
                && bias.DType == TensorDType.BFloat16)
            {
                for (int column = 0; column < outputWidth; column++)
                {
                    float value = TensorStorageCodec.RoundToBFloat16Compute(
                        bias._data[column]);
                    int weightOffset = column * inputWidth;
                    for (int inner = 0; inner < inputWidth; inner++)
                    {
                        float product = TensorStorageCodec.RoundToBFloat16Compute(
                            _data[inputOffset + inner]
                            * weight._data[weightOffset + inner]);
                        value = TensorStorageCodec.RoundToBFloat16Compute(
                            value + product);
                    }
                    output![outputOffset + column] =
                        applyRelu && value <= 0f ? 0f : value;
                }
                return;
            }
            if (transposedWeight is not null)
            {
                bias._data.CopyRangeTo(
                    0,
                    output!.AsSpan(outputOffset, outputWidth));
                for (int inner = 0; inner < inputWidth; inner++)
                {
                    AddScaledValues(
                        output!,
                        outputOffset,
                        transposedWeight,
                        inner * outputWidth,
                        _data[inputOffset + inner],
                        outputWidth);
                }
                if (applyRelu)
                    ReluValuesInPlace(output!, outputOffset, outputWidth);
                return;
            }

            for (int column = 0; column < outputWidth; column++)
            {
                float value;
                if (float16WeightCache is not null)
                {
                    float biasValue = float16BiasCache is not null
                        ? float16BiasCache[column]
                        : bias._data[column];
                    value = biasValue + DotProduct(
                        decodedForwardInput ?? _data.GetMutableFloat32Buffer(),
                        inputOffset,
                        float16WeightCache,
                        column * inputWidth,
                        inputWidth);
                }
                else
                {
                    value = bias._data[column] + DotProduct(
                        _data,
                        inputOffset,
                        weight._data,
                        column * inputWidth,
                        inputWidth);
                }
                output![outputOffset + column] = applyRelu && value <= 0f
                    ? 0f
                    : value;
            }
        }

        try
        {
            if (!useNativeFloat16Forward)
            {
                RunBatches(
                    rows,
                    (long)inputWidth * outputWidth,
                    ForwardRow);
            }
        }
        finally
        {
            if (decodedForwardInput is not null)
                ArrayPool<float>.Shared.Return(decodedForwardInput);
        }

        int[] outputShape = (int[])_shape.Clone();
        outputShape[^1] = outputWidth;
        Tensor result = nativeOutput is not null
            ? FromFloat16Result(nativeOutput, outputShape, [this, weight, bias])
            : new Tensor(output!, outputShape, [this, weight, bias]);
        result.Node.BackwardAction = () =>
        {
            Half[]? nativeBackwardInput = null;
            Half[]? nativeBackwardWeight = null;
            Half[]? nativeBackwardOutput = null;
            bool useNativeFloat16Backward = SimdEnabled
                && IsFloat16NativeAccelerated
                && _data.TryGetFloat16Buffer(out nativeBackwardInput)
                && weight._data.TryGetFloat16Buffer(out nativeBackwardWeight)
                && result._data.TryGetFloat16Buffer(out nativeBackwardOutput);

            // The portable path accumulates in Float32, so it uses a physical
            // Float16 decode cache. Native F16C backward widens directly from
            // the compact payload and does not need this extra cache.
            float[]? backwardFloat16WeightCache = weight.DType == TensorDType.Float16
                && !useNativeFloat16Backward
                ? weight.GetPhysicalFloat32ComputeCache()
                : null;

            // Weight gradients reuse each activation row for every output
            // column. Decode Float16 activations once into an ArrayPool
            // buffer instead of widening them outputWidth times.
            float[]? decodedInput = null;
            if (!useNativeFloat16Backward
                && DType == TensorDType.Float16
                && _data.TryGetFloat16Buffer(out Half[] inputHalf))
            {
                decodedInput = ArrayPool<float>.Shared.Rent(Numel);
                TensorStorageCodec.DecodeFloat16(
                    inputHalf,
                    decodedInput.AsSpan(0, Numel));
            }

            try
            {
            void BackwardInputRow(int row)
            {
                int inputOffset = row * inputWidth;
                int outputOffset = row * outputWidth;
                if (transposedWeight is not null)
                {
                    for (int inner = 0; inner < inputWidth; inner++)
                    {
                        float contribution = applyRelu
                            ? DotProductMaskedByPositiveStoredMask(
                                result._grad,
                                outputOffset,
                                result._data,
                                outputOffset,
                                transposedWeight,
                                inner * outputWidth,
                                outputWidth)
                            : DotProduct(
                                result._grad,
                                outputOffset,
                                transposedWeight,
                                inner * outputWidth,
                                outputWidth);
                        _grad[inputOffset + inner] += contribution;
                    }
                    return;
                }

                for (int column = 0; column < outputWidth; column++)
                {
                    int outputIndex = outputOffset + column;
                    if (applyRelu && result._data[outputIndex] <= 0f)
                        continue;

                    if (backwardFloat16WeightCache is not null)
                    {
                        AddScaledValues(
                            _grad,
                            inputOffset,
                            backwardFloat16WeightCache,
                            column * inputWidth,
                            result._grad[outputIndex],
                            inputWidth);
                    }
                    else
                    {
                        AddScaledValues(
                            _grad,
                            inputOffset,
                            weight._data,
                            column * inputWidth,
                            result._grad[outputIndex],
                            inputWidth);
                    }
                }
            }

            if (useNativeFloat16Backward)
            {
                int chunkCount = Math.Min(
                    rows,
                    Math.Max(1, EffectiveMaxDegreeOfParallelism * 4));
                int rowsPerChunk = (rows + chunkCount - 1) / chunkCount;
                RunParallel(0, chunkCount, chunk =>
                {
                    int rowStart = chunk * rowsPerChunk;
                    int rowCount = Math.Min(rowsPerChunk, rows - rowStart);
                    if (rowCount > 0)
                    {
                        TensorFloat16Native.LinearBackwardInputRows(
                            result._grad,
                            nativeBackwardOutput!,
                            nativeBackwardWeight!,
                            _grad,
                            rowStart,
                            rowCount,
                            inputWidth,
                            outputWidth,
                            applyRelu);
                    }
                });
            }
            else
            {
                RunBatches(
                    rows,
                    (long)inputWidth * outputWidth,
                    BackwardInputRow);
            }

            void BackwardWeightRow(int column)
            {
                int weightOffset = column * inputWidth;
                float biasGradient = 0f;
                for (int row = 0; row < rows; row++)
                {
                    int outputIndex = row * outputWidth + column;
                    if (applyRelu && result._data[outputIndex] <= 0f)
                        continue;

                    float gradient = result._grad[outputIndex];
                    biasGradient += gradient;
                    if (decodedInput is not null)
                    {
                        AddScaledValues(
                            weight._grad,
                            weightOffset,
                            decodedInput,
                            row * inputWidth,
                            gradient,
                            inputWidth);
                    }
                    else
                    {
                        AddScaledValues(
                            weight._grad,
                            weightOffset,
                            _data,
                            row * inputWidth,
                            gradient,
                            inputWidth);
                    }
                }

                bias._grad[column] += biasGradient;
            }

            if (useNativeFloat16Backward)
            {
                int chunkCount = Math.Min(
                    outputWidth,
                    Math.Max(1, EffectiveMaxDegreeOfParallelism * 4));
                int columnsPerChunk = (outputWidth + chunkCount - 1) / chunkCount;
                RunParallel(0, chunkCount, chunk =>
                {
                    int columnStart = chunk * columnsPerChunk;
                    int columnCount = Math.Min(
                        columnsPerChunk,
                        outputWidth - columnStart);
                    if (columnCount > 0)
                    {
                        TensorFloat16Native.LinearBackwardWeightColumns(
                            nativeBackwardInput!,
                            result._grad,
                            nativeBackwardOutput!,
                            weight._grad,
                            bias._grad,
                            columnStart,
                            columnCount,
                            rows,
                            inputWidth,
                            outputWidth,
                            applyRelu);
                    }
                });
            }
            else
            {
                RunBatches(
                    outputWidth,
                    (long)rows * inputWidth,
                    BackwardWeightRow);
            }
            }
            finally
            {
                if (decodedInput is not null)
                    ArrayPool<float>.Shared.Return(decodedInput);
            }
        };

        return result;
    }

    private static bool IsNativeDispatchFailure(Exception exception)
        => exception is DllNotFoundException
            or EntryPointNotFoundException
            or BadImageFormatException;
}
