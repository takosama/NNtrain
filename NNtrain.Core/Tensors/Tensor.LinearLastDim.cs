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
        bool bfloat16MatrixOperands = DType == TensorDType.BFloat16
            && weight.DType == TensorDType.BFloat16
            && bias.DType == TensorDType.BFloat16;
        bool bfp8MatrixOperands = DType == TensorDType.Bfp8
            && weight.DType == TensorDType.Bfp8
            && bias.DType == TensorDType.Bfp8;
        if (ExecutionDevice == TensorDevice.Cuda)
        {
            bool bfloat16Compute = bfloat16MatrixOperands;
            string forwardOperation = CudaOperationProfiler.IsEnabled
                ? $"forward.{(applyRelu ? "linear_relu" : "linear")}" +
                    $"[{inputWidth}->{outputWidth}]"
                : applyRelu ? "forward.linear_relu" : "forward.linear";
            int[] cudaOutputShape = (int[])_shape.Clone();
            cudaOutputShape[^1] = outputWidth;
            if (bfp8MatrixOperands)
            {
                Bfp8QuantizationDescriptor outputDescriptor =
                    SelectBfp8ResultDescriptor(this, weight, bias);
                using CudaBfp8OwnedBuffers bfp8Output =
                    CudaOperationProfiler.IsEnabled
                        ? CudaOperationProfiler.Measure(
                            forwardOperation,
                            () => CudaBfp8Gemm.LinearForward(
                                this,
                                weight,
                                bias,
                                outputDescriptor,
                                rows,
                                inputWidth,
                                outputWidth,
                                applyRelu))
                        : CudaBfp8Gemm.LinearForward(
                            this,
                            weight,
                            bias,
                            outputDescriptor,
                            rows,
                            inputWidth,
                            outputWidth,
                            applyRelu);
                Tensor bfp8Result = FromCudaBfp8Result(
                    bfp8Output,
                    CudaDeviceIndex,
                    cudaOutputShape,
                    [this, weight, bias]);
                if (AutogradContext.IsRecordingEnabled)
                {
                    string backwardOperation = CudaOperationProfiler.IsEnabled
                        ? $"backward.{(applyRelu ? "linear_relu" : "linear")}" +
                            $"[{inputWidth}->{outputWidth}]"
                        : applyRelu ? "backward.linear_relu" : "backward.linear";
                    bfp8Result.Node.BackwardAction = () =>
                    {
                        void Backward() => CudaBfp8Gemm.LinearBackward(
                            this,
                            weight,
                            bias,
                            bfp8Result,
                            rows,
                            inputWidth,
                            outputWidth,
                            applyRelu);
                        if (CudaOperationProfiler.IsEnabled)
                            CudaOperationProfiler.Measure(backwardOperation, Backward);
                        else
                            Backward();
                    };
                }
                return bfp8Result;
            }
            if (bfloat16Compute)
            {
                var bfloat16Output = CudaOperationProfiler.IsEnabled
                    ? CudaOperationProfiler.Measure(
                        forwardOperation,
                        () => TensorCudaKernels.LinearForwardBFloat16Resident(
                            this,
                            weight,
                            bias,
                            rows,
                            inputWidth,
                            outputWidth,
                            applyRelu))
                    : TensorCudaKernels.LinearForwardBFloat16Resident(
                        this,
                        weight,
                        bias,
                        rows,
                        inputWidth,
                        outputWidth,
                        applyRelu);
                Tensor bfloat16Result = FromCudaResult(
                    bfloat16Output,
                    CudaDeviceIndex,
                    cudaOutputShape,
                    [this, weight, bias],
                    TensorDType.BFloat16);
                if (AutogradContext.IsRecordingEnabled)
                {
                    string backwardOperation = CudaOperationProfiler.IsEnabled
                        ? $"backward.{(applyRelu ? "linear_relu" : "linear")}" +
                            $"[{inputWidth}->{outputWidth}]"
                        : applyRelu ? "backward.linear_relu" : "backward.linear";
                    bfloat16Result.Node.BackwardAction = () =>
                    {
                        if (CudaOperationProfiler.IsEnabled)
                        {
                            CudaOperationProfiler.Measure(
                                backwardOperation,
                                () => TensorCudaKernels.LinearBackwardBFloat16Resident(
                                    this,
                                    weight,
                                    bias,
                                    bfloat16Result,
                                    rows,
                                    inputWidth,
                                    outputWidth,
                                    applyRelu));
                        }
                        else
                        {
                            TensorCudaKernels.LinearBackwardBFloat16Resident(
                                this,
                                weight,
                                bias,
                                bfloat16Result,
                                rows,
                                inputWidth,
                                outputWidth,
                                applyRelu);
                        }
                    };
                }
                return bfloat16Result;
            }
            var outputBuffer = CudaOperationProfiler.IsEnabled
                ? CudaOperationProfiler.Measure(
                    forwardOperation,
                    () => TensorCudaKernels.LinearForwardResident(
                        this,
                        weight,
                        bias,
                        rows,
                        inputWidth,
                        outputWidth,
                        applyRelu,
                        bfloat16Compute))
                : TensorCudaKernels.LinearForwardResident(
                    this,
                    weight,
                    bias,
                    rows,
                    inputWidth,
                    outputWidth,
                    applyRelu,
                    bfloat16Compute);
            Tensor cudaResult = FromCudaResult(
                outputBuffer,
                CudaDeviceIndex,
                cudaOutputShape,
                [this, weight, bias]);
            if (AutogradContext.IsRecordingEnabled)
            {
                string backwardOperation = CudaOperationProfiler.IsEnabled
                    ? $"backward.{(applyRelu ? "linear_relu" : "linear")}" +
                        $"[{inputWidth}->{outputWidth}]"
                    : applyRelu ? "backward.linear_relu" : "backward.linear";
                cudaResult.Node.BackwardAction = () =>
                {
                    if (CudaOperationProfiler.IsEnabled)
                    {
                        CudaOperationProfiler.Measure(
                            backwardOperation,
                            () => TensorCudaKernels.LinearBackwardResident(
                                this,
                                weight,
                                bias,
                                cudaResult,
                                rows,
                                inputWidth,
                                outputWidth,
                                applyRelu,
                                bfloat16Compute));
                    }
                    else
                    {
                        TensorCudaKernels.LinearBackwardResident(
                            this,
                            weight,
                            bias,
                            cudaResult,
                            rows,
                            inputWidth,
                            outputWidth,
                            applyRelu,
                            bfloat16Compute);
                    }
                };
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
                    float value = bias._data[column];
                    int weightOffset = column * inputWidth;
                    for (int inner = 0; inner < inputWidth; inner++)
                        value += _data[inputOffset + inner]
                            * weight._data[weightOffset + inner];
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
                bool roundInputGradient = bfloat16MatrixOperands
                    && TensorExecutionContext.ActivePrecisionPolicy is null;
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
                    if (roundInputGradient)
                    {
                        for (int inner = 0; inner < inputWidth; inner++)
                        {
                            _grad[inputOffset + inner] =
                                TensorStorageCodec.RoundToBFloat16(
                                    _grad[inputOffset + inner]);
                        }
                    }
                    return;
                }

                for (int column = 0; column < outputWidth; column++)
                {
                    int outputIndex = outputOffset + column;
                    if (applyRelu && result._data[outputIndex] <= 0f)
                        continue;

                    float outputGradient = bfloat16MatrixOperands
                        ? TensorStorageCodec.RoundToBFloat16(
                            result._grad[outputIndex])
                        : result._grad[outputIndex];

                    if (backwardFloat16WeightCache is not null)
                    {
                        AddScaledValues(
                            _grad,
                            inputOffset,
                            backwardFloat16WeightCache,
                            column * inputWidth,
                            outputGradient,
                            inputWidth);
                    }
                    else
                    {
                        AddScaledValues(
                            _grad,
                            inputOffset,
                            weight._data,
                            column * inputWidth,
                            outputGradient,
                            inputWidth);
                    }
                }
                if (roundInputGradient)
                {
                    for (int inner = 0; inner < inputWidth; inner++)
                    {
                        _grad[inputOffset + inner] =
                            TensorStorageCodec.RoundToBFloat16(
                                _grad[inputOffset + inner]);
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

                    float gradient = bfloat16MatrixOperands
                        ? TensorStorageCodec.RoundToBFloat16(
                            result._grad[outputIndex])
                        : result._grad[outputIndex];
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
