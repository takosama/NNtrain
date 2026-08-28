namespace NNtrain;

partial class Tensor
{
    public Tensor Sin()
    {
        if (ExecutionDevice == TensorDevice.Cuda
            && DType is TensorDType.Float32
                or TensorDType.BFloat16
                or TensorDType.Bfp8)
        {
            return ApplyUnaryCuda(CudaPublicUnaryOperation.Sin);
        }
        ThrowIfCudaHostFallback(nameof(Sin));
        var output = new float[Numel];
        for (int index = 0; index < Numel; index++)
            output[index] = MathF.Sin(_data[index]);

        var result = new Tensor(output, _shape, [this]);
        result.Node.BackwardAction = () =>
        {
            for (int index = 0; index < Numel; index++)
            {
                _grad[index] +=
                    MathF.Cos(_data[index]) * result._grad[index];
            }
        };
        return result;
    }

    public Tensor Relu()
    {
        if (ExecutionDevice == TensorDevice.Cuda
            && DType is TensorDType.Float32
                or TensorDType.BFloat16
                or TensorDType.Bfp8)
        {
            return ApplyUnaryCuda(CudaPublicUnaryOperation.Relu);
        }
        ThrowIfCudaHostFallback(nameof(Relu));
        float[] y = new float[Numel];
        int i = 0;
        if (CanUseSimd(Numel))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = Numel - Numel % vectorWidth;
            Vector256<float> zero = Vector256<float>.Zero;

            for (; i < vectorizedLength; i += vectorWidth)
            {
                Vector256<float> value = LoadVector256(_data, i);
                StoreVector256(
                    Vector256.ConditionalSelect(
                        Vector256.GreaterThan(value, zero),
                        value,
                        zero),
                    y,
                    i);
            }
        }

        for (; i < Numel; i++)
            y[i] = _data[i] > 0f ? _data[i] : 0f;

        var t = new Tensor(y, _shape, new[] { this });

        t.Node.BackwardAction = () =>
        {
            int index = 0;
            if (CanUseSimd(Numel))
            {
                int vectorWidth = Vector256<float>.Count;
                int vectorizedLength = Numel - Numel % vectorWidth;
                Vector256<float> zero = Vector256<float>.Zero;

                for (; index < vectorizedLength; index += vectorWidth)
                {
                    Vector256<float> value = LoadVector256(
                        _data,
                        index);
                    Vector256<float> gradient = LoadVector256(
                        t._grad,
                        index);
                    Vector256<float> contribution =
                        Vector256.ConditionalSelect(
                            Vector256.GreaterThan(value, zero),
                            gradient,
                            zero);
                    StoreVector256(
                        LoadVector256(_grad, index) + contribution,
                        _grad,
                        index);
                }
            }

            for (; index < Numel; index++)
            {
                _grad[index] +=
                    (_data[index] > 0f ? 1f : 0f) * t._grad[index];
            }
        };

        return t;
    }

    /// <summary>Applies the tanh-approximated Gaussian error linear unit.</summary>
    public Tensor Gelu()
    {
        if (ExecutionDevice == TensorDevice.Cuda
            && DType is TensorDType.Float32
                or TensorDType.BFloat16
                or TensorDType.Bfp8)
        {
            return ApplyUnaryCuda(CudaPublicUnaryOperation.Gelu);
        }
        ThrowIfCudaHostFallback(nameof(Gelu));
        const float alpha = 0.7978845608028654f;
        const float beta = 0.044715f;
        var output = new float[Numel];
        for (int index = 0; index < Numel; index++)
        {
            float value = _data[index];
            float inner = alpha * (value + beta * value * value * value);
            output[index] = 0.5f * value * (1f + MathF.Tanh(inner));
        }
        var result = new Tensor(output, _shape, [this]);
        result.Node.BackwardAction = () =>
        {
            for (int index = 0; index < Numel; index++)
            {
                float value = _data[index];
                float square = value * value;
                float inner = alpha * (value + beta * value * square);
                float tanh = MathF.Tanh(inner);
                float derivative = 0.5f * (1f + tanh)
                    + 0.5f * value * (1f - tanh * tanh)
                        * alpha * (1f + 3f * beta * square);
                _grad[index] += derivative * result._grad[index];
            }
        };
        return result;
    }

    public Tensor Tanh()
        => ApplyElementaryUnary(
            CudaPublicUnaryOperation.Tanh,
            static value => MathF.Tanh(value),
            static (input, output) => 1f - output * output,
            nameof(Tanh));

    public Tensor Exp()
        => ApplyElementaryUnary(
            CudaPublicUnaryOperation.Exp,
            static value => MathF.Exp(value),
            static (_, output) => output,
            nameof(Exp));

    public Tensor Log()
        => ApplyElementaryUnary(
            CudaPublicUnaryOperation.Log,
            static value => MathF.Log(value),
            static (input, _) => 1f / input,
            nameof(Log));

    private Tensor ApplyElementaryUnary(
        CudaPublicUnaryOperation operation,
        Func<float, float> forward,
        Func<float, float, float> derivative,
        string operationName)
    {
        if (ExecutionDevice == TensorDevice.Cuda
            && DType is TensorDType.Float32
                or TensorDType.BFloat16
                or TensorDType.Bfp8)
        {
            return ApplyUnaryCuda(operation);
        }
        ThrowIfCudaHostFallback(operationName);
        var output = new float[Numel];
        for (int index = 0; index < Numel; index++)
            output[index] = forward(_data[index]);
        var result = new Tensor(output, _shape, [this]);
        result.Node.BackwardAction = () =>
        {
            for (int index = 0; index < Numel; index++)
            {
                _grad[index] += derivative(_data[index], output[index])
                    * result._grad[index];
            }
        };
        return result;
    }
    public Tensor AddRowWise(Tensor rowVec)
    {
        ArgumentNullException.ThrowIfNull(rowVec);
        CheckRank(2);
        rowVec.CheckRank(1);

        int rows = _shape[0];
        int cols = _shape[1];

        if (rowVec._shape[0] != cols)
            throw ShapeMismatch(this, rowVec, "Row-wise addition");

        if (ExecutionDevice == TensorDevice.Cuda
            && DType == rowVec.DType
            && DType is TensorDType.Float32
                or TensorDType.BFloat16
                or TensorDType.Bfp8)
        {
            return BroadcastAddCuda(rowVec, cols);
        }

        ThrowIfCudaHostFallback(nameof(AddRowWise));

        float[] y = new float[Numel];
        for (int r = 0; r < rows; r++)
        {
            int rowOffset = r * cols;
            int c = 0;
            if (CanUseSimd(cols))
            {
                int vectorWidth = Vector256<float>.Count;
                int vectorizedLength = cols - cols % vectorWidth;
                for (; c < vectorizedLength; c += vectorWidth)
                {
                    StoreVector256(
                        LoadVector256(_data, rowOffset + c)
                            + LoadVector256(rowVec._data, c),
                        y,
                        rowOffset + c);
                }
            }

            for (; c < cols; c++)
            {
                y[rowOffset + c] =
                    _data[rowOffset + c] + rowVec._data[c];
            }
        }

        var t = new Tensor(y, _shape, new[] { this, rowVec });

        t.Node.BackwardAction = () =>
        {
            for (int r = 0; r < rows; r++)
            {
                int rowOffset = r * cols;
                AddScaledValues(
                    _grad,
                    rowOffset,
                    t._grad,
                    rowOffset,
                    1f,
                    cols);
                AddScaledValues(
                    rowVec._grad,
                    0,
                    t._grad,
                    rowOffset,
                    1f,
                    cols);
            }
        };

        return t;
    }

    public Tensor SoftmaxLastDim()
    {
        if (ExecutionDevice == TensorDevice.Cuda
            && DType is TensorDType.Float32
                or TensorDType.BFloat16
                or TensorDType.Bfp8)
        {
            return SoftmaxCuda(logSoftmax: false);
        }
        ThrowIfCudaHostFallback(nameof(SoftmaxLastDim));
        if (Rank == 1)
        {
            int n = _shape[0];
            float max = MaxValues(_data, 0, n);

            float[] y = new float[n];
            float sum = ExpShiftedValues(_data, 0, max, y, 0, n);

            MultiplyValues(y, 0, 1f / sum, y, 0, n);

            var t = new Tensor(y, _shape, new[] { this });

            t.Node.BackwardAction = () =>
            {
                float dot = DotProduct(t._grad, 0, t._data, 0, n);
                AccumulateSoftmaxGradient(
                    _grad,
                    0,
                    t._data,
                    0,
                    t._grad,
                    0,
                    n,
                    dot);
            };

            return t;
        }

        if (Rank >= 2)
        {
            int cols = _shape[^1];
            int rows = Numel / cols;
            float[] y = new float[Numel];

            void ForwardRow(int r)
            {
                int rowOffset = r * cols;
                float max = MaxValues(_data, rowOffset, cols);
                float sum = ExpShiftedValues(
                    _data,
                    rowOffset,
                    max,
                    y,
                    rowOffset,
                    cols);

                MultiplyValues(
                    y,
                    rowOffset,
                    1f / sum,
                    y,
                    rowOffset,
                    cols);
            }

            RunBatches(rows, cols, ForwardRow);

            var t = new Tensor(y, _shape, new[] { this });

            t.Node.BackwardAction = () =>
            {
                void BackwardRow(int r)
                {
                    int rowOffset = r * cols;
                    float dot = DotProduct(
                        t._grad,
                        rowOffset,
                        t._data,
                        rowOffset,
                        cols);
                    AccumulateSoftmaxGradient(
                        _grad,
                        rowOffset,
                        t._data,
                        rowOffset,
                        t._grad,
                        rowOffset,
                        cols,
                        dot);
                }

                RunBatches(rows, cols, BackwardRow);
            };

            return t;
        }

        throw new NotSupportedException(
            "SoftmaxLastDim requires a tensor with at least one dimension.");
    }

    public Tensor LogSoftmaxLastDim()
    {
        if (ExecutionDevice == TensorDevice.Cuda
            && DType is TensorDType.Float32
                or TensorDType.BFloat16
                or TensorDType.Bfp8)
        {
            return SoftmaxCuda(logSoftmax: true);
        }
        ThrowIfCudaHostFallback(nameof(LogSoftmaxLastDim));
        if (Rank == 1)
        {
            int n = _shape[0];

            float max = MaxValues(_data, 0, n);

            float[] y = new float[n];
            float[] softmax = new float[n];
            float sumExp = ExpShiftedValues(
                _data,
                0,
                max,
                softmax,
                0,
                n);
            float logSumExpOfShiftedValues = MathF.Log(sumExp);
            SubtractShiftAndScalarValues(
                _data,
                0,
                max,
                logSumExpOfShiftedValues,
                y,
                0,
                n);
            MultiplyValues(softmax, 0, 1f / sumExp, softmax, 0, n);

            var t = new Tensor(y, _shape, new[] { this });

            t.Node.BackwardAction = () =>
            {
                float gradSum = SumValues(t._grad, 0, n);
                AccumulateLogSoftmaxGradient(
                    _grad,
                    0,
                    softmax,
                    0,
                    t._grad,
                    0,
                    n,
                    gradSum);
            };

            return t;
        }

        if (Rank >= 2)
        {
            int cols = _shape[^1];
            int rows = Numel / cols;

            float[] y = new float[Numel];
            float[] softmax = new float[Numel];

            void ForwardRow(int r)
            {
                int rowOffset = r * cols;
                float max = MaxValues(_data, rowOffset, cols);
                float sumExp = ExpShiftedValues(
                    _data,
                    rowOffset,
                    max,
                    softmax,
                    rowOffset,
                    cols);
                float logSumExpOfShiftedValues = MathF.Log(sumExp);
                SubtractShiftAndScalarValues(
                    _data,
                    rowOffset,
                    max,
                    logSumExpOfShiftedValues,
                    y,
                    rowOffset,
                    cols);
                MultiplyValues(
                    softmax,
                    rowOffset,
                    1f / sumExp,
                    softmax,
                    rowOffset,
                    cols);
            }

            RunBatches(rows, cols, ForwardRow);

            var t = new Tensor(y, _shape, new[] { this });

            t.Node.BackwardAction = () =>
            {
                void BackwardRow(int r)
                {
                    int rowOffset = r * cols;
                    float gradSum = SumValues(
                        t._grad,
                        rowOffset,
                        cols);
                    AccumulateLogSoftmaxGradient(
                        _grad,
                        rowOffset,
                        softmax,
                        rowOffset,
                        t._grad,
                        rowOffset,
                        cols,
                        gradSum);
                }

                RunBatches(rows, cols, BackwardRow);
            };

            return t;
        }

        throw new NotSupportedException(
            "LogSoftmaxLastDim requires a tensor with at least one dimension.");
    }
    public Tensor LayerNormLastDim(Tensor gamma, Tensor beta, float eps = 1e-5f)
    {
        ArgumentNullException.ThrowIfNull(gamma);
        ArgumentNullException.ThrowIfNull(beta);
        if (eps <= 0f || !float.IsFinite(eps))
            throw new ArgumentOutOfRangeException(nameof(eps), eps, "Epsilon must be finite and positive.");

        gamma.CheckRank(1);
        beta.CheckRank(1);

        if (ExecutionDevice == TensorDevice.Cuda)
        {
            int columns = _shape[^1];
            if (gamma._shape[0] != columns || beta._shape[0] != columns)
            {
                throw new ArgumentException(
                    $"LayerNorm parameters must have shape [{columns}], " +
                    $"but gamma is {ShapeText(gamma)} and beta is " +
                    $"{ShapeText(beta)}.");
            }
            int rows = Numel / columns;
            bool anyBfp8 = DType == TensorDType.Bfp8
                || gamma.DType == TensorDType.Bfp8
                || beta.DType == TensorDType.Bfp8;
            if (anyBfp8)
            {
                if (!AutogradContext.IsRecordingEnabled
                    && CudaBfp8InferenceComputeScope.IsActive
                    && DType == TensorDType.BFloat16
                    && gamma.DType == TensorDType.Bfp8
                    && beta.DType == TensorDType.Bfp8)
                {
                    TensorCudaKernels.BFloat16LayerNormResidentContext
                        mixedContext = CudaOperationProfiler.IsEnabled
                            ? CudaOperationProfiler.Measure(
                                "forward.layer_norm",
                                () => TensorCudaKernels
                                    .LayerNormForwardBFloat16ActivationBfp8ParametersInference(
                                        this,
                                        gamma,
                                        beta,
                                        rows,
                                        columns,
                                        eps))
                            : TensorCudaKernels
                                .LayerNormForwardBFloat16ActivationBfp8ParametersInference(
                                    this,
                                    gamma,
                                    beta,
                                    rows,
                                    columns,
                                    eps);
                    Tensor mixedResult = FromCudaResult(
                        mixedContext.Output,
                        CudaDeviceIndex,
                        _shape,
                        [this, gamma, beta],
                        TensorDType.BFloat16);
                    if (!CudaInferenceScope.TrackResource(mixedContext))
                        mixedContext.Dispose();
                    return mixedResult;
                }
                if (DType != TensorDType.Bfp8
                    || gamma.DType != TensorDType.Bfp8
                    || beta.DType != TensorDType.Bfp8)
                {
                    throw new InvalidOperationException(
                        "CUDA BFP8 LayerNorm requires input, gamma, and beta " +
                        "to use BFP8 storage; implicit host fallback is forbidden.");
                }
                return LayerNormLastDimBfp8Cuda(
                    gamma, beta, rows, columns, eps);
            }
            if (DType == TensorDType.BFloat16
                && gamma.DType == TensorDType.BFloat16
                && beta.DType == TensorDType.BFloat16)
            {
                TensorCudaKernels.BFloat16LayerNormResidentContext
                    bfloat16Context = CudaOperationProfiler.IsEnabled
                        ? CudaOperationProfiler.Measure(
                            "forward.layer_norm",
                            () => TensorCudaKernels
                                .LayerNormForwardBFloat16Resident(
                                    this,
                                    gamma,
                                    beta,
                                    rows,
                                    columns,
                                    eps))
                        : TensorCudaKernels.LayerNormForwardBFloat16Resident(
                            this,
                            gamma,
                            beta,
                            rows,
                            columns,
                            eps);
                Tensor bfloat16Result = FromCudaResult(
                    bfloat16Context.Output,
                    CudaDeviceIndex,
                    _shape,
                    [this, gamma, beta],
                    TensorDType.BFloat16);
                if (AutogradContext.IsRecordingEnabled)
                {
                    AutogradLease<TensorCudaKernels
                        .BFloat16LayerNormResidentContext> lease =
                        AutogradLease<TensorCudaKernels
                            .BFloat16LayerNormResidentContext>.Own(
                            bfloat16Context,
                            AutogradLeaseMetadata.CudaOwned(
                                CudaDeviceIndex,
                                TensorDType.BFloat16,
                                DataVersion),
                            static saved => saved.Dispose());
                    bfloat16Result.Node.SetBackward(lease, savedContext =>
                    {
                        if (CudaOperationProfiler.IsEnabled)
                        {
                            CudaOperationProfiler.Measure(
                                "backward.layer_norm",
                                () => TensorCudaKernels
                                    .LayerNormBackwardBFloat16Resident(
                                        this,
                                        gamma,
                                        beta,
                                        bfloat16Result,
                                        savedContext,
                                        rows,
                                        columns));
                        }
                        else
                        {
                            TensorCudaKernels.LayerNormBackwardBFloat16Resident(
                                this,
                                gamma,
                                beta,
                                bfloat16Result,
                                savedContext,
                                rows,
                                columns);
                        }
                    });
                }
                else if (!CudaInferenceScope.TrackResource(bfloat16Context))
                {
                    bfloat16Context.Dispose();
                }
                return bfloat16Result;
            }
            TensorCudaKernels.LayerNormResidentContext context =
                CudaOperationProfiler.IsEnabled
                ? CudaOperationProfiler.Measure(
                    "forward.layer_norm",
                    () => TensorCudaKernels.LayerNormForwardResident(
                        this,
                        gamma,
                        beta,
                        rows,
                        columns,
                        eps))
                : TensorCudaKernels.LayerNormForwardResident(
                    this,
                    gamma,
                    beta,
                    rows,
                    columns,
                    eps);
            Tensor cudaResult = FromCudaResult(
                context.Output,
                CudaDeviceIndex,
                _shape,
                [this, gamma, beta]);
            if (AutogradContext.IsRecordingEnabled)
            {
                AutogradLease<TensorCudaKernels.LayerNormResidentContext>
                    lease = AutogradLease<TensorCudaKernels
                        .LayerNormResidentContext>.Own(
                        context,
                        AutogradLeaseMetadata.CudaOwned(
                            CudaDeviceIndex,
                            TensorDType.Float32,
                            DataVersion),
                        static saved => saved.Dispose());
                cudaResult.Node.SetBackward(lease, savedContext =>
                {
                    if (CudaOperationProfiler.IsEnabled)
                    {
                        CudaOperationProfiler.Measure(
                            "backward.layer_norm",
                            () => TensorCudaKernels.LayerNormBackwardResident(
                                this,
                                gamma,
                                beta,
                                cudaResult,
                                savedContext,
                                rows,
                                columns));
                    }
                    else
                    {
                        TensorCudaKernels.LayerNormBackwardResident(
                            this,
                            gamma,
                            beta,
                            cudaResult,
                            savedContext,
                            rows,
                            columns);
                    }
                });
            }
            else
            {
                if (!CudaInferenceScope.TrackResource(context))
                    context.Dispose();
            }
            return cudaResult;
        }

        if (Rank == 1)
        {
            int n = _shape[0];
            if (gamma._shape[0] != n || beta._shape[0] != n)
                throw new ArgumentException(
                    $"LayerNorm parameters must have shape [{n}], but gamma is {ShapeText(gamma)} " +
                    $"and beta is {ShapeText(beta)}.");

            bool bfloat16Normalization = DType == TensorDType.BFloat16;
            float mean;
            float var;
            if (bfloat16Normalization)
            {
                ComputeCudaOrderedLayerNormMoments(
                    _data, 0, n, out mean, out var);
            }
            else
            {
                mean = SumValues(_data, 0, n) / n;
                var = SumSquaredDifferences(_data, 0, n, mean) / n;
            }

            float inv = 1f / MathF.Sqrt(var + eps);
            float[] xhat = new float[n];
            float[] y = new float[n];

            if (bfloat16Normalization)
            {
                NormalizeAffineBFloat16Reference(
                    _data, 0, gamma._data, beta._data, mean, inv,
                    xhat, y, n);
            }
            else
            {
                NormalizeAffineValues(
                    _data,
                    0,
                    gamma._data,
                    beta._data,
                    mean,
                    inv,
                    xhat,
                    0,
                    y,
                    0,
                    n);
            }

            var t = new Tensor(y, _shape, new[] { this, gamma, beta });

            t.Node.BackwardAction = () =>
            {
                AccumulateLayerNormParameterGradients(
                    t._grad,
                    0,
                    gamma._data,
                    xhat,
                    0,
                    gamma._grad,
                    beta._grad,
                    n,
                    out float sumDxhat,
                    out float sumDxhatXhat);
                if (bfloat16Normalization)
                {
                    ComputeCudaOrderedLayerNormGradientSums(
                        t._grad,
                        0,
                        gamma._data,
                        xhat,
                        0,
                        n,
                        out sumDxhat,
                        out sumDxhatXhat);
                    AccumulateCudaOrderedLayerNormInputGradient(
                        _grad,
                        0,
                        t._grad,
                        0,
                        gamma._data,
                        xhat,
                        0,
                        n,
                        inv,
                        sumDxhat,
                        sumDxhatXhat);
                }
                else
                {
                    AccumulateLayerNormInputGradient(
                        _grad,
                        0,
                        t._grad,
                        0,
                        gamma._data,
                        xhat,
                        0,
                        n,
                        inv,
                        sumDxhat,
                        sumDxhatXhat);
                }
            };

            return t;
        }

        if (Rank >= 2)
        {
            int cols = _shape[^1];
            int rows = Numel / cols;

            if (gamma._shape[0] != cols || beta._shape[0] != cols)
                throw new ArgumentException(
                    $"LayerNorm parameters must have shape [{cols}], but gamma is {ShapeText(gamma)} " +
                    $"and beta is {ShapeText(beta)}.");

            float[] y = new float[Numel];
            float[] xhat = new float[Numel];
            float[] invs = new float[rows];
            bool bfloat16Normalization = DType == TensorDType.BFloat16;

            void ForwardRow(int r)
            {
                int rowOffset = r * cols;
                float mean;
                float var;
                if (bfloat16Normalization)
                {
                    ComputeCudaOrderedLayerNormMoments(
                        _data, rowOffset, cols, out mean, out var);
                }
                else
                {
                    mean = SumValues(_data, rowOffset, cols) / cols;
                    var = SumSquaredDifferences(
                        _data,
                        rowOffset,
                        cols,
                        mean) / cols;
                }

                float inv = 1f / MathF.Sqrt(var + eps);
                invs[r] = inv;

                if (bfloat16Normalization)
                {
                    NormalizeAffineBFloat16Reference(
                        _data, rowOffset, gamma._data, beta._data,
                        mean, inv, xhat.AsSpan(rowOffset, cols),
                        y.AsSpan(rowOffset, cols), cols);
                }
                else
                {
                    NormalizeAffineValues(
                        _data,
                        rowOffset,
                        gamma._data,
                        beta._data,
                        mean,
                        inv,
                        xhat,
                        rowOffset,
                        y,
                        rowOffset,
                        cols);
                }
            }

            RunBatches(rows, cols, ForwardRow);

            var t = new Tensor(y, _shape, new[] { this, gamma, beta });

            t.Node.BackwardAction = () =>
            {
                void BackwardInputRow(int r)
                {
                    int rowOffset = r * cols;
                    if (bfloat16Normalization)
                    {
                        ComputeCudaOrderedLayerNormGradientSums(
                            t._grad,
                            rowOffset,
                            gamma._data,
                            xhat,
                            rowOffset,
                            cols,
                            out float sumDxhat,
                            out float sumDxhatXhat);
                        AccumulateCudaOrderedLayerNormInputGradient(
                            _grad,
                            rowOffset,
                            t._grad,
                            rowOffset,
                            gamma._data,
                            xhat,
                            rowOffset,
                            cols,
                            invs[r],
                            sumDxhat,
                            sumDxhatXhat);
                    }
                    else
                    {
                        ComputeLayerNormGradientSums(
                            t._grad,
                            rowOffset,
                            gamma._data,
                            xhat,
                            rowOffset,
                            cols,
                            out float sumDxhat,
                            out float sumDxhatXhat);
                        AccumulateLayerNormInputGradient(
                            _grad,
                            rowOffset,
                            t._grad,
                            rowOffset,
                            gamma._data,
                            xhat,
                            rowOffset,
                            cols,
                            invs[r],
                            sumDxhat,
                            sumDxhatXhat);
                    }
                }

                RunBatches(rows, cols, BackwardInputRow);

                void BackwardParameter(int c)
                {
                    if (bfloat16Normalization)
                    {
                        AccumulateCudaOrderedLayerNormParameterGradient(
                            t._grad,
                            xhat,
                            gamma._grad,
                            beta._grad,
                            rows,
                            cols,
                            c);
                        return;
                    }

                    float gammaGradient = 0f;
                    float betaGradient = 0f;
                    for (int r = 0; r < rows; r++)
                    {
                        int index = r * cols + c;
                        float gradient = t._grad[index];
                        betaGradient += gradient;
                        gammaGradient += gradient * xhat[index];
                    }

                    gamma._grad[c] += gammaGradient;
                    beta._grad[c] += betaGradient;
                }

                RunBatches(cols, rows, BackwardParameter);
            };

            return t;
        }

        throw new NotSupportedException(
            "LayerNormLastDim requires a tensor with at least one dimension.");
    }

    private static void ComputeCudaOrderedLayerNormMoments(
        TensorStorage values,
        int offset,
        int length,
        out float mean,
        out float variance)
    {
        const int threads = 256;
        Span<float> partials = stackalloc float[threads];
        for (int thread = 0; thread < threads; thread++)
        {
            float sum = 0f;
            for (int column = thread; column < length; column += threads)
                sum += values[offset + column];
            partials[thread] = sum;
        }
        mean = ReduceCudaLayerNormBlock(partials) / length;

        partials.Clear();
        for (int thread = 0; thread < threads; thread++)
        {
            float sum = 0f;
            for (int column = thread; column < length; column += threads)
            {
                float difference = values[offset + column] - mean;
                sum = MathF.FusedMultiplyAdd(difference, difference, sum);
            }
            partials[thread] = sum;
        }
        variance = ReduceCudaLayerNormBlock(partials) / length;
    }

    private static float ReduceCudaLayerNormBlock(Span<float> partials)
    {
        const int warpSize = 32;
        const int warps = 8;
        Span<float> warpSums = stackalloc float[warps];
        for (int warp = 0; warp < warps; warp++)
        {
            Span<float> values = partials.Slice(warp * warpSize, warpSize);
            for (int delta = warpSize / 2; delta > 0; delta >>= 1)
            {
                for (int lane = 0; lane < delta; lane++)
                    values[lane] += values[lane + delta];
            }
            warpSums[warp] = values[0];
        }
        for (int delta = warpSize / 2; delta > 0; delta >>= 1)
        {
            for (int lane = 0; lane < delta; lane++)
            {
                float right = lane + delta < warps
                    ? warpSums[lane + delta]
                    : 0f;
                if (lane < warps)
                    warpSums[lane] += right;
            }
        }
        return warpSums[0];
    }

    private static void ComputeCudaOrderedLayerNormGradientSums(
        float[] gradient,
        int gradientOffset,
        TensorStorage gamma,
        float[] normalized,
        int normalizedOffset,
        int length,
        out float sumGradientToNormalized,
        out float sumGradientToNormalizedTimesNormalized)
    {
        // Match layer_norm_backward_input_block: each of the 256 CUDA
        // threads owns columns thread + item * blockDim.x, then the block
        // reduces eight warp sums.  BF16 direct gradients can otherwise land
        // on the adjacent representable value solely because the CPU SIMD
        // reduction associates the same FP32 operands differently.
        const int threads = 256;
        Span<float> firstPartials = stackalloc float[threads];
        Span<float> secondPartials = stackalloc float[threads];
        for (int thread = 0; thread < threads; thread++)
        {
            float first = 0f;
            float second = 0f;
            for (int column = thread; column < length; column += threads)
            {
                float dxhat = gradient[gradientOffset + column]
                    * gamma[column];
                float xhat = normalized[normalizedOffset + column];
                first += dxhat;
                second = MathF.FusedMultiplyAdd(dxhat, xhat, second);
            }
            firstPartials[thread] = first;
            secondPartials[thread] = second;
        }

        sumGradientToNormalized = ReduceCudaLayerNormBlock(firstPartials);
        sumGradientToNormalizedTimesNormalized =
            ReduceCudaLayerNormBlock(secondPartials);
    }

    private static void AccumulateCudaOrderedLayerNormParameterGradient(
        float[] gradient,
        float[] normalized,
        float[] gammaGradient,
        float[] betaGradient,
        int rows,
        int columns,
        int column)
    {
        // Match layer_norm_backward_parameters_tiled followed by
        // layer_norm_backward_parameters_finalize.  The CUDA kernel assigns
        // rows to eight lanes inside each 1024-row tile, reduces those lanes
        // in lane order, then reduces tiles in ascending order.  Preserving
        // that association matters when the final pure-BF16 gradient lands
        // exactly on a rounding midpoint.
        const int parameterRows = 8;
        const int rowsPerTile = 1024;
        int rowTiles = (rows + rowsPerTile - 1) / rowsPerTile;
        float finalGamma = 0f;
        float finalBeta = 0f;
        Span<float> gammaPartials = stackalloc float[parameterRows];
        Span<float> betaPartials = stackalloc float[parameterRows];

        for (int tile = 0; tile < rowTiles; tile++)
        {
            int rowStart = tile * rowsPerTile;
            int rowEnd = Math.Min(rows, rowStart + rowsPerTile);
            for (int lane = 0; lane < parameterRows; lane++)
            {
                float gammaSum = 0f;
                float betaSum = 0f;
                for (int row = rowStart + lane;
                    row < rowEnd;
                    row += parameterRows)
                {
                    int index = row * columns + column;
                    float value = gradient[index];
                    betaSum += value;
                    gammaSum = MathF.FusedMultiplyAdd(
                        value,
                        normalized[index],
                        gammaSum);
                }
                gammaPartials[lane] = gammaSum;
                betaPartials[lane] = betaSum;
            }

            float tileGamma = 0f;
            float tileBeta = 0f;
            for (int lane = 0; lane < parameterRows; lane++)
            {
                tileGamma += gammaPartials[lane];
                tileBeta += betaPartials[lane];
            }
            finalGamma += tileGamma;
            finalBeta += tileBeta;
        }

        gammaGradient[column] += finalGamma;
        betaGradient[column] += finalBeta;
    }

    private static void AccumulateCudaOrderedLayerNormInputGradient(
        float[] destination,
        int destinationOffset,
        float[] gradient,
        int gradientOffset,
        TensorStorage gamma,
        float[] normalized,
        int normalizedOffset,
        int length,
        float inverseStandardDeviation,
        float sumGradientToNormalized,
        float sumGradientToNormalizedTimesNormalized)
    {
        float inverseOverColumns = inverseStandardDeviation / length;
        for (int column = 0; column < length; column++)
        {
            float dxhat = gradient[gradientOffset + column] * gamma[column];
            float xhat = normalized[normalizedOffset + column];
            float centered = MathF.FusedMultiplyAdd(
                length, dxhat, -sumGradientToNormalized);
            centered = MathF.FusedMultiplyAdd(
                -xhat,
                sumGradientToNormalizedTimesNormalized,
                centered);
            float inputGradient = inverseOverColumns * centered;
            destination[destinationOffset + column] += inputGradient;
        }
    }

    private static void NormalizeAffineBFloat16Reference(
        TensorStorage values,
        int offset,
        TensorStorage gamma,
        TensorStorage beta,
        float mean,
        float inverse,
        Span<float> normalized,
        Span<float> output,
        int length)
    {
        for (int column = 0; column < length; column++)
        {
            float xhat = (values[offset + column] - mean) * inverse;
            normalized[column] = xhat;
            output[column] = MathF.FusedMultiplyAdd(
                xhat, gamma[column], beta[column]);
        }
    }

    public Tensor CausalMask(float fillValue = -1e9f)
    {
        if (Rank < 2)
        {
            throw new InvalidOperationException(
                "CausalMask requires a tensor with at least two " +
                "dimensions.");
        }

        int rows = _shape[^2];
        int cols = _shape[^1];
        if (ExecutionDevice == TensorDevice.Cuda
            && DType is TensorDType.Float32
                or TensorDType.BFloat16
                or TensorDType.Bfp8)
        {
            return CausalMaskCuda(rows, cols, fillValue);
        }

        ThrowIfCudaHostFallback(nameof(CausalMask));

        int matrixCount = Numel / (rows * cols);
        float[] y = new float[Numel];
        _data.CopyTo(y);

        for (int matrix = 0; matrix < matrixCount; matrix++)
        {
            int matrixOffset = matrix * rows * cols;
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    if (c > r)
                        y[matrixOffset + r * cols + c] = fillValue;
                }
            }
        }

        var t = new Tensor(y, _shape, new[] { this });

        t.Node.BackwardAction = () =>
        {
            for (int matrix = 0; matrix < matrixCount; matrix++)
            {
                int matrixOffset = matrix * rows * cols;
                for (int r = 0; r < rows; r++)
                {
                    int copiedColumns = Math.Min(r + 1, cols);
                    AddScaledValues(
                        _grad,
                        matrixOffset + r * cols,
                        t._grad,
                        matrixOffset + r * cols,
                        1f,
                        copiedColumns);
                }
            }
        };

        return t;
    }
}
