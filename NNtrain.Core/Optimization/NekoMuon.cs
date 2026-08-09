namespace NNtrain;

public sealed class NekoMuon : IOptimizer, ILearningRateAdjustable
{
    private const float NewtonSchulzA = 3.4445f;
    private const float NewtonSchulzB = -4.7750f;
    private const float NewtonSchulzC = 2.0315f;

    private readonly List<Parameter> _parameters;
    private NekoMuonState _state;

    public NekoMuon(
        IEnumerable<Parameter> parameters,
        NekoMuonOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(parameters);

        _parameters = [];
        var seenParameters =
            new HashSet<Parameter>(ReferenceEqualityComparer.Instance);

        foreach (Parameter parameter in parameters)
        {
            if (parameter is null)
            {
                throw new ArgumentException(
                    "Optimizer parameters cannot contain null.",
                    nameof(parameters));
            }

            if (!seenParameters.Add(parameter))
            {
                throw new ArgumentException(
                    $"Parameter '{parameter.Name}' was supplied to " +
                    "NekoMuon more than once.",
                    nameof(parameters));
            }

            _parameters.Add(parameter);
        }

        NekoMuonOptions effectiveOptions = options ?? new NekoMuonOptions();
        ValidateOptions(effectiveOptions, nameof(options));
        _state = CreateInitialState(_parameters, effectiveOptions);
    }

    public NekoMuonState CaptureState()
        => CloneState(_state);

    public float LearningRate => _state.Options.LearningRate;

    public void SetLearningRate(float learningRate)
    {
        if (!float.IsFinite(learningRate) || learningRate <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(learningRate),
                learningRate,
                "NekoMuon learning rate must be finite and positive.");
        }

        _state = _state with
        {
            Options = _state.Options with { LearningRate = learningRate },
        };
    }

    public void RestoreState(NekoMuonState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        ValidateState(state);
        _state = CloneState(state);
    }

    public void ZeroGrad()
    {
        foreach (Parameter parameter in _parameters)
            parameter.ZeroGrad();
    }

    public void Step()
    {
        if (_state.Step == int.MaxValue)
        {
            throw new InvalidOperationException(
                "NekoMuon cannot advance beyond Int32.MaxValue steps.");
        }

        _state = _state with { Step = _state.Step + 1 };
        NekoMuonOptions options = _state.Options;
        float fastCorrection =
            1f - MathF.Pow(options.BetaFast, _state.Step);
        float slowCorrection =
            1f - MathF.Pow(options.BetaSlow, _state.Step);

        void UpdateParameter(int parameterIndex)
        {
            Parameter parameter = _parameters[parameterIndex];
            NekoMuonParameterState parameterState =
                _state.ParameterStates[parameterIndex];
            float[] gradientBuffer = parameter.T.GradientBuffer;
            float[] fast = parameterState.FastMoment;
            float[] slow = parameterState.SlowMoment;
            int length = parameter.T.Numel;
            var fastHat = new float[length];
            var slowHat = new float[length];

            UpdateMoments(
                gradientBuffer,
                fast,
                slow,
                fastHat,
                slowHat,
                options,
                fastCorrection,
                slowCorrection);

            float confidenceRaw = CalculateConfidenceRaw(
                fastHat,
                slowHat,
                options.Epsilon);
            float confidence = Math.Clamp(
                options.Rho * parameterState.Confidence
                    + (1f - options.Rho) * confidenceRaw,
                0f,
                1f);
            _state.ParameterStates[parameterIndex] =
                parameterState with { Confidence = confidence };

            float depth = options.MaxNewtonSchulzSteps * confidence;
            int wholeSteps = Math.Min(
                options.MaxNewtonSchulzSteps,
                (int)MathF.Floor(depth));
            float fraction = depth - wholeSteps;

            GetMatrixShape(
                parameter,
                out int originalRows,
                out int originalColumns);
            bool transpose = originalRows > originalColumns;
            int rows = Math.Min(originalRows, originalColumns);
            int columns = Math.Max(originalRows, originalColumns);
            var x = new float[length];
            InitializeMuonMatrix(
                fastHat,
                x,
                originalRows,
                originalColumns,
                transpose,
                options.Epsilon);

            var next = new float[length];
            var gram = new float[rows * rows];
            var gramSquared = new float[rows * rows];
            for (int step = 0; step < wholeSteps; step++)
            {
                NewtonSchulz(
                    x,
                    next,
                    gram,
                    gramSquared,
                    rows,
                    columns);
                (x, next) = (next, x);
            }

            if (fraction > 0f)
            {
                NewtonSchulz(
                    x,
                    next,
                    gram,
                    gramSquared,
                    rows,
                    columns);
                InterpolateInPlace(x, next, fraction);
            }

            float finalScale = MathF.Sqrt(MathF.Max(
                1f,
                (float)originalRows / originalColumns));
            float[] update = transpose ? slowHat : x;
            if (transpose)
            {
                TransposeBack(
                    x,
                    update,
                    originalRows,
                    originalColumns);
            }

            ApplyUpdate(parameter, update, finalScale, options);
        }

        long totalElements = 0;
        foreach (Parameter parameter in _parameters)
            totalElements += parameter.T.Numel;

        if (_parameters.Count > 1 && totalElements >= 32_768)
            Parallel.For(0, _parameters.Count, UpdateParameter);
        else
            for (int index = 0; index < _parameters.Count; index++)
                UpdateParameter(index);
    }

    private static void UpdateMoments(
        float[] gradientBuffer,
        float[] fast,
        float[] slow,
        float[] fastHat,
        float[] slowHat,
        NekoMuonOptions options,
        float fastCorrection,
        float slowCorrection)
    {
        int length = fast.Length;
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> betaFast = Vector256.Create(options.BetaFast);
            Vector256<float> betaSlow = Vector256.Create(options.BetaSlow);
            Vector256<float> oneMinusBetaFast =
                Vector256.Create(1f - options.BetaFast);
            Vector256<float> oneMinusBetaSlow =
                Vector256.Create(1f - options.BetaSlow);
            Vector256<float> inverseFastCorrection =
                Vector256.Create(1f / fastCorrection);
            Vector256<float> inverseSlowCorrection =
                Vector256.Create(1f / slowCorrection);

            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> gradient = gradientBuffer.Length == 0
                    ? Vector256<float>.Zero
                    : Vector256.LoadUnsafe(ref gradientBuffer[index]);
                Vector256<float> nextFast =
                    betaFast * Vector256.LoadUnsafe(ref fast[index])
                    + oneMinusBetaFast * gradient;
                Vector256<float> nextSlow =
                    betaSlow * Vector256.LoadUnsafe(ref slow[index])
                    + oneMinusBetaSlow * gradient;
                nextFast.StoreUnsafe(ref fast[index]);
                nextSlow.StoreUnsafe(ref slow[index]);
                (nextFast * inverseFastCorrection)
                    .StoreUnsafe(ref fastHat[index]);
                (nextSlow * inverseSlowCorrection)
                    .StoreUnsafe(ref slowHat[index]);
            }
        }

        for (; index < length; index++)
        {
            float gradient = gradientBuffer.Length == 0
                ? 0f
                : gradientBuffer[index];
            fast[index] = options.BetaFast * fast[index]
                + (1f - options.BetaFast) * gradient;
            slow[index] = options.BetaSlow * slow[index]
                + (1f - options.BetaSlow) * gradient;
            fastHat[index] = fast[index] / fastCorrection;
            slowHat[index] = slow[index] / slowCorrection;
        }
    }

    private static float CalculateConfidenceRaw(
        float[] fastHat,
        float[] slowHat,
        float epsilon)
    {
        double dot = 0d;
        double fastNormSquared = 0d;
        double slowNormSquared = 0d;
        double residualNormSquared = 0d;

        for (int index = 0; index < fastHat.Length; index++)
        {
            double fast = fastHat[index];
            double slow = slowHat[index];
            double residual = fast - slow;
            dot += fast * slow;
            fastNormSquared += fast * fast;
            slowNormSquared += slow * slow;
            residualNormSquared += residual * residual;
        }

        double alignmentDenominator =
            Math.Sqrt(fastNormSquared)
            * Math.Sqrt(slowNormSquared)
            + epsilon;
        double alignment = Math.Max(0d, dot / alignmentDenominator);
        double persistence = slowNormSquared
            / (slowNormSquared + residualNormSquared + epsilon);
        return (float)Math.Clamp(alignment * persistence, 0d, 1d);
    }

    private static void GetMatrixShape(
        Parameter parameter,
        out int rows,
        out int columns)
    {
        if (parameter.T.Rank >= 2)
        {
            rows = parameter.T.Shape[0];
            columns = parameter.T.Numel / rows;
            return;
        }

        rows = 1;
        columns = parameter.T.Numel;
    }

    private static void InitializeMuonMatrix(
        float[] source,
        float[] destination,
        int rows,
        int columns,
        bool transpose,
        float epsilon)
    {
        double normSquared = 0d;
        foreach (float value in source)
            normSquared += (double)value * value;
        float inverseNorm = 1f / ((float)Math.Sqrt(normSquared) + epsilon);

        if (!transpose)
        {
            Scale(source, destination, inverseNorm);
            return;
        }

        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                destination[column * rows + row] =
                    source[row * columns + column] * inverseNorm;
            }
        }
    }

    private static void NewtonSchulz(
        float[] source,
        float[] destination,
        float[] gram,
        float[] gramSquared,
        int rows,
        int columns)
    {
        for (int row = 0; row < rows; row++)
        {
            int rowOffset = row * columns;
            for (int other = 0; other <= row; other++)
            {
                float dot = Dot(
                    source,
                    rowOffset,
                    other * columns,
                    columns);
                gram[row * rows + other] = dot;
                gram[other * rows + row] = dot;
            }
        }

        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < rows; column++)
            {
                double sum = 0d;
                for (int inner = 0; inner < rows; inner++)
                {
                    sum += gram[row * rows + inner]
                        * gram[inner * rows + column];
                }

                gramSquared[row * rows + column] = (float)sum;
            }
        }

        Scale(source, destination, NewtonSchulzA);
        for (int row = 0; row < rows; row++)
        {
            int destinationOffset = row * columns;
            for (int inner = 0; inner < rows; inner++)
            {
                float coefficient =
                    NewtonSchulzB * gram[row * rows + inner]
                    + NewtonSchulzC
                        * gramSquared[row * rows + inner];
                AddScaled(
                    source,
                    inner * columns,
                    destination,
                    destinationOffset,
                    columns,
                    coefficient);
            }
        }
    }

    private static float Dot(
        float[] values,
        int firstOffset,
        int secondOffset,
        int length)
    {
        int index = 0;
        float sum = 0f;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> vectorSum = Vector256<float>.Zero;
            for (; index < vectorizedLength; index += width)
            {
                vectorSum +=
                    Vector256.LoadUnsafe(ref values[firstOffset + index])
                    * Vector256.LoadUnsafe(ref values[secondOffset + index]);
            }

            sum = Vector256.Sum(vectorSum);
        }

        for (; index < length; index++)
        {
            sum += values[firstOffset + index]
                * values[secondOffset + index];
        }

        return sum;
    }

    private static void Scale(
        float[] source,
        float[] destination,
        float scale)
    {
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && source.Length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = source.Length - source.Length % width;
            Vector256<float> vectorScale = Vector256.Create(scale);
            for (; index < vectorizedLength; index += width)
            {
                (Vector256.LoadUnsafe(ref source[index]) * vectorScale)
                    .StoreUnsafe(ref destination[index]);
            }
        }

        for (; index < source.Length; index++)
            destination[index] = source[index] * scale;
    }

    private static void AddScaled(
        float[] source,
        int sourceOffset,
        float[] destination,
        int destinationOffset,
        int length,
        float scale)
    {
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> vectorScale = Vector256.Create(scale);
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> result =
                    Vector256.LoadUnsafe(ref destination[
                        destinationOffset + index])
                    + vectorScale
                        * Vector256.LoadUnsafe(ref source[
                            sourceOffset + index]);
                result.StoreUnsafe(ref destination[
                    destinationOffset + index]);
            }
        }

        for (; index < length; index++)
        {
            destination[destinationOffset + index] +=
                scale * source[sourceOffset + index];
        }
    }

    private static void InterpolateInPlace(
        float[] current,
        float[] next,
        float fraction)
    {
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && current.Length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = current.Length - current.Length % width;
            Vector256<float> vectorFraction = Vector256.Create(fraction);
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> currentValues =
                    Vector256.LoadUnsafe(ref current[index]);
                Vector256<float> nextValues =
                    Vector256.LoadUnsafe(ref next[index]);
                (currentValues
                    + vectorFraction * (nextValues - currentValues))
                    .StoreUnsafe(ref current[index]);
            }
        }

        for (; index < current.Length; index++)
        {
            current[index] += fraction * (next[index] - current[index]);
        }
    }

    private static void TransposeBack(
        float[] transposed,
        float[] destination,
        int rows,
        int columns)
    {
        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                destination[row * columns + column] =
                    transposed[column * rows + row];
            }
        }
    }

    private static void ApplyUpdate(
        Parameter parameter,
        float[] update,
        float finalScale,
        NekoMuonOptions options)
    {
        using Tensor.DataMutation mutation = parameter.BeginUpdate();
        Span<float> data = mutation.Values;
        bool applyWeightDecay =
            parameter.WeightDecay == WeightDecayPolicy.Apply
            || (options.Decay1D && parameter.T.Rank == 1);
        int index = 0;

        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && data.Length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = data.Length - data.Length % width;
            Vector256<float> learningRate =
                Vector256.Create(options.LearningRate);
            Vector256<float> updateScale =
                Vector256.Create(options.LearningRate * finalScale);
            Vector256<float> weightDecay =
                Vector256.Create(options.WeightDecay);
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> parameterValues =
                    Vector256.LoadUnsafe(ref data[index]);
                if (applyWeightDecay)
                {
                    parameterValues -= learningRate
                        * weightDecay
                        * parameterValues;
                }

                parameterValues -= updateScale
                    * Vector256.LoadUnsafe(ref update[index]);
                parameterValues.StoreUnsafe(ref data[index]);
            }
        }

        for (; index < data.Length; index++)
        {
            if (applyWeightDecay)
            {
                data[index] -= options.LearningRate
                    * options.WeightDecay
                    * data[index];
            }

            data[index] -= options.LearningRate
                * finalScale
                * update[index];
        }
    }

    private static NekoMuonState CreateInitialState(
        IReadOnlyList<Parameter> parameters,
        NekoMuonOptions options)
    {
        NekoMuonParameterState[] parameterStates = parameters
            .Select((parameter, index) =>
                new NekoMuonParameterState(
                    index,
                    parameter.Name,
                    parameter.T.Shape.ToArray(),
                    new float[parameter.T.Numel],
                    new float[parameter.T.Numel],
                    0f))
            .ToArray();

        return new NekoMuonState(
            NekoMuonState.CurrentFormatVersion,
            0,
            options with { },
            parameterStates);
    }

    private void ValidateState(NekoMuonState state)
    {
        if (state.FormatVersion != NekoMuonState.CurrentFormatVersion)
        {
            throw new ArgumentException(
                $"Unsupported NekoMuon state format version " +
                $"'{state.FormatVersion}'. Expected " +
                $"'{NekoMuonState.CurrentFormatVersion}'.",
                nameof(state));
        }

        if (state.Step < 0 || state.Step == int.MaxValue)
        {
            throw new ArgumentException(
                "NekoMuon state step must be non-negative and leave room " +
                "for another optimizer step.",
                nameof(state));
        }

        if (state.Options is null)
        {
            throw new ArgumentException(
                "NekoMuon state options cannot be null.",
                nameof(state));
        }

        ValidateOptions(state.Options, nameof(state));

        if (state.ParameterStates is null
            || state.ParameterStates.Length != _parameters.Count)
        {
            throw new ArgumentException(
                "NekoMuon state parameter count does not match the " +
                "optimizer.",
                nameof(state));
        }

        for (int index = 0; index < _parameters.Count; index++)
        {
            Parameter parameter = _parameters[index];
            NekoMuonParameterState parameterState =
                state.ParameterStates[index];
            if (parameterState is null
                || parameterState.Index != index
                || !string.Equals(
                    parameterState.Name,
                    parameter.Name,
                    StringComparison.Ordinal)
                || parameterState.Shape is null
                || !parameterState.Shape.SequenceEqual(parameter.T.Shape)
                || parameterState.FastMoment is null
                || parameterState.FastMoment.Length != parameter.T.Numel
                || parameterState.SlowMoment is null
                || parameterState.SlowMoment.Length != parameter.T.Numel)
            {
                throw new ArgumentException(
                    $"NekoMuon parameter state for slot {index} is " +
                    "incompatible.",
                    nameof(state));
            }

            if (!float.IsFinite(parameterState.Confidence)
                || parameterState.Confidence < 0f
                || parameterState.Confidence > 1f)
            {
                throw new ArgumentException(
                    $"NekoMuon confidence for slot {index} must be " +
                    "finite and in [0, 1].",
                    nameof(state));
            }
        }
    }

    private static void ValidateOptions(
        NekoMuonOptions options,
        string parameterName)
    {
        if (!float.IsFinite(options.LearningRate)
            || options.LearningRate <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.LearningRate,
                "NekoMuon learning rate must be finite and positive.");
        }

        ValidateUnitInterval(
            options.BetaFast,
            parameterName,
            "betaFast");
        ValidateUnitInterval(
            options.BetaSlow,
            parameterName,
            "betaSlow");
        ValidateUnitInterval(options.Rho, parameterName, "rho");

        if (!float.IsFinite(options.Epsilon) || options.Epsilon <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.Epsilon,
                "NekoMuon epsilon must be finite and positive.");
        }

        if (options.MaxNewtonSchulzSteps <= 0)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.MaxNewtonSchulzSteps,
                "NekoMuon maximum Newton-Schulz steps must be positive.");
        }

        if (!float.IsFinite(options.WeightDecay)
            || options.WeightDecay < 0f)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.WeightDecay,
                "NekoMuon weight decay must be finite and non-negative.");
        }
    }

    private static void ValidateUnitInterval(
        float value,
        string parameterName,
        string optionName)
    {
        if (!float.IsFinite(value) || value < 0f || value >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                value,
                $"NekoMuon {optionName} must be finite and in [0, 1).");
        }
    }

    private static NekoMuonState CloneState(NekoMuonState state)
    {
        return new NekoMuonState(
            state.FormatVersion,
            state.Step,
            state.Options with { },
            state.ParameterStates
                .Select(parameterState =>
                    new NekoMuonParameterState(
                        parameterState.Index,
                        parameterState.Name,
                        parameterState.Shape.ToArray(),
                        parameterState.FastMoment.ToArray(),
                        parameterState.SlowMoment.ToArray(),
                        parameterState.Confidence))
                .ToArray());
    }
}
