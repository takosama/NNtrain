using System.Diagnostics;

namespace NNtrain;

public sealed partial class NekoMuon : IOptimizer, ILearningRateAdjustable
{
    private const float NewtonSchulzA = 3.4445f;
    private const float NewtonSchulzB = -4.7750f;
    private const float NewtonSchulzC = 2.0315f;

    private readonly List<Parameter> _parameters;
    private readonly long _totalElements;
    private readonly NekoMuonWorkspace[] _workspaces;
    private NekoMuonState _state;

    public bool ProfilingEnabled { get; set; }

    public NekoMuonStepProfile LastStepProfile { get; private set; }

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

        _totalElements = _parameters.Sum(
            parameter => (long)parameter.T.Numel);

        NekoMuonOptions effectiveOptions = options ?? new NekoMuonOptions();
        ValidateOptions(effectiveOptions, nameof(options));
        _state = CreateInitialState(_parameters, effectiveOptions);
        _workspaces = _parameters.Select(CreateWorkspace).ToArray();
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
        if (_parameters.Count > 1 && _totalElements >= 32_768)
        {
            Tensor.RunParallel(
                0,
                _parameters.Count,
                index => _parameters[index].ZeroGrad());
            return;
        }

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
        long[]? profileTicks = ProfilingEnabled ? new long[9] : null;

        void UpdateParameter(int parameterIndex)
        {
            Parameter parameter = _parameters[parameterIndex];
            NekoMuonParameterState parameterState =
                _state.ParameterStates[parameterIndex];
            float[] gradientBuffer = parameter.T.GradientBuffer;
            float[] fast = parameterState.FastMoment;
            float[] slow = parameterState.SlowMoment;
            NekoMuonWorkspace workspace = _workspaces[parameterIndex];
            float[] fastHat = workspace.FastHat;
            float[] slowHat = workspace.SlowHat;

            long phaseStart = profileTicks is null
                ? 0L
                : Stopwatch.GetTimestamp();
            UpdateMoments(
                gradientBuffer,
                fast,
                slow,
                fastHat,
                slowHat,
                options,
                fastCorrection,
                slowCorrection);
            AddProfileTicks(profileTicks, 0, phaseStart);

            phaseStart = profileTicks is null ? 0L : Stopwatch.GetTimestamp();
            float confidenceRaw = CalculateConfidenceRaw(
                fastHat,
                slowHat,
                options.Epsilon);
            float confidence = Math.Clamp(
                options.Rho * parameterState.Confidence
                    + (1f - options.Rho) * confidenceRaw,
                0f,
                1f);
            AddProfileTicks(profileTicks, 1, phaseStart);
            _state.ParameterStates[parameterIndex] =
                parameterState with { Confidence = confidence };

            bool runNewtonSchulz =
                _state.Step % options.NewtonSchulzInterval == 0;
            // Moments and weights still advance on the intervening steps.
            // They use the normalized current momentum matrix; only the
            // expensive orthogonalization is cadence-limited.
            float depth = runNewtonSchulz
                ? options.MaxNewtonSchulzSteps * confidence
                : 0f;
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
            float[] x = workspace.X;
            phaseStart = profileTicks is null ? 0L : Stopwatch.GetTimestamp();
            InitializeMuonMatrix(
                fastHat,
                x,
                originalRows,
                originalColumns,
                transpose,
                options.Epsilon);
            AddProfileTicks(profileTicks, 2, phaseStart);

            float[] next = workspace.Next;
            float[] gram = workspace.Gram;
            float[] gramSquared = workspace.GramSquared;
            phaseStart = profileTicks is null ? 0L : Stopwatch.GetTimestamp();
            for (int step = 0; step < wholeSteps; step++)
            {
                NewtonSchulz(
                    x,
                    next,
                    gram,
                    gramSquared,
                    rows,
                    columns,
                    profileTicks);
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
                    columns,
                    profileTicks);
                InterpolateInPlace(x, next, fraction);
            }
            AddProfileTicks(profileTicks, 3, phaseStart);

            float finalScale = MathF.Sqrt(MathF.Max(
                1f,
                (float)originalRows / originalColumns));
            float[] update = transpose ? slowHat : x;
            if (transpose)
            {
                phaseStart = profileTicks is null
                    ? 0L
                    : Stopwatch.GetTimestamp();
                TransposeBack(
                    x,
                    update,
                    originalRows,
                    originalColumns);
                AddProfileTicks(profileTicks, 4, phaseStart);
            }

            phaseStart = profileTicks is null ? 0L : Stopwatch.GetTimestamp();
            ApplyUpdate(parameter, update, finalScale, options);
            AddProfileTicks(profileTicks, 5, phaseStart);
        }

        if (_parameters.Count > 1 && _totalElements >= 32_768)
            Tensor.RunParallel(0, _parameters.Count, UpdateParameter);
        else
            for (int index = 0; index < _parameters.Count; index++)
                UpdateParameter(index);

        if (profileTicks is not null)
        {
            LastStepProfile = new NekoMuonStepProfile(
                TicksToMilliseconds(profileTicks[0]),
                TicksToMilliseconds(profileTicks[1]),
                TicksToMilliseconds(profileTicks[2]),
                TicksToMilliseconds(profileTicks[3]),
                TicksToMilliseconds(profileTicks[4]),
                TicksToMilliseconds(profileTicks[5]),
                TicksToMilliseconds(profileTicks[6]),
                TicksToMilliseconds(profileTicks[7]),
                TicksToMilliseconds(profileTicks[8]));
        }
    }

    private static void AddProfileTicks(
        long[]? profileTicks,
        int index,
        long start)
    {
        if (profileTicks is null)
            return;
        Interlocked.Add(
            ref profileTicks[index],
            Stopwatch.GetTimestamp() - start);
    }

    private static double TicksToMilliseconds(long ticks)
        => ticks * 1000d / Stopwatch.Frequency;

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

    private static NekoMuonWorkspace CreateWorkspace(Parameter parameter)
    {
        int length = parameter.T.Numel;
        GetMatrixShape(parameter, out int originalRows, out int originalColumns);
        int rows = Math.Min(originalRows, originalColumns);
        int gramLength = checked(rows * rows);
        return new NekoMuonWorkspace(
            new float[length],
            new float[length],
            new float[length],
            new float[length],
            new float[gramLength],
            new float[gramLength]);
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

        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && fastHat.Length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = fastHat.Length - fastHat.Length % width;
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> fast =
                    Vector256.LoadUnsafe(ref fastHat[index]);
                Vector256<float> slow =
                    Vector256.LoadUnsafe(ref slowHat[index]);
                Vector256<float> residual = fast - slow;
                dot += Vector256.Sum(fast * slow);
                fastNormSquared += Vector256.Sum(fast * fast);
                slowNormSquared += Vector256.Sum(slow * slow);
                residualNormSquared += Vector256.Sum(residual * residual);
            }
        }

        for (; index < fastHat.Length; index++)
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
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && source.Length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = source.Length - source.Length % width;
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> values =
                    Vector256.LoadUnsafe(ref source[index]);
                normSquared += Vector256.Sum(values * values);
            }
        }

        for (; index < source.Length; index++)
            normSquared += (double)source[index] * source[index];
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
        int columns,
        long[]? profileTicks)
    {
        long phaseStart = profileTicks is null
            ? 0L
            : Stopwatch.GetTimestamp();
        ComputeSymmetricGram(source, gram, rows, columns);
        AddProfileTicks(profileTicks, 6, phaseStart);

        phaseStart = profileTicks is null ? 0L : Stopwatch.GetTimestamp();
        ComputeSymmetricGram(gram, gramSquared, rows, rows);
        AddProfileTicks(profileTicks, 7, phaseStart);

        phaseStart = profileTicks is null ? 0L : Stopwatch.GetTimestamp();
        Scale(source, destination, NewtonSchulzA);
        int outputRow = 0;
        for (; outputRow + 7 < rows; outputRow += 8)
        {
            int destination0 = outputRow * columns;
            int destination1 = destination0 + columns;
            int destination2 = destination1 + columns;
            int destination3 = destination2 + columns;
            int destination4 = destination3 + columns;
            int destination5 = destination4 + columns;
            int destination6 = destination5 + columns;
            int destination7 = destination6 + columns;
            int coefficient0 = outputRow * rows;
            int coefficient1 = coefficient0 + rows;
            int coefficient2 = coefficient1 + rows;
            int coefficient3 = coefficient2 + rows;
            int coefficient4 = coefficient3 + rows;
            int coefficient5 = coefficient4 + rows;
            int coefficient6 = coefficient5 + rows;
            int coefficient7 = coefficient6 + rows;

            for (int inner = 0; inner < rows; inner++)
            {
                AddScaledEightRows(
                    source,
                    inner * columns,
                    destination,
                    destination0,
                    destination1,
                    destination2,
                    destination3,
                    destination4,
                    destination5,
                    destination6,
                    destination7,
                    columns,
                    NewtonSchulzB * gram[coefficient0 + inner]
                        + NewtonSchulzC * gramSquared[coefficient0 + inner],
                    NewtonSchulzB * gram[coefficient1 + inner]
                        + NewtonSchulzC * gramSquared[coefficient1 + inner],
                    NewtonSchulzB * gram[coefficient2 + inner]
                        + NewtonSchulzC * gramSquared[coefficient2 + inner],
                    NewtonSchulzB * gram[coefficient3 + inner]
                        + NewtonSchulzC * gramSquared[coefficient3 + inner],
                    NewtonSchulzB * gram[coefficient4 + inner]
                        + NewtonSchulzC * gramSquared[coefficient4 + inner],
                    NewtonSchulzB * gram[coefficient5 + inner]
                        + NewtonSchulzC * gramSquared[coefficient5 + inner],
                    NewtonSchulzB * gram[coefficient6 + inner]
                        + NewtonSchulzC * gramSquared[coefficient6 + inner],
                    NewtonSchulzB * gram[coefficient7 + inner]
                        + NewtonSchulzC * gramSquared[coefficient7 + inner]);
            }
        }

        for (; outputRow + 3 < rows; outputRow += 4)
        {
            int destination0 = outputRow * columns;
            int destination1 = destination0 + columns;
            int destination2 = destination1 + columns;
            int destination3 = destination2 + columns;
            int coefficient0 = outputRow * rows;
            int coefficient1 = coefficient0 + rows;
            int coefficient2 = coefficient1 + rows;
            int coefficient3 = coefficient2 + rows;

            for (int inner = 0; inner < rows; inner++)
            {
                AddScaledFourRows(
                    source,
                    inner * columns,
                    destination,
                    destination0,
                    destination1,
                    destination2,
                    destination3,
                    columns,
                    NewtonSchulzB * gram[coefficient0 + inner]
                        + NewtonSchulzC
                            * gramSquared[coefficient0 + inner],
                    NewtonSchulzB * gram[coefficient1 + inner]
                        + NewtonSchulzC
                            * gramSquared[coefficient1 + inner],
                    NewtonSchulzB * gram[coefficient2 + inner]
                        + NewtonSchulzC
                            * gramSquared[coefficient2 + inner],
                    NewtonSchulzB * gram[coefficient3 + inner]
                        + NewtonSchulzC
                            * gramSquared[coefficient3 + inner]);
            }
        }

        for (; outputRow < rows; outputRow++)
        {
            int destinationOffset = outputRow * columns;
            for (int inner = 0; inner < rows; inner++)
            {
                float coefficient =
                    NewtonSchulzB * gram[outputRow * rows + inner]
                    + NewtonSchulzC
                        * gramSquared[outputRow * rows + inner];
                AddScaled(
                    source,
                    inner * columns,
                    destination,
                    destinationOffset,
                    columns,
                    coefficient);
            }
        }
        AddProfileTicks(profileTicks, 8, phaseStart);
    }

    private static void ComputeSymmetricGram(
        float[] source,
        float[] destination,
        int rows,
        int columns)
    {
        if (!Tensor.SimdEnabled
            || !Vector256.IsHardwareAccelerated
            || rows % 4 != 0
            || columns < Vector256<float>.Count)
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
                    destination[row * rows + other] = dot;
                    destination[other * rows + row] = dot;
                }
            }
            return;
        }

        // Four rows and two comparison rows share six vector loads across
        // eight dot products. This removes most repeated reads in the two
        // Gram products that dominate Newton-Schulz on wide FFN matrices.
        for (int rowBase = 0; rowBase < rows; rowBase += 4)
        {
            for (int otherBase = 0;
                otherBase <= rowBase + 2;
                otherBase += 2)
            {
                ComputeGramFourByTwo(
                    source,
                    destination,
                    rows,
                    columns,
                    rowBase,
                    otherBase);
            }
        }
    }

    private static void ComputeGramFourByTwo(
        float[] source,
        float[] destination,
        int rows,
        int columns,
        int rowBase,
        int otherBase)
    {
        int row0 = rowBase * columns;
        int row1 = row0 + columns;
        int row2 = row1 + columns;
        int row3 = row2 + columns;
        int other0 = otherBase * columns;
        int other1 = other0 + columns;
        int index = 0;
        int width = Vector256<float>.Count;
        int vectorizedLength = columns - columns % width;
        Vector256<float> sum00 = Vector256<float>.Zero;
        Vector256<float> sum01 = Vector256<float>.Zero;
        Vector256<float> sum10 = Vector256<float>.Zero;
        Vector256<float> sum11 = Vector256<float>.Zero;
        Vector256<float> sum20 = Vector256<float>.Zero;
        Vector256<float> sum21 = Vector256<float>.Zero;
        Vector256<float> sum30 = Vector256<float>.Zero;
        Vector256<float> sum31 = Vector256<float>.Zero;

        for (; index < vectorizedLength; index += width)
        {
            Vector256<float> a0 = Vector256.LoadUnsafe(ref source[row0 + index]);
            Vector256<float> a1 = Vector256.LoadUnsafe(ref source[row1 + index]);
            Vector256<float> a2 = Vector256.LoadUnsafe(ref source[row2 + index]);
            Vector256<float> a3 = Vector256.LoadUnsafe(ref source[row3 + index]);
            Vector256<float> b0 = Vector256.LoadUnsafe(
                ref source[other0 + index]);
            Vector256<float> b1 = Vector256.LoadUnsafe(
                ref source[other1 + index]);
            sum00 = Vector256.FusedMultiplyAdd(a0, b0, sum00);
            sum01 = Vector256.FusedMultiplyAdd(a0, b1, sum01);
            sum10 = Vector256.FusedMultiplyAdd(a1, b0, sum10);
            sum11 = Vector256.FusedMultiplyAdd(a1, b1, sum11);
            sum20 = Vector256.FusedMultiplyAdd(a2, b0, sum20);
            sum21 = Vector256.FusedMultiplyAdd(a2, b1, sum21);
            sum30 = Vector256.FusedMultiplyAdd(a3, b0, sum30);
            sum31 = Vector256.FusedMultiplyAdd(a3, b1, sum31);
        }

        float scalar00 = Vector256.Sum(sum00);
        float scalar01 = Vector256.Sum(sum01);
        float scalar10 = Vector256.Sum(sum10);
        float scalar11 = Vector256.Sum(sum11);
        float scalar20 = Vector256.Sum(sum20);
        float scalar21 = Vector256.Sum(sum21);
        float scalar30 = Vector256.Sum(sum30);
        float scalar31 = Vector256.Sum(sum31);
        for (; index < columns; index++)
        {
            float a0 = source[row0 + index];
            float a1 = source[row1 + index];
            float a2 = source[row2 + index];
            float a3 = source[row3 + index];
            float b0 = source[other0 + index];
            float b1 = source[other1 + index];
            scalar00 += a0 * b0;
            scalar01 += a0 * b1;
            scalar10 += a1 * b0;
            scalar11 += a1 * b1;
            scalar20 += a2 * b0;
            scalar21 += a2 * b1;
            scalar30 += a3 * b0;
            scalar31 += a3 * b1;
        }

        StoreSymmetricGram(destination, rows, rowBase, otherBase, scalar00);
        StoreSymmetricGram(destination, rows, rowBase, otherBase + 1, scalar01);
        StoreSymmetricGram(destination, rows, rowBase + 1, otherBase, scalar10);
        StoreSymmetricGram(
            destination,
            rows,
            rowBase + 1,
            otherBase + 1,
            scalar11);
        StoreSymmetricGram(destination, rows, rowBase + 2, otherBase, scalar20);
        StoreSymmetricGram(
            destination,
            rows,
            rowBase + 2,
            otherBase + 1,
            scalar21);
        StoreSymmetricGram(destination, rows, rowBase + 3, otherBase, scalar30);
        StoreSymmetricGram(
            destination,
            rows,
            rowBase + 3,
            otherBase + 1,
            scalar31);
    }

    private static void StoreSymmetricGram(
        float[] destination,
        int rows,
        int row,
        int column,
        float value)
    {
        destination[row * rows + column] = value;
        destination[column * rows + row] = value;
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
            int unrolledLength = vectorizedLength - 3 * width;
            Vector256<float> sum0 = Vector256<float>.Zero;
            Vector256<float> sum1 = Vector256<float>.Zero;
            Vector256<float> sum2 = Vector256<float>.Zero;
            Vector256<float> sum3 = Vector256<float>.Zero;
            for (; index < unrolledLength; index += 4 * width)
            {
                sum0 +=
                    Vector256.LoadUnsafe(ref values[firstOffset + index])
                    * Vector256.LoadUnsafe(ref values[secondOffset + index]);
                sum1 +=
                    Vector256.LoadUnsafe(
                        ref values[firstOffset + index + width])
                    * Vector256.LoadUnsafe(
                        ref values[secondOffset + index + width]);
                sum2 +=
                    Vector256.LoadUnsafe(
                        ref values[firstOffset + index + 2 * width])
                    * Vector256.LoadUnsafe(
                        ref values[secondOffset + index + 2 * width]);
                sum3 +=
                    Vector256.LoadUnsafe(
                        ref values[firstOffset + index + 3 * width])
                    * Vector256.LoadUnsafe(
                        ref values[secondOffset + index + 3 * width]);
            }

            sum = Vector256.Sum((sum0 + sum1) + (sum2 + sum3));
            for (; index < vectorizedLength; index += width)
            {
                sum += Vector256.Sum(
                    Vector256.LoadUnsafe(ref values[firstOffset + index])
                    * Vector256.LoadUnsafe(
                        ref values[secondOffset + index]));
            }
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

    private static void AddScaledFourRows(
        float[] source,
        int sourceOffset,
        float[] destination,
        int destination0,
        int destination1,
        int destination2,
        int destination3,
        int length,
        float scale0,
        float scale1,
        float scale2,
        float scale3)
    {
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> vectorScale0 = Vector256.Create(scale0);
            Vector256<float> vectorScale1 = Vector256.Create(scale1);
            Vector256<float> vectorScale2 = Vector256.Create(scale2);
            Vector256<float> vectorScale3 = Vector256.Create(scale3);
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> sourceVector = Vector256.LoadUnsafe(
                    ref source[sourceOffset + index]);
                (Vector256.LoadUnsafe(ref destination[destination0 + index])
                    + vectorScale0 * sourceVector)
                    .StoreUnsafe(ref destination[destination0 + index]);
                (Vector256.LoadUnsafe(ref destination[destination1 + index])
                    + vectorScale1 * sourceVector)
                    .StoreUnsafe(ref destination[destination1 + index]);
                (Vector256.LoadUnsafe(ref destination[destination2 + index])
                    + vectorScale2 * sourceVector)
                    .StoreUnsafe(ref destination[destination2 + index]);
                (Vector256.LoadUnsafe(ref destination[destination3 + index])
                    + vectorScale3 * sourceVector)
                    .StoreUnsafe(ref destination[destination3 + index]);
            }
        }

        for (; index < length; index++)
        {
            float sourceValue = source[sourceOffset + index];
            destination[destination0 + index] += scale0 * sourceValue;
            destination[destination1 + index] += scale1 * sourceValue;
            destination[destination2 + index] += scale2 * sourceValue;
            destination[destination3 + index] += scale3 * sourceValue;
        }
    }

    private static void AddScaledEightRows(
        float[] source,
        int sourceOffset,
        float[] destination,
        int destination0,
        int destination1,
        int destination2,
        int destination3,
        int destination4,
        int destination5,
        int destination6,
        int destination7,
        int length,
        float scale0,
        float scale1,
        float scale2,
        float scale3,
        float scale4,
        float scale5,
        float scale6,
        float scale7)
    {
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> vectorScale0 = Vector256.Create(scale0);
            Vector256<float> vectorScale1 = Vector256.Create(scale1);
            Vector256<float> vectorScale2 = Vector256.Create(scale2);
            Vector256<float> vectorScale3 = Vector256.Create(scale3);
            Vector256<float> vectorScale4 = Vector256.Create(scale4);
            Vector256<float> vectorScale5 = Vector256.Create(scale5);
            Vector256<float> vectorScale6 = Vector256.Create(scale6);
            Vector256<float> vectorScale7 = Vector256.Create(scale7);
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> sourceVector = Vector256.LoadUnsafe(
                    ref source[sourceOffset + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale0,
                    Vector256.LoadUnsafe(ref destination[destination0 + index]))
                    .StoreUnsafe(ref destination[destination0 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale1,
                    Vector256.LoadUnsafe(ref destination[destination1 + index]))
                    .StoreUnsafe(ref destination[destination1 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale2,
                    Vector256.LoadUnsafe(ref destination[destination2 + index]))
                    .StoreUnsafe(ref destination[destination2 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale3,
                    Vector256.LoadUnsafe(ref destination[destination3 + index]))
                    .StoreUnsafe(ref destination[destination3 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale4,
                    Vector256.LoadUnsafe(ref destination[destination4 + index]))
                    .StoreUnsafe(ref destination[destination4 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale5,
                    Vector256.LoadUnsafe(ref destination[destination5 + index]))
                    .StoreUnsafe(ref destination[destination5 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale6,
                    Vector256.LoadUnsafe(ref destination[destination6 + index]))
                    .StoreUnsafe(ref destination[destination6 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale7,
                    Vector256.LoadUnsafe(ref destination[destination7 + index]))
                    .StoreUnsafe(ref destination[destination7 + index]);
            }
        }

        for (; index < length; index++)
        {
            float sourceValue = source[sourceOffset + index];
            destination[destination0 + index] += scale0 * sourceValue;
            destination[destination1 + index] += scale1 * sourceValue;
            destination[destination2 + index] += scale2 * sourceValue;
            destination[destination3 + index] += scale3 * sourceValue;
            destination[destination4 + index] += scale4 * sourceValue;
            destination[destination5 + index] += scale5 * sourceValue;
            destination[destination6 + index] += scale6 * sourceValue;
            destination[destination7 + index] += scale7 * sourceValue;
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

    private sealed record NekoMuonWorkspace(
        float[] FastHat,
        float[] SlowHat,
        float[] X,
        float[] Next,
        float[] Gram,
        float[] GramSquared);

}

public readonly record struct NekoMuonStepProfile(
    double UpdateMomentsMilliseconds,
    double ConfidenceMilliseconds,
    double InitializeMilliseconds,
    double NewtonSchulzMilliseconds,
    double TransposeMilliseconds,
    double ApplyUpdateMilliseconds,
    double FirstGramMilliseconds,
    double GramSquaredMilliseconds,
    double PolynomialMilliseconds)
{
    public double TotalCpuMilliseconds
        => UpdateMomentsMilliseconds
            + ConfidenceMilliseconds
            + InitializeMilliseconds
            + NewtonSchulzMilliseconds
            + TransposeMilliseconds
            + ApplyUpdateMilliseconds;
}
