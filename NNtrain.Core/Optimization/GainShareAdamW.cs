namespace NNtrain;

public sealed class GainShareAdamW : IOptimizer, ILearningRateAdjustable
{
    private readonly List<Parameter> _parameters = [];
    private readonly int[][] _groups;
    private readonly int[] _parameterGroupIndices;
    private GainShareAdamWState _state;

    internal IReadOnlyList<Parameter> Parameters => _parameters;

    public GainShareAdamW(
        IEnumerable<IEnumerable<Parameter>> parameterGroups,
        GainShareAdamWOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(parameterGroups);

        var groups = new List<int[]>();
        var seenParameters =
            new HashSet<Parameter>(ReferenceEqualityComparer.Instance);
        foreach (IEnumerable<Parameter>? sourceGroup in parameterGroups)
        {
            if (sourceGroup is null)
            {
                throw new ArgumentException(
                    "GainShare parameter groups cannot contain null.",
                    nameof(parameterGroups));
            }

            var indices = new List<int>();
            foreach (Parameter? parameter in sourceGroup)
            {
                if (parameter is null)
                {
                    throw new ArgumentException(
                        "GainShare parameter groups cannot contain null " +
                        "parameters.",
                        nameof(parameterGroups));
                }

                if (!seenParameters.Add(parameter))
                {
                    throw new ArgumentException(
                        $"Parameter '{parameter.Name}' was supplied to " +
                        "GainShareAdamW more than once.",
                        nameof(parameterGroups));
                }

                indices.Add(_parameters.Count);
                _parameters.Add(parameter);
            }

            if (indices.Count == 0)
            {
                throw new ArgumentException(
                    "GainShare parameter groups cannot be empty.",
                    nameof(parameterGroups));
            }

            groups.Add(indices.ToArray());
        }

        if (groups.Count == 0)
        {
            throw new ArgumentException(
                "GainShareAdamW requires at least one parameter group.",
                nameof(parameterGroups));
        }

        GainShareAdamWOptions effectiveOptions =
            options ?? new GainShareAdamWOptions();
        ValidateOptions(effectiveOptions, nameof(options));

        _groups = groups.ToArray();
        _parameterGroupIndices = new int[_parameters.Count];
        for (int groupIndex = 0; groupIndex < _groups.Length; groupIndex++)
        {
            foreach (int parameterIndex in _groups[groupIndex])
                _parameterGroupIndices[parameterIndex] = groupIndex;
        }

        _state = CreateInitialState(
            _parameters,
            _groups,
            effectiveOptions);
    }

    public float LearningRate => _state.Options.LearningRate;

    public void SetLearningRate(float learningRate)
    {
        if (!float.IsFinite(learningRate) || learningRate <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(learningRate),
                learningRate,
                "GainShareAdamW learning rate must be finite and positive.");
        }

        _state = _state with
        {
            Options = _state.Options with { LearningRate = learningRate },
        };
    }

    public GainShareAdamWState CaptureState() => CloneState(_state);

    public void RestoreState(GainShareAdamWState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        ValidateState(state);
        _state = CloneState(state);
    }

    internal void ZeroGrad()
    {
        foreach (Parameter parameter in _parameters)
            parameter.ZeroGrad();
    }

    public void zero_grad() => ZeroGrad();

    internal void Step()
    {
        if (_state.Step == int.MaxValue)
        {
            throw new InvalidOperationException(
                "GainShareAdamW cannot advance beyond Int32.MaxValue steps.");
        }

        _state = _state with { Step = _state.Step + 1 };
        GainShareAdamWOptions options = _state.Options;
        float biasCorrection1 =
            1f - MathF.Pow(options.Beta1, _state.Step);
        float biasCorrection2 =
            1f - MathF.Pow(options.Beta2, _state.Step);
        var directions = new float[_parameters.Count][];

        void ComputeDirection(int parameterIndex)
        {
            Parameter parameter = _parameters[parameterIndex];
            GainShareAdamWParameterState parameterState =
                _state.ParameterStates[parameterIndex];
            float[] gradient = parameter.T.GradientBuffer;
            float[] firstMoment = parameterState.FirstMoment;
            float[] secondMoment = parameterState.SecondMoment;
            var direction = new float[parameter.T.Numel];
            directions[parameterIndex] = direction;

            int index = 0;
            if (Tensor.SimdEnabled
                && Vector256.IsHardwareAccelerated
                && parameter.T.Numel >= Vector256<float>.Count)
            {
                int width = Vector256<float>.Count;
                int vectorizedLength = parameter.T.Numel
                    - parameter.T.Numel % width;
                Vector256<float> beta1 = Vector256.Create(options.Beta1);
                Vector256<float> beta2 = Vector256.Create(options.Beta2);
                Vector256<float> oneMinusBeta1 =
                    Vector256.Create(1f - options.Beta1);
                Vector256<float> oneMinusBeta2 =
                    Vector256.Create(1f - options.Beta2);
                Vector256<float> inverseBiasCorrection1 =
                    Vector256.Create(1f / biasCorrection1);
                Vector256<float> inverseBiasCorrection2 =
                    Vector256.Create(1f / biasCorrection2);
                Vector256<float> epsilon = Vector256.Create(options.Epsilon);

                for (; index < vectorizedLength; index += width)
                {
                    Vector256<float> gradientVector = gradient.Length == 0
                        ? Vector256<float>.Zero
                        : Vector256.LoadUnsafe(ref gradient[index]);
                    Vector256<float> first =
                        beta1 * Vector256.LoadUnsafe(ref firstMoment[index])
                        + oneMinusBeta1 * gradientVector;
                    Vector256<float> second =
                        beta2 * Vector256.LoadUnsafe(ref secondMoment[index])
                        + oneMinusBeta2
                            * gradientVector
                            * gradientVector;
                    first.StoreUnsafe(ref firstMoment[index]);
                    second.StoreUnsafe(ref secondMoment[index]);
                    Vector256<float> update =
                        (first * inverseBiasCorrection1)
                        / (Vector256.Sqrt(
                                second * inverseBiasCorrection2)
                            + epsilon);
                    update.StoreUnsafe(ref direction[index]);
                }
            }

            for (; index < parameter.T.Numel; index++)
            {
                float value = gradient.Length == 0 ? 0f : gradient[index];
                firstMoment[index] = options.Beta1 * firstMoment[index]
                    + (1f - options.Beta1) * value;
                secondMoment[index] = options.Beta2 * secondMoment[index]
                    + (1f - options.Beta2) * value * value;
                float firstHat = firstMoment[index] / biasCorrection1;
                float secondHat = secondMoment[index] / biasCorrection2;
                direction[index] = firstHat
                    / (MathF.Sqrt(secondHat) + options.Epsilon);
            }
        }

        long totalElements = 0;
        foreach (Parameter parameter in _parameters)
            totalElements += parameter.T.Numel;
        if (_parameters.Count > 1 && totalElements >= 32_768)
            Tensor.RunParallel(0, _parameters.Count, ComputeDirection);
        else
            for (int index = 0; index < _parameters.Count; index++)
                ComputeDirection(index);

        var energies = new double[_groups.Length];
        var smoothedAlignments = new double[_groups.Length];
        double totalEnergy = 0d;
        for (int groupIndex = 0; groupIndex < _groups.Length; groupIndex++)
        {
            double alignment = 0d;
            double energy = 0d;
            foreach (int parameterIndex in _groups[groupIndex])
            {
                float[] gradient =
                    _parameters[parameterIndex].T.GradientBuffer;
                float[] direction = directions[parameterIndex];
                AccumulateAlignmentAndEnergy(
                    gradient,
                    direction,
                    ref alignment,
                    ref energy);
            }

            double ratio = Math.Max(alignment, 0d)
                / (energy + options.Epsilon);
            double? previous =
                _state.GroupStates[groupIndex].AlignmentEma;
            double smoothed = previous is null
                ? ratio
                : options.Rho * previous.Value
                    + (1d - options.Rho) * ratio;
            _state.GroupStates[groupIndex] =
                _state.GroupStates[groupIndex] with
                {
                    AlignmentEma = smoothed,
                };
            energies[groupIndex] = energy;
            smoothedAlignments[groupIndex] = smoothed;
            totalEnergy += energy;
        }

        double weightedAlignment = 0d;
        for (int groupIndex = 0; groupIndex < _groups.Length; groupIndex++)
        {
            weightedAlignment += energies[groupIndex]
                * smoothedAlignments[groupIndex];
        }

        double targetAlignment = weightedAlignment
            / (totalEnergy + options.Epsilon);
        var rawScales = new double[_groups.Length];
        if (!double.IsFinite(targetAlignment)
            || targetAlignment <= options.Epsilon)
        {
            Array.Fill(rawScales, 1d);
        }
        else
        {
            for (int groupIndex = 0;
                groupIndex < _groups.Length;
                groupIndex++)
            {
                double relative = Math.Max(
                    smoothedAlignments[groupIndex],
                    0d) / targetAlignment;
                rawScales[groupIndex] = Math.Clamp(
                    Math.Pow(relative, options.Gamma),
                    options.MinScale,
                    options.MaxScale);
            }
        }

        double scaledEnergy = 0d;
        for (int groupIndex = 0; groupIndex < _groups.Length; groupIndex++)
        {
            scaledEnergy += rawScales[groupIndex]
                * rawScales[groupIndex]
                * energies[groupIndex];
        }

        double normalization = totalEnergy > 0d
            ? Math.Sqrt(totalEnergy / (scaledEnergy + options.Epsilon))
            : 1d;
        var scales = new float[_groups.Length];
        for (int groupIndex = 0; groupIndex < _groups.Length; groupIndex++)
            scales[groupIndex] = (float)(normalization * rawScales[groupIndex]);

        void UpdateParameter(int parameterIndex)
        {
            Parameter parameter = _parameters[parameterIndex];
            float scale = scales[_parameterGroupIndices[parameterIndex]];
            float[] direction = directions[parameterIndex];
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
                Vector256<float> updateScale = Vector256.Create(
                    options.LearningRate * scale);
                Vector256<float> decay = Vector256.Create(
                    1f - options.LearningRate * options.WeightDecay);
                for (; index < vectorizedLength; index += width)
                {
                    Vector256<float> values =
                        Vector256.LoadUnsafe(ref data[index]);
                    if (applyWeightDecay)
                        values *= decay;
                    values -= updateScale
                        * Vector256.LoadUnsafe(ref direction[index]);
                    values.StoreUnsafe(ref data[index]);
                }
            }

            float decayFactor =
                1f - options.LearningRate * options.WeightDecay;
            for (; index < data.Length; index++)
            {
                if (applyWeightDecay)
                    data[index] *= decayFactor;
                data[index] -= options.LearningRate
                    * scale
                    * direction[index];
            }
        }

        if (_parameters.Count > 1 && totalElements >= 32_768)
            Tensor.RunParallel(0, _parameters.Count, UpdateParameter);
        else
            for (int index = 0; index < _parameters.Count; index++)
                UpdateParameter(index);
    }

    public void step() => Step();

    public OptimizerStateDictionary state_dict()
        => OptimizerStateDictionary.Create(
            "GainShareAdamW",
            CaptureState());

    public void load_state_dict(OptimizerStateDictionary state)
    {
        ArgumentNullException.ThrowIfNull(state);
        RestoreState(state.Read<GainShareAdamWState>("GainShareAdamW"));
    }

    private static void AccumulateAlignmentAndEnergy(
        float[] gradient,
        float[] direction,
        ref double alignment,
        ref double energy)
    {
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && direction.Length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = direction.Length
                - direction.Length % width;
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> directionVector =
                    Vector256.LoadUnsafe(ref direction[index]);
                Vector256<float> gradientVector = gradient.Length == 0
                    ? Vector256<float>.Zero
                    : Vector256.LoadUnsafe(ref gradient[index]);
                alignment += Vector256.Sum(
                    gradientVector * directionVector);
                energy += Vector256.Sum(
                    directionVector * directionVector);
            }
        }

        for (; index < direction.Length; index++)
        {
            float gradientValue = gradient.Length == 0
                ? 0f
                : gradient[index];
            alignment += gradientValue * direction[index];
            energy += direction[index] * direction[index];
        }
    }

    private static GainShareAdamWState CreateInitialState(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int[]> groups,
        GainShareAdamWOptions options)
    {
        GainShareAdamWParameterState[] parameterStates = parameters
            .Select((parameter, index) =>
                new GainShareAdamWParameterState(
                    index,
                    parameter.Name,
                    parameter.T.Shape.ToArray(),
                    new float[parameter.T.Numel],
                    new float[parameter.T.Numel]))
            .ToArray();
        GainShareAdamWGroupState[] groupStates = groups
            .Select((indices, index) =>
                new GainShareAdamWGroupState(
                    index,
                    indices.ToArray(),
                    null))
            .ToArray();
        return new GainShareAdamWState(
            GainShareAdamWState.CurrentFormatVersion,
            0,
            options with { },
            parameterStates,
            groupStates);
    }

    private void ValidateState(GainShareAdamWState state)
    {
        if (state.FormatVersion != GainShareAdamWState.CurrentFormatVersion)
        {
            throw new ArgumentException(
                $"Unsupported GainShareAdamW state format version " +
                $"'{state.FormatVersion}'. Expected " +
                $"'{GainShareAdamWState.CurrentFormatVersion}'.",
                nameof(state));
        }
        if (state.Step < 0 || state.Step == int.MaxValue)
        {
            throw new ArgumentException(
                "GainShareAdamW state step must be non-negative and leave " +
                "room for another optimizer step.",
                nameof(state));
        }
        if (state.Options is null)
        {
            throw new ArgumentException(
                "GainShareAdamW state options cannot be null.",
                nameof(state));
        }
        ValidateOptions(state.Options, nameof(state));
        if (state.ParameterStates is null
            || state.ParameterStates.Length != _parameters.Count)
        {
            throw new ArgumentException(
                "GainShareAdamW state parameter count does not match the " +
                "optimizer.",
                nameof(state));
        }
        if (state.GroupStates is null
            || state.GroupStates.Length != _groups.Length)
        {
            throw new ArgumentException(
                "GainShareAdamW state group count does not match the " +
                "optimizer.",
                nameof(state));
        }

        for (int index = 0; index < _parameters.Count; index++)
        {
            Parameter parameter = _parameters[index];
            GainShareAdamWParameterState parameterState =
                state.ParameterStates[index];
            if (parameterState is null
                || parameterState.Index != index
                || !string.Equals(
                    parameterState.Name,
                    parameter.Name,
                    StringComparison.Ordinal)
                || parameterState.Shape is null
                || !parameterState.Shape.SequenceEqual(parameter.T.Shape)
                || parameterState.FirstMoment is null
                || parameterState.FirstMoment.Length != parameter.T.Numel
                || parameterState.SecondMoment is null
                || parameterState.SecondMoment.Length != parameter.T.Numel)
            {
                throw new ArgumentException(
                    $"GainShareAdamW parameter state for slot {index} is " +
                    "incompatible.",
                    nameof(state));
            }
        }

        for (int index = 0; index < _groups.Length; index++)
        {
            GainShareAdamWGroupState groupState = state.GroupStates[index];
            if (groupState is null
                || groupState.Index != index
                || groupState.ParameterIndices is null
                || !groupState.ParameterIndices.SequenceEqual(_groups[index])
                || (groupState.AlignmentEma is { } alignment
                    && (!double.IsFinite(alignment) || alignment < 0d)))
            {
                throw new ArgumentException(
                    $"GainShareAdamW group state for slot {index} is " +
                    "incompatible.",
                    nameof(state));
            }
        }
    }

    private static void ValidateOptions(
        GainShareAdamWOptions options,
        string parameterName)
    {
        if (!float.IsFinite(options.LearningRate)
            || options.LearningRate <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.LearningRate,
                "GainShareAdamW learning rate must be finite and positive.");
        }
        ValidateUnitInterval(options.Beta1, parameterName, "beta1");
        ValidateUnitInterval(options.Beta2, parameterName, "beta2");
        ValidateUnitInterval(options.Rho, parameterName, "rho");
        if (!float.IsFinite(options.Epsilon) || options.Epsilon <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.Epsilon,
                "GainShareAdamW epsilon must be finite and positive.");
        }
        if (!float.IsFinite(options.Gamma) || options.Gamma < 0f)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.Gamma,
                "GainShareAdamW gamma must be finite and non-negative.");
        }
        if (!float.IsFinite(options.MinScale) || options.MinScale <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.MinScale,
                "GainShareAdamW minimum scale must be finite and positive.");
        }
        if (!float.IsFinite(options.MaxScale)
            || options.MaxScale < options.MinScale)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.MaxScale,
                "GainShareAdamW maximum scale must be finite and not less " +
                "than the minimum scale.");
        }
        if (!float.IsFinite(options.WeightDecay)
            || options.WeightDecay < 0f)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.WeightDecay,
                "GainShareAdamW weight decay must be finite and " +
                "non-negative.");
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
                $"GainShareAdamW {optionName} must be finite and in " +
                "[0, 1).");
        }
    }

    private static GainShareAdamWState CloneState(
        GainShareAdamWState state)
    {
        return new GainShareAdamWState(
            state.FormatVersion,
            state.Step,
            state.Options with { },
            state.ParameterStates
                .Select(parameterState =>
                    new GainShareAdamWParameterState(
                        parameterState.Index,
                        parameterState.Name,
                        parameterState.Shape.ToArray(),
                        parameterState.FirstMoment.ToArray(),
                        parameterState.SecondMoment.ToArray()))
                .ToArray(),
            state.GroupStates
                .Select(groupState =>
                    new GainShareAdamWGroupState(
                        groupState.Index,
                        groupState.ParameterIndices.ToArray(),
                        groupState.AlignmentEma))
                .ToArray());
    }
}
