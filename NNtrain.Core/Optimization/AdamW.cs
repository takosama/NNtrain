namespace NNtrain;

public class AdamW : IOptimizer, ILearningRateAdjustable
{
    private readonly List<Parameter> _parameters;
    private readonly long _totalElements;
    private AdamWState _state;

    public AdamW(
        IEnumerable<Parameter> parameters,
        AdamWOptions? options = null)
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
                    $"Parameter '{parameter.Name}' was supplied to AdamW " +
                    "more than once.",
                    nameof(parameters));
            }

            _parameters.Add(parameter);
        }

        _totalElements = _parameters.Sum(
            parameter => (long)parameter.T.Numel);

        AdamWOptions effectiveOptions = options ?? new AdamWOptions();
        ValidateOptions(effectiveOptions, nameof(options));
        _state = CreateInitialState(_parameters, effectiveOptions);
    }

    public AdamWState CaptureState()
    {
        return CloneState(_state);
    }

    public float LearningRate => _state.Options.LearningRate;

    public void SetLearningRate(float learningRate)
    {
        if (!float.IsFinite(learningRate) || learningRate <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(learningRate),
                learningRate,
                "AdamW learning rate must be finite and positive.");
        }

        _state = _state with
        {
            Options = _state.Options with { LearningRate = learningRate },
        };
    }

    public void RestoreState(AdamWState state)
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
                "AdamW cannot advance beyond Int32.MaxValue steps.");
        }

        _state = _state with { Step = _state.Step + 1 };
        AdamWOptions options = _state.Options;

        float bc1 = 1f - MathF.Pow(options.Beta1, _state.Step);
        float bc2 = 1f - MathF.Pow(options.Beta2, _state.Step);

        void UpdateParameter(int parameterIndex)
        {
            Parameter p = _parameters[parameterIndex];
            AdamWParameterState parameterState =
                _state.ParameterStates[parameterIndex];
            using Tensor.DataMutation mutation = p.BeginUpdate();
            Span<float> data = mutation.Values;
            float[] grad = p.T.GradientBuffer;
            float[] m = parameterState.FirstMoment;
            float[] v = parameterState.SecondMoment;

            bool applyWeightDecay =
                p.WeightDecay == WeightDecayPolicy.Apply
                || (options.Decay1D && p.T.Rank == 1);
            int length = p.T.Numel;
            int index = 0;

            if (Tensor.SimdEnabled
                && Vector256.IsHardwareAccelerated
                && length >= Vector256<float>.Count)
            {
                int vectorWidth = Vector256<float>.Count;
                int vectorizedLength = length - length % vectorWidth;
                Vector256<float> beta1 = Vector256.Create(options.Beta1);
                Vector256<float> beta2 = Vector256.Create(options.Beta2);
                Vector256<float> oneMinusBeta1 =
                    Vector256.Create(1f - options.Beta1);
                Vector256<float> oneMinusBeta2 =
                    Vector256.Create(1f - options.Beta2);
                Vector256<float> inverseBc1 = Vector256.Create(1f / bc1);
                Vector256<float> inverseBc2 = Vector256.Create(1f / bc2);
                Vector256<float> learningRate =
                    Vector256.Create(options.LearningRate);
                Vector256<float> epsilon =
                    Vector256.Create(options.Epsilon);
                Vector256<float> decay = Vector256.Create(
                    options.LearningRate * options.WeightDecay);

                for (; index < vectorizedLength; index += vectorWidth)
                {
                    Vector256<float> gradient = grad.Length == 0
                        ? Vector256<float>.Zero
                        : Vector256.LoadUnsafe(ref grad[index]);
                    Vector256<float> firstMoment =
                        beta1 * Vector256.LoadUnsafe(ref m[index])
                        + oneMinusBeta1 * gradient;
                    Vector256<float> secondMoment =
                        beta2 * Vector256.LoadUnsafe(ref v[index])
                        + oneMinusBeta2 * gradient * gradient;
                    firstMoment.StoreUnsafe(ref m[index]);
                    secondMoment.StoreUnsafe(ref v[index]);

                    Vector256<float> parameter =
                        Vector256.LoadUnsafe(ref data[index]);
                    if (applyWeightDecay)
                        parameter -= decay * parameter;
                    parameter -= learningRate
                        * (firstMoment * inverseBc1)
                        / (Vector256.Sqrt(secondMoment * inverseBc2)
                            + epsilon);
                    parameter.StoreUnsafe(ref data[index]);
                }
            }

            for (; index < length; index++)
            {
                float g = grad.Length == 0 ? 0f : grad[index];

                // Adam
                m[index] = options.Beta1 * m[index]
                    + (1f - options.Beta1) * g;
                v[index] = options.Beta2 * v[index]
                    + (1f - options.Beta2) * g * g;

                float mHat = m[index] / bc1;
                float vHat = v[index] / bc2;

                // AdamW: decoupled weight decay
                if (applyWeightDecay)
                    data[index] -= options.LearningRate
                        * options.WeightDecay
                        * data[index];

                data[index] -=
                    options.LearningRate
                    * mHat
                    / (MathF.Sqrt(vHat) + options.Epsilon);
            }
        }

        if (_parameters.Count > 1 && _totalElements >= 32_768)
            Tensor.RunParallel(0, _parameters.Count, UpdateParameter);
        else
            for (int index = 0; index < _parameters.Count; index++)
                UpdateParameter(index);
    }

    private static AdamWState CreateInitialState(
        IReadOnlyList<Parameter> parameters,
        AdamWOptions options)
    {
        AdamWParameterState[] parameterStates = parameters
            .Select((parameter, index) =>
                new AdamWParameterState(
                    index,
                    parameter.Name,
                    parameter.T.Shape.ToArray(),
                    new float[parameter.T.Numel],
                    new float[parameter.T.Numel]))
            .ToArray();

        return new AdamWState(
            AdamWState.CurrentFormatVersion,
            0,
            options with { },
            parameterStates);
    }

    private void ValidateState(AdamWState state)
    {
        if (state.FormatVersion != AdamWState.CurrentFormatVersion)
        {
            throw new ArgumentException(
                $"Unsupported AdamW state format version " +
                $"'{state.FormatVersion}'. Expected " +
                $"'{AdamWState.CurrentFormatVersion}'.",
                nameof(state));
        }

        if (state.Step < 0 || state.Step == int.MaxValue)
        {
            throw new ArgumentException(
                "AdamW state step must be non-negative and leave room for " +
                "another optimizer step.",
                nameof(state));
        }

        if (state.Options is null)
        {
            throw new ArgumentException(
                "AdamW state options cannot be null.",
                nameof(state));
        }

        ValidateOptions(state.Options, nameof(state));

        if (state.ParameterStates is null)
        {
            throw new ArgumentException(
                "AdamW parameter states cannot be null.",
                nameof(state));
        }

        if (state.ParameterStates.Length != _parameters.Count)
        {
            throw new ArgumentException(
                $"AdamW state contains {state.ParameterStates.Length} " +
                $"parameter slots, but the optimizer manages " +
                $"{_parameters.Count}.",
                nameof(state));
        }

        for (int index = 0; index < _parameters.Count; index++)
        {
            Parameter parameter = _parameters[index];
            AdamWParameterState parameterState =
                state.ParameterStates[index];

            if (parameterState is null)
            {
                throw new ArgumentException(
                    $"AdamW parameter state at index {index} cannot be null.",
                    nameof(state));
            }

            if (parameterState.Index != index)
            {
                throw new ArgumentException(
                    $"AdamW parameter state index '{parameterState.Index}' " +
                    $"does not match slot '{index}'.",
                    nameof(state));
            }

            if (!string.Equals(
                parameterState.Name,
                parameter.Name,
                StringComparison.Ordinal))
            {
                throw new ArgumentException(
                    $"AdamW parameter state at index {index} is named " +
                    $"'{parameterState.Name}', but the optimizer parameter " +
                    $"is named '{parameter.Name}'.",
                    nameof(state));
            }

            if (parameterState.Shape is null
                || !parameterState.Shape.SequenceEqual(parameter.T.Shape))
            {
                throw new ArgumentException(
                    $"AdamW parameter state for '{parameter.Name}' has an " +
                    "incompatible shape.",
                    nameof(state));
            }

            if (parameterState.FirstMoment is null
                || parameterState.FirstMoment.Length != parameter.T.Numel)
            {
                throw new ArgumentException(
                    $"AdamW first moment for '{parameter.Name}' has an " +
                    "incompatible length.",
                    nameof(state));
            }

            if (parameterState.SecondMoment is null
                || parameterState.SecondMoment.Length != parameter.T.Numel)
            {
                throw new ArgumentException(
                    $"AdamW second moment for '{parameter.Name}' has an " +
                    "incompatible length.",
                    nameof(state));
            }
        }
    }

    private static void ValidateOptions(
        AdamWOptions options,
        string parameterName)
    {
        if (!float.IsFinite(options.LearningRate)
            || options.LearningRate <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.LearningRate,
                "AdamW learning rate must be finite and positive.");
        }

        if (!float.IsFinite(options.Beta1)
            || options.Beta1 < 0f
            || options.Beta1 >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.Beta1,
                "AdamW beta1 must be finite and in the range [0, 1).");
        }

        if (!float.IsFinite(options.Beta2)
            || options.Beta2 < 0f
            || options.Beta2 >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.Beta2,
                "AdamW beta2 must be finite and in the range [0, 1).");
        }

        if (!float.IsFinite(options.Epsilon) || options.Epsilon <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.Epsilon,
                "AdamW epsilon must be finite and positive.");
        }

        if (!float.IsFinite(options.WeightDecay)
            || options.WeightDecay < 0f)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.WeightDecay,
                "AdamW weight decay must be finite and non-negative.");
        }
    }

    private static AdamWState CloneState(AdamWState state)
    {
        return new AdamWState(
            state.FormatVersion,
            state.Step,
            state.Options with { },
            state.ParameterStates
                .Select(parameterState =>
                    new AdamWParameterState(
                        parameterState.Index,
                        parameterState.Name,
                        parameterState.Shape.ToArray(),
                        parameterState.FirstMoment.ToArray(),
                        parameterState.SecondMoment.ToArray()))
                .ToArray());
    }
}
