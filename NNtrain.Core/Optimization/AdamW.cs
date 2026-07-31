namespace NNtrain;

public class AdamW : IOptimizer
{
    private readonly List<Parameter> _parameters;
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

        _state = CreateInitialState(
            _parameters,
            options ?? new AdamWOptions());
    }

    public AdamWState CaptureState()
    {
        return CloneState(_state);
    }

    public void RestoreState(AdamWState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        ValidateState(state);
        _state = CloneState(state);
    }

    public void ZeroGrad()
    {
        foreach (var p in _parameters)
            p.ZeroGrad();
    }

    public void Step()
    {
        _state = _state with { Step = _state.Step + 1 };
        AdamWOptions options = _state.Options;

        float bc1 = 1f - MathF.Pow(options.Beta1, _state.Step);
        float bc2 = 1f - MathF.Pow(options.Beta2, _state.Step);

        for (int parameterIndex = 0;
            parameterIndex < _parameters.Count;
            parameterIndex++)
        {
            Parameter p = _parameters[parameterIndex];
            AdamWParameterState parameterState =
                _state.ParameterStates[parameterIndex];
            using Tensor.DataMutation mutation = p.BeginUpdate();
            Span<float> data = mutation.Values;
            IReadOnlyList<float> grad = p.T.Grad;
            float[] m = parameterState.FirstMoment;
            float[] v = parameterState.SecondMoment;

            bool applyWeightDecay =
                p.WeightDecay == WeightDecayPolicy.Apply
                || (options.Decay1D && p.T.Rank == 1);

            for (int i = 0; i < p.T.Numel; i++)
            {
                float g = grad[i];

                // Adam
                m[i] = options.Beta1 * m[i] + (1f - options.Beta1) * g;
                v[i] = options.Beta2 * v[i] + (1f - options.Beta2) * g * g;

                float mHat = m[i] / bc1;
                float vHat = v[i] / bc2;

                // AdamW: decoupled weight decay
                if (applyWeightDecay)
                    data[i] -=
                        options.LearningRate * options.WeightDecay * data[i];

                data[i] -=
                    options.LearningRate
                    * mHat
                    / (MathF.Sqrt(vHat) + options.Epsilon);
            }
        }
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

        if (state.Step < 0)
        {
            throw new ArgumentException(
                "AdamW state step cannot be negative.",
                nameof(state));
        }

        if (state.Options is null)
        {
            throw new ArgumentException(
                "AdamW state options cannot be null.",
                nameof(state));
        }

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
