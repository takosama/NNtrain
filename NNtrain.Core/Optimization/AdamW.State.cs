namespace NNtrain;

public partial class AdamW
{
    private static AdamWParameterRuntime[] CreateParameterRuntime(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<AdamWParameterState> states,
        AdamWOptions options)
    {
        var result = new AdamWParameterRuntime[parameters.Count];
        for (int index = 0; index < result.Length; index++)
        {
            Parameter parameter = parameters[index];
            result[index] = new AdamWParameterRuntime(
                parameter,
                parameter.DataBuffer,
                parameter.T.GradientBuffer,
                states[index].FirstMoment,
                states[index].SecondMoment,
                options.UseBFloat16FirstMoment
                    ? new short[parameter.T.Numel]
                    : null,
                options.UseBFloat16SecondMoment
                    ? new short[parameter.T.Numel]
                    : null,
                ShouldApplyWeightDecay(parameter, options));
        }
        return result;
    }

    private void RefreshWeightDecayFlags()
    {
        for (int index = 0; index < _parameterRuntime.Length; index++)
        {
            _parameterRuntime[index].ApplyWeightDecay =
                ShouldApplyWeightDecay(_parameters[index], _options);
        }
    }

    private static bool ShouldApplyWeightDecay(
        Parameter parameter,
        AdamWOptions options)
        => parameter.WeightDecay == WeightDecayPolicy.Apply
            || (options.Decay1D && parameter.T.Rank == 1);

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
                    options.UseBFloat16FirstMoment
                        ? []
                        : new float[parameter.T.Numel],
                    options.UseBFloat16SecondMoment
                        ? []
                        : new float[parameter.T.Numel]))
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

    private readonly record struct AdamWWorkItem(
        int ParameterIndex,
        int Start,
        int Length);

    private sealed class AdamWParameterRuntime(
        Parameter parameter,
        float[] data,
        float[] gradient,
        float[] firstMoment,
        float[] secondMoment,
        short[]? firstMomentBFloat16,
        short[]? secondMomentBFloat16,
        bool applyWeightDecay)
    {
        internal Parameter Parameter { get; } = parameter;
        internal float[] Data { get; } = data;
        internal float[] Gradient { get; set; } = gradient;
        internal float[] FirstMoment { get; set; } = firstMoment;
        internal short[]? FirstMomentBFloat16 { get; set; } =
            firstMomentBFloat16;
        internal float[] SecondMoment { get; set; } = secondMoment;
        internal short[]? SecondMomentBFloat16 { get; set; } =
            secondMomentBFloat16;
        internal bool ApplyWeightDecay { get; set; } = applyWeightDecay;
        internal CudaOptimizerKernels.AdamWResidentState? CudaState
        {
            get;
            set;
        }
        internal CudaOptimizerKernels.AdamWBFloat16ResidentState?
            CudaBFloat16State
        {
            get;
            set;
        }
    }
}

internal readonly record struct AdamWStreamingParameterState(
    int Index,
    string Name,
    int[] Shape,
    float[] FirstMoment,
    short[]? FirstMomentBFloat16,
    float[] SecondMoment,
    short[]? SecondMomentBFloat16);
