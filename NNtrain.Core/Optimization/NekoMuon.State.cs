namespace NNtrain;

public sealed partial class NekoMuon
{
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

        if (options.NewtonSchulzInterval <= 0)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.NewtonSchulzInterval,
                "NekoMuon Newton-Schulz interval must be positive.");
        }

        if (!Enum.IsDefined(options.NewtonSchulzDepthMode))
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.NewtonSchulzDepthMode,
                "NekoMuon Newton-Schulz depth mode is invalid.");
        }

        if (!float.IsFinite(options.NewtonSchulzDepth)
            || options.NewtonSchulzDepth < 0f
            || options.NewtonSchulzDepth > options.MaxNewtonSchulzSteps)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.NewtonSchulzDepth,
                "NekoMuon Newton-Schulz depth must be finite and in " +
                $"[0, {options.MaxNewtonSchulzSteps}].");
        }

        if (options.NewtonSchulzDepthMode
                == NekoMuonNewtonSchulzDepthMode.Adaptive
            && options.NewtonSchulzDepth != 0f)
        {
            throw new ArgumentException(
                "NekoMuon adaptive Newton-Schulz depth must not specify a " +
                "fixed depth.",
                parameterName);
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
