namespace NNtrain;

public sealed class Lion : IOptimizer, ILearningRateAdjustable, IDisposable
{
    private readonly List<Parameter> _parameters;
    private readonly object _cudaSync = new();
    private LionState _state;
    private CudaLionOptimizer? _cudaOptimizer;

    internal IReadOnlyList<Parameter> Parameters => _parameters;

    public Lion(
        IEnumerable<Parameter> parameters,
        LionOptions? options = null)
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
                    $"Parameter '{parameter.Name}' was supplied to Lion " +
                    "more than once.",
                    nameof(parameters));
            }

            _parameters.Add(parameter);
        }

        LionOptions effectiveOptions = options ?? new LionOptions();
        ValidateOptions(effectiveOptions, nameof(options));
        _state = CreateInitialState(_parameters, effectiveOptions);
    }

    public LionState CaptureState()
        => CloneState(CaptureStateForStreaming());

    internal LionState CaptureStateForStreaming()
    {
        SynchronizeCudaStateForCheckpoint();
        return _state;
    }

    public float LearningRate => _state.Options.LearningRate;

    public void SetLearningRate(float learningRate)
    {
        if (!float.IsFinite(learningRate) || learningRate <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(learningRate),
                learningRate,
                "Lion learning rate must be finite and positive.");
        }

        _state = _state with
        {
            Options = _state.Options with { LearningRate = learningRate },
        };
    }

    public void RestoreState(LionState state)
        => RestoreState(state, takeOwnership: false);

    private void RestoreState(LionState state, bool takeOwnership)
    {
        ArgumentNullException.ThrowIfNull(state);
        ValidateState(state);
        DisposeCudaResources();
        _state = takeOwnership ? state : CloneState(state);
    }

    internal void RestoreStateOwned(LionState state)
        => RestoreState(state, takeOwnership: true);

    internal void ZeroGrad()
    {
        if (Tensor.ExecutionDevice == TensorDevice.Cuda)
        {
            foreach (Parameter parameter in _parameters)
                parameter.T.ClearGradient();
            return;
        }
        foreach (Parameter parameter in _parameters)
            parameter.ZeroGrad();
    }

    public void zero_grad() => ZeroGrad();

    internal void Step()
    {
        if (_state.Step == int.MaxValue)
        {
            throw new InvalidOperationException(
                "Lion cannot advance beyond Int32.MaxValue steps.");
        }

        if (Tensor.ExecutionDevice == TensorDevice.Cuda)
        {
            CudaGradientOptimizerGuard.ValidateAndConsume(
                _parameters,
                Tensor.CudaDeviceIndices);
        }
        else
        {
            MakeCpuStateAuthoritative();
        }

        _state = _state with { Step = _state.Step + 1 };
        LionOptions options = _state.Options;

        if (Tensor.ExecutionDevice == TensorDevice.Cuda)
        {
            GetOrCreateCudaOptimizer().Step(
                Tensor.CudaDeviceIndices,
                options,
                _state.Step);
            return;
        }

        void UpdateParameter(int parameterIndex)
        {
            Parameter parameter = _parameters[parameterIndex];
            LionParameterState parameterState =
                _state.ParameterStates[parameterIndex];
            using Tensor.DataMutation mutation = parameter.BeginUpdate();
            Span<float> data = mutation.Values;
            float[] gradientBuffer = parameter.T.GradientBuffer;
            float[] momentum = parameterState.Momentum;

            bool applyWeightDecay =
                parameter.WeightDecay == WeightDecayPolicy.Apply
                || (options.Decay1D && parameter.T.Rank == 1);
            int length = parameter.T.Numel;
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
                Vector256<float> learningRate =
                    Vector256.Create(options.LearningRate);
                Vector256<float> decay = Vector256.Create(
                    options.LearningRate * options.WeightDecay);
                Vector256<float> zero = Vector256<float>.Zero;
                Vector256<float> one = Vector256.Create(1f);
                Vector256<float> minusOne = Vector256.Create(-1f);

                for (; index < vectorizedLength; index += vectorWidth)
                {
                    Vector256<float> gradient = gradientBuffer.Length == 0
                        ? zero
                        : Vector256.LoadUnsafe(ref gradientBuffer[index]);
                    Vector256<float> previousMomentum =
                        Vector256.LoadUnsafe(ref momentum[index]);
                    Vector256<float> direction =
                        beta1 * previousMomentum
                        + oneMinusBeta1 * gradient;
                    Vector256<float> sign = Vector256.ConditionalSelect(
                        Vector256.GreaterThan(direction, zero),
                        one,
                        Vector256.ConditionalSelect(
                            Vector256.LessThan(direction, zero),
                            minusOne,
                            direction));

                    Vector256<float> parameterValues =
                        Vector256.LoadUnsafe(ref data[index]);
                    if (applyWeightDecay)
                        parameterValues -= decay * parameterValues;
                    parameterValues -= learningRate * sign;
                    parameterValues.StoreUnsafe(ref data[index]);

                    Vector256<float> nextMomentum =
                        beta2 * previousMomentum
                        + oneMinusBeta2 * gradient;
                    nextMomentum.StoreUnsafe(ref momentum[index]);
                }
            }

            for (; index < length; index++)
            {
                float gradient = gradientBuffer.Length == 0
                    ? 0f
                    : gradientBuffer[index];
                float direction = options.Beta1 * momentum[index]
                    + (1f - options.Beta1) * gradient;
                float sign = direction > 0f
                    ? 1f
                    : direction < 0f
                        ? -1f
                        : direction;

                if (applyWeightDecay)
                {
                    data[index] -= options.LearningRate
                        * options.WeightDecay
                        * data[index];
                }

                data[index] -= options.LearningRate * sign;
                momentum[index] = options.Beta2 * momentum[index]
                    + (1f - options.Beta2) * gradient;
            }
        }

        long totalElements = 0;
        foreach (Parameter parameter in _parameters)
            totalElements += parameter.T.Numel;

        if (_parameters.Count > 1 && totalElements >= 32_768)
            Tensor.RunParallel(0, _parameters.Count, UpdateParameter);
        else
            for (int index = 0; index < _parameters.Count; index++)
                UpdateParameter(index);
    }

    public void step() => Step();

    /// <summary>
    /// Materializes persistent Lion moment storage and fused multi-tensor
    /// plans before a transfer-guarded CUDA training step.
    /// </summary>
    public void prepare()
    {
        if (Tensor.ExecutionDevice != TensorDevice.Cuda)
            return;
        GetOrCreateCudaOptimizer().Prepare(Tensor.CudaDeviceIndices);
    }

    public OptimizerStateDictionary state_dict()
        => OptimizerStateDictionary.Create("Lion", CaptureState());

    public void load_state_dict(OptimizerStateDictionary state)
    {
        ArgumentNullException.ThrowIfNull(state);
        RestoreStateOwned(state.Read<LionState>("Lion"));
    }

    internal void DisposeCudaResources()
    {
        CudaLionOptimizer? optimizer;
        lock (_cudaSync)
        {
            optimizer = _cudaOptimizer;
            _cudaOptimizer = null;
        }
        optimizer?.Dispose();
    }

    public void Dispose() => DisposeCudaResources();

    private CudaLionOptimizer GetOrCreateCudaOptimizer()
    {
        lock (_cudaSync)
        {
            return _cudaOptimizer ??= new CudaLionOptimizer(
                _parameters,
                _state.ParameterStates);
        }
    }

    private void SynchronizeCudaStateForCheckpoint()
    {
        CudaLionOptimizer? optimizer;
        lock (_cudaSync)
            optimizer = _cudaOptimizer;
        if (optimizer is null)
            return;
        int primaryDevice = Tensor.CudaDeviceIndices.Count > 0
            ? Tensor.CudaDeviceIndices[0]
            : Tensor.CudaDeviceIndex;
        optimizer.SynchronizeHost(primaryDevice);
    }

    private void MakeCpuStateAuthoritative()
    {
        CudaLionOptimizer? optimizer;
        lock (_cudaSync)
        {
            optimizer = _cudaOptimizer;
            _cudaOptimizer = null;
        }
        if (optimizer is null)
            return;
        try
        {
            int primaryDevice = Tensor.CudaDeviceIndices.Count > 0
                ? Tensor.CudaDeviceIndices[0]
                : Tensor.CudaDeviceIndex;
            optimizer.SynchronizeHost(primaryDevice);
        }
        finally
        {
            optimizer.Dispose();
        }
    }

    private static LionState CreateInitialState(
        IReadOnlyList<Parameter> parameters,
        LionOptions options)
    {
        LionParameterState[] parameterStates = parameters
            .Select((parameter, index) =>
                new LionParameterState(
                    index,
                    parameter.Name,
                    parameter.T.Shape.ToArray(),
                    new float[parameter.T.Numel]))
            .ToArray();

        return new LionState(
            LionState.CurrentFormatVersion,
            0,
            options with { },
            parameterStates);
    }

    private void ValidateState(LionState state)
    {
        if (state.FormatVersion != LionState.CurrentFormatVersion)
        {
            throw new ArgumentException(
                $"Unsupported Lion state format version " +
                $"'{state.FormatVersion}'. Expected " +
                $"'{LionState.CurrentFormatVersion}'.",
                nameof(state));
        }

        if (state.Step < 0 || state.Step == int.MaxValue)
        {
            throw new ArgumentException(
                "Lion state step must be non-negative and leave room for " +
                "another optimizer step.",
                nameof(state));
        }

        if (state.Options is null)
        {
            throw new ArgumentException(
                "Lion state options cannot be null.",
                nameof(state));
        }

        ValidateOptions(state.Options, nameof(state));

        if (state.ParameterStates is null)
        {
            throw new ArgumentException(
                "Lion parameter states cannot be null.",
                nameof(state));
        }

        if (state.ParameterStates.Length != _parameters.Count)
        {
            throw new ArgumentException(
                $"Lion state contains {state.ParameterStates.Length} " +
                $"parameter slots, but the optimizer manages " +
                $"{_parameters.Count}.",
                nameof(state));
        }

        for (int index = 0; index < _parameters.Count; index++)
        {
            Parameter parameter = _parameters[index];
            LionParameterState parameterState = state.ParameterStates[index];

            if (parameterState is null)
            {
                throw new ArgumentException(
                    $"Lion parameter state at index {index} cannot be null.",
                    nameof(state));
            }

            if (parameterState.Index != index)
            {
                throw new ArgumentException(
                    $"Lion parameter state index '{parameterState.Index}' " +
                    $"does not match slot '{index}'.",
                    nameof(state));
            }

            if (!string.Equals(
                parameterState.Name,
                parameter.Name,
                StringComparison.Ordinal))
            {
                throw new ArgumentException(
                    $"Lion parameter state at index {index} is named " +
                    $"'{parameterState.Name}', but the optimizer parameter " +
                    $"is named '{parameter.Name}'.",
                    nameof(state));
            }

            if (parameterState.Shape is null
                || !parameterState.Shape.SequenceEqual(parameter.T.Shape))
            {
                throw new ArgumentException(
                    $"Lion parameter state for '{parameter.Name}' has an " +
                    "incompatible shape.",
                    nameof(state));
            }

            if (parameterState.Momentum is null
                || parameterState.Momentum.Length != parameter.T.Numel)
            {
                throw new ArgumentException(
                    $"Lion momentum for '{parameter.Name}' has an " +
                    "incompatible length.",
                    nameof(state));
            }
        }
    }

    private static void ValidateOptions(
        LionOptions options,
        string parameterName)
    {
        if (!float.IsFinite(options.LearningRate)
            || options.LearningRate <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.LearningRate,
                "Lion learning rate must be finite and positive.");
        }

        if (!float.IsFinite(options.Beta1)
            || options.Beta1 < 0f
            || options.Beta1 >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.Beta1,
                "Lion beta1 must be finite and in the range [0, 1).");
        }

        if (!float.IsFinite(options.Beta2)
            || options.Beta2 < 0f
            || options.Beta2 >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.Beta2,
                "Lion beta2 must be finite and in the range [0, 1).");
        }

        if (!float.IsFinite(options.WeightDecay)
            || options.WeightDecay < 0f)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                options.WeightDecay,
                "Lion weight decay must be finite and non-negative.");
        }
    }

    private static LionState CloneState(LionState state)
    {
        return new LionState(
            state.FormatVersion,
            state.Step,
            state.Options with { },
            state.ParameterStates
                .Select(parameterState =>
                    new LionParameterState(
                        parameterState.Index,
                        parameterState.Name,
                        parameterState.Shape.ToArray(),
                        parameterState.Momentum.ToArray()))
                .ToArray());
    }
}
