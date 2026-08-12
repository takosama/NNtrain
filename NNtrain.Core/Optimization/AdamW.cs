namespace NNtrain;

public class AdamW : IOptimizer, ILearningRateAdjustable
{
    private readonly List<Parameter> _parameters;
    private readonly long _totalElements;
    private readonly AdamWParameterRuntime[] _parameterRuntime;
    private readonly AdamWWorkItem[] _workItems;
    private AdamWOptions _options;
    private AdamWParameterState[] _parameterStates;
    private readonly Action<int> _updateWorkItemAction;
    private readonly Action<int> _clearWorkItemAction;
    private AdamWOptions _stepOptions = null!;
    private float _stepUpdateScale;
    private float _stepScaledEpsilon;
    private int _step;

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
        _workItems = CreateWorkItems(_parameters);

        AdamWOptions effectiveOptions = options ?? new AdamWOptions();
        ValidateOptions(effectiveOptions, nameof(options));
        AdamWState initialState = CreateInitialState(
            _parameters,
            effectiveOptions);
        _options = initialState.Options;
        _parameterStates = initialState.ParameterStates;
        _parameterRuntime = CreateParameterRuntime(
            _parameters,
            _parameterStates,
            _options);
        _updateWorkItemAction = UpdateWorkItem;
        _clearWorkItemAction = ClearWorkItem;
    }

    public AdamWState CaptureState()
    {
        if (_options.UseBFloat16FirstMoment
            || _options.UseBFloat16SecondMoment)
        {
            var states = new AdamWParameterState[_parameters.Count];
            for (int index = 0; index < states.Length; index++)
            {
                Parameter parameter = _parameters[index];
                states[index] = new AdamWParameterState(
                    index,
                    parameter.Name,
                    parameter.T.Shape.ToArray(),
                    _options.UseBFloat16FirstMoment
                        ? DecodeBFloat16(
                            _parameterRuntime[index]
                                .FirstMomentBFloat16!)
                        : _parameterRuntime[index].FirstMoment.ToArray(),
                    _options.UseBFloat16SecondMoment
                        ? DecodeBFloat16(
                            _parameterRuntime[index]
                                .SecondMomentBFloat16!)
                        : _parameterRuntime[index].SecondMoment.ToArray());
            }
            return new AdamWState(
                AdamWState.CurrentFormatVersion,
                _step,
                _options with { },
                states);
        }

        return CloneState(
            new AdamWState(
                AdamWState.CurrentFormatVersion,
                _step,
                _options,
                _parameterStates));
    }

    public float LearningRate => _options.LearningRate;

    public void SetLearningRate(float learningRate)
    {
        if (!float.IsFinite(learningRate) || learningRate <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(learningRate),
                learningRate,
                "AdamW learning rate must be finite and positive.");
        }

        _options = _options with { LearningRate = learningRate };
        RefreshWeightDecayFlags();
    }

    public void RestoreState(AdamWState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        ValidateState(state);
        AdamWState clone = CloneState(state);
        _step = clone.Step;
        _options = clone.Options;
        _parameterStates = clone.ParameterStates;
        for (int index = 0; index < _parameterRuntime.Length; index++)
        {
            if (_options.UseBFloat16FirstMoment)
            {
                _parameterRuntime[index].FirstMoment = [];
                _parameterRuntime[index].FirstMomentBFloat16 =
                    EncodeBFloat16(_parameterStates[index].FirstMoment);
                _parameterStates[index] = _parameterStates[index] with
                {
                    FirstMoment = [],
                };
            }
            else
            {
                _parameterRuntime[index].FirstMoment =
                    _parameterStates[index].FirstMoment;
                _parameterRuntime[index].FirstMomentBFloat16 = null;
            }
            if (_options.UseBFloat16SecondMoment)
            {
                _parameterRuntime[index].SecondMoment = [];
                _parameterRuntime[index].SecondMomentBFloat16 =
                    EncodeBFloat16(_parameterStates[index].SecondMoment);
                _parameterStates[index] = _parameterStates[index] with
                {
                    SecondMoment = [],
                };
            }
            else
            {
                _parameterRuntime[index].SecondMoment =
                    _parameterStates[index].SecondMoment;
                _parameterRuntime[index].SecondMomentBFloat16 = null;
            }
        }
        RefreshWeightDecayFlags();
    }

    public void ZeroGrad()
    {
        if (_workItems.Length > 1 && _totalElements >= 32_768)
        {
            Tensor.RunParallel(
                0,
                _workItems.Length,
                _clearWorkItemAction);
            return;
        }

        for (int index = 0; index < _workItems.Length; index++)
            ClearWorkItem(index);
    }

    private void ClearWorkItem(int workItemIndex)
    {
        AdamWWorkItem workItem = _workItems[workItemIndex];
        _parameterRuntime[workItem.ParameterIndex]
            .Parameter
            .T
            .ClearGradientRange(workItem.Start, workItem.Length);
    }

    public void Step()
    {
        if (_step == int.MaxValue)
        {
            throw new InvalidOperationException(
                "AdamW cannot advance beyond Int32.MaxValue steps.");
        }

        _step++;
        AdamWOptions options = _options;

        float bc1 = 1f - MathF.Pow(options.Beta1, _step);
        float bc2 = 1f - MathF.Pow(options.Beta2, _step);
        float sqrtBc2 = MathF.Sqrt(bc2);
        _stepOptions = options;
        _stepUpdateScale = options.LearningRate * sqrtBc2 / bc1;
        _stepScaledEpsilon = options.Epsilon * sqrtBc2;

        for (int parameterIndex = 0;
            parameterIndex < _parameters.Count;
            parameterIndex++)
        {
            AdamWParameterRuntime runtime =
                _parameterRuntime[parameterIndex];
            runtime.Parameter.MarkUpdated();
            runtime.Gradient = runtime.Parameter.T.GradientBuffer;
        }

        if (_workItems.Length > 1 && _totalElements >= 32_768)
            Tensor.RunParallel(0, _workItems.Length, _updateWorkItemAction);
        else
            for (int index = 0; index < _workItems.Length; index++)
                UpdateWorkItem(index);
    }

    private void UpdateWorkItem(int workItemIndex)
    {
        AdamWOptions options = _stepOptions;
        float updateScale = _stepUpdateScale;
        float scaledEpsilon = _stepScaledEpsilon;
        AdamWWorkItem workItem = _workItems[workItemIndex];
        AdamWParameterRuntime runtime =
            _parameterRuntime[workItem.ParameterIndex];
        float[] data = runtime.Data;
        float[] grad = runtime.Gradient;
        float[] m = runtime.FirstMoment;
        short[]? mBFloat16 = runtime.FirstMomentBFloat16;
        float[] v = runtime.SecondMoment;
        short[]? vBFloat16 = runtime.SecondMomentBFloat16;
        bool applyWeightDecay = runtime.ApplyWeightDecay;
        int end = workItem.Start + workItem.Length;
        int index = workItem.Start;

        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && (mBFloat16 is null
                && vBFloat16 is null
                || System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            && workItem.Length >= Vector256<float>.Count)
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = end - workItem.Length % vectorWidth;
            Vector256<float> beta1 = Vector256.Create(options.Beta1);
            Vector256<float> beta2 = Vector256.Create(options.Beta2);
            Vector256<float> oneMinusBeta1 =
                Vector256.Create(1f - options.Beta1);
            Vector256<float> oneMinusBeta2 =
                Vector256.Create(1f - options.Beta2);
            Vector256<float> updateScaleVector =
                Vector256.Create(updateScale);
            Vector256<float> epsilon =
                Vector256.Create(scaledEpsilon);
            Vector256<float> parameterScale = Vector256.Create(
                applyWeightDecay
                    ? 1f - options.LearningRate * options.WeightDecay
                    : 1f);
            Vector256<float> one = Vector256.Create(1f);
            Vector256<float> two = Vector256.Create(2f);
            Vector256<float> half = Vector256.Create(0.5f);
            Vector256<float> three = Vector256.Create(3f);
            Vector256<float> inverseZeroDenominator =
                Vector256.Create(1f / scaledEpsilon);
            ref float dataStart = ref System.Runtime.InteropServices
                .MemoryMarshal.GetArrayDataReference(data);
            ref float gradientStart = ref System.Runtime.InteropServices
                .MemoryMarshal.GetArrayDataReference(grad);
            ref float firstMomentStart = ref System.Runtime.InteropServices
                .MemoryMarshal.GetArrayDataReference(m);
            ref float secondMomentStart = ref System.Runtime.InteropServices
                .MemoryMarshal.GetArrayDataReference(v);

            for (; index < vectorizedLength; index += vectorWidth)
            {
                Vector256<float> gradient = grad.Length == 0
                    ? Vector256<float>.Zero
                    : Vector256.LoadUnsafe(
                        ref System.Runtime.CompilerServices.Unsafe.Add(
                            ref gradientStart,
                            index));
                Vector256<float> firstMoment =
                    Vector256.FusedMultiplyAdd(
                        oneMinusBeta1,
                        gradient,
                        beta1 * (mBFloat16 is null
                            ? Vector256.LoadUnsafe(
                                ref System.Runtime.CompilerServices.Unsafe.Add(
                                    ref firstMomentStart,
                                    index))
                            : LoadBFloat16(mBFloat16, index)));
                Vector256<float> secondMoment =
                    Vector256.FusedMultiplyAdd(
                        oneMinusBeta2 * gradient,
                        gradient,
                        beta2 * (vBFloat16 is null
                            ? Vector256.LoadUnsafe(
                                ref System.Runtime.CompilerServices.Unsafe.Add(
                                    ref secondMomentStart,
                                    index))
                            : LoadBFloat16(vBFloat16, index)));
                if (mBFloat16 is null)
                {
                    firstMoment.StoreUnsafe(
                        ref System.Runtime.CompilerServices.Unsafe.Add(
                            ref firstMomentStart,
                            index));
                }
                else
                {
                    StoreBFloat16(firstMoment, mBFloat16, index);
                }
                if (vBFloat16 is null)
                {
                    secondMoment.StoreUnsafe(
                        ref System.Runtime.CompilerServices.Unsafe.Add(
                            ref secondMomentStart,
                            index));
                }
                else
                {
                    StoreBFloat16(secondMoment, vBFloat16, index);
                }

                Vector256<float> parameter =
                    Vector256.LoadUnsafe(
                        ref System.Runtime.CompilerServices.Unsafe.Add(
                            ref dataStart,
                            index))
                    * parameterScale;
                Vector256<float> inverseRoot =
                    System.Runtime.Intrinsics.X86.Avx
                        .ReciprocalSqrt(secondMoment);
                inverseRoot *= half
                    * (three - secondMoment * inverseRoot * inverseRoot);
                Vector256<float> epsilonCorrection =
                    one + epsilon * inverseRoot;
                Vector256<float> inverseCorrection =
                    System.Runtime.Intrinsics.X86.Avx.Reciprocal(
                        epsilonCorrection);
                inverseCorrection *= two
                    - epsilonCorrection * inverseCorrection;
                Vector256<float> inverseDenominator =
                    inverseRoot * inverseCorrection;
                inverseDenominator = Vector256.ConditionalSelect(
                    Vector256.Equals(
                        secondMoment,
                        Vector256<float>.Zero),
                    inverseZeroDenominator,
                    inverseDenominator);
                parameter -= updateScaleVector
                    * firstMoment
                    * inverseDenominator;
                parameter.StoreUnsafe(
                    ref System.Runtime.CompilerServices.Unsafe.Add(
                        ref dataStart,
                        index));
            }
        }

        for (; index < end; index++)
        {
            float g = grad.Length == 0 ? 0f : grad[index];
            float previousFirstMoment = mBFloat16 is null
                ? m[index]
                : BFloat16ToSingle(mBFloat16[index]);
            float firstMoment = options.Beta1 * previousFirstMoment
                + (1f - options.Beta1) * g;
            if (mBFloat16 is null)
                m[index] = firstMoment;
            else
                mBFloat16[index] = SingleToBFloat16(firstMoment);
            float previousSecondMoment = vBFloat16 is null
                ? v[index]
                : BFloat16ToSingle(vBFloat16[index]);
            float secondMoment = options.Beta2 * previousSecondMoment
                + (1f - options.Beta2) * g * g;
            if (vBFloat16 is null)
                v[index] = secondMoment;
            else
                vBFloat16[index] = SingleToBFloat16(secondMoment);

            if (applyWeightDecay)
                data[index] *= 1f - options.LearningRate
                    * options.WeightDecay;

            data[index] -= updateScale * firstMoment
                / (MathF.Sqrt(secondMoment) + scaledEpsilon);
        }
    }

    private static AdamWWorkItem[] CreateWorkItems(
        IReadOnlyList<Parameter> parameters)
    {
        // Split large matrices so one embedding or projection cannot leave
        // the remaining workers idle near the end of an optimizer step.
        const int ChunkElements = 65_536;
        var workItems = new List<AdamWWorkItem>();
        for (int parameterIndex = 0;
            parameterIndex < parameters.Count;
            parameterIndex++)
        {
            int length = parameters[parameterIndex].T.Numel;
            for (int start = 0; start < length; start += ChunkElements)
            {
                workItems.Add(
                    new AdamWWorkItem(
                        parameterIndex,
                        start,
                        Math.Min(ChunkElements, length - start)));
            }
        }
        return workItems.ToArray();
    }

    private static Vector256<float> LoadBFloat16(
        short[] source,
        int offset)
    {
        Vector128<short> packed = Vector128.LoadUnsafe(ref source[offset]);
        Vector256<int> widened = System.Runtime.Intrinsics.X86.Avx2
            .ConvertToVector256Int32(packed);
        return System.Runtime.Intrinsics.X86.Avx2
            .ShiftLeftLogical(widened.AsUInt32(), 16)
            .AsSingle();
    }

    private static void StoreBFloat16(
        Vector256<float> values,
        short[] destination,
        int offset)
    {
        Vector256<int> bits = values.AsInt32();
        Vector256<int> leastSignificantBit = System.Runtime.Intrinsics.X86.Avx2
            .ShiftRightLogical(bits.AsUInt32(), 16)
            .AsInt32()
            & Vector256.Create(1);
        Vector256<int> rounded = bits
            + Vector256.Create(0x7FFF)
            + leastSignificantBit;
        Vector256<int> upper = System.Runtime.Intrinsics.X86.Avx2
            .ShiftRightArithmetic(rounded, 16);
        Vector256<short> duplicated = System.Runtime.Intrinsics.X86.Avx2
            .PackSignedSaturate(upper, upper);
        Vector256<short> ordered = System.Runtime.Intrinsics.X86.Avx2
            .Permute4x64(duplicated.AsInt64(), 0xD8)
            .AsInt16();
        ordered.GetLower().StoreUnsafe(ref destination[offset]);
    }

    private static short SingleToBFloat16(float value)
    {
        uint bits = unchecked((uint)BitConverter.SingleToInt32Bits(value));
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        return unchecked((short)(bits >> 16));
    }

    private static float BFloat16ToSingle(short value)
        => BitConverter.Int32BitsToSingle(value << 16);

    private static short[] EncodeBFloat16(float[] source)
    {
        var result = new short[source.Length];
        for (int index = 0; index < source.Length; index++)
            result[index] = SingleToBFloat16(source[index]);
        return result;
    }

    private static float[] DecodeBFloat16(short[] source)
    {
        var result = new float[source.Length];
        int index = 0;
        if (Vector256.IsHardwareAccelerated
            && System.Runtime.Intrinsics.X86.Avx2.IsSupported)
        {
            int end = source.Length
                - source.Length % Vector256<float>.Count;
            for (; index < end; index += Vector256<float>.Count)
                LoadBFloat16(source, index).StoreUnsafe(ref result[index]);
        }
        for (; index < source.Length; index++)
            result[index] = BFloat16ToSingle(source[index]);
        return result;
    }

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
    }
}
