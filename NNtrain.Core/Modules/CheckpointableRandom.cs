namespace NNtrain;

/// <summary>
/// Serializable runtime random state used by training-only stochastic
/// modules. Model initialization is delegated to <see cref="Random"/> so the
/// established seeded initialization sequence is preserved. Once the model
/// is fully constructed, runtime draws use explicit host/device states that
/// can be restored without replaying every preceding training step.
/// </summary>
internal sealed record TrainingRandomState(
    int FormatVersion,
    ulong RootSeed,
    ulong HostState,
    TrainingRandomDeviceState[] DeviceStates)
{
    internal const int CurrentFormatVersion = 1;
}

internal sealed record TrainingRandomDeviceState(
    int DeviceIndex,
    ulong State);

internal sealed class CheckpointableRandom : Random
{
    private const ulong GoldenRatio = 0x9E3779B97F4A7C15UL;
    private readonly object _sync = new();
    private readonly Random _initializationRandom;
    private readonly Dictionary<int, ulong> _deviceStates = [];
    private ulong _rootSeed;
    private ulong _hostState;
    private bool _runtimeMode;

    internal CheckpointableRandom(int seed)
    {
        _initializationRandom = new Random(seed);
    }

    internal ulong RootSeed
    {
        get
        {
            lock (_sync)
            {
                EnsureRuntimeMode();
                return _rootSeed;
            }
        }
    }

    /// <summary>
    /// Ends the initialization-compatible phase. Runtime state is seeded once
    /// from the same System.Random stream that direct model construction used.
    /// </summary>
    internal void BeginRuntime()
    {
        lock (_sync)
        {
            if (_runtimeMode)
                return;
            ulong low = unchecked((ulong)_initializationRandom.NextInt64());
            ulong high = unchecked((ulong)_initializationRandom.NextInt64());
            _rootSeed = Mix(low ^ (high << 1) ^ GoldenRatio);
            _hostState = _rootSeed;
            _runtimeMode = true;
        }
    }

    internal TrainingRandomState CaptureRuntimeState()
    {
        lock (_sync)
        {
            EnsureRuntimeMode();
            return new TrainingRandomState(
                TrainingRandomState.CurrentFormatVersion,
                _rootSeed,
                _hostState,
                _deviceStates
                    .OrderBy(pair => pair.Key)
                    .Select(pair => new TrainingRandomDeviceState(
                        pair.Key,
                        pair.Value))
                    .ToArray());
        }
    }

    internal void RestoreRuntimeState(TrainingRandomState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (state.FormatVersion != TrainingRandomState.CurrentFormatVersion
            || state.DeviceStates is null
            || state.DeviceStates.Any(item => item is null
                || item.DeviceIndex < 0)
            || state.DeviceStates
                .Select(item => item.DeviceIndex)
                .Distinct()
                .Count() != state.DeviceStates.Length)
        {
            throw new InvalidDataException(
                "Checkpoint training random state is invalid.");
        }

        lock (_sync)
        {
            EnsureRuntimeMode();
            _rootSeed = state.RootSeed;
            _hostState = state.HostState;
            _deviceStates.Clear();
            foreach (TrainingRandomDeviceState device in state.DeviceStates)
                _deviceStates.Add(device.DeviceIndex, device.State);
        }
    }

    public override int Next()
        => !_runtimeMode
            ? _initializationRandom.Next()
            : (int)(NextRuntimeUInt64() & 0x7FFF_FFFFUL);

    public override int Next(int maxValue)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(maxValue);
        if (!_runtimeMode)
            return _initializationRandom.Next(maxValue);
        return maxValue == 0
            ? 0
            : (int)(NextRuntimeUInt64() % (uint)maxValue);
    }

    public override int Next(int minValue, int maxValue)
    {
        if (minValue > maxValue)
            throw new ArgumentOutOfRangeException(nameof(minValue));
        if (!_runtimeMode)
            return _initializationRandom.Next(minValue, maxValue);
        ulong range = (ulong)((long)maxValue - minValue);
        return range == 0
            ? minValue
            : checked((int)(minValue + (long)(NextRuntimeUInt64() % range)));
    }

    public override long NextInt64()
        => !_runtimeMode
            ? _initializationRandom.NextInt64()
            : (long)(NextRuntimeUInt64() >> 1);

    public override long NextInt64(long maxValue)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(maxValue);
        if (!_runtimeMode)
            return _initializationRandom.NextInt64(maxValue);
        return maxValue == 0
            ? 0
            : (long)(NextRuntimeUInt64() % (ulong)maxValue);
    }

    public override long NextInt64(long minValue, long maxValue)
    {
        if (minValue > maxValue)
            throw new ArgumentOutOfRangeException(nameof(minValue));
        if (!_runtimeMode)
            return _initializationRandom.NextInt64(minValue, maxValue);
        ulong range = unchecked((ulong)(maxValue - minValue));
        return range == 0
            ? minValue
            : unchecked((long)(NextRuntimeUInt64() % range) + minValue);
    }

    public override double NextDouble()
        => !_runtimeMode
            ? _initializationRandom.NextDouble()
            : (NextRuntimeUInt64() >> 11) * (1d / (1UL << 53));

    public override float NextSingle()
        => !_runtimeMode
            ? _initializationRandom.NextSingle()
            : (float)((NextRuntimeUInt64() >> 40) * (1d / (1UL << 24)));

    public override void NextBytes(byte[] buffer)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        NextBytes(buffer.AsSpan());
    }

    public override void NextBytes(Span<byte> buffer)
    {
        if (!_runtimeMode)
        {
            _initializationRandom.NextBytes(buffer);
            return;
        }

        int offset = 0;
        while (offset < buffer.Length)
        {
            ulong value = NextRuntimeUInt64();
            int count = Math.Min(sizeof(ulong), buffer.Length - offset);
            for (int index = 0; index < count; index++)
                buffer[offset + index] = (byte)(value >> (index * 8));
            offset += count;
        }
    }

    protected override double Sample() => NextDouble();

    private ulong NextRuntimeUInt64()
    {
        if (!_runtimeMode)
        {
            // Preserve System.Random exactly while parameters are being
            // initialized. All runtime callers begin after BeginRuntime().
            return unchecked((ulong)_initializationRandom.NextInt64());
        }

        lock (_sync)
        {
            TorchDevice device = TensorExecutionContext.Device;
            if (!device.IsCuda)
            {
                _hostState = unchecked(_hostState + GoldenRatio);
                return Mix(_hostState);
            }

            if (!_deviceStates.TryGetValue(
                device.Index,
                out ulong state))
            {
                state = Mix(
                    _rootSeed
                    ^ unchecked((ulong)(uint)device.Index * GoldenRatio));
            }
            state = unchecked(state + GoldenRatio);
            _deviceStates[device.Index] = state;
            return Mix(state);
        }
    }

    private void EnsureRuntimeMode()
    {
        if (!_runtimeMode)
        {
            throw new InvalidOperationException(
                "Training random runtime state is not active.");
        }
    }

    private static ulong Mix(ulong value)
    {
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9UL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBUL;
        return value ^ (value >> 31);
    }
}
