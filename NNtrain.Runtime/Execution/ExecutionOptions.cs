namespace NNtrain.Runtime.Execution;

/// <summary>
/// Immutable options used to construct an <see cref="ExecutionSession"/>.
/// CUDA device availability is intentionally independent from the selected
/// execution device.
/// </summary>
public sealed record ExecutionOptions
{
    public ExecutionDeviceKind Device { get; init; } = ExecutionDeviceKind.Cpu;

    public DeviceSet CudaDevices { get; init; } = DeviceSet.Default;
    public int ArcDeviceIndex { get; init; }

    /// <summary>
    /// Arc devices available to this session. When omitted, the legacy
    /// <see cref="ArcDeviceIndex"/> selects the sole Arc device.
    /// </summary>
    public DeviceSet? ArcDevices { get; init; }

    public PrecisionPolicy Precision { get; init; } = PrecisionPolicy.Float32;

    /// <summary>
    /// Rejects implicit host materialization of parameters, activations and
    /// gradients while executing on an accelerator.
    /// </summary>
    public bool RequireDeviceResidency { get; init; } = true;

    public bool IncludesArcDevice(int deviceIndex)
        => ArcDevices?.Contains(deviceIndex) ?? ArcDeviceIndex == deviceIndex;

    public ExecutionOptions Validate()
    {
        ArgumentNullException.ThrowIfNull(CudaDevices);
        ArgumentNullException.ThrowIfNull(Precision);
        ArgumentOutOfRangeException.ThrowIfNegative(ArcDeviceIndex);
        if (ArcDevices is not null && !ArcDevices.Contains(ArcDeviceIndex))
        {
            throw new ArgumentException(
                $"Primary Arc device {ArcDeviceIndex} is not in the Arc device set.",
                nameof(ArcDevices));
        }
        if (!Enum.IsDefined(Device))
            throw new ArgumentOutOfRangeException(nameof(Device));
        return this;
    }
}
