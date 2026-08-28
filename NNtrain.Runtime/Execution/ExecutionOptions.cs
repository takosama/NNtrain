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

    public PrecisionPolicy Precision { get; init; } = PrecisionPolicy.Float32;

    /// <summary>
    /// Rejects implicit host materialization of parameters, activations and
    /// gradients while executing on an accelerator.
    /// </summary>
    public bool RequireDeviceResidency { get; init; } = true;

    public ExecutionOptions Validate()
    {
        ArgumentNullException.ThrowIfNull(CudaDevices);
        ArgumentNullException.ThrowIfNull(Precision);
        if (!Enum.IsDefined(Device))
            throw new ArgumentOutOfRangeException(nameof(Device));
        return this;
    }
}
