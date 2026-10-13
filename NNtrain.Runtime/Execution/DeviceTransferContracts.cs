namespace NNtrain.Runtime.Execution;

/// <summary>Transfer counters observed inside one guarded training step.</summary>
public readonly record struct DeviceTransferSnapshot(
    long HostToDeviceCopyCount,
    long HostToDeviceBytes,
    long DeviceToHostCopyCount,
    long DeviceToHostBytes);

/// <summary>Explicit transport classes that may cross the host boundary.</summary>
public enum DeviceTransferTransportCategory
{
    GradientCollective = 0,
}

/// <summary>Physical transfer counters attributed to one CUDA device.</summary>
public readonly record struct DeviceTransferDeviceSnapshot(
    int DeviceIndex,
    long HostToDeviceCopyCount,
    long HostToDeviceBytes,
    long DeviceToHostCopyCount,
    long DeviceToHostBytes);

/// <summary>
/// Physical transfer counters for an explicitly authorized transport class.
/// These counters are separate from the four-field implicit/batch snapshot so
/// existing telemetry consumers retain their original contract.
/// </summary>
public sealed record DeviceTransferTransportSnapshot(
    DeviceTransferTransportCategory Category,
    DeviceTransferSnapshot Totals,
    IReadOnlyList<DeviceTransferDeviceSnapshot> Devices);
