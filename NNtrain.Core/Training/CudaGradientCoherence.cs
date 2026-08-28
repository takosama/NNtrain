namespace NNtrain;

/// <summary>
/// Logical state of a gradient generation. This is deliberately independent
/// from the physical payload format (Float32, BF16, or BFP8).
/// </summary>
internal enum CudaGradientCoherenceKind
{
    Host,
    Local,
    Reduced,
}

/// <summary>
/// Uniquely identifies one successfully completed reducer generation/step.
/// A step id alone is insufficient because a rebuilt reducer starts its own
/// sequence at one.
/// </summary>
internal readonly record struct CudaGradientReductionStamp(
    long ReducerGeneration,
    long StepId)
{
    internal bool IsValid => ReducerGeneration > 0 && StepId > 0;
}

internal static class CudaGradientReductionStampSource
{
    private static long _reducerGeneration;

    internal static long CreateReducerGeneration()
    {
        long generation = Interlocked.Increment(ref _reducerGeneration);
        if (generation <= 0)
        {
            throw new InvalidOperationException(
                "CUDA gradient reducer generation space was exhausted.");
        }
        return generation;
    }

    internal static CudaGradientReductionStamp CreateStandalone()
        => new(CreateReducerGeneration(), 1);
}

internal readonly record struct CudaGradientCoherenceSnapshot(
    CudaGradientCoherenceKind Kind,
    int LocalDeviceIndex,
    int[] ReducedDevices,
    CudaGradientReductionStamp ReductionStamp,
    CudaGradientReductionStamp PendingStamp,
    long GradientVersion,
    long ConsumedGradientVersion,
    CudaGradientReductionStamp ConsumedReductionStamp);

/// <summary>
/// Claims a current CUDA gradient generation before an optimizer can mutate
/// weights. On multi-GPU runs every parameter must carry the same completed
/// reduction stamp. Claiming is fail-closed: a partially failed update cannot
/// be retried against the same gradients.
/// </summary>
internal static class CudaGradientOptimizerGuard
{
    private static readonly object Sync = new();

    internal static void ValidateAndConsume(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> devices)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(devices);
        if (devices.Count == 0)
        {
            throw new InvalidOperationException(
                "A CUDA optimizer requires at least one execution device.");
        }
        if (devices.Any(device => device < 0)
            || devices.Distinct().Count() != devices.Count)
        {
            throw new InvalidOperationException(
                "CUDA optimizer devices must be unique and non-negative.");
        }

        lock (Sync)
        {
            CudaGradientReductionStamp expectedStamp = default;
            var snapshots = new (Tensor Tensor,
                CudaGradientCoherenceSnapshot Snapshot)[parameters.Count];
            for (int index = 0; index < parameters.Count; index++)
            {
                Parameter parameter = parameters[index]
                    ?? throw new InvalidOperationException(
                        "CUDA optimizer parameters cannot contain null.");
                Tensor tensor = parameter.T;
                CudaGradientCoherenceSnapshot snapshot =
                    tensor.GetCudaGradientCoherenceSnapshot();
                snapshots[index] = (tensor, snapshot);

                if (snapshot.PendingStamp.IsValid)
                {
                    throw new InvalidOperationException(
                        $"CUDA gradient '{parameter.Name}' belongs to an " +
                        $"incomplete reduction step " +
                        $"{snapshot.PendingStamp.StepId}.");
                }
                if (snapshot.GradientVersion
                    == snapshot.ConsumedGradientVersion)
                {
                    throw new InvalidOperationException(
                        $"CUDA gradient '{parameter.Name}' was already " +
                        "consumed by an optimizer update.");
                }

                if (devices.Count == 1)
                {
                    bool local = snapshot.Kind
                            == CudaGradientCoherenceKind.Local
                        && snapshot.LocalDeviceIndex == devices[0];
                    bool reduced = snapshot.Kind
                            == CudaGradientCoherenceKind.Reduced
                        && snapshot.ReducedDevices.Contains(devices[0]);
                    if (!local && !reduced)
                    {
                        throw new InvalidOperationException(
                            $"CUDA gradient '{parameter.Name}' is not " +
                            $"current on device {devices[0]} " +
                            $"(kind={snapshot.Kind}, " +
                            $"local={snapshot.LocalDeviceIndex}, " +
                            $"reduced=[{string.Join(", ", snapshot.ReducedDevices)}], " +
                            $"version={snapshot.GradientVersion}).");
                    }
                    continue;
                }

                if (snapshot.Kind != CudaGradientCoherenceKind.Reduced
                    || !snapshot.ReductionStamp.IsValid
                    || !snapshot.ReducedDevices.SequenceEqual(devices))
                {
                    throw new InvalidOperationException(
                        $"CUDA gradient '{parameter.Name}' has not been " +
                        $"reduced across the active device set " +
                        $"[{string.Join(", ", devices)}].");
                }
                if (snapshot.ReductionStamp
                    == snapshot.ConsumedReductionStamp)
                {
                    throw new InvalidOperationException(
                        $"CUDA gradient '{parameter.Name}' reduction step " +
                        "was already consumed by an optimizer update.");
                }
                if (!expectedStamp.IsValid)
                {
                    expectedStamp = snapshot.ReductionStamp;
                }
                else if (snapshot.ReductionStamp != expectedStamp)
                {
                    throw new InvalidOperationException(
                        "CUDA optimizer parameters do not share one " +
                        "completed gradient reduction step.");
                }
            }

            // Validate the full set first, then claim it. A concurrent
            // mutation between phases is detected by the expected-version
            // compare and fails closed.
            foreach ((Tensor tensor, CudaGradientCoherenceSnapshot snapshot)
                in snapshots)
            {
                tensor.ConsumeCudaGradientForOptimizer(
                    snapshot.GradientVersion,
                    devices.Count > 1 ? expectedStamp : default);
            }
        }
    }
}
