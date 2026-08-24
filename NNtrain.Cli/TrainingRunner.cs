using System.Runtime;

namespace NNtrain;

/// <summary>
/// Shared deterministic lifecycle primitives for CLI training tasks.
/// Task implementations retain ownership of batches and numerical work.
/// </summary>
internal static class TrainingRunner
{
    internal static NoGcTrainingWindow BeginNoGcTrainingWindow()
        // CUDA training still builds a short-lived managed autograd graph.
        // Suppressing collection only lets those objects accumulate until the
        // fixed budget is exhausted, producing a step-time cliff with no
        // long-run throughput benefit. Keep normal generational GC enabled.
        => new(enabled: false);

    internal static IEnumerable<TrainingEpoch> Epochs(
        int firstEpoch,
        int lastEpoch,
        int firstResumeUnit = 0)
    {
        if (firstEpoch <= 0 || firstEpoch > lastEpoch)
            throw new ArgumentOutOfRangeException(nameof(firstEpoch));
        if (firstResumeUnit < 0)
            throw new ArgumentOutOfRangeException(nameof(firstResumeUnit));

        for (int epoch = firstEpoch; epoch <= lastEpoch; epoch++)
        {
            yield return new TrainingEpoch(
                epoch,
                epoch == firstEpoch ? firstResumeUnit : 0);
        }
    }

    internal static int DivideRoundUp(int value, int divisor)
    {
        if (value < 0)
            throw new ArgumentOutOfRangeException(nameof(value));
        if (divisor <= 0)
            throw new ArgumentOutOfRangeException(nameof(divisor));
        return value / divisor + (value % divisor == 0 ? 0 : 1);
    }

    internal static bool ShouldSaveCheckpoint(
        int completedUnits,
        int totalUnits)
    {
        if (completedUnits <= 0 || completedUnits > totalUnits)
            throw new ArgumentOutOfRangeException(nameof(completedUnits));
        if (totalUnits <= 0)
            throw new ArgumentOutOfRangeException(nameof(totalUnits));
        int previousTenth = (completedUnits - 1) * 10 / totalUnits;
        int currentTenth = completedUnits * 10 / totalUnits;
        return currentTenth > previousTenth;
    }

    internal static void Shuffle<T>(Span<T> values, Random random)
    {
        ArgumentNullException.ThrowIfNull(random);
        for (int index = values.Length - 1; index > 0; index--)
        {
            int swapIndex = random.Next(index + 1);
            (values[index], values[swapIndex]) =
                (values[swapIndex], values[index]);
        }
    }
}

internal sealed class NoGcTrainingWindow : IDisposable
{
    private const long ManagedBudgetBytes = 512L * 1024 * 1024;
    private readonly bool _enabled;
    private bool _disposed;

    internal NoGcTrainingWindow(bool enabled)
    {
        _enabled = enabled;
        Pulse();
    }

    internal void Pulse()
    {
        if (!_enabled || _disposed
            || GCSettings.LatencyMode == GCLatencyMode.NoGCRegion)
            return;
        _ = GC.TryStartNoGCRegion(
            ManagedBudgetBytes,
            disallowFullBlockingGC: true);
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        if (GCSettings.LatencyMode == GCLatencyMode.NoGCRegion)
            GC.EndNoGCRegion();
    }
}

internal readonly record struct TrainingEpoch(int Number, int ResumeUnit);
