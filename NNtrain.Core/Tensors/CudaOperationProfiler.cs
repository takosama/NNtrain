using System.Diagnostics;

namespace NNtrain;

/// <summary>
/// Diagnostic-only CUDA operation timing. It deliberately synchronizes the
/// selected device around each measured operation, so it must never be enabled
/// by the normal training path.
/// </summary>
internal static class CudaOperationProfiler
{
    private static readonly object Sync = new();
    private static readonly Dictionary<(int Device, string Operation), Sample>
        Samples = [];
    private static int _enabled;

    internal static bool IsEnabled => Volatile.Read(ref _enabled) != 0;

    internal static IDisposable Begin()
    {
        lock (Sync)
            Samples.Clear();
        Volatile.Write(ref _enabled, 1);
        return new Session();
    }

    internal static T Measure<T>(string operation, Func<T> action)
    {
        ArgumentNullException.ThrowIfNull(operation);
        ArgumentNullException.ThrowIfNull(action);
        if (!IsEnabled)
            return action();

        int device = Tensor.CudaDeviceIndex;
        var accelerator = ForgetMemoryV2Cuda.GetAccelerator(device);
        accelerator.Synchronize();
        var timer = Stopwatch.StartNew();
        T result = action();
        accelerator.Synchronize();
        timer.Stop();
        Add(device, operation, timer.Elapsed.TotalMilliseconds);
        return result;
    }

    internal static void Measure(string operation, Action action)
    {
        ArgumentNullException.ThrowIfNull(action);
        _ = Measure<object?>(operation, () =>
        {
            action();
            return null;
        });
    }

    internal static void MeasureDevices(
        string operation,
        IReadOnlyList<int> devices,
        Action action)
    {
        ArgumentNullException.ThrowIfNull(operation);
        ArgumentNullException.ThrowIfNull(devices);
        ArgumentNullException.ThrowIfNull(action);
        if (!IsEnabled)
        {
            action();
            return;
        }

        foreach (int device in devices)
            ForgetMemoryV2Cuda.GetAccelerator(device).Synchronize();
        var timer = Stopwatch.StartNew();
        action();
        foreach (int device in devices)
            ForgetMemoryV2Cuda.GetAccelerator(device).Synchronize();
        timer.Stop();
        Add(-1, operation, timer.Elapsed.TotalMilliseconds);
    }

    internal static IReadOnlyList<CudaOperationProfileSample> Snapshot()
    {
        lock (Sync)
        {
            return Samples
                .Select(pair => new CudaOperationProfileSample(
                    pair.Key.Device,
                    pair.Key.Operation,
                    pair.Value.Count,
                    pair.Value.TotalMilliseconds,
                    pair.Value.MaximumMilliseconds))
                .OrderBy(sample => sample.Device)
                .ThenByDescending(sample => sample.TotalMilliseconds)
                .ToArray();
        }
    }

    private static void Add(int device, string operation, double milliseconds)
    {
        lock (Sync)
        {
            var key = (device, operation);
            Samples.TryGetValue(key, out Sample current);
            Samples[key] = new Sample(
                current.Count + 1,
                current.TotalMilliseconds + milliseconds,
                Math.Max(current.MaximumMilliseconds, milliseconds));
        }
    }

    private sealed class Session : IDisposable
    {
        private int _disposed;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
                Volatile.Write(ref _enabled, 0);
        }
    }

    private readonly record struct Sample(
        int Count,
        double TotalMilliseconds,
        double MaximumMilliseconds);
}

internal readonly record struct CudaOperationProfileSample(
    int Device,
    string Operation,
    int Count,
    double TotalMilliseconds,
    double MaximumMilliseconds);
