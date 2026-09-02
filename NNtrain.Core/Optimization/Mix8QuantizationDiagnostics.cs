using NNtrain.Cuda.Interop;

namespace NNtrain;

/// <summary>
/// Quantization-boundary diagnostics for one committed mix8_32 optimizer
/// update. Ratios are measured in units of the parameter's block quantization
/// step, so values remain comparable across layers with different scales.
/// </summary>
public readonly record struct Mix8QuantizationDiagnostics(
    ulong ChangedWeightCount,
    ulong ElementCount,
    double ResidualStepRatioSquaredSum,
    double UpdateStepRatioSquaredSum)
{
    public bool HasValues => ElementCount != 0;

    public double QuantizedWeightChangeRate => ElementCount == 0
        ? 0d
        : (double)ChangedWeightCount / ElementCount;

    public double ResidualRmsPerQuantStep => ElementCount == 0
        ? 0d
        : Math.Sqrt(ResidualStepRatioSquaredSum / ElementCount);

    public double UpdateRmsPerQuantStep => ElementCount == 0
        ? 0d
        : Math.Sqrt(UpdateStepRatioSquaredSum / ElementCount);

    public static Mix8QuantizationDiagnostics Combine(
        IEnumerable<Mix8QuantizationDiagnostics> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        ulong changed = 0;
        ulong elements = 0;
        double residual = 0d;
        double update = 0d;
        foreach (Mix8QuantizationDiagnostics value in values)
        {
            checked
            {
                changed += value.ChangedWeightCount;
                elements += value.ElementCount;
            }
            residual += value.ResidualStepRatioSquaredSum;
            update += value.UpdateStepRatioSquaredSum;
        }
        return new Mix8QuantizationDiagnostics(
            changed, elements, residual, update);
    }
}

/// <summary>
/// Implemented by optimizers which can expose device-aggregated mix8_32
/// publication diagnostics without copying parameter-sized buffers to host.
/// </summary>
public interface IMix8QuantizationDiagnosticsProvider
{
    bool TryGetMix8QuantizationDiagnostics(
        out Mix8QuantizationDiagnostics diagnostics);
}

/// <summary>
/// Owns one 32-byte device aggregate per participating GPU. The optimizer
/// resets it once per commit; CUDA kernels reduce into it without any
/// parameter-sized diagnostic allocation or per-parameter D2H transfer.
/// </summary>
internal sealed class CudaMix8QuantizationDiagnostics : IDisposable
{
    private readonly object _sync = new();
    private readonly Dictionary<int,
        NativeCudaBuffer<CudaMix8DiagnosticAccumulator>>
        _buffers = [];
    private int _disposed;

    internal NativeCudaBuffer<CudaMix8DiagnosticAccumulator>
        Reset(int deviceIndex)
    {
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0, this);
        NativeCudaBuffer<CudaMix8DiagnosticAccumulator> buffer;
        lock (_sync)
        {
            if (!_buffers.TryGetValue(deviceIndex, out buffer!))
            {
                buffer = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex)
                    .Allocate1D<CudaMix8DiagnosticAccumulator>(1);
                _buffers.Add(deviceIndex, buffer);
            }
        }
        buffer.MemSetToZero();
        return buffer;
    }

    internal bool TryRead(
        int deviceIndex,
        out Mix8QuantizationDiagnostics diagnostics)
    {
        NativeCudaBuffer<CudaMix8DiagnosticAccumulator>? buffer;
        lock (_sync)
            _buffers.TryGetValue(deviceIndex, out buffer);
        if (buffer is null)
        {
            diagnostics = default;
            return false;
        }

        Span<CudaMix8DiagnosticAccumulator> host =
            stackalloc CudaMix8DiagnosticAccumulator[1];
        buffer.CopyToCPU(host);
        CudaMix8DiagnosticAccumulator value = host[0];
        diagnostics = new Mix8QuantizationDiagnostics(
            value.ChangedCodeCount,
            value.ElementCount,
            value.ResidualStepRatioSquaredSum,
            value.UpdateStepRatioSquaredSum);
        return diagnostics.HasValues;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        List<Exception>? failures = null;
        lock (_sync)
        {
            foreach (IDisposable buffer in _buffers.Values)
            {
                try
                {
                    buffer.Dispose();
                }
                catch (Exception exception)
                {
                    (failures ??= []).Add(exception);
                }
            }
            _buffers.Clear();
        }
        if (failures is not null)
        {
            throw new AggregateException(
                "mix8_32 diagnostic buffer cleanup failed.", failures);
        }
    }
}
