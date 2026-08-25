using System.Runtime.CompilerServices;

namespace NNtrain;

/// <summary>
/// Keeps immutable Float32 compute views on every CUDA adapter that consumes
/// them. Entries follow the lifetime of their host array and are invalidated
/// explicitly when a tensor is updated.
/// </summary>
internal static class CudaResidentArrayCache
{
    private static readonly ConditionalWeakTable<float[], Entry> Entries = new();

    internal static NativeCudaBuffer<float> GetOrUpload(
        NativeCudaDevice accelerator,
        float[] values)
    {
        Entry entry = Entries.GetValue(values, static _ => new Entry());
        return entry.GetOrUpload(accelerator, values);
    }

    internal static void Invalidate(float[]? values)
    {
        if (values is null)
            return;
        if (Entries.TryGetValue(values, out Entry? entry))
            entry.Dispose();
        Entries.Remove(values);
    }

    private sealed class Entry : IDisposable
    {
        private readonly object _sync = new();
        private readonly Dictionary<NativeCudaDevice,
            NativeCudaBuffer<float>> _buffers =
            new(ReferenceEqualityComparer.Instance);
        private bool _disposed;

        internal NativeCudaBuffer<float> GetOrUpload(
            NativeCudaDevice accelerator,
            float[] values)
        {
            lock (_sync)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                if (!_buffers.TryGetValue(accelerator, out var buffer))
                {
                    buffer = accelerator.Allocate1D(values);
                    _buffers.Add(accelerator, buffer);
                }
                return buffer;
            }
        }

        public void Dispose()
        {
            lock (_sync)
            {
                if (_disposed)
                    return;
                foreach (var buffer in _buffers.Values)
                    buffer.Dispose();
                _buffers.Clear();
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        ~Entry() => Dispose();
    }
}
