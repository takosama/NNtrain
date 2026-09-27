using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("NNtrain.Core.Tests")]

namespace NNtrain.Arc;

public sealed partial class ArcExecutionLane
{
    private readonly Dictionary<nint, CachedKernelArguments> _kernelArgumentCaches = [];

    public long KernelArgumentSetCount { get; private set; }
    public long KernelArgumentCacheHitCount { get; private set; }

    // RunDimensions holds _sync and obtains this once before its argument loop.
    private CachedKernelArguments? GetKernelArgumentCache(nint kernel, int argumentCount)
    {
        if (!Options.CacheKernelArguments) return null;
        if (!_kernelArgumentCaches.TryGetValue(kernel, out CachedKernelArguments? cache))
        {
            cache = new CachedKernelArguments(argumentCount);
            _kernelArgumentCaches.Add(kernel, cache);
        }
        else cache.EnsureCapacity(argumentCount);
        return cache;
    }

    // Direct P/Invoke on a miss: no per-call delegate allocation or virtual setter.
    // Buffer ownership/liveness checks must still happen before invoking this.
    private int SetKernelArgument(nint kernel, uint index, nuint size, nint value,
        CachedKernelArguments? cache)
    {
        KernelArgumentBytes key = default;
        bool cacheable = cache is not null && KernelArgumentBytes.TryRead(size, value, out key);
        if (cacheable && cache!.Matches(index, key))
        {
            KernelArgumentCacheHitCount++;
            return 0;
        }
        // A failed setter, unsupported argument size, or thrown native call
        // must never leave a previous successful fingerprint usable.
        cache?.Invalidate(index);
        KernelArgumentSetCount++;
        int status = OpenClNative.clSetKernelArg(kernel, index, size, value);
        if (status == 0 && cacheable) cache!.Store(index, key);
        return status;
    }

    // Call before native cl_mem release. Some runtimes recycle handle values;
    // equal pointer bytes alone cannot establish that a binding still refers to
    // the same allocation. Clear the objects in place: a dispatch may hold one
    // while an intervening HostArray allocation evicts a pooled buffer.
    private void InvalidateKernelArgumentCaches()
    {
        foreach (CachedKernelArguments cache in _kernelArgumentCaches.Values)
            cache.Clear();
    }

    internal readonly record struct KernelArgumentBytes(nuint Size, ulong Bits, bool IsLocal)
    {
        internal static unsafe bool TryRead(nuint size, nint pointer, out KernelArgumentBytes key)
        {
            if (pointer == 0)
            {
                key = new(size, 0, true);
                return true;
            }
            if (size == 4)
            {
                key = new(size, Unsafe.ReadUnaligned<uint>((void*)pointer), false);
                return true;
            }
            if (size == 8)
            {
                key = new(size, Unsafe.ReadUnaligned<ulong>((void*)pointer), false);
                return true;
            }
            key = default;
            return false;
        }
    }

    internal sealed class CachedKernelArguments(int capacity)
    {
        private struct Entry
        {
            internal KernelArgumentBytes Key;
            internal bool Valid;
        }

        private Entry[] _entries = new Entry[capacity];

        internal void EnsureCapacity(int capacity)
        {
            if (capacity > _entries.Length) Array.Resize(ref _entries, capacity);
        }

        internal bool Matches(uint index, KernelArgumentBytes key)
            => index < _entries.Length && _entries[index].Valid && _entries[index].Key == key;

        internal void Store(uint index, KernelArgumentBytes key)
        {
            EnsureCapacity(checked((int)index + 1));
            _entries[index] = new Entry { Key = key, Valid = true };
        }

        internal void Invalidate(uint index)
        {
            if (index < _entries.Length) _entries[index].Valid = false;
        }

        internal void Clear() => Array.Clear(_entries);
    }

    // A CPU-only test seam for the same raw-byte matching/state transitions.
    // Production dispatch calls SetKernelArgument above, never this delegate.
    internal static int SetKernelArgumentForTesting(nint kernel, uint index, nuint size, nint value,
        CachedKernelArguments? cache, Func<nint, uint, nuint, nint, int> setter)
    {
        KernelArgumentBytes key = default;
        bool cacheable = cache is not null && KernelArgumentBytes.TryRead(size, value, out key);
        if (cacheable && cache!.Matches(index, key)) return 0;
        cache?.Invalidate(index);
        int status = setter(kernel, index, size, value);
        if (status == 0 && cacheable) cache!.Store(index, key);
        return status;
    }
}
