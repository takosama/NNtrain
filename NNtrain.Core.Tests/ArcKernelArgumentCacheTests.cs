using System.Runtime.InteropServices;
using NNtrain.Arc;
using Xunit;
using Cache = NNtrain.Arc.ArcExecutionLane.CachedKernelArguments;

namespace NNtrain.Core.Tests;

public sealed class ArcKernelArgumentCacheTests
{
    [Fact]
    public void ScalarFingerprintPreservesSignedZeroNanPayloadAndAllUnsignedBits()
    {
        using var data = new NativeBytes();
        var cache = new Cache(1);
        int calls = 0;
        int Setter(nint kernel, uint index, nuint size, nint value)
        {
            Assert.Equal((nint)7, kernel);
            Assert.Equal(0u, index);
            Assert.Equal((nuint)4, size);
            Assert.Equal(data.Pointer, value);
            calls++;
            return 0;
        }
        foreach (uint bits in new uint[] { 0, 0x80000000, 0x7fc00001, 0x7fc00002, 0xffffffff, 0x3f800000 })
        {
            data.Write32(bits);
            int before = calls;
            Assert.Equal(0, ArcExecutionLane.SetKernelArgumentForTesting(7, 0, 4, data.Pointer, cache, Setter));
            Assert.Equal(before + 1, calls);
            Assert.Equal(0, ArcExecutionLane.SetKernelArgumentForTesting(7, 0, 4, data.Pointer, cache, Setter));
            Assert.Equal(before + 1, calls);
        }
    }

    [Fact]
    public void HandleWidthLocalMemorySizeAndArgumentIndexAreDistinct()
    {
        using var data = new NativeBytes();
        var cache = new Cache(1);
        int calls = 0;
        int Setter(nint kernel, uint index, nuint size, nint value) { calls++; return 0; }
        void Set(uint index, nuint size, nint pointer)
            => Assert.Equal(0, ArcExecutionLane.SetKernelArgumentForTesting(11, index, size, pointer, cache, Setter));
        data.Write64(0x0000000112345678);
        Set(0, 8, data.Pointer); Set(0, 8, data.Pointer);
        Assert.Equal(1, calls);
        data.Write64(0x0000000212345678); // Same lower word, distinct pointer bytes.
        Set(0, 8, data.Pointer);
        Assert.Equal(2, calls);
        Set(0, 4, data.Pointer); // Same lower bits are not the same argument size.
        Assert.Equal(3, calls);
        Set(4, 4, data.Pointer); Set(4, 4, data.Pointer); // Capacity growth and index isolation.
        Assert.Equal(4, calls);
        data.Write32(0);
        Set(0, 4, data.Pointer);
        Set(0, 4, 0); // Local memory is distinct from a zero-valued scalar.
        Set(0, 4, 0);
        Assert.Equal(6, calls);
        Set(0, 8192, 0); Set(0, 8192, 0);
        Set(0, 16384, 0);
        Assert.Equal(8, calls);
    }

    [Fact]
    public void FailedOrThrowingSetterNeverCommitsCandidateOrRetainsStaleFingerprint()
    {
        using var data = new NativeBytes();
        var cache = new Cache(1);
        int calls = 0, status = 0;
        bool throws = false;
        int Setter(nint kernel, uint index, nuint size, nint value)
        {
            calls++;
            if (throws) throw new InvalidOperationException("Injected setter failure");
            return status;
        }
        int Set() => ArcExecutionLane.SetKernelArgumentForTesting(13, 0, 4, data.Pointer, cache, Setter);
        data.Write32(1);
        Assert.Equal(0, Set());
        data.Write32(2); status = -51;
        Assert.Equal(-51, Set()); Assert.Equal(-51, Set());
        Assert.Equal(3, calls);
        status = 0;
        data.Write32(1); // A failed mutation also invalidated the older successful entry.
        Assert.Equal(0, Set()); Assert.Equal(0, Set());
        Assert.Equal(4, calls);
        data.Write32(3); throws = true;
        Assert.Throws<InvalidOperationException>(() => Set());
        throws = false;
        Assert.Equal(0, Set()); Assert.Equal(0, Set());
        Assert.Equal(6, calls);
    }

    [Fact]
    public void InPlaceInvalidationRebindsReusedBufferHandlesAndKeepsKernelCachesSeparate()
    {
        using var data = new NativeBytes();
        data.Write64(0x12345678);
        var first = new Cache(1);
        var second = new Cache(1);
        int calls = 0;
        int Setter(nint kernel, uint index, nuint size, nint value) { calls++; return 0; }
        int Set(nint kernel, Cache cache)
            => ArcExecutionLane.SetKernelArgumentForTesting(kernel, 0, (nuint)IntPtr.Size, data.Pointer, cache, Setter);
        Assert.Equal(0, Set(17, first)); Assert.Equal(0, Set(17, first));
        Assert.Equal(0, Set(19, second)); Assert.Equal(0, Set(19, second));
        Assert.Equal(2, calls);
        Cache heldByDispatch = first;
        first.Clear(); // Native release invalidates the already-held object in place.
        Assert.Equal(0, Set(17, heldByDispatch)); Assert.Equal(0, Set(19, second));
        Assert.Equal(3, calls);
    }

    [Fact]
    public void DisabledCachingAndUnsupportedSizesAlwaysCallSetter()
    {
        using var data = new NativeBytes();
        data.Write64(42);
        var cache = new Cache(1);
        int calls = 0;
        int Setter(nint kernel, uint index, nuint size, nint value) { calls++; return 0; }
        int Set(nuint size, Cache? current)
            => ArcExecutionLane.SetKernelArgumentForTesting(23, 0, size, data.Pointer, current, Setter);
        Assert.Equal(0, Set(4, null)); Assert.Equal(0, Set(4, null));
        Assert.Equal(2, calls);
        Assert.Equal(0, Set(4, cache)); Assert.Equal(0, Set(4, cache));
        Assert.Equal(3, calls);
        Assert.Equal(0, Set(16, cache)); Assert.Equal(0, Set(16, cache));
        Assert.Equal(5, calls);
        Assert.Equal(0, Set(4, cache)); // Unsupported-size mutation invalidated the old fingerprint.
        Assert.Equal(6, calls);
    }

    private sealed class NativeBytes : IDisposable
    {
        internal nint Pointer { get; } = Marshal.AllocHGlobal(16);
        internal void Write32(uint bits) => Marshal.WriteInt32(Pointer, unchecked((int)bits));
        internal void Write64(ulong bits) => Marshal.WriteInt64(Pointer, unchecked((long)bits));
        public void Dispose() => Marshal.FreeHGlobal(Pointer);
    }
}
