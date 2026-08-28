using System.Collections.Concurrent;
using NNtrain.Cuda.Memory;
using Xunit;

public sealed class CudaMemoryManagerLifecycleTests
{
    [Theory]
    [InlineData(CudaMemoryKind.Persistent)]
    [InlineData(CudaMemoryKind.Transient)]
    [InlineData(CudaMemoryKind.Workspace)]
    [InlineData(CudaMemoryKind.PinnedStaging)]
    public void IncompleteFenceMovesLeaseFromActiveToPending(
        CudaMemoryKind kind)
    {
        var allocator = new FakeAllocator();
        using var manager = new CudaMemoryManager(1, allocator);
        var fence = new FakeFence();
        CudaMemoryLease lease = manager.Allocate(64, kind);

        lease.DisposeAfter(fence);

        Assert.Equal(0, manager.ActiveAllocationCount);
        Assert.Equal(1, manager.PendingAllocationCount);
        Assert.Equal(0, manager.ActiveBytes);
        Assert.Equal(64, manager.PendingBytes);
        Assert.Equal(1, manager.AllocationCount);
        Assert.Equal(64, manager.AllocatedBytes);
        Assert.Empty(allocator.Releases);

        Assert.Equal(0, manager.CollectCompleted());
        fence.Complete();
        Assert.Equal(1, manager.CollectCompleted());

        Assert.Equal(0, manager.AllocationCount);
        Assert.Equal(0, manager.AllocatedBytes);
        FakeRelease release = Assert.Single(allocator.Releases);
        Assert.Equal(1, release.DeviceIndex);
        Assert.Equal((nuint)64, release.ByteLength);
        Assert.Equal(kind, release.Kind);
        Assert.Equal(1, fence.DisposeCount);

        lease.Dispose();
        Assert.Equal(0, manager.CollectCompleted());
        Assert.Single(allocator.Releases);
    }

    [Fact]
    public void ConcurrentCollectionClaimsEveryPendingAllocationExactlyOnce()
    {
        const int count = 128;
        var allocator = new FakeAllocator();
        using var manager = new CudaMemoryManager(0, allocator);
        var fences = new FakeFence[count];
        var leases = new CudaMemoryLease[count];

        for (int index = 0; index < count; index++)
        {
            fences[index] = new FakeFence();
            leases[index] = manager.Allocate(32, CudaMemoryKind.Transient);
            leases[index].SetReleaseFence(fences[index]);
        }

        Parallel.ForEach(leases, lease => lease.Dispose());
        Parallel.ForEach(fences, fence => fence.Complete());
        Parallel.For(0, 8, _ => manager.CollectCompleted());

        Assert.Equal(0, manager.AllocationCount);
        Assert.Equal(0, manager.AllocatedBytes);
        Assert.Equal(count, allocator.Releases.Count);
        Assert.Equal(
            count,
            allocator.Releases.Select(static release => release.Pointer).Distinct().Count());
        Assert.All(fences, fence => Assert.Equal(1, fence.DisposeCount));
    }

    [Fact]
    public void OomCollectsCompletedThenRunsRecoveryAndRetriesOnlyOnce()
    {
        var allocator = new FakeAllocator();
        bool recoveryObservedPriorRelease = false;
        int recoveryCount = 0;
        using var manager = new CudaMemoryManager(
            2,
            allocator,
            (device, bytes, kind) =>
            {
                recoveryCount++;
                recoveryObservedPriorRelease = allocator.Releases.Count == 1;
                Assert.Equal(2, device);
                Assert.Equal((nuint)128, bytes);
                Assert.Equal(CudaMemoryKind.Workspace, kind);
            });

        var fence = new FakeFence();
        CudaMemoryLease oldLease = manager.Allocate(16, CudaMemoryKind.Transient);
        oldLease.DisposeAfter(fence);
        fence.Complete();
        allocator.FailAllocationAttempts.Enqueue(true);

        using CudaMemoryLease recovered =
            manager.Allocate(128, CudaMemoryKind.Workspace);

        Assert.Equal(1, recoveryCount);
        Assert.True(recoveryObservedPriorRelease);
        Assert.Equal(3, allocator.AllocationAttempts);
        Assert.Empty(manager.RecoveryErrors);
    }

    [Fact]
    public void OomFailurePerformsNoMoreThanOneRetry()
    {
        var allocator = new FakeAllocator();
        allocator.FailAllocationAttempts.Enqueue(true);
        allocator.FailAllocationAttempts.Enqueue(true);
        int recoveryCount = 0;
        using var manager = new CudaMemoryManager(
            0,
            allocator,
            (_, _, _) => recoveryCount++);

        Assert.Throws<OutOfMemoryException>(
            () => manager.Allocate(256, CudaMemoryKind.Persistent));

        Assert.Equal(2, allocator.AllocationAttempts);
        Assert.Equal(1, recoveryCount);
        Assert.Equal(0, manager.AllocationCount);
    }

    [Fact]
    public void RecoveryFailureIsRecordedAndDoesNotSuppressTheRetry()
    {
        var allocator = new FakeAllocator();
        allocator.FailAllocationAttempts.Enqueue(true);
        using var manager = new CudaMemoryManager(
            0,
            allocator,
            (_, _, _) => throw new InvalidOperationException("recovery failure"));

        using CudaMemoryLease lease =
            manager.Allocate(32, CudaMemoryKind.PinnedStaging);

        Assert.Equal(2, allocator.AllocationAttempts);
        Assert.Single(manager.RecoveryErrors);
    }

    [Fact]
    public async Task DisposeWaitsForFenceAndThenReleasesPendingAllocation()
    {
        var allocator = new FakeAllocator();
        var manager = new CudaMemoryManager(0, allocator);
        var fence = new FakeFence();
        CudaMemoryLease lease = manager.Allocate(48, CudaMemoryKind.Workspace);
        lease.DisposeAfter(fence);

        CancellationToken cancellationToken = TestContext.Current.CancellationToken;
        Task disposing = Task.Run(manager.Dispose, cancellationToken);
        Assert.True(
            fence.WaitEntered.Wait(
                TimeSpan.FromSeconds(5),
                cancellationToken));
        Assert.False(disposing.IsCompleted);

        fence.Complete();
        await disposing.WaitAsync(
            TimeSpan.FromSeconds(5),
            cancellationToken);

        Assert.Equal(1, fence.WaitCount);
        Assert.Equal(1, fence.DisposeCount);
        Assert.Single(allocator.Releases);
        Assert.Equal(0, manager.AllocationCount);
    }

    [Fact]
    public void DisposeContinuesAfterFenceAndNativeReleaseFailures()
    {
        var allocator = new FakeAllocator
        {
            ThrowOnEveryRelease = true,
        };
        var manager = new CudaMemoryManager(0, allocator);
        var fences = Enumerable.Range(0, 3)
            .Select(_ => new FakeFence { ThrowOnWait = true, ThrowOnDispose = true })
            .ToArray();

        foreach (FakeFence fence in fences)
        {
            CudaMemoryLease lease =
                manager.Allocate(24, CudaMemoryKind.Persistent);
            lease.DisposeAfter(fence);
        }

        Exception? exception = Record.Exception(manager.Dispose);

        Assert.Null(exception);
        Assert.Equal(3, allocator.ReleaseAttempts);
        Assert.Equal(0, manager.AllocationCount);
        Assert.Equal(0, manager.AllocatedBytes);
        Assert.Equal(9, manager.ReleaseErrors.Count);
        Assert.All(fences, fence => Assert.Equal(1, fence.WaitCount));
        Assert.All(fences, fence => Assert.Equal(1, fence.DisposeCount));
    }

    [Fact]
    public void FenceCanOnlyBeAttachedOnceAndNeverAfterReleaseStarts()
    {
        using var manager = new CudaMemoryManager(0, new FakeAllocator());
        using CudaMemoryLease lease =
            manager.Allocate(8, CudaMemoryKind.Transient);
        var first = new FakeFence();
        lease.SetReleaseFence(first);

        Assert.Throws<InvalidOperationException>(
            () => lease.SetReleaseFence(new FakeFence()));

        first.Complete();
        lease.Dispose();
        Assert.Throws<ObjectDisposedException>(
            () => lease.SetReleaseFence(new FakeFence()));
    }

    [Fact]
    public void ExactSizeRentIsReusedWithinLaneAndFreedOnManagerDispose()
    {
        var allocator = new FakeAllocator();
        var manager = new CudaMemoryManager(3, allocator);

        CudaMemoryLease first = manager.Rent(
            96,
            CudaMemoryKind.Transient);
        nint pointer = first.Pointer;
        first.Dispose();

        Assert.Equal(1, manager.CachedAllocationCount);
        Assert.Equal(96, manager.CachedBytes);
        Assert.Equal(1, manager.AllocationCount);
        Assert.Empty(allocator.Releases);

        using (CudaMemoryLease second = manager.Rent(
                   96,
                   CudaMemoryKind.Transient))
        {
            Assert.Equal(pointer, second.Pointer);
            Assert.Equal(1, allocator.AllocationAttempts);
            Assert.Equal(1, manager.ActiveAllocationCount);
            Assert.Equal(0, manager.CachedAllocationCount);
        }

        Assert.Equal(1, manager.CachedAllocationCount);
        manager.Dispose();

        FakeRelease release = Assert.Single(allocator.Releases);
        Assert.Equal(pointer, release.Pointer);
        Assert.Equal(CudaMemoryKind.Transient, release.Kind);
        Assert.Equal(0, manager.AllocationCount);
        Assert.Equal(0, manager.AllocatedBytes);
    }

    [Fact]
    public void ReusableFenceBecomesCacheOnlyAfterCompletion()
    {
        var allocator = new FakeAllocator();
        using var manager = new CudaMemoryManager(0, allocator);
        var fence = new FakeFence();
        CudaMemoryLease lease = manager.Rent(
            128,
            CudaMemoryKind.Workspace);

        lease.DisposeAfter(fence);
        Assert.Equal(1, manager.PendingAllocationCount);
        Assert.Equal(0, manager.CachedAllocationCount);

        fence.Complete();
        Assert.Equal(1, manager.CollectCompleted());
        Assert.Equal(0, manager.PendingAllocationCount);
        Assert.Equal(1, manager.CachedAllocationCount);
        Assert.Empty(allocator.Releases);
        Assert.Equal(1, fence.DisposeCount);
    }

    [Fact]
    public void OomTrimsReusableCacheBeforeSingleRetry()
    {
        var allocator = new FakeAllocator();
        using var manager = new CudaMemoryManager(0, allocator);
        CudaMemoryLease cached = manager.Rent(
            64,
            CudaMemoryKind.Transient);
        cached.Dispose();
        allocator.FailAllocationAttempts.Enqueue(true);

        using CudaMemoryLease result = manager.Rent(
            256,
            CudaMemoryKind.Workspace);

        Assert.Equal(3, allocator.AllocationAttempts);
        Assert.Single(allocator.Releases);
        Assert.Equal(0, manager.CachedAllocationCount);
        Assert.Equal(1, manager.ActiveAllocationCount);
    }

    [Fact]
    public void RentRejectsPersistentAndPinnedKinds()
    {
        using var manager = new CudaMemoryManager(0, new FakeAllocator());

        Assert.Throws<ArgumentException>(
            () => manager.Rent(8, CudaMemoryKind.Persistent));
        Assert.Throws<ArgumentException>(
            () => manager.Rent(8, CudaMemoryKind.PinnedStaging));
    }

    [Fact]
    public void ReusableCacheEvictsOldestEntriesWithinByteAndCountBudgets()
    {
        var allocator = new FakeAllocator();
        using var manager = new CudaMemoryManager(
            0,
            allocator,
            oomRecovery: null,
            maximumReusableCacheBytes: 192,
            maximumReusableCacheEntries: 2);

        CudaMemoryLease first = manager.Rent(
            64, CudaMemoryKind.Transient);
        nint firstPointer = first.Pointer;
        first.Dispose();
        CudaMemoryLease second = manager.Rent(
            80, CudaMemoryKind.Transient);
        second.Dispose();
        CudaMemoryLease third = manager.Rent(
            96, CudaMemoryKind.Workspace);
        third.Dispose();

        Assert.Equal(2, manager.CachedAllocationCount);
        Assert.Equal(176, manager.CachedBytes);
        FakeRelease evicted = Assert.Single(allocator.Releases);
        Assert.Equal(firstPointer, evicted.Pointer);
        Assert.Equal((nuint)64, evicted.ByteLength);
    }

    [Fact]
    public void OversizedReusableAllocationIsReleasedInsteadOfCached()
    {
        var allocator = new FakeAllocator();
        using var manager = new CudaMemoryManager(
            0,
            allocator,
            oomRecovery: null,
            maximumReusableCacheBytes: 128,
            maximumReusableCacheEntries: 4);

        CudaMemoryLease lease = manager.Rent(
            256, CudaMemoryKind.Transient);
        nint pointer = lease.Pointer;
        lease.Dispose();

        Assert.Equal(0, manager.CachedAllocationCount);
        Assert.Equal(0, manager.CachedBytes);
        FakeRelease released = Assert.Single(allocator.Releases);
        Assert.Equal(pointer, released.Pointer);
        Assert.Equal((nuint)256, released.ByteLength);
    }

    [Fact]
    public void ProtectedLanguageModelHeadSurvivesSmallCachePressure()
    {
        const long mib = 1024L * 1024;
        const long headBytes = 404 * mib;
        const long smallBytes = 4 * mib;
        var allocator = new FakeAllocator();
        using var manager = new CudaMemoryManager(0, allocator);

        CudaMemoryLease head = manager.Rent(
            (nuint)headBytes,
            CudaMemoryKind.Transient);
        nint headPointer = head.Pointer;
        head.Dispose();

        // Keep all small leases active until the head is cached, then return
        // more than the remaining 108 MiB byte budget.
        CudaMemoryLease[] small = Enumerable.Range(0, 40)
            .Select(_ => manager.Rent(
                (nuint)smallBytes,
                CudaMemoryKind.Transient))
            .ToArray();
        foreach (CudaMemoryLease lease in small)
        {
            lease.Dispose();
            Assert.InRange(
                manager.CachedBytes,
                0,
                manager.MaximumReusableCacheBytes);
            Assert.InRange(
                manager.CachedAllocationCount,
                0,
                manager.MaximumReusableCacheEntries);
        }

        Assert.Equal(512 * mib, manager.CachedBytes);
        Assert.Equal(28, manager.CachedAllocationCount);
        int allocationAttemptsBeforeReuse = allocator.AllocationAttempts;

        using CudaMemoryLease reusedHead = manager.Rent(
            (nuint)headBytes,
            CudaMemoryKind.Transient);

        Assert.Equal(headPointer, reusedHead.Pointer);
        Assert.Equal(
            allocationAttemptsBeforeReuse,
            allocator.AllocationAttempts);
        Assert.DoesNotContain(
            allocator.Releases,
            release => release.Pointer == headPointer);
    }

    [Fact]
    public void DefaultEntryBudgetRetainsTransformerActivationHighWater()
    {
        const int activationCount = 314;
        const int activationBytes = 4096;
        var allocator = new FakeAllocator();
        using var manager = new CudaMemoryManager(0, allocator);

        CudaMemoryLease[] warmup = Enumerable.Range(0, activationCount)
            .Select(_ => manager.Rent(
                activationBytes,
                CudaMemoryKind.Transient))
            .ToArray();
        foreach (CudaMemoryLease lease in warmup)
            lease.Dispose();

        Assert.Equal(activationCount, manager.CachedAllocationCount);
        int nativeAllocationsAfterWarmup = allocator.AllocationAttempts;

        CudaMemoryLease[] steady = Enumerable.Range(0, activationCount)
            .Select(_ => manager.Rent(
                activationBytes,
                CudaMemoryKind.Transient))
            .ToArray();

        Assert.Equal(
            nativeAllocationsAfterWarmup,
            allocator.AllocationAttempts);
        foreach (CudaMemoryLease lease in steady)
            lease.Dispose();
        Assert.Equal(activationCount, manager.CachedAllocationCount);
    }

    [Fact]
    public void NewProtectedLargeShapeReplacesPreviousLargeShape()
    {
        const long mib = 1024L * 1024;
        const long oldShapeBytes = 404 * mib;
        const long newShapeBytes = 396 * mib;
        var allocator = new FakeAllocator();
        using var manager = new CudaMemoryManager(0, allocator);

        CudaMemoryLease oldShape = manager.Rent(
            (nuint)oldShapeBytes,
            CudaMemoryKind.Transient);
        nint oldPointer = oldShape.Pointer;
        oldShape.Dispose();

        CudaMemoryLease newShape = manager.Rent(
            (nuint)newShapeBytes,
            CudaMemoryKind.Transient);
        nint newPointer = newShape.Pointer;
        newShape.Dispose();

        FakeRelease replaced = Assert.Single(
            allocator.Releases,
            release => release.Pointer == oldPointer);
        Assert.Equal((nuint)oldShapeBytes, replaced.ByteLength);
        Assert.Equal(newShapeBytes, manager.CachedBytes);
        Assert.Equal(1, manager.CachedAllocationCount);

        int allocationAttemptsBeforeReuse = allocator.AllocationAttempts;
        using CudaMemoryLease reusedNewShape = manager.Rent(
            (nuint)newShapeBytes,
            CudaMemoryKind.Transient);
        Assert.Equal(newPointer, reusedNewShape.Pointer);
        Assert.Equal(
            allocationAttemptsBeforeReuse,
            allocator.AllocationAttempts);
    }

    [Fact]
    public void TrimAndDisposeReleaseProtectedAndSmallCacheEntries()
    {
        const long mib = 1024L * 1024;
        var allocator = new FakeAllocator();
        var manager = new CudaMemoryManager(0, allocator);

        CudaMemoryLease protectedLarge = manager.Rent(
            (nuint)(404 * mib),
            CudaMemoryKind.Transient);
        protectedLarge.Dispose();
        CudaMemoryLease[] small = Enumerable.Range(0, 4)
            .Select(_ => manager.Rent(
                (nuint)(8 * mib),
                CudaMemoryKind.Workspace))
            .ToArray();
        foreach (CudaMemoryLease lease in small)
            lease.Dispose();

        long cachedBeforeTrim = manager.CachedAllocationCount;
        Assert.Equal(cachedBeforeTrim, manager.TrimReusableCache());
        Assert.Equal(0, manager.CachedAllocationCount);
        Assert.Equal(0, manager.CachedBytes);
        Assert.Equal(0, manager.AllocationCount);
        Assert.Equal(
            cachedBeforeTrim,
            allocator.Releases.Select(static release => release.Pointer)
                .Distinct()
                .Count());

        CudaMemoryLease afterTrim = manager.Rent(
            (nuint)(404 * mib),
            CudaMemoryKind.Transient);
        nint afterTrimPointer = afterTrim.Pointer;
        afterTrim.Dispose();
        manager.Dispose();

        Assert.Equal(0, manager.AllocationCount);
        Assert.Equal(0, manager.AllocatedBytes);
        Assert.Single(
            allocator.Releases,
            release => release.Pointer == afterTrimPointer);
    }

    private sealed class FakeAllocator : ICudaMemoryAllocator
    {
        private long _nextPointer;
        private int _allocationAttempts;
        private int _releaseAttempts;

        public ConcurrentQueue<bool> FailAllocationAttempts { get; } = new();

        public ConcurrentQueue<FakeRelease> Releases { get; } = new();

        public bool ThrowOnEveryRelease { get; init; }

        public int AllocationAttempts => Volatile.Read(ref _allocationAttempts);

        public int ReleaseAttempts => Volatile.Read(ref _releaseAttempts);

        public nint Allocate(
            int deviceIndex,
            nuint byteLength,
            CudaMemoryKind kind)
        {
            Interlocked.Increment(ref _allocationAttempts);
            if (FailAllocationAttempts.TryDequeue(out bool fail) && fail)
                throw new OutOfMemoryException("scripted allocation failure");
            return (nint)Interlocked.Increment(ref _nextPointer);
        }

        public void Release(
            int deviceIndex,
            nint pointer,
            nuint byteLength,
            CudaMemoryKind kind)
        {
            Interlocked.Increment(ref _releaseAttempts);
            if (ThrowOnEveryRelease)
                throw new InvalidOperationException("scripted release failure");
            Releases.Enqueue(
                new FakeRelease(deviceIndex, pointer, byteLength, kind));
        }
    }

    private sealed class FakeFence : ICudaCompletionFence
    {
        private readonly ManualResetEventSlim _completion = new(false);
        private int _waitCount;
        private int _disposeCount;

        public ManualResetEventSlim WaitEntered { get; } = new(false);

        public bool ThrowOnWait { get; init; }

        public bool ThrowOnDispose { get; init; }

        public int WaitCount => Volatile.Read(ref _waitCount);

        public int DisposeCount => Volatile.Read(ref _disposeCount);

        public bool IsCompleted => _completion.IsSet;

        public void Complete() => _completion.Set();

        public void Wait()
        {
            Interlocked.Increment(ref _waitCount);
            WaitEntered.Set();
            if (ThrowOnWait)
                throw new InvalidOperationException("scripted wait failure");
            _completion.Wait();
        }

        public void Dispose()
        {
            Interlocked.Increment(ref _disposeCount);
            if (ThrowOnDispose)
                throw new InvalidOperationException("scripted fence dispose failure");
            _completion.Dispose();
            WaitEntered.Dispose();
        }
    }

    private readonly record struct FakeRelease(
        int DeviceIndex,
        nint Pointer,
        nuint ByteLength,
        CudaMemoryKind Kind);
}
