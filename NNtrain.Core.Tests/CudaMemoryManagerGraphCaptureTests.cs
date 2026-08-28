using System.Collections.Concurrent;
using NNtrain.Cuda.Memory;
using Xunit;

public sealed class CudaMemoryManagerGraphCaptureTests
{
    [Fact]
    public void CaptureLocalExactSizeReuseMatchesLivePeak()
    {
        var allocator = new FakeAllocator();
        using var manager = NoCacheManager(allocator);
        using CudaGraphCaptureScope capture =
            manager.BeginGraphCaptureReservation();

        CudaMemoryLease first = manager.Rent(
            64,
            CudaMemoryKind.Transient);
        CudaMemoryLease second = manager.Rent(
            64,
            CudaMemoryKind.Transient);
        nint firstPointer = first.Pointer;
        nint secondPointer = second.Pointer;
        Assert.NotEqual(firstPointer, secondPointer);

        first.Dispose();
        using (CudaMemoryLease reused = manager.Rent(
                   64,
                   CudaMemoryKind.Transient))
        {
            Assert.Equal(firstPointer, reused.Pointer);
            Assert.Equal(2, allocator.AllocationAttempts);
        }
        second.Dispose();

        Assert.Equal(0, manager.ActiveAllocationCount);
        Assert.Equal(2, manager.GraphPinnedAllocationCount);
        Assert.Equal(128, manager.GraphPinnedBytes);
        Assert.Equal(128, manager.AllocatedBytes);
        CudaMemoryManagerTelemetry telemetry = manager.Telemetry;
        Assert.Equal(2, telemetry.AllocationCount);
        Assert.Equal(0, telemetry.ActiveAllocationCount);
        Assert.Equal(0, telemetry.CachedAllocationCount);
        Assert.Equal(2, telemetry.GraphPinnedAllocationCount);
        Assert.Equal(128, telemetry.GraphPinnedBytes);

        using IDisposable reservation = capture.Commit();
        Assert.Equal(2, manager.GraphPinnedAllocationCount);
        reservation.Dispose();

        Assert.Equal(0, manager.AllocationCount);
        Assert.Equal(0, manager.AllocatedBytes);
        Assert.Equal(2, allocator.ReleaseAttempts);
    }

    [Fact]
    public void HundredsOfLogicalRentsUseOneNativeAllocationDuringCapture()
    {
        var allocator = new FakeAllocator();
        using var manager = NoCacheManager(allocator);
        using CudaGraphCaptureScope capture =
            manager.BeginGraphCaptureReservation();
        nint pointer = nint.Zero;

        for (int iteration = 0; iteration < 500; iteration++)
        {
            using CudaMemoryLease lease = manager.Rent(
                4096,
                CudaMemoryKind.Workspace);
            if (iteration == 0)
                pointer = lease.Pointer;
            else
                Assert.Equal(pointer, lease.Pointer);
        }

        Assert.Equal(1, allocator.AllocationAttempts);
        Assert.Equal(1, manager.GraphPinnedAllocationCount);
        Assert.Equal(4096, manager.GraphPinnedBytes);
        using IDisposable reservation = capture.Commit();
    }

    [Fact]
    public void CapturePoolSeparatesSizeAndMemoryKind()
    {
        var allocator = new FakeAllocator();
        using var manager = NoCacheManager(allocator);
        using CudaGraphCaptureScope capture =
            manager.BeginGraphCaptureReservation();

        nint transient64 = RentAndReturn(
            manager,
            64,
            CudaMemoryKind.Transient);
        nint workspace64 = RentAndReturn(
            manager,
            64,
            CudaMemoryKind.Workspace);
        nint transient96 = RentAndReturn(
            manager,
            96,
            CudaMemoryKind.Transient);

        Assert.Equal(
            transient64,
            RentAndReturn(manager, 64, CudaMemoryKind.Transient));
        Assert.Equal(
            workspace64,
            RentAndReturn(manager, 64, CudaMemoryKind.Workspace));
        Assert.Equal(
            transient96,
            RentAndReturn(manager, 96, CudaMemoryKind.Transient));
        Assert.Equal(3, allocator.AllocationAttempts);
        Assert.Equal(3, manager.GraphPinnedAllocationCount);

        using IDisposable reservation = capture.Commit();
    }

    [Fact]
    public void UncommittedDisposeRollsBackActiveAndIdleAllocations()
    {
        var allocator = new FakeAllocator();
        using var manager = NoCacheManager(allocator);
        var fence = new FakeFence();
        CudaGraphCaptureScope capture =
            manager.BeginGraphCaptureReservation();
        CudaMemoryLease idle = manager.Rent(
            32,
            CudaMemoryKind.Transient);
        idle.SetReleaseFence(fence);
        idle.Dispose();
        CudaMemoryLease active = manager.Allocate(
            48,
            CudaMemoryKind.Workspace);

        capture.Dispose();

        Assert.True(active.IsClosed);
        Assert.Equal(0, manager.ActiveAllocationCount);
        Assert.Equal(0, manager.GraphPinnedAllocationCount);
        Assert.Equal(0, manager.AllocationCount);
        Assert.Equal(2, allocator.ReleaseAttempts);
        Assert.Equal(1, fence.DisposeCount);
        Assert.Equal(0, fence.WaitCount);
    }

    [Fact]
    public void CommitRejectsOutstandingLeaseAndScopeCanStillRollback()
    {
        var allocator = new FakeAllocator();
        using var manager = NoCacheManager(allocator);
        CudaGraphCaptureScope capture =
            manager.BeginGraphCaptureReservation();
        CudaMemoryLease lease = manager.Rent(
            64,
            CudaMemoryKind.Transient);

        InvalidOperationException exception =
            Assert.Throws<InvalidOperationException>(capture.Commit);
        Assert.Contains("active", exception.Message);

        capture.Dispose();
        Assert.True(lease.IsClosed);
        Assert.Equal(0, manager.AllocationCount);
    }

    [Fact]
    public void PersistentAllocationNestedCaptureAndDoubleCommitAreRejected()
    {
        using var manager = NoCacheManager(new FakeAllocator());
        using CudaMemoryLease persistent = manager.Allocate(
            8,
            CudaMemoryKind.Persistent);
        using CudaGraphCaptureScope capture =
            manager.BeginGraphCaptureReservation();

        Assert.Throws<InvalidOperationException>(() =>
            manager.Allocate(8, CudaMemoryKind.Persistent));
        Assert.Throws<InvalidOperationException>(() =>
            manager.Allocate(8, CudaMemoryKind.PinnedStaging));
        Assert.Throws<InvalidOperationException>(() =>
            manager.BeginGraphCaptureReservation());

        using IDisposable reservation = capture.Commit();
        Assert.Throws<InvalidOperationException>(capture.Commit);
    }

    [Fact]
    public void OtherThreadAndOtherManagerDoNotJoinCapturePool()
    {
        var allocator = new FakeAllocator();
        using var firstManager = NoCacheManager(allocator, deviceIndex: 0);
        using var secondManager = NoCacheManager(allocator, deviceIndex: 1);
        using CudaGraphCaptureScope firstCapture =
            firstManager.BeginGraphCaptureReservation();
        nint captured = RentAndReturn(
            firstManager,
            128,
            CudaMemoryKind.Transient);

        nint otherThread = nint.Zero;
        Exception? workerFailure = null;
        var worker = new Thread(() =>
        {
            try
            {
                using CudaMemoryLease lease = firstManager.Rent(
                    128,
                    CudaMemoryKind.Transient);
                otherThread = lease.Pointer;
            }
            catch (Exception exception)
            {
                workerFailure = exception;
            }
        });
        worker.Start();
        Assert.True(worker.Join(TimeSpan.FromSeconds(5)));
        Assert.Null(workerFailure);
        Assert.NotEqual(captured, otherThread);
        Assert.Equal(1, firstManager.GraphPinnedAllocationCount);

        using CudaGraphCaptureScope secondCapture =
            secondManager.BeginGraphCaptureReservation();
        nint secondCaptured = RentAndReturn(
            secondManager,
            128,
            CudaMemoryKind.Transient);
        Assert.NotEqual(captured, secondCaptured);
        Assert.Equal(1, secondManager.GraphPinnedAllocationCount);

        using IDisposable firstReservation = firstCapture.Commit();
        using IDisposable secondReservation = secondCapture.Commit();
    }

    [Fact]
    public void CommitIsOwnerThreadAffineAndCanRecoverAfterRejection()
    {
        using var manager = NoCacheManager(new FakeAllocator());
        using CudaGraphCaptureScope capture =
            manager.BeginGraphCaptureReservation();
        _ = RentAndReturn(manager, 64, CudaMemoryKind.Transient);
        Exception? workerResult = null;
        var worker = new Thread(() =>
        {
            workerResult = Record.Exception(() => capture.Commit());
        });

        worker.Start();
        Assert.True(worker.Join(TimeSpan.FromSeconds(5)));
        InvalidOperationException exception = Assert.IsType<InvalidOperationException>(
            workerResult);
        Assert.Contains("thread", exception.Message);

        using IDisposable reservation = capture.Commit();
        Assert.Equal(1, manager.GraphPinnedAllocationCount);
    }

    [Fact]
    public void ExceptionLeavingCaptureScopeRollsBackEveryAllocation()
    {
        var allocator = new FakeAllocator();
        using var manager = NoCacheManager(allocator);

        Action failCapture = () =>
        {
            using CudaGraphCaptureScope capture =
                manager.BeginGraphCaptureReservation();
            _ = RentAndReturn(manager, 64, CudaMemoryKind.Transient);
            _ = manager.Allocate(96, CudaMemoryKind.Workspace);
            throw new InvalidOperationException("scripted capture failure");
        };
        InvalidOperationException exception = Assert.Throws<InvalidOperationException>(
            failCapture);

        Assert.Equal("scripted capture failure", exception.Message);
        Assert.Equal(0, manager.AllocationCount);
        Assert.Equal(0, manager.AllocatedBytes);
        Assert.Equal(2, allocator.ReleaseAttempts);
    }

    [Fact]
    public void ConcurrentManagersOwnIndependentCaptureReservations()
    {
        var allocator = new FakeAllocator();
        using var firstManager = NoCacheManager(allocator, deviceIndex: 0);
        using var secondManager = NoCacheManager(allocator, deviceIndex: 1);
        using var ready = new CountdownEvent(2);
        using var start = new ManualResetEventSlim();
        var failures = new ConcurrentQueue<Exception>();

        static void RunCapture(
            CudaMemoryManager manager,
            CountdownEvent ready,
            ManualResetEventSlim start,
            ConcurrentQueue<Exception> failures)
        {
            try
            {
                using CudaGraphCaptureScope capture =
                    manager.BeginGraphCaptureReservation();
                ready.Signal();
                start.Wait(TimeSpan.FromSeconds(5));
                _ = RentAndReturn(manager, 128, CudaMemoryKind.Workspace);
                using IDisposable reservation = capture.Commit();
                Assert.Equal(1, manager.GraphPinnedAllocationCount);
            }
            catch (Exception exception)
            {
                failures.Enqueue(exception);
            }
        }

        var first = new Thread(() => RunCapture(
            firstManager,
            ready,
            start,
            failures));
        var second = new Thread(() => RunCapture(
            secondManager,
            ready,
            start,
            failures));
        first.Start();
        second.Start();
        bool bothReady = ready.Wait(
            TimeSpan.FromSeconds(5),
            TestContext.Current.CancellationToken);
        start.Set();
        Assert.True(bothReady);
        Assert.True(first.Join(TimeSpan.FromSeconds(5)));
        Assert.True(second.Join(TimeSpan.FromSeconds(5)));

        Assert.Empty(failures);
        Assert.Equal(2, allocator.AllocationAttempts);
        Assert.Equal(2, allocator.ReleaseAttempts);
    }

    [Fact]
    public void ManagerDisposeOwnsCommittedReservationCleanup()
    {
        var allocator = new FakeAllocator();
        var manager = NoCacheManager(allocator);
        using CudaGraphCaptureScope capture =
            manager.BeginGraphCaptureReservation();
        _ = RentAndReturn(manager, 64, CudaMemoryKind.Transient);
        _ = RentAndReturn(manager, 96, CudaMemoryKind.Workspace);
        IDisposable reservation = capture.Commit();

        manager.Dispose();
        reservation.Dispose();

        Assert.Equal(0, manager.AllocationCount);
        Assert.Equal(0, manager.AllocatedBytes);
        Assert.Equal(2, allocator.ReleaseAttempts);
    }

    [Fact]
    public void TrimAndOomRecoveryPreserveGraphPinnedAllocations()
    {
        var allocator = new FakeAllocator();
        int recoveryCount = 0;
        using var manager = new CudaMemoryManager(
            0,
            allocator,
            (_, _, _) => Interlocked.Increment(ref recoveryCount),
            maximumReusableCacheBytes: 0,
            maximumReusableCacheEntries: 0);
        using CudaGraphCaptureScope capture =
            manager.BeginGraphCaptureReservation();
        nint first = RentAndReturn(
            manager,
            64,
            CudaMemoryKind.Transient);

        Assert.Equal(0, manager.TrimReusableCache());
        Assert.Equal(1, manager.GraphPinnedAllocationCount);
        allocator.FailNextAllocation = true;
        nint second = RentAndReturn(
            manager,
            96,
            CudaMemoryKind.Workspace);

        Assert.NotEqual(first, second);
        Assert.Equal(1, recoveryCount);
        Assert.Equal(3, allocator.AllocationAttempts);
        Assert.Equal(2, manager.GraphPinnedAllocationCount);
        Assert.Equal(0, allocator.ReleaseAttempts);
        using IDisposable reservation = capture.Commit();
    }

    [Fact]
    public void ReservationDisposeReturnsReusableStorageToBoundedCache()
    {
        var allocator = new FakeAllocator();
        using var manager = new CudaMemoryManager(
            0,
            allocator,
            oomRecovery: null,
            maximumReusableCacheBytes: 1024,
            maximumReusableCacheEntries: 2);
        using CudaGraphCaptureScope capture =
            manager.BeginGraphCaptureReservation();
        nint captured = RentAndReturn(
            manager,
            128,
            CudaMemoryKind.Workspace);
        IDisposable reservation = capture.Commit();

        reservation.Dispose();

        Assert.Equal(0, manager.GraphPinnedAllocationCount);
        Assert.Equal(1, manager.CachedAllocationCount);
        Assert.Equal(0, allocator.ReleaseAttempts);
        Assert.Equal(
            captured,
            RentAndReturn(manager, 128, CudaMemoryKind.Workspace));
        Assert.Equal(1, allocator.AllocationAttempts);
    }

    [Fact]
    public void CleanupContinuesAfterIndividualFenceAndReleaseFailures()
    {
        var allocator = new FakeAllocator
        {
            ThrowOnEveryRelease = true,
        };
        var manager = NoCacheManager(allocator);
        using CudaGraphCaptureScope capture =
            manager.BeginGraphCaptureReservation();
        var fences = Enumerable.Range(0, 3)
            .Select(_ => new FakeFence { ThrowOnDispose = true })
            .ToArray();
        foreach (FakeFence fence in fences)
        {
            CudaMemoryLease lease = manager.Rent(
                24,
                CudaMemoryKind.Transient);
            lease.SetReleaseFence(fence);
            lease.Dispose();
        }
        using IDisposable reservation = capture.Commit();

        Exception? exception = Record.Exception(reservation.Dispose);

        Assert.Null(exception);
        Assert.Equal(1, allocator.ReleaseAttempts);
        Assert.Equal(4, manager.ReleaseErrors.Count);
        Assert.Equal(0, manager.AllocationCount);
        Assert.All(fences, fence => Assert.Equal(1, fence.DisposeCount));
    }

    private static CudaMemoryManager NoCacheManager(
        FakeAllocator allocator,
        int deviceIndex = 0)
        => new(
            deviceIndex,
            allocator,
            oomRecovery: null,
            maximumReusableCacheBytes: 0,
            maximumReusableCacheEntries: 0);

    private static nint RentAndReturn(
        CudaMemoryManager manager,
        nuint bytes,
        CudaMemoryKind kind)
    {
        using CudaMemoryLease lease = manager.Rent(bytes, kind);
        return lease.Pointer;
    }

    private sealed class FakeAllocator : ICudaMemoryAllocator
    {
        private long _nextPointer;
        private int _allocationAttempts;
        private int _releaseAttempts;

        public bool ThrowOnEveryRelease { get; init; }

        public bool FailNextAllocation { get; set; }

        public int AllocationAttempts => Volatile.Read(ref _allocationAttempts);

        public int ReleaseAttempts => Volatile.Read(ref _releaseAttempts);

        public ConcurrentQueue<nint> Releases { get; } = new();

        public nint Allocate(
            int deviceIndex,
            nuint byteLength,
            CudaMemoryKind kind)
        {
            Interlocked.Increment(ref _allocationAttempts);
            if (FailNextAllocation)
            {
                FailNextAllocation = false;
                throw new OutOfMemoryException("scripted allocation failure");
            }
            return (nint)Interlocked.Increment(ref _nextPointer);
        }

        public void Release(
            int deviceIndex,
            nint pointer,
            nuint byteLength,
            CudaMemoryKind kind)
        {
            Interlocked.Increment(ref _releaseAttempts);
            Releases.Enqueue(pointer);
            if (ThrowOnEveryRelease)
                throw new InvalidOperationException("scripted release failure");
        }
    }

    private sealed class FakeFence : ICudaCompletionFence
    {
        private int _waitCount;
        private int _disposeCount;

        public bool ThrowOnDispose { get; init; }

        public bool IsCompleted => false;

        public int WaitCount => Volatile.Read(ref _waitCount);

        public int DisposeCount => Volatile.Read(ref _disposeCount);

        public void Wait() => Interlocked.Increment(ref _waitCount);

        public void Dispose()
        {
            Interlocked.Increment(ref _disposeCount);
            if (ThrowOnDispose)
                throw new InvalidOperationException("scripted fence failure");
        }
    }
}
