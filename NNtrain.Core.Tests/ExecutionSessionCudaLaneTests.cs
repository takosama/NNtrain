using System.Collections.Concurrent;
using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class ExecutionSessionCudaLaneTests
{
    [Fact]
    public void NativeLaneFactoryOwnsDistinctStreamsMemoryAndCapabilities()
    {
        var runtime = new FakeCudaRuntime();
        CudaExecutionLane lane = CudaExecutionLaneFactory.Create(3, runtime);

        Assert.Equal(3, lane.DeviceIndex);
        Assert.NotEqual(nint.Zero, lane.ComputeStreamHandle);
        Assert.NotEqual(nint.Zero, lane.CommunicationStreamHandle);
        Assert.NotEqual(
            lane.ComputeStreamHandle,
            lane.CommunicationStreamHandle);
        Assert.Equal(3, lane.Memory.DeviceIndex);
        Assert.True(lane.CudaCapabilities.Supports(
            CudaKernelFeature.TensorCores));

        nint computeStream = lane.ComputeStreamHandle;
        lane.ActivateComputeStream();
        lane.Dispose();
        lane.Dispose();

        Assert.Equal(
            [$"activate:3:{computeStream}", "activate:3:0"],
            runtime.Events.Where(static value =>
                value.StartsWith("activate:", StringComparison.Ordinal)));
        Assert.Equal(2, runtime.SynchronizedStreams.Count);
        Assert.Equal(2, runtime.DestroyedStreams.Count);
        Assert.Equal(
            runtime.CreatedStreams.AsEnumerable().Reverse(),
            runtime.DestroyedStreams);
    }

    [Fact]
    public void LaneFactoryCleansPartialConstructionInReverseOrder()
    {
        var runtime = new FakeCudaRuntime
        {
            ThrowOnCapabilities = true,
        };

        Assert.Throws<InvalidOperationException>(
            () => CudaExecutionLaneFactory.Create(1, runtime));

        Assert.Equal(2, runtime.CreatedStreams.Count);
        Assert.Equal(
            runtime.CreatedStreams.AsEnumerable().Reverse(),
            runtime.DestroyedStreams);
    }

    [Fact]
    public void LaneDisposeContinuesAndAggregatesEveryCleanupFailure()
    {
        var runtime = new FakeCudaRuntime
        {
            ThrowOnSynchronize = true,
            ThrowOnDestroy = true,
            ThrowOnRelease = true,
        };
        var profiler = new ThrowingProfiler();
        CudaExecutionLane lane = CudaExecutionLaneFactory.Create(
            0,
            runtime,
            profiler);
        _ = lane.Memory.Allocate(64, CudaMemoryKind.Persistent);

        AggregateException failure = Assert.Throws<AggregateException>(
            lane.Dispose);

        Assert.True(failure.Flatten().InnerExceptions.Count >= 6);
        Assert.Equal(2, runtime.SynchronizedStreams.Count);
        Assert.Equal(2, runtime.DestroyedStreams.Count);
        Assert.Equal(1, runtime.ReleaseAttempts);
        Assert.True(profiler.DisposeCalled);
        lane.Dispose();
    }

    [Fact]
    public async Task ResourceAttachRaceWithLaneDisposeNeverLeaksRejectedResources()
    {
        var runtime = new FakeCudaRuntime();
        CudaExecutionLane lane = CudaExecutionLaneFactory.Create(0, runtime);
        var resources = new ConcurrentBag<TrackingDisposable>();
        using var start = new ManualResetEventSlim(false);
        Task[] attachers = Enumerable.Range(0, 8)
            .Select(workerIndex => Task.Run(() =>
            {
                start.Wait(TestContext.Current.CancellationToken);
                for (int index = 0; index < 128; index++)
                {
                    var resource = new TrackingDisposable();
                    resources.Add(resource);
                    try
                    {
                    ExecutionLaneResources.Attach(lane, resource);
                    }
                    catch (ObjectDisposedException)
                    {
                    }
                }
            }, TestContext.Current.CancellationToken))
            .ToArray();
        Task disposer = Task.Run(() =>
        {
            start.Wait(TestContext.Current.CancellationToken);
            lane.Dispose();
        }, TestContext.Current.CancellationToken);

        start.Set();
        await Task.WhenAll(attachers.Append(disposer));

        Assert.Equal(8 * 128, resources.Count);
        Assert.All(resources, resource => Assert.True(resource.Disposed));
    }

    [Fact]
    public void CpuSessionKeepsCpuAuthorityWithConfiguredCudaDevices()
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [7, 9];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using var session = new ExecutionSession(new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cpu,
                CudaDevices = new DeviceSet(3, 5),
                Precision = PrecisionPolicy.Mix16_32,
            });
            using IDisposable scope = session.Enter();

            Assert.Equal(TensorDevice.Cpu, Tensor.ExecutionDevice);
            Assert.Equal([3, 5], Tensor.CudaDeviceIndices);
            Assert.Same(
                PrecisionPolicy.Mix16_32,
                TensorExecutionContext.ActivePrecisionPolicy);
            Assert.Empty(session.Lanes);
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void CudaSessionSelectsAndRestoresLanesForOutOfOrderScopes()
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [7];
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            var first = new FakeStreamLane(3, (nint)301, (nint)302);
            var second = new FakeStreamLane(5, (nint)501, (nint)502);
            using var session = new ExecutionSession(
                new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet(3, 5),
                    Precision = PrecisionPolicy.BFloat16,
                },
                [first, second]);
            using IDisposable sessionScope = session.Enter();

            Assert.Equal(TensorDevice.Cuda, Tensor.ExecutionDevice);
            Assert.Equal(3, Tensor.CudaDeviceIndex);
            Assert.Equal(1, first.ActivationCount);

            IDisposable outer = TensorExecutionContext.Push(
                new TorchDevice(TensorDevice.Cuda, 5));
            IDisposable inner = TensorExecutionContext.Push(
                new TorchDevice(TensorDevice.Cuda, 3));
            Assert.Equal(3, Tensor.CudaDeviceIndex);

            outer.Dispose();
            outer.Dispose();
            Assert.Equal(3, Tensor.CudaDeviceIndex);

            inner.Dispose();
            inner.Dispose();
            Assert.Equal(3, Tensor.CudaDeviceIndex);
            // Enter activates the default lane once.  Switching to device 5
            // and back activates each changed binding exactly once; repeated
            // scope restoration must not rebind an unchanged lane.
            Assert.Equal(2, first.ActivationCount);
            Assert.Equal(1, second.ActivationCount);
            Assert.True(NativeCudaRuntime.TryResolveCommunicationStream(
                5,
                out nint communication));
            Assert.Equal((nint)502, communication);
            Assert.Throws<InvalidOperationException>(() =>
                TensorExecutionContext.Push(
                    new TorchDevice(TensorDevice.Cuda, 8)));
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void RepeatedComputeStreamResolutionKeepsUnchangedLaneBound()
    {
        var lane = new FakeStreamLane(4, (nint)401, (nint)402);
        using var session = new ExecutionSession(
            new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda,
                CudaDevices = new DeviceSet(4),
            },
            [lane]);
        using IDisposable scope = session.Enter();

        Assert.Equal((nint)401, NativeCudaRuntime.ResolveComputeStream(4));
        int activationCount = lane.ActivationCount;
        for (int iteration = 0; iteration < 128; iteration++)
        {
            Assert.Equal(
                (nint)401,
                NativeCudaRuntime.ResolveComputeStream(4));
        }

        Assert.Equal(activationCount, lane.ActivationCount);
    }

    [Fact]
    public async Task ComputeStreamIsReboundAfterManagedThreadHop()
    {
        var lane = new FakeStreamLane(2, (nint)201, (nint)202);
        using var session = new ExecutionSession(
            new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda,
                CudaDevices = new DeviceSet(2),
            },
            [lane]);
        using IDisposable scope = session.Enter();
        int before = lane.ActivationCount;

        nint stream = await Task.Run(
            () => NativeCudaRuntime.ResolveComputeStream(2),
            TestContext.Current.CancellationToken);

        Assert.Equal((nint)201, stream);
        Assert.True(lane.ActivationCount > before);
        Assert.NotEmpty(lane.ActivationThreads);
    }

    [Fact]
    public void ExecutionSessionDisposesEveryLaneAfterMultipleFailures()
    {
        var events = new List<int>();
        var first = new ThrowingLane(0, events);
        var second = new ThrowingLane(1, events);
        var session = new ExecutionSession(
            new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda,
                CudaDevices = new DeviceSet(0, 1),
            },
            [first, second]);

        AggregateException failure = Assert.Throws<AggregateException>(
            session.Dispose);

        Assert.Equal([1, 0], events);
        Assert.Equal(2, failure.Flatten().InnerExceptions.Count);
        Assert.True(first.Disposed);
        Assert.True(second.Disposed);
        session.Dispose();
    }

    [Fact]
    public async Task SameDeviceDistinctStreamsKeepGemmZeroAndCopiesOrdered()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        int baselineHandles = CudaBlas.ActiveLaneHandleCount;
        ExecutionSession first = CreateNativeCudaSession(0);
        ExecutionSession second = CreateNativeCudaSession(0);
        try
        {
            var firstLane = Assert.IsType<CudaExecutionLane>(
                first.GetRequiredLane(ExecutionDeviceKind.Cuda, 0));
            var secondLane = Assert.IsType<CudaExecutionLane>(
                second.GetRequiredLane(ExecutionDeviceKind.Cuda, 0));
            Assert.NotEqual(
                firstLane.ComputeStreamHandle,
                secondLane.ComputeStreamHandle);

            CancellationToken cancellationToken =
                TestContext.Current.CancellationToken;
            Task firstWork = Task.Factory.StartNew(
                () => RunOrderedGemmCopyLoop(first, 1f),
                cancellationToken,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default);
            Task secondWork = Task.Factory.StartNew(
                () => RunOrderedGemmCopyLoop(second, 3f),
                cancellationToken,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default);

            await Task.WhenAll(firstWork, secondWork);
            Assert.Equal(baselineHandles + 2, CudaBlas.ActiveLaneHandleCount);
        }
        finally
        {
            second.Dispose();
            first.Dispose();
        }
        Assert.Equal(baselineHandles, CudaBlas.ActiveLaneHandleCount);
    }

    [Fact]
    public async Task LaneDisposeReleasesLtPlansHandlesAndWorkspaces()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        int baselineLt = CudaBlasLt.ActiveLaneResourceCount;
        int baselineInt8 = CudaBlasLtInt8.ActiveLaneResourceCount;
        var session = CreateNativeCudaSession(0);
        try
        {
            using var start = new Barrier(4);
            Task[] workers = Enumerable.Range(0, 4)
                .Select(_ => Task.Factory.StartNew(
                    () => RunLtInitializationOnSession(session, start),
                    TestContext.Current.CancellationToken,
                    TaskCreationOptions.LongRunning,
                    TaskScheduler.Default))
                .ToArray();
            await Task.WhenAll(workers);

            Assert.Equal(baselineLt + 1, CudaBlasLt.ActiveLaneResourceCount);
            Assert.Equal(
                baselineInt8 + 1,
                CudaBlasLtInt8.ActiveLaneResourceCount);
        }
        finally
        {
            session.Dispose();
        }

        Assert.Equal(baselineLt, CudaBlasLt.ActiveLaneResourceCount);
        Assert.Equal(baselineInt8, CudaBlasLtInt8.ActiveLaneResourceCount);
    }

    [Fact]
    public async Task SameLaneConcurrentFirstGemmCreatesExactlyOneOwnedHandle()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        int baseline = CudaBlas.ActiveLaneHandleCount;
        var session = CreateNativeCudaSession(0);
        try
        {
            using var start = new Barrier(4);
            Task[] workers = Enumerable.Range(0, 4)
                .Select(index => Task.Factory.StartNew(
                    () =>
                    {
                        using IDisposable scope = session.Enter();
                        start.SignalAndWait(
                            TestContext.Current.CancellationToken);
                        RunOrderedGemmCopyLoop(session, index + 1f,
                            enterSession: false,
                            iterations: 1);
                    },
                    TestContext.Current.CancellationToken,
                    TaskCreationOptions.LongRunning,
                    TaskScheduler.Default))
                .ToArray();
            await Task.WhenAll(workers);
            Assert.Equal(baseline + 1, CudaBlas.ActiveLaneHandleCount);
        }
        finally
        {
            session.Dispose();
        }
        Assert.Equal(baseline, CudaBlas.ActiveLaneHandleCount);
    }

    [Fact]
    public void SameThreadSameDeviceSessionSwitchRebindsTensorKernelStream()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        using ExecutionSession first = CreateNativeCudaSession(0);
        using ExecutionSession second = CreateNativeCudaSession(0);
        for (int iteration = 0; iteration < 32; iteration++)
        {
            RunTensorAddOnSession(first, iteration + 1f);
            RunTensorAddOnSession(second, iteration + 3f);
        }
    }

    [Fact]
    public void RepeatedLaneLifecycleReturnsNativeAllocationsAndVramToBaseline()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        // Exclude one-time CUDA/cuBLAS driver caches from the measured region.
        ExecutionSession warmup = CreateNativeCudaSession(0);
        RunOrderedGemmCopyLoop(warmup, 1f, iterations: 1);
        RunLtInitializationOnSession(warmup, start: null);
        warmup.Dispose();

        NativeCudaDevice device = NativeCudaRuntime.GetDevice(0);
        device.Synchronize();
        long baselineFreeBytes = device.GetFreeMemory();
        NativeCudaAllocationTelemetry before =
            NativeCudaRuntime.AllocationTelemetry;

        const int repetitions = 16;
        for (int index = 0; index < repetitions; index++)
        {
            ExecutionSession session = CreateNativeCudaSession(0);
            try
            {
                RunOrderedGemmCopyLoop(
                    session,
                    index + 1f,
                    iterations: 1);
                RunLtInitializationOnSession(session, start: null);
            }
            finally
            {
                session.Dispose();
            }
        }

        device.Synchronize();
        NativeCudaAllocationTelemetry allocationDelta =
            NativeCudaRuntime.AllocationTelemetry - before;
        long finalFreeBytes = device.GetFreeMemory();
        const long vramToleranceBytes = 8L * 1024 * 1024;

        Assert.Equal(
            allocationDelta.AllocationCount,
            allocationDelta.FreeCount);
        Assert.Equal(
            allocationDelta.AllocationBytes,
            allocationDelta.FreeBytes);
        Assert.True(
            finalFreeBytes >= baselineFreeBytes - vramToleranceBytes,
            $"CUDA lane lifecycle retained " +
            $"{baselineFreeBytes - finalFreeBytes:N0} bytes; " +
            $"tolerance is {vramToleranceBytes:N0} bytes.");
    }

    [Fact]
    public void LaneTransientRentReusesExactNativePointerAndTrimsAtShutdown()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        ExecutionSession session = CreateNativeCudaSession(0);
        CudaExecutionLane lane = Assert.IsType<CudaExecutionLane>(
            session.GetRequiredLane(ExecutionDeviceKind.Cuda, 0));
        nint firstPointer;
        try
        {
            using IDisposable scope = session.Enter();
            NativeCudaDevice device = NativeCudaRuntime.GetDevice(0);
            using (NativeCudaBuffer<float> first = device.Allocate1D<float>(
                       257,
                       CudaMemoryKind.Transient))
            {
                firstPointer = first.NativePtr;
                Assert.Equal(1, lane.Memory.ActiveAllocationCount);
            }

            Assert.Equal(0, lane.Memory.ActiveAllocationCount);
            Assert.Equal(1, lane.Memory.CachedAllocationCount);
            Assert.Equal(257L * sizeof(float), lane.Memory.CachedBytes);

            using (NativeCudaBuffer<float> second = device.Allocate1D<float>(
                       257,
                       CudaMemoryKind.Transient))
            {
                Assert.Equal(firstPointer, second.NativePtr);
                Assert.Equal(1, lane.Memory.ActiveAllocationCount);
                Assert.Equal(0, lane.Memory.CachedAllocationCount);
            }
            Assert.Equal(1, lane.Memory.CachedAllocationCount);
        }
        finally
        {
            session.Dispose();
        }

        Assert.Equal(0, lane.Memory.AllocationCount);
        Assert.Equal(0, lane.Memory.AllocatedBytes);
    }

    private static ExecutionSession CreateNativeCudaSession(int deviceIndex)
        => new(
            new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda,
                CudaDevices = new DeviceSet(deviceIndex),
            },
            [CudaExecutionLaneFactory.Create(deviceIndex)]);

    private static void RunOrderedGemmCopyLoop(
        ExecutionSession session,
        float scale,
        bool enterSession = true,
        int iterations = 32)
    {
        using IDisposable? scope = enterSession ? session.Enter() : null;
        NativeCudaDevice device = NativeCudaRuntime.GetDevice(0);
        float[] leftValues = [1f * scale, 2f, 3f, 4f * scale];
        float[] rightValues = [5f, 6f * scale, 7f, 8f];
        using NativeCudaBuffer<float> left = device.Allocate(leftValues);
        using NativeCudaBuffer<float> right = device.Allocate(rightValues);
        using NativeCudaBuffer<float> output = device.Allocate1D<float>(4);
        float[] expected =
        [
            leftValues[0] * rightValues[0]
                + leftValues[1] * rightValues[2],
            leftValues[0] * rightValues[1]
                + leftValues[1] * rightValues[3],
            leftValues[2] * rightValues[0]
                + leftValues[3] * rightValues[2],
            leftValues[2] * rightValues[1]
                + leftValues[3] * rightValues[3],
        ];
        var actual = new float[4];

        for (int iteration = 0; iteration < iterations; iteration++)
        {
            output.MemSetToZero();
            CudaBlas.MatMulForward(
                device,
                deviceIndex: 0,
                left,
                right,
                output,
                batch: 1,
                m: 2,
                k: 2,
                n: 2);
            output.CopyToCPU(actual);
            Assert.Equal(expected, actual);
        }
    }

    private static void RunLtInitializationOnSession(
        ExecutionSession session,
        Barrier? start)
    {
        using IDisposable scope = session.Enter();
        start?.SignalAndWait(TestContext.Current.CancellationToken);
        NativeCudaDevice device = NativeCudaRuntime.GetDevice(0);
        const int width = 16;
        const int length = width * width;
        var bf16 = new ushort[length];
        TensorStorageCodec.EncodeBFloat16(
            Enumerable.Repeat(1f, length).ToArray(),
            bf16);
        var biasValues = new ushort[width];
        TensorStorageCodec.EncodeBFloat16(new float[width], biasValues);
        using NativeCudaBuffer<ushort> input = device.Allocate(bf16);
        using NativeCudaBuffer<ushort> weight = device.Allocate(bf16);
        using NativeCudaBuffer<ushort> bias = device.Allocate(biasValues);
        using NativeCudaBuffer<ushort> output =
            device.Allocate1D<ushort>(length);
        _ = CudaBlasLt.TryLinearForwardBFloat16(
            device,
            0,
            input,
            weight,
            bias,
            output,
            rows: width,
            inputWidth: width,
            outputWidth: width,
            applyRelu: false);

        using NativeCudaBuffer<sbyte> int8Left =
            device.Allocate(new sbyte[length]);
        using NativeCudaBuffer<sbyte> int8Right =
            device.Allocate(new sbyte[length]);
        using NativeCudaBuffer<int> int8Output =
            device.Allocate1D<int>(length);
        _ = CudaBlasLtInt8.TryMatMul(
            device,
            0,
            int8Left,
            int8Right,
            int8Output,
            width,
            width,
            width);
    }

    private static void RunTensorAddOnSession(
        ExecutionSession session,
        float offset)
    {
        using IDisposable scope = session.Enter();
        NativeCudaDevice device = NativeCudaRuntime.GetDevice(0);
        float[] leftValues = [offset, offset + 1f, offset + 2f, offset + 3f];
        float[] rightValues = [4f, 3f, 2f, 1f];
        using NativeCudaBuffer<float> left = device.Allocate(leftValues);
        using NativeCudaBuffer<float> right = device.Allocate(rightValues);
        using NativeCudaBuffer<float> output = device.Allocate1D<float>(4);
        CudaTensorNative.Add(
            0,
            left.NativePtr,
            right.NativePtr,
            output.NativePtr,
            length: 4,
            bfloat16: false);
        var actual = new float[4];
        output.CopyToCPU(actual);
        Assert.Equal(
            leftValues.Zip(rightValues, static (leftValue, rightValue) =>
                leftValue + rightValue),
            actual);
    }

    private sealed class FakeCudaRuntime : ICudaExecutionRuntime
    {
        private long _nextStream = 100;
        private long _nextPointer = 1000;

        internal bool ThrowOnCapabilities { get; init; }
        internal bool ThrowOnSynchronize { get; init; }
        internal bool ThrowOnDestroy { get; init; }
        internal bool ThrowOnRelease { get; init; }
        internal List<string> Events { get; } = [];
        internal List<nint> CreatedStreams { get; } = [];
        internal List<nint> DestroyedStreams { get; } = [];
        internal List<nint> SynchronizedStreams { get; } = [];
        internal int ReleaseAttempts { get; private set; }

        public nint CreateStream(int deviceIndex)
        {
            nint stream = (nint)Interlocked.Increment(ref _nextStream);
            CreatedStreams.Add(stream);
            Events.Add($"create:{deviceIndex}:{stream}");
            return stream;
        }

        public void DestroyStream(int deviceIndex, nint stream)
        {
            DestroyedStreams.Add(stream);
            Events.Add($"destroy:{deviceIndex}:{stream}");
            if (ThrowOnDestroy)
                throw new InvalidOperationException("scripted stream destroy failure");
        }

        public void ActivateStream(int deviceIndex, nint stream)
            => Events.Add($"activate:{deviceIndex}:{stream}");

        public void SynchronizeStream(int deviceIndex, nint stream)
        {
            SynchronizedStreams.Add(stream);
            Events.Add($"synchronize:{deviceIndex}:{stream}");
            if (ThrowOnSynchronize)
                throw new InvalidOperationException("scripted stream sync failure");
        }

        public CudaKernelCapabilities GetKernelCapabilities(int deviceIndex)
        {
            if (ThrowOnCapabilities)
                throw new InvalidOperationException("scripted capability failure");
            return new CudaKernelCapabilities(
                8,
                6,
                CudaKernelFeature.TensorCores);
        }

        public nint Allocate(
            int deviceIndex,
            nuint byteLength,
            CudaMemoryKind kind)
            => (nint)Interlocked.Increment(ref _nextPointer);

        public void Release(
            int deviceIndex,
            nint pointer,
            nuint byteLength,
            CudaMemoryKind kind)
        {
            ReleaseAttempts++;
            if (ThrowOnRelease)
                throw new InvalidOperationException("scripted memory release failure");
        }
    }

    private sealed class ThrowingProfiler : IExecutionProfiler
    {
        internal bool DisposeCalled { get; private set; }
        public IDisposable Measure(string operation) => EmptyDisposable.Instance;
        public void RecordCounter(string name, long value) { }
        public void Dispose()
        {
            DisposeCalled = true;
            throw new InvalidOperationException("scripted profiler failure");
        }
    }

    private sealed class EmptyDisposable : IDisposable
    {
        internal static EmptyDisposable Instance { get; } = new();
        public void Dispose() { }
    }

    private sealed class TrackingDisposable : IDisposable
    {
        private int _disposed;
        internal bool Disposed => Volatile.Read(ref _disposed) != 0;
        public void Dispose() => Interlocked.Exchange(ref _disposed, 1);
    }

    private sealed class FakeStreamLane(
        int deviceIndex,
        nint computeStream,
        nint communicationStream) : IStreamExecutionLane
    {
        private int _activationCount;
        internal ConcurrentBag<int> ActivationThreads { get; } = [];
        internal int ActivationCount => Volatile.Read(ref _activationCount);
        public ExecutionDeviceKind DeviceKind => ExecutionDeviceKind.Cuda;
        public int DeviceIndex { get; } = deviceIndex;
        public IDeviceMemoryManager MemoryManager { get; } =
            new FakeMemoryManager(deviceIndex);
        public IKernelCapabilitySet Capabilities { get; } =
            new CudaKernelCapabilities(8, 6, CudaKernelFeature.TensorCores);
        public IExecutionProfiler Profiler => NullExecutionProfiler.Instance;
        public nint ComputeStreamHandle { get; } = computeStream;
        public nint CommunicationStreamHandle { get; } = communicationStream;
        public void ActivateComputeStream()
        {
            Interlocked.Increment(ref _activationCount);
            ActivationThreads.Add(Environment.CurrentManagedThreadId);
        }
        public void SynchronizeComputeStream() { }
        public void SynchronizeCommunicationStream() { }
        public T OwnResource<T>(T resource) where T : class, IDisposable
            => resource;
        public void Dispose() => MemoryManager.Dispose();
    }

    private sealed class ThrowingLane(
        int deviceIndex,
        List<int> events) : IExecutionLane
    {
        public bool Disposed { get; private set; }
        public ExecutionDeviceKind DeviceKind => ExecutionDeviceKind.Cuda;
        public int DeviceIndex { get; } = deviceIndex;
        public IDeviceMemoryManager MemoryManager { get; } =
            new FakeMemoryManager(deviceIndex);
        public IKernelCapabilitySet Capabilities { get; } =
            new CudaKernelCapabilities(8, 6, CudaKernelFeature.None);
        public IExecutionProfiler Profiler => NullExecutionProfiler.Instance;
        public void Dispose()
        {
            Disposed = true;
            events.Add(DeviceIndex);
            throw new InvalidOperationException($"lane {DeviceIndex} failure");
        }
    }

    private sealed class FakeMemoryManager(int deviceIndex)
        : IDeviceMemoryManager
    {
        public int DeviceIndex { get; } = deviceIndex;
        public long AllocationCount => 0;
        public long AllocatedBytes => 0;
        public void Dispose() { }
    }
}
