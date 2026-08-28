using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaReplicaExecutorTests
{
    [Fact]
    public async Task DedicatedWorkersReuseThreadsAndRunConcurrently()
    {
        CancellationToken cancellationToken =
            TestContext.Current.CancellationToken;
        using var executor = new CudaReplicaExecutor([3, 7], _ => { });
        using var work = new ConcurrentRecordingWork(replicaCount: 2);
        Task first = Task.Run(() => executor.Execute(
            work,
            replicaCount: 2,
            session: null,
            PrecisionPolicy.Float32,
            cancellationToken), cancellationToken);
        try
        {
            Assert.True(work.AllEntered.Wait(
                TimeSpan.FromSeconds(5), cancellationToken));
            Assert.False(first.IsCompleted);
        }
        finally
        {
            work.Release.Set();
        }
        await first.WaitAsync(TimeSpan.FromSeconds(5), cancellationToken);
        int[] firstThreads = work.ThreadIds.ToArray();

        var second = new ThreadRecordingWork(replicaCount: 2);
        executor.Execute(
            second,
            replicaCount: 2,
            session: null,
            PrecisionPolicy.Float32,
            cancellationToken);
        CudaReplicaExecutorTelemetrySnapshot telemetry = executor.Telemetry;

        Assert.Equal([3, 7], work.AmbientDeviceIndices);
        Assert.Equal(firstThreads, second.ThreadIds);
        Assert.Equal(2, firstThreads.Distinct().Count());
        Assert.Equal(2, telemetry.WorkerThreadCreationCount);
        Assert.Equal(2, telemetry.LiveWorkerCount);
        Assert.Equal(2, telemetry.MaxConcurrentReplicaCount);
        Assert.Equal(2, telemetry.DispatchCount);
        Assert.Equal(2, telemetry.CompletedDispatchCount);
        Assert.Equal([2L, 2L], telemetry.WorkerExecutionCounts);
        Assert.Equal([1L, 1L], telemetry.WorkerContextBindingCounts);
    }

    [Fact]
    public void FailuresAreOrderedAndWorkersRecoverForTheNextDispatch()
    {
        CancellationToken cancellationToken =
            TestContext.Current.CancellationToken;
        using var executor = new CudaReplicaExecutor([1, 4], _ => { });
        var failing = new FailingWork();

        AggregateException exception = Assert.Throws<AggregateException>(() =>
            executor.Execute(
                failing,
                replicaCount: 2,
                session: null,
                PrecisionPolicy.BFloat16,
                cancellationToken));
        Assert.Collection(
            exception.InnerExceptions,
            first => Assert.Equal("replica 0 failed", first.Message),
            second => Assert.Equal("replica 1 failed", second.Message));

        var recovery = new ThreadRecordingWork(replicaCount: 2);
        executor.Execute(
            recovery,
            replicaCount: 2,
            session: null,
            PrecisionPolicy.BFloat16,
            cancellationToken);
        CudaReplicaExecutorTelemetrySnapshot telemetry = executor.Telemetry;
        Assert.Equal(2, telemetry.DispatchCount);
        Assert.Equal(1, telemetry.FailedDispatchCount);
        Assert.Equal(1, telemetry.CompletedDispatchCount);
        Assert.Equal(2, telemetry.LiveWorkerCount);
        Assert.All(recovery.ThreadIds, id => Assert.NotEqual(0, id));
    }

    [Fact]
    public async Task DisposeWaitsForActiveWorkAndLeavesNoWorkerAlive()
    {
        CancellationToken cancellationToken =
            TestContext.Current.CancellationToken;
        var executor = new CudaReplicaExecutor([0, 1], _ => { });
        using var work = new ConcurrentRecordingWork(replicaCount: 2);
        Task execution = Task.Run(() => executor.Execute(
            work,
            replicaCount: 2,
            session: null,
            PrecisionPolicy.Mix16_32,
            cancellationToken), cancellationToken);
        Assert.True(work.AllEntered.Wait(
            TimeSpan.FromSeconds(5), cancellationToken));

        Task dispose = Task.Run(executor.Dispose, cancellationToken);
        await Task.Delay(50, cancellationToken);
        Assert.False(dispose.IsCompleted);
        work.Release.Set();
        await execution.WaitAsync(TimeSpan.FromSeconds(5), cancellationToken);
        await dispose.WaitAsync(TimeSpan.FromSeconds(5), cancellationToken);

        CudaReplicaExecutorTelemetrySnapshot telemetry = executor.Telemetry;
        Assert.Equal(0, telemetry.LiveWorkerCount);
        Assert.Equal(0, telemetry.ActiveReplicaCount);
        Assert.Throws<ObjectDisposedException>(() => executor.Execute(
            new ThreadRecordingWork(replicaCount: 1),
            replicaCount: 1,
            session: null,
            PrecisionPolicy.Float32,
            cancellationToken));
    }

    [Fact]
    public void PreCanceledDispatchStartsNoWorkAndDoesNotPoisonExecutor()
    {
        using var executor = new CudaReplicaExecutor([0, 1], _ => { });
        var work = new ThreadRecordingWork(replicaCount: 2);
        using var cancellation = CancellationTokenSource
            .CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        cancellation.Cancel();

        Assert.Throws<OperationCanceledException>(() => executor.Execute(
            work,
            replicaCount: 2,
            session: null,
            PrecisionPolicy.Float32,
            cancellation.Token));
        Assert.Equal(0, executor.Telemetry.DispatchCount);

        executor.Execute(
            work,
            replicaCount: 2,
            session: null,
            PrecisionPolicy.Float32,
            TestContext.Current.CancellationToken);
        Assert.Equal(1, executor.Telemetry.CompletedDispatchCount);
    }

    [Fact]
    public void DedicatedWorkersShareGuardedTransferBudgetAndTelemetry()
    {
        using var executor = new CudaReplicaExecutor([0, 1], _ => { });
        var work = new AllowedTransferWork(replicaCount: 2);
        using IDisposable guard = DeviceTransferGuard.EnterTrainingStep(
            cudaDeviceCount: 2,
            maximumDeviceToHostCopies: 2);

        executor.Execute(
            work,
            replicaCount: 2,
            session: null,
            PrecisionPolicy.Mix16_32,
            TestContext.Current.CancellationToken);

        Assert.Equal([true, true], work.ObservedGuard);
        DeviceTransferSnapshot snapshot = Assert.NotNull(
            DeviceTransferGuard.CurrentSnapshot);
        Assert.Equal(2, snapshot.HostToDeviceCopyCount);
        Assert.Equal(64, snapshot.HostToDeviceBytes);
        Assert.Equal(2, snapshot.DeviceToHostCopyCount);
        Assert.Equal(8, snapshot.DeviceToHostBytes);
    }

    [Fact]
    public void DedicatedWorkersRejectImplicitTransfersWithoutCountingThem()
    {
        using var executor = new CudaReplicaExecutor([0, 1], _ => { });
        using IDisposable guard = DeviceTransferGuard.EnterTrainingStep(
            cudaDeviceCount: 2);

        AggregateException failure = Assert.Throws<AggregateException>(() =>
            executor.Execute(
                new ForbiddenTransferWork(),
                replicaCount: 2,
                session: null,
                PrecisionPolicy.Mix16_32,
                TestContext.Current.CancellationToken));

        Assert.Collection(
            failure.InnerExceptions,
            first => Assert.Contains("unclassified H2D", first.Message),
            second => Assert.Contains("implicit D2H", second.Message));
        DeviceTransferSnapshot snapshot = Assert.NotNull(
            DeviceTransferGuard.CurrentSnapshot);
        Assert.Equal(0, snapshot.HostToDeviceCopyCount);
        Assert.Equal(0, snapshot.HostToDeviceBytes);
        Assert.Equal(0, snapshot.DeviceToHostCopyCount);
        Assert.Equal(0, snapshot.DeviceToHostBytes);
    }

    [Fact]
    public void EngineKeepsEachCudaLaneOnOneWorkerAcrossStableSteps()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0, 1];
            using var execution = new ExecutionSession(new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda,
                CudaDevices = new DeviceSet([0, 1]),
                Precision = PrecisionPolicy.Mix16_32,
            });
            execution.AttachLane(CudaExecutionLaneFactory.Create(0));
            execution.AttachLane(CudaExecutionLaneFactory.Create(1));
            using IDisposable executionScope = execution.Enter();
            var model = new GptRinWikiJp(
                vocabularySize: 64,
                contextLength: 4,
                dModel: 16,
                numHeads: 2,
                dHidden: 32,
                numLayers: 1,
                rng: new Random(251),
                dtype: TensorDType.BFloat16);
            model.SetPrecisionMode(TensorPrecisionMode.Mix16_32);
            using var engine = new CudaDataParallelEngine(model, [0, 1]);
            engine.PrepareForTraining(batchSize: 2);
            int[] input = [1, 2, 3, 4, 5, 6, 7, 8];
            int[] target = [2, 3, 4, 5, 6, 7, 8, 9];

            model.ZeroGrad();
            float firstLoss = engine.ForwardBackward(
                input, target, batchSize: 2, sequenceLength: 4);
            CudaReplicaExecutorTelemetrySnapshot first =
                engine.ReplicaExecutorTelemetry;
            model.ZeroGrad();
            float secondLoss = engine.ForwardBackward(
                input, target, batchSize: 2, sequenceLength: 4);
            CudaReplicaExecutorTelemetrySnapshot second =
                engine.ReplicaExecutorTelemetry;

            Assert.True(float.IsFinite(firstLoss));
            Assert.True(float.IsFinite(secondLoss));
            Assert.Equal(2, second.WorkerThreadCreationCount);
            Assert.Equal(first.WorkerThreadIds, second.WorkerThreadIds);
            Assert.Equal([1L, 1L], second.WorkerContextBindingCounts);
            Assert.All(second.WorkerExecutionCounts, count =>
                Assert.True(count >= 2));
            Assert.Equal(1, engine.CachedTrainingShapePlanCount);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private sealed class ThreadRecordingWork(int replicaCount)
        : ICudaReplicaWorkDescriptor
    {
        internal int[] ThreadIds { get; } = new int[replicaCount];

        public void Execute(
            int replicaIndex,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ThreadIds[replicaIndex] = Environment.CurrentManagedThreadId;
        }
    }

    private sealed class ConcurrentRecordingWork(int replicaCount)
        : ICudaReplicaWorkDescriptor, IDisposable
    {
        internal CountdownEvent AllEntered { get; } = new(replicaCount);
        internal ManualResetEventSlim Release { get; } = new(false);
        internal int[] ThreadIds { get; } = new int[replicaCount];
        internal int[] AmbientDeviceIndices { get; } = new int[replicaCount];

        public void Execute(
            int replicaIndex,
            CancellationToken cancellationToken)
        {
            ThreadIds[replicaIndex] = Environment.CurrentManagedThreadId;
            AmbientDeviceIndices[replicaIndex] = Tensor.CudaDeviceIndex;
            AllEntered.Signal();
            Release.Wait(cancellationToken);
        }

        public void Dispose()
        {
            AllEntered.Dispose();
            Release.Dispose();
        }
    }

    private sealed class FailingWork : ICudaReplicaWorkDescriptor
    {
        public void Execute(
            int replicaIndex,
            CancellationToken cancellationToken)
            => throw new InvalidOperationException(
                $"replica {replicaIndex} failed");
    }

    private sealed class AllowedTransferWork(int replicaCount)
        : ICudaReplicaWorkDescriptor
    {
        internal bool[] ObservedGuard { get; } = new bool[replicaCount];

        public void Execute(
            int replicaIndex,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ObservedGuard[replicaIndex] =
                DeviceTransferGuard.CurrentSnapshot is not null;
            using (DeviceTransferGuard.AllowBatchHostToDevice())
            {
                DeviceTransferGuard.BeforeHostToDevice(
                    32,
                    $"replica {replicaIndex} batch input");
                DeviceTransferGuard.RecordHostToDevice(32);
            }
            DeviceTransferGuard.BeforeDeviceToHost(
                4,
                $"replica {replicaIndex} scalar loss");
        }
    }

    private sealed class ForbiddenTransferWork : ICudaReplicaWorkDescriptor
    {
        public void Execute(
            int replicaIndex,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (replicaIndex == 0)
            {
                DeviceTransferGuard.BeforeHostToDevice(
                    4096,
                    "worker optimizer upload");
                return;
            }

            DeviceTransferGuard.BeforeDeviceToHost(
                4096,
                "worker activation materialization");
        }
    }
}
