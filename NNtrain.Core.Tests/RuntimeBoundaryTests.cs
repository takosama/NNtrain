using NNtrain;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;
using NNtrain.Training.Execution;
using Xunit;

public sealed class RuntimeBoundaryTests
{
    [Fact]
    public void PrecisionPoliciesKeepPureBFloat16AndMixedGradientsDistinct()
    {
        Assert.Equal(
            NumericFormat.BFloat16,
            PrecisionPolicy.BFloat16.Gradient);
        Assert.Equal(
            NumericFormat.Float32,
            PrecisionPolicy.Mix16_32.Gradient);
    }

    [Fact]
    public void CudaDeviceSetDoesNotSelectCudaExecution()
    {
        var options = new ExecutionOptions
        {
            Device = ExecutionDeviceKind.Cpu,
            CudaDevices = new DeviceSet(0, 1),
            Precision = PrecisionPolicy.Mix16_32,
        };

        Assert.Equal(ExecutionDeviceKind.Cpu, options.Validate().Device);
        Assert.Equal([0, 1], options.CudaDevices);
        Assert.Equal(NumericFormat.BFloat16, options.Precision.ParameterStorage);
        Assert.Equal(NumericFormat.Float32, options.Precision.OptimizerState);
    }

    [Fact]
    public void ExecutionScopesSupportNestedOutOfOrderAndDoubleDispose()
    {
        using var first = new ExecutionSession(new ExecutionOptions());
        using var second = new ExecutionSession(new ExecutionOptions());
        IDisposable outer = first.Enter();
        IDisposable inner = second.Enter();

        Assert.Same(second, ExecutionSession.Current);
        outer.Dispose();
        outer.Dispose();
        Assert.Same(second, ExecutionSession.Current);

        inner.Dispose();
        inner.Dispose();
        Assert.Null(ExecutionSession.Current);
    }

    [Fact]
    public void MemoryManagerReleasesEveryLeaseAfterOneReleaseFailure()
    {
        var allocator = new TrackingAllocator(failPointer: (nint)1);
        var manager = new CudaMemoryManager(0, allocator);
        _ = manager.Allocate(16, CudaMemoryKind.Persistent);
        _ = manager.Allocate(32, CudaMemoryKind.Workspace);

        manager.Dispose();

        Assert.Equal(2, allocator.ReleaseCount);
        Assert.Equal(0, manager.AllocationCount);
        Assert.Equal(0, manager.AllocatedBytes);
        Assert.Single(manager.ReleaseErrors);
    }

    [Fact]
    public void TrainingStepRequiresOrderedCommitBeforeCheckpoint()
    {
        using var execution = new ExecutionSession(new ExecutionOptions());
        using var training = new TrainingSession(execution);
        using TrainingStep step = training.BeginStep(0);

        Assert.Throws<InvalidOperationException>(
            () => step.Advance(TrainingStepPhase.ForwardCompleted));
        foreach (TrainingStepPhase phase in new[]
        {
            TrainingStepPhase.BatchAcquired,
            TrainingStepPhase.GradientsCleared,
            TrainingStepPhase.ForwardCompleted,
            TrainingStepPhase.BackwardCompleted,
            TrainingStepPhase.GradientsReduced,
            TrainingStepPhase.GradientsClipped,
            TrainingStepPhase.ScheduleApplied,
            TrainingStepPhase.OptimizerCommitted,
            TrainingStepPhase.MetricsCommitted,
        })
        {
            step.Advance(phase);
        }

        Assert.True(step.CanPublishCheckpoint);
        Assert.True(training.CanPublishCheckpoint);
        Assert.Equal(0, training.LastCommittedStep);
    }

    [Fact]
    public void TrainingSessionDisposesOwnedDataParallelEngine()
    {
        using var execution = new ExecutionSession(new ExecutionOptions());
        var training = new TrainingSession(execution);
        var model = new GptRinWikiJp(
            vocabularySize: 16,
            contextLength: 2,
            dModel: 4,
            numHeads: 1,
            dHidden: 8,
            numLayers: 1,
            rng: new Random(101));

        CudaDataParallelEngine engine =
            training.OwnCudaDataParallel(model);
        Assert.False(engine.IsDisposed);

        training.Dispose();

        Assert.True(engine.IsDisposed);
    }

    [Fact]
    public void DataParallelEngineOwnsAnImmutableCudaDeviceSet()
    {
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            var model = new GptRinWikiJp(
                vocabularySize: 16,
                contextLength: 2,
                dModel: 4,
                numHeads: 1,
                dHidden: 8,
                numLayers: 1,
                rng: new Random(103));
            using var engine = new CudaDataParallelEngine(model, [1, 0]);

            Tensor.CudaDeviceIndices = [7];

            Assert.Equal([1, 0], engine.CudaDeviceIndices);
            Assert.Throws<ArgumentException>(
                () => new CudaDataParallelEngine(model, [0, 0]));
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private sealed class TrackingAllocator(nint failPointer) : ICudaMemoryAllocator
    {
        private nint _nextPointer;
        public int ReleaseCount { get; private set; }

        public nint Allocate(
            int deviceIndex,
            nuint byteLength,
            CudaMemoryKind kind)
            => ++_nextPointer;

        public void Release(
            int deviceIndex,
            nint pointer,
            nuint byteLength,
            CudaMemoryKind kind)
        {
            ReleaseCount++;
            if (pointer == failPointer)
                throw new InvalidOperationException("Expected test failure.");
        }
    }
}
