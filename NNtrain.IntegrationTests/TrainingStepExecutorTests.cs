using NNtrain;
using NNtrain.Runtime.Execution;
using NNtrain.Training.Execution;
using Xunit;

public sealed class TrainingStepExecutorTests
{
    private static readonly string[] PhaseNames =
    [
        "batch",
        "zero",
        "forward",
        "backward",
        "reduce",
        "clip",
        "schedule",
        "optimizer",
        "metrics",
    ];

    [Fact]
    public void ExecutesPhasesInSessionOrderAndCommitsMetricsLast()
    {
        using var fixture = new Fixture();
        var observed = new List<(string Phase, TrainingStepPhase State)>();
        TrainingStepOperations operations = CreateOperations(phase =>
        {
            observed.Add((phase, fixture.Executor.State!.Phase));
        });

        TrainingStepState committed = fixture.Executor.Execute(operations);

        Assert.Equal(
            [
                ("batch", TrainingStepPhase.Initialized),
                ("zero", TrainingStepPhase.BatchAcquired),
                ("forward", TrainingStepPhase.GradientsCleared),
                ("backward", TrainingStepPhase.ForwardCompleted),
                ("reduce", TrainingStepPhase.BackwardCompleted),
                ("clip", TrainingStepPhase.GradientsReduced),
                ("schedule", TrainingStepPhase.GradientsClipped),
                ("optimizer", TrainingStepPhase.ScheduleApplied),
                ("metrics", TrainingStepPhase.OptimizerCommitted),
            ],
            observed);
        Assert.Equal(TrainingStepPhase.MetricsCommitted, committed.Phase);
        Assert.Equal(committed, fixture.Executor.State);
        Assert.Equal(0, fixture.Session.LastCommittedStep);
        Assert.True(fixture.Session.CanPublishCheckpoint);
        Assert.False(fixture.Session.IsFaulted);
        Assert.Null(fixture.Session.Failure);
    }

    [Fact]
    public void CommittedStepCanBeginTheNextGlobalStep()
    {
        using var fixture = new Fixture();
        var starts = new List<(long Step, TrainingStepPhase Phase)>();
        TrainingStepOperations operations = CreateOperations(phase =>
        {
            if (phase == "batch")
            {
                TrainingStepState state = fixture.Executor.State!;
                starts.Add((state.GlobalStep, state.Phase));
            }
        });

        TrainingStepState first = fixture.Executor.Execute(operations);
        TrainingStepState second = fixture.Executor.Execute(operations);

        Assert.Equal(
            [
                (0L, TrainingStepPhase.Initialized),
                (1L, TrainingStepPhase.Initialized),
            ],
            starts);
        Assert.Equal(0, first.GlobalStep);
        Assert.Equal(1, second.GlobalStep);
        Assert.Equal(TrainingStepPhase.MetricsCommitted, second.Phase);
        Assert.Equal(1, fixture.Session.LastCommittedStep);
    }

    [Theory]
    [InlineData("batch")]
    [InlineData("zero")]
    [InlineData("forward")]
    [InlineData("backward")]
    [InlineData("reduce")]
    [InlineData("clip")]
    [InlineData("schedule")]
    [InlineData("optimizer")]
    [InlineData("metrics")]
    public void PhaseFailureFaultsSessionAndRejectsReuse(string failedPhase)
    {
        using var fixture = new Fixture();
        var visited = new List<string>();
        var failure = new TestStepException(failedPhase);
        TrainingStepOperations operations = CreateOperations(phase =>
        {
            visited.Add(phase);
            if (phase == failedPhase)
                throw failure;
        });

        TestStepException thrown = Assert.Throws<TestStepException>(
            () => fixture.Executor.Execute(operations));

        Assert.Same(failure, thrown);
        TrainingStepState faulted = fixture.Executor.State!;
        Assert.Equal(TrainingStepPhase.Faulted, faulted.Phase);
        Assert.Same(failure, faulted.Failure);
        Assert.True(fixture.Session.IsFaulted);
        Assert.Same(failure, fixture.Session.Failure);
        Assert.False(fixture.Session.CanPublishCheckpoint);
        Assert.Equal(-1, fixture.Session.LastCommittedStep);
        Assert.Equal(
            PhaseNames.Take(Array.IndexOf(PhaseNames, failedPhase) + 1),
            visited);

        InvalidOperationException reuse =
            Assert.Throws<InvalidOperationException>(
                () => fixture.Executor.Execute(CreateOperations(_ => { })));
        Assert.Same(failure, reuse.InnerException);
        Assert.Equal(TrainingStepPhase.Faulted, fixture.Executor.State!.Phase);
    }

    [Fact]
    public void FailureAfterCommitRevokesCheckpointPublishing()
    {
        using var fixture = new Fixture();
        fixture.Executor.Execute(CreateOperations(_ => { }));
        var failure = new TestStepException("backward");
        TrainingStepOperations failing = CreateOperations(phase =>
        {
            if (phase == "backward")
                throw failure;
        });

        Assert.Throws<TestStepException>(
            () => fixture.Executor.Execute(failing));

        Assert.Equal(0, fixture.Session.LastCommittedStep);
        Assert.True(fixture.Session.IsFaulted);
        Assert.Same(failure, fixture.Session.Failure);
        Assert.False(fixture.Session.CanPublishCheckpoint);
    }

    [Fact]
    public void InvalidOperationsDoNotStartOrFaultTheSession()
    {
        using var fixture = new Fixture();
        TrainingStepOperations invalid = CreateOperations(_ => { }) with
        {
            CommitMetrics = null!,
        };

        Assert.Throws<ArgumentNullException>(
            () => fixture.Executor.Execute(invalid));

        Assert.Null(fixture.Executor.State);
        Assert.Equal(-1, fixture.Session.LastCommittedStep);
        Assert.False(fixture.Session.IsFaulted);
        Assert.Null(fixture.Session.Failure);
    }

    [Fact]
    public void TypedFusedGradientOperationRunsOnceAndCommitsAllPhases()
    {
        using var fixture = new Fixture();
        var operations = new TypedOperations(
            TrainingGradientExecutionMode.FusedForwardBackwardReduced);

        TrainingStepState committed = fixture.Executor.Execute(operations);

        Assert.Equal(1, operations.AcquireCalls);
        Assert.Equal(1, operations.ClearCalls);
        Assert.Equal(0, operations.ForwardCalls);
        Assert.Equal(0, operations.BackwardCalls);
        Assert.Equal(0, operations.ReduceCalls);
        Assert.Equal(1, operations.FusedCalls);
        Assert.Equal(1, operations.ClipCalls);
        Assert.Equal(1, operations.ScheduleCalls);
        Assert.Equal(1, operations.OptimizerCalls);
        Assert.Equal(1, operations.MetricsCalls);
        Assert.Equal(TrainingStepPhase.MetricsCommitted, committed.Phase);
    }

    [Fact]
    public void PreparationRunsOnceBeforeCudaTransferGuard()
    {
        using var execution = new ExecutionSession(new ExecutionOptions
        {
            Device = ExecutionDeviceKind.Cuda,
            CudaDevices = new DeviceSet(0),
        });
        using var session = new TrainingSession(execution);
        var executor = new TrainingStepExecutor(session);
        var operations = new TypedOperations(
            TrainingGradientExecutionMode.Separate);

        executor.Execute(operations);
        executor.Execute(operations);

        Assert.Equal(1, operations.PrepareCalls);
        Assert.True(operations.PreparationObservedOutsideGuard);
        Assert.Equal(2, operations.OptimizerCalls);
        Assert.Null(DeviceTransferGuard.CurrentSnapshot);
    }

    [Fact]
    public void FailedPreparationCanRetryWithoutStartingOrFaultingStep()
    {
        using var fixture = new Fixture();
        var operations = new TypedOperations(
            TrainingGradientExecutionMode.Separate)
        {
            PreparationFailuresRemaining = 1,
        };

        Assert.Throws<TestStepException>(
            () => fixture.Executor.Execute(operations));
        Assert.Null(fixture.Executor.State);
        Assert.False(fixture.Session.IsFaulted);

        TrainingStepState committed = fixture.Executor.Execute(operations);

        Assert.Equal(2, operations.PrepareCalls);
        Assert.Equal(TrainingStepPhase.MetricsCommitted, committed.Phase);
    }

    [Fact]
    public void TypedFusedGradientFailureDoesNotRunCommitOperations()
    {
        using var fixture = new Fixture();
        var failure = new TestStepException("fused");
        var operations = new TypedOperations(
            TrainingGradientExecutionMode.FusedForwardBackwardReduced)
        {
            Failure = failure,
        };

        TestStepException thrown = Assert.Throws<TestStepException>(
            () => fixture.Executor.Execute(operations));

        Assert.Same(failure, thrown);
        Assert.Equal(1, operations.FusedCalls);
        Assert.Equal(0, operations.ClipCalls);
        Assert.Equal(0, operations.ScheduleCalls);
        Assert.Equal(0, operations.OptimizerCalls);
        Assert.Equal(0, operations.MetricsCalls);
        Assert.Equal(TrainingStepPhase.Faulted, fixture.Executor.State!.Phase);
        Assert.False(fixture.Session.CanPublishCheckpoint);
    }

    [Fact]
    public void PublicationGuardAcceptsOnlyTheLatestMetricsCommit()
    {
        using var fixture = new Fixture();

        Assert.Throws<InvalidOperationException>(() =>
            ProductionTrainingSessionFactory.EnsureCanPublishCheckpoint(
                fixture.Session,
                globalStep: 0));

        TrainingStepState committed = fixture.Executor.Execute(
            CreateOperations(_ => { }));

        ProductionTrainingSessionFactory.EnsureCanPublishCheckpoint(
            fixture.Session,
            committed.GlobalStep);
        Assert.Throws<InvalidOperationException>(() =>
            ProductionTrainingSessionFactory.EnsureCanPublishCheckpoint(
                fixture.Session,
                committed.GlobalStep + 1));
    }

    [Fact]
    public void ResumedSessionContinuesWithTheNextGlobalStep()
    {
        using var execution = new ExecutionSession(new ExecutionOptions());
        using var session = new TrainingSession(
            execution,
            lastCommittedStep: 510);
        var executor = new TrainingStepExecutor(session);
        var operations = new TypedOperations(
            TrainingGradientExecutionMode.Separate);

        TrainingStepState committed = executor.Execute(operations);

        Assert.Equal(511, committed.GlobalStep);
        Assert.Equal(511, session.LastCommittedStep);
        Assert.Equal(1, operations.ForwardCalls);
        Assert.Equal(1, operations.BackwardCalls);
        Assert.Equal(1, operations.ReduceCalls);
        Assert.Equal(0, operations.FusedCalls);
    }

    [Fact]
    public void CudaTrainingStepGuardsEveryPhaseAgainstImplicitD2h()
    {
        using var execution = new ExecutionSession(new ExecutionOptions
        {
            Device = ExecutionDeviceKind.Cuda,
            CudaDevices = new DeviceSet(0, 1),
        });
        using var session = new TrainingSession(execution);
        var executor = new TrainingStepExecutor(session);
        int guardedPhases = 0;
        TrainingStepOperations operations = CreateOperations(_ =>
        {
            Assert.NotNull(DeviceTransferGuard.CurrentSnapshot);
            guardedPhases++;
        });

        TrainingStepState committed = executor.Execute(operations);

        Assert.Equal(9, guardedPhases);
        Assert.Equal(TrainingStepPhase.MetricsCommitted, committed.Phase);
        Assert.Null(DeviceTransferGuard.CurrentSnapshot);
    }

    [Fact]
    public void ImplicitTensorDownloadFaultsCudaStepBeforeOptimizerCommit()
    {
        using var execution = new ExecutionSession(new ExecutionOptions
        {
            Device = ExecutionDeviceKind.Cuda,
            CudaDevices = new DeviceSet(0),
        });
        using var session = new TrainingSession(execution);
        var executor = new TrainingStepExecutor(session);
        bool optimizerCommitted = false;
        TrainingStepOperations operations = CreateOperations(phase =>
        {
            if (phase == "forward")
            {
                DeviceTransferGuard.BeforeDeviceToHost(
                    4096,
                    "activation materialization");
            }
            if (phase == "optimizer")
                optimizerCommitted = true;
        });

        InvalidOperationException failure = Assert.Throws<InvalidOperationException>(
            () => executor.Execute(operations));

        Assert.Contains("implicit D2H", failure.Message);
        Assert.False(optimizerCommitted);
        Assert.Equal(TrainingStepPhase.Faulted, executor.State!.Phase);
        Assert.False(session.CanPublishCheckpoint);
        Assert.Null(DeviceTransferGuard.CurrentSnapshot);
    }

    private static TrainingStepOperations CreateOperations(
        Action<string> execute)
        => new(
            () => execute("batch"),
            () => execute("zero"),
            () => execute("forward"),
            () => execute("backward"),
            () => execute("reduce"),
            () => execute("clip"),
            () => execute("schedule"),
            () => execute("optimizer"),
            () => execute("metrics"));

    private sealed class Fixture : IDisposable
    {
        internal Fixture()
        {
            Execution = new ExecutionSession(new ExecutionOptions());
            Session = new TrainingSession(Execution);
            Executor = new TrainingStepExecutor(Session);
        }

        internal ExecutionSession Execution { get; }
        internal TrainingSession Session { get; }
        internal TrainingStepExecutor Executor { get; }

        public void Dispose()
        {
            Session.Dispose();
            Execution.Dispose();
        }
    }

    private sealed class TestStepException(string phase)
        : Exception(phase);

    private sealed class TypedOperations(
        TrainingGradientExecutionMode mode)
        : ITrainingStepOperations
    {
        internal Exception? Failure { get; init; }
        internal int PreparationFailuresRemaining { get; set; }
        internal int PrepareCalls { get; private set; }
        internal bool PreparationObservedOutsideGuard { get; private set; }
        internal int AcquireCalls { get; private set; }
        internal int ClearCalls { get; private set; }
        internal int ForwardCalls { get; private set; }
        internal int BackwardCalls { get; private set; }
        internal int ReduceCalls { get; private set; }
        internal int FusedCalls { get; private set; }
        internal int ClipCalls { get; private set; }
        internal int ScheduleCalls { get; private set; }
        internal int OptimizerCalls { get; private set; }
        internal int MetricsCalls { get; private set; }

        public TrainingGradientExecutionMode GradientExecutionMode
            => mode;

        public void Prepare()
        {
            PrepareCalls++;
            PreparationObservedOutsideGuard =
                DeviceTransferGuard.CurrentSnapshot is null;
            if (PreparationFailuresRemaining > 0)
            {
                PreparationFailuresRemaining--;
                throw new TestStepException("prepare");
            }
        }

        public void AcquireBatch() => AcquireCalls++;

        public void ClearGradients() => ClearCalls++;

        public void Forward() => ForwardCalls++;

        public void Backward() => BackwardCalls++;

        public void ReduceGradients() => ReduceCalls++;

        public void ForwardBackwardReduced()
        {
            FusedCalls++;
            if (Failure is not null)
                throw Failure;
        }

        public void ClipGradients() => ClipCalls++;

        public void ApplySchedule() => ScheduleCalls++;

        public void CommitOptimizer() => OptimizerCalls++;

        public void CommitMetrics() => MetricsCalls++;
    }
}
