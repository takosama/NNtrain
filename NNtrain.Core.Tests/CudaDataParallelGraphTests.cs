using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Interop;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaDataParallelGraphTests
{
    [Fact]
    public void CapturedReducerReadyEventCanBeSynchronizedAfterReplay()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        CudaExecutionLane lane = CudaExecutionLaneFactory.Create(0);
        using var execution = new ExecutionSession(
            new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda,
                CudaDevices = new DeviceSet(0),
                Precision = PrecisionPolicy.Mix16_32,
            },
            [lane]);
        using IDisposable executionScope = execution.Enter();
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(0);
        nint ready = CudaGradientBuckets.CreateReadyEvent(accelerator, 0);
        try
        {
            bool external = !CudaDispatchPolicy.Current
                    .DisableExternalGradientReadyEvents
                && CudaNativeGateway.AbiVersion.Minor
                    >= CudaAbiVersion.ExternalGradientReadyEventMinor;
            using CudaGraphExecutable graph = CudaGraphExecutable.Capture(
                lane,
                () =>
                {
                    if (external)
                    {
                        CudaGradientBuckets.RecordReadyExternal(
                            0, accelerator, ready);
                    }
                    else
                    {
                        CudaGradientBuckets.RecordReady(
                            0, accelerator, ready);
                    }
                });
            graph.Launch();
            if (!external)
                CudaGradientBuckets.RecordReady(0, accelerator, ready);
            Assert.Equal(0, NativeCudaRuntime.EventSynchronizeNative(0, ready));
            lane.SynchronizeComputeStream();
        }
        finally
        {
            CudaGradientBuckets.DestroyEvent(accelerator, 0, ready);
        }
    }

    [Fact]
    public void Mix8SingleGpuCapturesThenReplaysWithoutHotAllocations()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            var random = new CheckpointableRandom(1701);
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 4,
                dModel: 8,
                numHeads: 2,
                dHidden: 16,
                numLayers: 1,
                rng: random,
                dropout: 0.2f,
                dtype: TensorDType.Float32);
            random.BeginRuntime();
            model.AttachTrainingRandom(random);
            model.to(TensorPrecisionMode.Mix8_32);

            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            using var execution = new ExecutionSession(
                new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet(0),
                    Precision = PrecisionPolicy.Mix8_32,
                },
                [CudaExecutionLaneFactory.Create(0)]);
            using IDisposable sessionScope = execution.Enter();
            using var engine = new CudaDataParallelEngine(model, [0]);
            int[] input = [1, 2, 3, 4];
            int[] target = [2, 3, 4, 5];
            engine.PrepareForTraining(batchSize: 1);

            model.ZeroGrad();
            NativeCudaTransferTelemetry firstTransfersBefore =
                NativeCudaRuntime.TransferTelemetry;
            float first;
            try
            {
                first = engine.ForwardBackward(
                    input,
                    target,
                    1,
                    4,
                    Tensor.DefaultCrossEntropyIgnoreIndex,
                    globalStep: 0);
            }
            catch (Exception executionFailure)
            {
                throw new Xunit.Sdk.XunitException(
                    $"Graph failure: {engine.LastGraphFailure}\n" +
                    $"Fallback failure: {executionFailure}");
            }
            NativeCudaTransferTelemetry firstTransfers =
                NativeCudaRuntime.TransferTelemetry - firstTransfersBefore;
            CudaTrainingGraphTelemetry compiled = engine.TrainingGraphTelemetry;

            model.ZeroGrad();
            NativeCudaAllocationTelemetry allocationsBefore =
                NativeCudaRuntime.AllocationTelemetry;
            NativeCudaTransferTelemetry transfersBefore =
                NativeCudaRuntime.TransferTelemetry;
            float second = engine.ForwardBackward(
                input,
                target,
                1,
                4,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep: 1);
            NativeCudaAllocationTelemetry allocations =
                NativeCudaRuntime.AllocationTelemetry - allocationsBefore;
            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - transfersBefore;
            CudaTrainingGraphTelemetry replayed = engine.TrainingGraphTelemetry;

            Assert.True(float.IsFinite(first));
            Assert.True(float.IsFinite(second));
            Assert.True(
                compiled.CaptureCount == 1,
                engine.LastGraphFailure?.ToString()
                    ?? $"capture count was {compiled.CaptureCount}");
            Assert.Equal(1, compiled.ReplayCount);
            Assert.Equal(1, compiled.CachedCompiledPlanCount);
            Assert.True(compiled.GraphPinnedBytes > 0);
            Assert.Equal(2, firstTransfers.HostToDeviceCopyCount);
            Assert.Equal(
                2L * input.Length * sizeof(int),
                firstTransfers.HostToDeviceBytes);
            Assert.Equal(1, firstTransfers.DeviceToHostCopyCount);
            Assert.Equal(sizeof(float), firstTransfers.DeviceToHostBytes);
            Assert.Equal(2, replayed.ReplayCount);
            Assert.Equal(0, allocations.AllocationCount);
            Assert.Equal(0, allocations.FreeCount);
            Assert.Equal(2, transfers.HostToDeviceCopyCount);
            Assert.Equal(2L * input.Length * sizeof(int), transfers.HostToDeviceBytes);
            Assert.Equal(1, transfers.DeviceToHostCopyCount);
            Assert.Equal(sizeof(float), transfers.DeviceToHostBytes);

            int retiredPlans =
                engine.ReleaseCheckpointTransientMemory();
            Assert.Equal(1, retiredPlans);
            Assert.Equal(0, engine.CachedTrainingShapePlanCount);
            Assert.Equal(0, engine.TrainingGraphTelemetry.GraphPinnedBytes);

            model.ZeroGrad();
            float afterCheckpoint = engine.ForwardBackward(
                input,
                target,
                1,
                4,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep: 2);
            Assert.True(float.IsFinite(afterCheckpoint));
            Assert.Equal(2, engine.TrainingShapePlanBuildCount);
            Assert.Equal(1, engine.CachedTrainingShapePlanCount);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void Mix8GraphReplayKeepsLossStableAcrossInferenceTransition()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        AdamW? optimizer = null;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            var random = new CheckpointableRandom(1753);
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 4,
                dModel: 8,
                numHeads: 2,
                dHidden: 16,
                numLayers: 1,
                rng: random,
                dropout: 0f,
                dtype: TensorDType.Float32,
                tieWordEmbeddings: true);
            random.BeginRuntime();
            model.AttachTrainingRandom(random);
            model.to(TensorPrecisionMode.Mix8_32);

            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0, 1];
            using var execution = new ExecutionSession(
                new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet(0, 1),
                    Precision = PrecisionPolicy.Mix8_32,
                },
                [
                    CudaExecutionLaneFactory.Create(0),
                    CudaExecutionLaneFactory.Create(1),
                ]);
            using IDisposable sessionScope = execution.Enter();
            using var engine = new CudaDataParallelEngine(
                model,
                [0, 1],
                new CudaAdaptiveShardingOptions { Enabled = false });
            optimizer = new AdamW(
                model.Parameters(),
                new AdamWOptions
                {
                    LearningRate = 0.02f,
                    WeightDecay = 0f,
                });
            optimizer.prepare();
            int[] input = [1, 2, 3, 4, 5, 6, 7, 8];
            int[] target = [2, 3, 4, 5, 6, 7, 8, 9];
            engine.PrepareForTraining(batchSize: 2);

            // Move the encoded weights far enough from their capture-time
            // values that refreshing a stale BF16 leaf cache is observable.
            for (int step = 0; step < 12; step++)
            {
                optimizer.zero_grad();
                _ = engine.ForwardBackward(
                    input,
                    target,
                    2,
                    4,
                    Tensor.DefaultCrossEntropyIgnoreIndex,
                    globalStep: step);
                optimizer.step();
            }

            optimizer.zero_grad();
            float beforeInference = engine.ForwardBackward(
                input,
                target,
                2,
                4,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep: 100);
            optimizer.zero_grad();
            int[] generated = model.GenerateTokenIds(
                input[..4],
                maxNewTokens: 1,
                temperature: 0f,
                topK: 1,
                stopTokenId: null,
                random: new Random(1753));
            float afterInference = engine.ForwardBackward(
                input,
                target,
                2,
                4,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep: 100);

            Assert.Equal(5, generated.Length);
            Assert.True(model.IsTraining);
            Assert.Equal(beforeInference, afterInference);
            Assert.Equal(1, engine.TrainingGraphTelemetry.CaptureCount);
            Assert.Equal(0, engine.TrainingGraphTelemetry.FallbackCount);
        }
        finally
        {
            optimizer?.DisposeCudaResources();
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void ProfiledStepRetiresCompiledGraphBeforeEagerExecution()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            var random = new CheckpointableRandom(1811);
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 4,
                dModel: 8,
                numHeads: 2,
                dHidden: 16,
                numLayers: 1,
                rng: random,
                dropout: 0.2f,
                dtype: TensorDType.Float32);
            random.BeginRuntime();
            model.AttachTrainingRandom(random);
            model.to(TensorPrecisionMode.Mix8_32);

            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            using var execution = new ExecutionSession(
                new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet(0),
                    Precision = PrecisionPolicy.Mix8_32,
                },
                [CudaExecutionLaneFactory.Create(0)]);
            using IDisposable sessionScope = execution.Enter();
            using var engine = new CudaDataParallelEngine(model, [0]);
            int[] input = [1, 2, 3, 4];
            int[] target = [2, 3, 4, 5];

            model.ZeroGrad();
            _ = engine.ForwardBackward(
                input,
                target,
                1,
                4,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep: 0);
            CudaTrainingGraphTelemetry compiled =
                engine.TrainingGraphTelemetry;
            Assert.Equal(1, compiled.CachedCompiledPlanCount);
            Assert.True(compiled.GraphPinnedBytes > 0);

            model.ZeroGrad();
            CudaDataParallelProfile profile =
                engine.ForwardBackwardProfiled(input, target, 1, 4);
            CudaTrainingGraphTelemetry afterProbe =
                engine.TrainingGraphTelemetry;

            Assert.True(float.IsFinite(profile.Loss));
            Assert.Equal(0, afterProbe.CachedCompiledPlanCount);
            Assert.Equal(0, afterProbe.GraphPinnedBytes);
            Assert.Equal(compiled.CaptureCount, afterProbe.CaptureCount);
            Assert.Equal(compiled.ReplayCount, afterProbe.ReplayCount);

            model.ZeroGrad();
            _ = engine.ForwardBackward(
                input,
                target,
                1,
                4,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep: 1);
            CudaTrainingGraphTelemetry recaptured =
                engine.TrainingGraphTelemetry;
            Assert.Equal(2, recaptured.CaptureCount);
            Assert.Equal(1, recaptured.CachedCompiledPlanCount);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void Mix8TwoGpuCapturePublishesReducedGradientsWithoutRepack()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            var random = new CheckpointableRandom(1907);
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 4,
                dModel: 8,
                numHeads: 2,
                dHidden: 16,
                numLayers: 1,
                rng: random,
                dropout: 0.2f,
                dtype: TensorDType.Float32);
            random.BeginRuntime();
            model.AttachTrainingRandom(random);
            model.to(TensorPrecisionMode.Mix8_32);

            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0, 1];
            using var execution = new ExecutionSession(
                new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet(0, 1),
                    Precision = PrecisionPolicy.Mix8_32,
                },
                [
                    CudaExecutionLaneFactory.Create(0),
                    CudaExecutionLaneFactory.Create(1),
                ]);
            using IDisposable sessionScope = execution.Enter();
            using var engine = new CudaDataParallelEngine(
                model,
                [0, 1],
                new CudaAdaptiveShardingOptions { Enabled = false });
            int[] input = [1, 2, 3, 4, 5, 6, 7, 8];
            int[] target = [2, 3, 4, 5, 6, 7, 8, 9];
            engine.PrepareForTraining(batchSize: 2);

            model.ZeroGrad();
            NativeCudaTransferTelemetry firstTransfersBefore =
                NativeCudaRuntime.TransferTelemetry;
            NativeCudaTransferTelemetry firstGradientTransfersBefore =
                NativeCudaRuntime.GradientCollectiveTransferTelemetry;
            float first = engine.ForwardBackward(
                input,
                target,
                2,
                4,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep: 0);
            NativeCudaTransferTelemetry firstTransfers =
                NativeCudaRuntime.TransferTelemetry - firstTransfersBefore;
            NativeCudaTransferTelemetry firstGradientTransfers =
                NativeCudaRuntime.GradientCollectiveTransferTelemetry
                    - firstGradientTransfersBefore;
            CudaTrainingGraphTelemetry compiled = engine.TrainingGraphTelemetry;
            model.ZeroGrad();
            NativeCudaAllocationTelemetry allocationsBefore =
                NativeCudaRuntime.AllocationTelemetry;
            NativeCudaTransferTelemetry transfersBefore =
                NativeCudaRuntime.TransferTelemetry;
            NativeCudaTransferTelemetry gradientTransfersBefore =
                NativeCudaRuntime.GradientCollectiveTransferTelemetry;
            float second = engine.ForwardBackward(
                input,
                target,
                2,
                4,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep: 1);
            NativeCudaAllocationTelemetry allocations =
                NativeCudaRuntime.AllocationTelemetry - allocationsBefore;
            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - transfersBefore;
            NativeCudaTransferTelemetry gradientTransfers =
                NativeCudaRuntime.GradientCollectiveTransferTelemetry
                    - gradientTransfersBefore;
            CudaTrainingGraphTelemetry replayed = engine.TrainingGraphTelemetry;

            Assert.True(float.IsFinite(first));
            Assert.True(float.IsFinite(second));
            Assert.True(
                compiled.CaptureCount == 1,
                engine.LastGraphFailure?.ToString()
                    ?? $"capture count was {compiled.CaptureCount}");
            Assert.Equal(1, compiled.ReplayCount);
            Assert.Equal(1, compiled.CachedCompiledPlanCount);
            Assert.True(compiled.GraphPinnedBytes > 0);
            Assert.True(firstGradientTransfers.HostToDeviceCopyCount > 0);
            Assert.Equal(
                firstGradientTransfers.HostToDeviceCopyCount,
                firstGradientTransfers.DeviceToHostCopyCount);
            Assert.Equal(
                4 + firstGradientTransfers.HostToDeviceCopyCount,
                firstTransfers.HostToDeviceCopyCount);
            Assert.Equal(
                2L * input.Length * sizeof(int)
                    + firstGradientTransfers.HostToDeviceBytes,
                firstTransfers.HostToDeviceBytes);
            Assert.Equal(
                3 + firstGradientTransfers.DeviceToHostCopyCount,
                firstTransfers.DeviceToHostCopyCount);
            Assert.Equal(
                2L * sizeof(float) + sizeof(double)
                    + firstGradientTransfers.DeviceToHostBytes,
                firstTransfers.DeviceToHostBytes);
            if (!CudaDispatchPolicy.Current.DisableExternalGradientReadyEvents
                && CudaNativeGateway.AbiVersion.Minor
                    >= CudaAbiVersion.ExternalGradientReadyEventMinor)
            {
                Assert.Equal(
                    compiled.CapturedReadyEventRecordCount,
                    replayed.CapturedReadyEventRecordCount);
                Assert.Equal(
                    compiled.CapturedReadyEventRecordMilliseconds,
                    replayed.CapturedReadyEventRecordMilliseconds);
            }
            else
            {
                Assert.True(
                    replayed.CapturedReadyEventRecordCount
                        > compiled.CapturedReadyEventRecordCount);
                Assert.True(
                    replayed.CapturedReadyEventRecordMilliseconds
                        >= compiled.CapturedReadyEventRecordMilliseconds);
            }
            Assert.Equal(0, allocations.AllocationCount);
            Assert.Equal(0, allocations.FreeCount);
            Assert.True(gradientTransfers.HostToDeviceCopyCount > 0);
            Assert.Equal(
                gradientTransfers.HostToDeviceCopyCount,
                gradientTransfers.DeviceToHostCopyCount);
            Assert.Equal(
                4 + gradientTransfers.HostToDeviceCopyCount,
                transfers.HostToDeviceCopyCount);
            Assert.Equal(
                2L * input.Length * sizeof(int)
                    + gradientTransfers.HostToDeviceBytes,
                transfers.HostToDeviceBytes);
            // Two device-local loss scalars and one reduced gradient-norm
            // scalar are the only replay-step readbacks.
            Assert.Equal(
                3 + gradientTransfers.DeviceToHostCopyCount,
                transfers.DeviceToHostCopyCount);
            Assert.Equal(
                2L * sizeof(float) + sizeof(double)
                    + gradientTransfers.DeviceToHostBytes,
                transfers.DeviceToHostBytes);
            Assert.All(
                model.Parameters(),
                parameter => Assert.Equal(
                    CudaGradientCoherenceKind.Reduced,
                    parameter.T.GetCudaGradientCoherenceSnapshot().Kind));
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void PureBFloat16TwoGpuGraphUsesDirectGradientArenaAndStableRng()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            var random = new CheckpointableRandom(2017);
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 4,
                dModel: 8,
                numHeads: 2,
                dHidden: 16,
                numLayers: 1,
                rng: random,
                dropout: 0.2f,
                dtype: TensorDType.BFloat16,
                tieWordEmbeddings: true);
            random.BeginRuntime();
            model.AttachTrainingRandom(random);
            model.to(TensorPrecisionMode.BFloat16);

            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0, 1];
            using var execution = new ExecutionSession(
                new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet(0, 1),
                    Precision = PrecisionPolicy.BFloat16,
                },
                [
                    CudaExecutionLaneFactory.Create(0),
                    CudaExecutionLaneFactory.Create(1),
                ]);
            using IDisposable sessionScope = execution.Enter();
            using var engine = new CudaDataParallelEngine(
                model,
                [0, 1],
                new CudaAdaptiveShardingOptions { Enabled = false });
            int[] input = [1, 2, 3, 4, 5, 6, 7, 8];
            int[] target = [2, 3, 4, 5, 6, 7, 8, 9];

            model.ZeroGrad();
            float first = engine.ForwardBackward(
                input,
                target,
                2,
                4,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep: 0);
            CudaTrainingGraphTelemetry compiled = engine.TrainingGraphTelemetry;

            model.ZeroGrad();
            NativeCudaAllocationTelemetry allocationsBefore =
                NativeCudaRuntime.AllocationTelemetry;
            float second = engine.ForwardBackward(
                input,
                target,
                2,
                4,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep: 1);
            NativeCudaAllocationTelemetry allocations =
                NativeCudaRuntime.AllocationTelemetry - allocationsBefore;

            model.ZeroGrad();
            float repeated = engine.ForwardBackward(
                input,
                target,
                2,
                4,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep: 1);

            Assert.True(float.IsFinite(first));
            Assert.True(float.IsFinite(second));
            Assert.NotEqual(first, second);
            Assert.Equal(second, repeated);
            Assert.True(
                compiled.CaptureCount == 1,
                engine.LastGraphFailure?.ToString()
                    ?? $"capture count was {compiled.CaptureCount}");
            Assert.Equal(1, compiled.CachedCompiledPlanCount);
            Assert.Equal(0, allocations.AllocationCount);
            Assert.Equal(0, allocations.FreeCount);
            Assert.Equal(
                0,
                engine.BFloat16GradientManagedLocalPackSubmissionCount);
            foreach (Parameter parameter in model.Parameters())
            {
                Assert.Null(parameter.T.GetCudaGradientArena(0));
                Assert.Null(parameter.T.GetCudaGradientArena(1));
                Assert.NotNull(parameter.T.GetCudaBFloat16GradientArena(0));
                Assert.NotNull(parameter.T.GetCudaBFloat16GradientArena(1));
                Assert.Equal(
                    CudaGradientCoherenceKind.Reduced,
                    parameter.T.GetCudaGradientCoherenceSnapshot().Kind);
            }
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void StableGraphShapeCacheEvictsToMostRecentThreePlans()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            var random = new CheckpointableRandom(2213);
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 6,
                dModel: 8,
                numHeads: 2,
                dHidden: 16,
                numLayers: 1,
                rng: random,
                dropout: 0.1f,
                dtype: TensorDType.Float32);
            random.BeginRuntime();
            model.AttachTrainingRandom(random);
            model.to(TensorPrecisionMode.Mix8_32);

            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            using var execution = new ExecutionSession(
                new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet(0),
                    Precision = PrecisionPolicy.Mix8_32,
                },
                [CudaExecutionLaneFactory.Create(0)]);
            using IDisposable sessionScope = execution.Enter();
            using var engine = new CudaDataParallelEngine(
                model,
                [0],
                new CudaAdaptiveShardingOptions { Enabled = false });
            engine.PrepareForTraining(batchSize: 1);

            for (int shape = 0; shape < 4; shape++)
            {
                int sequence = shape + 2;
                int[] input = Enumerable.Range(1, sequence).ToArray();
                int[] target = Enumerable.Range(2, sequence).ToArray();
                model.ZeroGrad();
                float loss = engine.ForwardBackward(
                    input,
                    target,
                    1,
                    sequence,
                    Tensor.DefaultCrossEntropyIgnoreIndex,
                    globalStep: shape);
                Assert.True(float.IsFinite(loss));
            }

            CudaTrainingGraphTelemetry telemetry =
                engine.TrainingGraphTelemetry;
            Assert.Equal(4, telemetry.CaptureCount);
            Assert.Equal(4, telemetry.ReplayCount);
            Assert.Equal(0, telemetry.FallbackCount);
            Assert.Equal(3, telemetry.CachedCompiledPlanCount);
            Assert.Equal(3, engine.CachedTrainingShapePlanCount);
            Assert.Equal(4, engine.TrainingShapePlanBuildCount);
            Assert.True(telemetry.GraphPinnedBytes > 0);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void GraphShapeCacheEvictsCompletedPlansWhenVramBudgetIsExceeded()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            var random = new CheckpointableRandom(2281);
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 5,
                dModel: 8,
                numHeads: 2,
                dHidden: 16,
                numLayers: 1,
                rng: random,
                dropout: 0.1f,
                dtype: TensorDType.Float32);
            random.BeginRuntime();
            model.AttachTrainingRandom(random);
            model.to(TensorPrecisionMode.Mix8_32);

            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            using var execution = new ExecutionSession(
                new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet(0),
                    Precision = PrecisionPolicy.Mix8_32,
                },
                [CudaExecutionLaneFactory.Create(0)]);
            using IDisposable sessionScope = execution.Enter();
            using var engine = new CudaDataParallelEngine(
                model,
                [0],
                new CudaAdaptiveShardingOptions
                {
                    Enabled = false,
                    // Every compiled graph pins more than one byte. Keeping
                    // the active plan while evicting older completed plans is
                    // therefore deterministic and independent of GPU model.
                    GraphCacheBudgetBytes = 1,
                });
            engine.PrepareForTraining(batchSize: 1);

            for (int shape = 0; shape < 3; shape++)
            {
                int sequence = shape + 2;
                int[] input = Enumerable.Range(1, sequence).ToArray();
                int[] target = Enumerable.Range(2, sequence).ToArray();
                model.ZeroGrad();
                float loss = engine.ForwardBackward(
                    input,
                    target,
                    1,
                    sequence,
                    Tensor.DefaultCrossEntropyIgnoreIndex,
                    globalStep: shape);
                Assert.True(float.IsFinite(loss));
            }

            Assert.Equal(1, engine.GraphCacheBudgetBytes);
            Assert.Equal(1, engine.CachedTrainingShapePlanCount);
            Assert.Equal(2, engine.TrainingShapePlanEvictionCount);
            Assert.Equal(1,
                engine.TrainingGraphTelemetry.CachedCompiledPlanCount);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void OversizedCompiledShapeSurvivesRestoreWithoutMarginalRecapture()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            var random = new CheckpointableRandom(2311);
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 2,
                dModel: 8,
                numHeads: 2,
                dHidden: 16,
                numLayers: 1,
                rng: random,
                dropout: 0.1f,
                dtype: TensorDType.Float32);
            random.BeginRuntime();
            model.AttachTrainingRandom(random);
            model.to(TensorPrecisionMode.Mix8_32);

            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0, 1];
            using var execution = new ExecutionSession(
                new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet(0, 1),
                    Precision = PrecisionPolicy.Mix8_32,
                },
                [
                    CudaExecutionLaneFactory.Create(0),
                    CudaExecutionLaneFactory.Create(1),
                ]);
            using IDisposable sessionScope = execution.Enter();
            var options = new CudaAdaptiveShardingOptions
            {
                GraphCacheBudgetBytes = 1,
                MinimumObservationsBeforeAdjustment = 1,
                RequiredConsecutiveCandidateObservations = 1,
                MinimumStepsBetweenAdjustments = 0,
                MinimumPredictedStepTimeImprovement = 0d,
                OversizedGraphMinimumPredictedImprovement = 0.15d,
            };
            using var engine = new CudaDataParallelEngine(
                model,
                [0, 1],
                options);
            const int batch = 72;
            int[] input = Enumerable.Range(0, batch * 2)
                .Select(index => index % 31 + 1)
                .ToArray();
            int[] target = Enumerable.Range(0, batch * 2)
                .Select(index => (index + 1) % 31 + 1)
                .ToArray();
            engine.PrepareForTraining(batch);

            model.ZeroGrad();
            _ = engine.ForwardBackward(
                input,
                target,
                batch,
                sequenceLength: 2,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep: 0);
            Assert.Equal(1, engine.TrainingGraphTelemetry.CaptureCount);
            Assert.True(engine.TrainingGraphTelemetry.GraphPinnedBytes > 1);

            // This 8.3% projected gain formerly changed [36,36] to [37,35]
            // immediately, evicting and rebuilding an over-budget graph.
            engine.RestoreAdaptiveShardingState(
                new CudaAdaptiveShardState(
                    CudaAdaptiveShardState.CurrentFormatVersion,
                    Devices: [0, 1],
                    LastAllocation: [36, 36],
                    ThroughputEma: [0.1d, 1d / 12d],
                    HasObservation: true)
                {
                    ObservationCount = 1,
                });
            model.ZeroGrad();
            _ = engine.ForwardBackward(
                input,
                target,
                batch,
                sequenceLength: 2,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep: 1);

            Assert.Equal([36, 36], engine.LastShardBatchSizes);
            Assert.Equal(1, engine.TrainingGraphTelemetry.CaptureCount);
            Assert.Equal(1, engine.TrainingShapePlanBuildCount);
            Assert.Equal(1, engine.CachedTrainingShapePlanCount);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }
}
