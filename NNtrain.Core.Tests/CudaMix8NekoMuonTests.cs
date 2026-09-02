using System.Diagnostics;
using NNtrain;
using NNtrain.Runtime.Execution;
using NNtrain.Training.Execution;
using Xunit;

[Collection(TensorSimdCollection.Name)]
public sealed class CudaMix8NekoMuonTests
{
    [Fact]
    public void OrdinaryMuonReportsExactMix8QuantizationDiagnostics()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            Bfp8QuantizationDescriptor descriptor =
                Bfp8QuantizationDescriptor.Block(32);
            Parameter parameter = CreateParameter(
                Values(32 * 33, 17, 0.18f),
                [32, 33],
                "muon.diagnostics",
                descriptor);
            var optimizer = new NekoMuon(
                [parameter],
                FixedFiveOptions() with
                {
                    LearningRate = 0.001f,
                    WeightDecay = 0.01f,
                });
            optimizer.SetOrdinaryMuonPolicy();
            try
            {
                optimizer.prepare();
                float[] masterBefore = Read(
                    parameter.T.EnsureCudaMasterFloat32Buffer(0));
                Bfp8EncodedStorage encodedBefore = Read(
                    parameter.T.EnsureCudaBfp8Buffer(0));
                parameter.T.SetCudaGradient(
                    Values(parameter.T.Numel, 41, 0.035f), 0);

                optimizer.step();

                Assert.True(optimizer.TryGetMix8QuantizationDiagnostics(
                    out Mix8QuantizationDiagnostics diagnostics));
                float[] masterAfter = Read(
                    parameter.T.EnsureCudaMasterFloat32Buffer(0));
                Bfp8EncodedStorage encodedAfter = Read(
                    parameter.T.EnsureCudaBfp8Buffer(0));
                double expectedUpdateSum = 0d;
                double expectedResidualSum = 0d;
                ulong expectedChanged = 0;
                int blockSize = descriptor.GetEffectiveBlockSize(
                    parameter.T.Numel);
                for (int index = 0; index < parameter.T.Numel; index++)
                {
                    float oldStep = encodedBefore.Scales.Span[
                        index / blockSize];
                    float newStep = encodedAfter.Scales.Span[
                        index / blockSize];
                    double update =
                        (masterAfter[index] - masterBefore[index]) / oldStep;
                    double residual = (masterAfter[index]
                        - encodedAfter.Payload.Span[index] * newStep)
                        / newStep;
                    expectedUpdateSum += update * update;
                    expectedResidualSum += residual * residual;
                    if (encodedBefore.Payload.Span[index]
                        != encodedAfter.Payload.Span[index])
                    {
                        expectedChanged++;
                    }
                }

                Assert.Equal((ulong)parameter.T.Numel,
                    diagnostics.ElementCount);
                Assert.Equal(expectedChanged,
                    diagnostics.ChangedWeightCount);
                AssertRelativeClose(
                    expectedUpdateSum,
                    diagnostics.UpdateStepRatioSquaredSum,
                    2e-4);
                AssertRelativeClose(
                    expectedResidualSum,
                    diagnostics.ResidualStepRatioSquaredSum,
                    2e-4);
                Assert.InRange(
                    diagnostics.ResidualRmsPerQuantStep,
                    0d,
                    0.5001d);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void OrdinaryMuonMix8DiagnosticsCountPrimaryReplicaOnce()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithCuda([0, 1], () =>
        {
            Parameter parameter = CreateParameter(
                Values(32 * 32, 23, 0.15f),
                [32, 32],
                "muon.replica.diagnostics",
                Bfp8QuantizationDescriptor.Block(32),
                [0, 1]);
            var optimizer = new NekoMuon([parameter], FixedFiveOptions());
            optimizer.SetOrdinaryMuonPolicy();
            try
            {
                optimizer.prepare();
                SetSynchronizedGradient(
                    parameter,
                    Values(parameter.T.Numel, 51, 0.03f),
                    [0, 1]);

                optimizer.step();

                Assert.True(optimizer.TryGetMix8QuantizationDiagnostics(
                    out Mix8QuantizationDiagnostics diagnostics));
                Assert.Equal((ulong)parameter.T.Numel,
                    diagnostics.ElementCount);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void OrdinaryMuonBlock32KeepsFp32StateAndReplicasIdentical()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithCuda([0, 1], () =>
        {
            using IDisposable dispatch = CudaDispatchPolicy.Push(
                CudaDispatchPolicy.Defaults with
                {
                    EnableBlockBfp8OptimizerState = true,
                });
            Parameter parameter = CreateParameter(
                Values(128 * 128, 11, 0.12f), [128, 128], "muon.weight",
                Bfp8QuantizationDescriptor.Block(32), [0, 1]);
            var optimizer = new NekoMuon([parameter], FixedFiveOptions());
            optimizer.SetOrdinaryMuonPolicy();
            try
            {
                optimizer.prepare();
                for (int step = 0; step < 3; step++)
                {
                    SetSynchronizedGradient(parameter,
                        Values(parameter.T.Numel, 23 + step, 0.03f), [0, 1]);
                    optimizer.step();
                    optimizer.zero_grad();
                }
                AssertClose(
                    Read(parameter.T.EnsureCudaMasterFloat32Buffer(0)),
                    Read(parameter.T.EnsureCudaMasterFloat32Buffer(1)),
                    5e-5f);
                Assert.Throws<InvalidOperationException>(
                    () => optimizer.GetCudaBfp8Moments(0, 0));
                var state0 = optimizer.GetCudaMix8Moments(0, 0);
                var state1 = optimizer.GetCudaMix8Moments(0, 1);
                AssertClose(Read(state0.Fast), Read(state1.Fast), 1e-6f);
                AssertClose(Read(state0.Slow), Read(state1.Slow), 1e-6f);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void OrdinaryMuonMix8AccumulatesSubQuantizationUpdatesInMaster()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            using IDisposable dispatch = CudaDispatchPolicy.Push(
                CudaDispatchPolicy.Defaults with
                {
                    EnableBlockBfp8OptimizerState = true,
                });
            Bfp8QuantizationDescriptor descriptor =
                Bfp8QuantizationDescriptor.Block(32);
            float[] initial = Values(32 * 32, 71, 0.16f);
            float[] gradient = Values(32 * 32, 113, 0.025f);
            Parameter parameter = CreateParameter(
                initial, [32, 32], "muon.sub_quantization", descriptor);
            var optimizer = new NekoMuon(
                [parameter],
                FixedFiveOptions() with
                {
                    LearningRate = 1e-6f,
                    WeightDecay = 0f,
                });
            optimizer.SetOrdinaryMuonPolicy();
            try
            {
                optimizer.prepare();
                float[] masterBefore = Read(
                    parameter.T.EnsureCudaMasterFloat32Buffer(0));
                float previousRms = 0f;
                for (int step = 0; step < 8; step++)
                {
                    parameter.T.SetCudaGradient(gradient, 0);
                    optimizer.step();
                    optimizer.zero_grad();
                    float[] master = Read(
                        parameter.T.EnsureCudaMasterFloat32Buffer(0));
                    float rms = MathF.Sqrt(master.Zip(
                            masterBefore,
                            (current, before) =>
                                (current - before) * (current - before))
                        .Average());
                    Assert.True(
                        rms > previousRms,
                        $"FP32 master update did not accumulate at step " +
                        $"{step + 1}: {previousRms:G9} -> {rms:G9}.");
                    previousRms = rms;
                }

                Bfp8EncodedStorage published = Read(
                    parameter.T.EnsureCudaBfp8Buffer(0));
                Assert.True(
                    previousRms < published.Scales.ToArray().Min() * 0.5f);
                AssertEncoded(
                    Bfp8QuantizationCodec.Default.Encode(
                        Read(parameter.T.EnsureCudaMasterFloat32Buffer(0)),
                        descriptor),
                    published);
                Assert.Throws<InvalidOperationException>(
                    () => optimizer.GetCudaBfp8Moments(0, 0));
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void FixedFiveEqualShapesUseOneBatchedNs5DispatchAndMatchScalar()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            const int parameterCount = 8;
            const int rows = 48;
            const int columns = 64;
            float[] initial = Values(rows * columns, 7, 0.18f);
            float[] gradient = Values(rows * columns, 29, 0.035f);

            (float[][] Masters,
                NekoMuonFixedNs5TelemetrySnapshot Telemetry,
                double Milliseconds) Run(bool disableBatching)
            {
                Parameter[] parameters = Enumerable.Range(
                        0, parameterCount)
                    .Select(index => CreateParameter(
                        initial.ToArray(),
                        [rows, columns],
                        $"hidden.{index}",
                        Bfp8QuantizationDescriptor.Block(128)))
                    .ToArray();
                CudaDispatchPolicy dispatch = CudaDispatchPolicy.Defaults with
                {
                    DisableBatchedNekoMuon = disableBatching,
                    NekoMuonBatchSize = 8,
                };
                var optimizer = new NekoMuon(
                    parameters,
                    FixedFiveOptions(),
                    dispatch);
                try
                {
                    optimizer.prepare();
                    NekoMuonFixedNs5TelemetrySnapshot before =
                        NekoMuonFixedNs5Telemetry.Snapshot;
                    double elapsed = 0d;
                    for (int iteration = 0; iteration < 6; iteration++)
                    {
                        foreach (Parameter parameter in parameters)
                            parameter.T.SetCudaGradient(gradient, 0);
                        var timer = Stopwatch.StartNew();
                        optimizer.step();
                        timer.Stop();
                        if (iteration > 0)
                            elapsed += timer.Elapsed.TotalMilliseconds;
                        optimizer.zero_grad();
                    }
                    NekoMuonFixedNs5TelemetrySnapshot telemetry =
                        NekoMuonFixedNs5Telemetry.Snapshot - before;
                    return (
                        parameters.Select(parameter => Read(
                            parameter.T.EnsureCudaMasterFloat32Buffer(0)))
                            .ToArray(),
                        telemetry,
                        elapsed / 5d);
                }
                finally
                {
                    optimizer.DisposeCudaResources();
                    foreach (Parameter parameter in parameters)
                        parameter.T.InvalidateCudaBuffers();
                }
            }

            _ = CudaDispatchPolicy.Startup;
            CudaDispatchEnvironmentTelemetrySnapshot environmentBefore =
                CudaDispatchEnvironmentTelemetry.Snapshot;
            var scalar = Run(disableBatching: true);
            var grouped = Run(disableBatching: false);
            CudaDispatchEnvironmentTelemetrySnapshot environmentDelta =
                CudaDispatchEnvironmentTelemetry.Snapshot
                - environmentBefore;

            for (int index = 0; index < parameterCount; index++)
            {
                AssertClose(
                    scalar.Masters[index],
                    grouped.Masters[index],
                    8e-4f);
            }
            Assert.Equal(6 * parameterCount,
                scalar.Telemetry.ScalarDispatchCount);
            Assert.Equal(0, scalar.Telemetry.BatchedDispatchCount);
            Assert.Equal(6 * parameterCount * 15,
                scalar.Telemetry.GemmLaunchCount);
            Assert.Equal(0, grouped.Telemetry.ScalarDispatchCount);
            Assert.Equal(6, grouped.Telemetry.BatchedDispatchCount);
            Assert.Equal(6 * 15,
                grouped.Telemetry.GemmLaunchCount);
            Assert.Equal(
                scalar.Telemetry.LogicalMatrixCount,
                grouped.Telemetry.LogicalMatrixCount);
            Assert.Equal(0, environmentDelta.EnvironmentReads);
            Console.WriteLine(
                $"fixed NS5 scalar={scalar.Milliseconds:F3} ms, " +
                $"grouped={grouped.Milliseconds:F3} ms, " +
                $"GEMM launches {scalar.Telemetry.GemmLaunchCount}->" +
                $"{grouped.Telemetry.GemmLaunchCount}");
        });
    }

    [Fact]
    public void DefaultCudaScratchCapacityHonorsThirtyTwoMebibyteBudget()
    {
        var parameter = new Parameter(
            new float[512 * 2048],
            [512, 2048],
            "production-shape",
            WeightDecayPolicy.Apply);
        var optimizer = new NekoMuon(
            [parameter],
            FixedFiveOptions(),
            CudaDispatchPolicy.Defaults);

        Assert.Equal(32L * 1024L * 1024L,
            optimizer.CudaScratchBudgetBytes);
        Assert.Equal(3, optimizer.CudaBatchCapacity);
        Assert.True(
            optimizer.ConfiguredCudaScratchBytesPerDevice
                <= optimizer.CudaScratchBudgetBytes);
    }

    [Fact]
    public void FirstGuardedStepPrewarmsAllMix8ResidencyOutsideGuard()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithCuda([0, 1], () =>
        {
            Parameter parameter = CreateParameter(
                Values(257, 7, 0.5f),
                [1, 257],
                "hidden",
                Bfp8QuantizationDescriptor.Block(96),
                [0, 1]);
            var optimizer = new NekoMuon(
                [parameter],
                FixedFiveOptions());
            using var execution = new ExecutionSession(new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda,
                CudaDevices = new DeviceSet(0, 1),
                Precision = PrecisionPolicy.Mix8_32,
            });
            using IDisposable executionScope = execution.Enter();
            using var session = new TrainingSession(execution);
            var executor = new TrainingStepExecutor(session);
            var operations = new OptimizerTrainingOperations(optimizer);
            try
            {
                SetSynchronizedGradient(
                    parameter,
                    Values(257, 29, 0.06f),
                    [0, 1]);

                executor.Execute(operations);

                DeviceTransferSnapshot snapshot = Assert.NotNull(
                    operations.GuardedSnapshot);
                Assert.Equal(0, snapshot.HostToDeviceCopyCount);
                Assert.Equal(0, snapshot.HostToDeviceBytes);
                Assert.Equal(2 * sizeof(int), snapshot.DeviceToHostBytes);
                Assert.True(parameter.T.HasCudaMasterFloat32Buffer(0));
                Assert.True(parameter.T.HasCudaMasterFloat32Buffer(1));
                _ = optimizer.GetCudaMix8Moments(0, 0);
                _ = optimizer.GetCudaMix8Moments(0, 1);

                NativeCudaTransferTelemetry before =
                    NativeCudaRuntime.TransferTelemetry;
                optimizer.prepare();
                NativeCudaTransferTelemetry repeated =
                    NativeCudaRuntime.TransferTelemetry - before;
                Assert.Equal(0, repeated.HostToDeviceBytes);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void PureBfp8PrewarmDoesNotCreateFloat32Master()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCudaPolicy([0], PrecisionPolicy.Bfp8, () =>
        {
            var parameter = new Parameter(
                Values(129, 5, 0.4f),
                [1, 129],
                "hidden",
                WeightDecayPolicy.Apply);
            parameter.T.ConvertStorageInPlace(
                TensorDType.Bfp8,
                Bfp8QuantizationDescriptor.TensorWide,
                preserveFloat32Master: false);
            _ = parameter.T.EnsureCudaBfp8Buffer(0);
            parameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));
            var optimizer = new NekoMuon(
                [parameter],
                FixedFiveOptions());
            try
            {
                optimizer.prepare();

                Assert.False(parameter.T.HasCudaMasterFloat32Buffer(0));
                _ = optimizer.GetCudaBfp8Moments(0, 0);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void NekoMuonOptInKeepsBlockBfp8MomentsResidentAndFinite()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            using IDisposable dispatch = CudaDispatchPolicy.Push(
                CudaDispatchPolicy.Defaults with
                {
                    EnableBlockBfp8OptimizerState = true,
                });
            Parameter parameter = CreateParameter(
                Values(16 * 32, 17, 0.4f),
                [16, 32],
                "hidden",
                Bfp8QuantizationDescriptor.Mix8_32);
            var optimizer = new NekoMuon(
                [parameter],
                FixedFiveOptions());
            try
            {
                parameter.T.SetCudaGradient(
                    Values(16 * 32, 53, 0.05f),
                    0);
                optimizer.step();

                var moments = optimizer.GetCudaBfp8Moments(0, 0);
                Assert.Equal(
                    Bfp8QuantizationDescriptor.Mix8_32,
                    moments.Fast.Descriptor);
                Assert.Equal(4, moments.Fast.Scales.Length);
                Assert.All(Read(moments.Fast).Scales.ToArray(),
                    scale => Assert.True(float.IsFinite(scale) && scale > 0f));
                Assert.All(Read(moments.Slow).Scales.ToArray(),
                    scale => Assert.True(float.IsFinite(scale) && scale > 0f));
                Assert.All(Read(parameter.T.EnsureCudaMasterFloat32Buffer(0)),
                    value => Assert.True(float.IsFinite(value)));
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Theory]
    [InlineData(1, 128)]
    [InlineData(257, 128)]
    [InlineData(515, 96)]
    public void OneGpuKeepsBlockParameterAndFp32MasterGradientAndMoments(
        int length,
        int blockSize)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            float[] values = Values(length, 3, 0.55f);
            float[] gradient = Values(length, 19, 0.08f);
            Bfp8QuantizationDescriptor descriptor =
                Bfp8QuantizationDescriptor.Block(blockSize);
            Parameter parameter = CreateParameter(
                values, [1, length], "hidden", descriptor);
            NekoMuonOptions options = Options();
            var optimizer = new NekoMuon([parameter], options);
            try
            {
                parameter.T.SetCudaGradient(gradient, 0);
                optimizer.step();

                Assert.Equal(descriptor, parameter.T.Bfp8Quantization);
                Assert.True(parameter.T.HasCudaMasterFloat32Buffer(0));
                Assert.False(parameter.T.HasAuthoritativeCudaBfp8Gradient);
                Assert.Throws<InvalidOperationException>(
                    () => optimizer.GetCudaBfp8Moments(0, 0));
                var moments = optimizer.GetCudaMix8Moments(0, 0);
                float[] expectedFast = gradient.Select(value =>
                    (1f - options.BetaFast) * value).ToArray();
                float[] expectedSlow = gradient.Select(value =>
                    (1f - options.BetaSlow) * value).ToArray();
                AssertClose(expectedFast, Read(moments.Fast), 2e-6f);
                AssertClose(expectedSlow, Read(moments.Slow), 2e-6f);

                float[] master = Read(
                    parameter.T.EnsureCudaMasterFloat32Buffer(0));
                Assert.All(master,
                    value => Assert.True(float.IsFinite(value)));
                Assert.NotEqual(values, master);
                AssertEncoded(
                    Bfp8QuantizationCodec.Default.Encode(master, descriptor),
                    Read(parameter.T.EnsureCudaBfp8Buffer(0)));
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void WarmTwoGpuStepPublishesIdenticalReplicasWithoutPayloadTransfer()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithCuda([0, 1], () =>
        {
            Bfp8QuantizationDescriptor descriptor =
                Bfp8QuantizationDescriptor.Block(96);
            Parameter parameter = CreateParameter(
                Values(257, 7, 0.5f),
                [1, 257],
                "hidden",
                descriptor,
                [0, 1]);
            var optimizer = new NekoMuon([parameter], FixedFiveOptions());
            try
            {
                SetSynchronizedGradient(
                    parameter, Values(257, 29, 0.07f), [0, 1]);
                optimizer.step();
                optimizer.zero_grad();
                SetSynchronizedGradient(
                    parameter, Values(257, 43, 0.05f), [0, 1]);
                NativeCudaTransferTelemetry before =
                    NativeCudaRuntime.TransferTelemetry;

                optimizer.step();

                NativeCudaTransferTelemetry transfer =
                    NativeCudaRuntime.TransferTelemetry - before;
                Assert.Equal(0, transfer.HostToDeviceBytes);
                // Fixed NS5 keeps statistics and confidence device-resident;
                // only one aggregate finite-status scalar returns per GPU.
                Assert.Equal(2 * sizeof(int), transfer.DeviceToHostBytes);
                AssertEncoded(
                    Read(parameter.T.EnsureCudaBfp8Buffer(0)),
                    Read(parameter.T.EnsureCudaBfp8Buffer(1)));
                Assert.Equal(
                    Read(parameter.T.EnsureCudaMasterFloat32Buffer(0)),
                    Read(parameter.T.EnsureCudaMasterFloat32Buffer(1)));
                var primary = optimizer.GetCudaMix8Moments(0, 0);
                var secondary = optimizer.GetCudaMix8Moments(0, 1);
                Assert.Equal(Read(primary.Fast), Read(secondary.Fast));
                Assert.Equal(Read(primary.Slow), Read(secondary.Slow));
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void BinaryCheckpointResumePreservesFp32MasterStateAndConfidence()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            Bfp8QuantizationDescriptor descriptor =
                Bfp8QuantizationDescriptor.Block(128);
            Parameter source = CreateParameter(
                Values(257, 5, 0.52f), [1, 257], "hidden", descriptor);
            NekoMuonOptions options = FixedFiveOptions();
            var uninterrupted = new NekoMuon([source], options);
            NekoMuon? resumed = null;
            Parameter? restored = null;
            try
            {
                source.T.SetCudaGradient(Values(257, 17, 0.07f), 0);
                uninterrupted.step();
                float[] savedMaster = Read(
                    source.T.EnsureCudaMasterFloat32Buffer(0));
                using var checkpoint = new MemoryStream();
                OptimizerStateStream.SaveStateBinary(
                    uninterrupted,
                    checkpoint);

                restored = CreateParameter(
                    savedMaster,
                    [1, 257],
                    "hidden",
                    descriptor);
                resumed = new NekoMuon([restored], options);
                checkpoint.Position = 0;
                OptimizerStateStream.LoadStateBinary(resumed, checkpoint);

                float[] gradient = Values(257, 37, 0.05f);
                uninterrupted.zero_grad();
                resumed.zero_grad();
                source.T.SetCudaGradient(gradient, 0);
                restored.T.SetCudaGradient(gradient, 0);
                uninterrupted.step();
                resumed.step();

                Assert.Equal(
                    Read(source.T.EnsureCudaMasterFloat32Buffer(0)),
                    Read(restored.T.EnsureCudaMasterFloat32Buffer(0)));
                AssertEncoded(
                    Read(source.T.EnsureCudaBfp8Buffer(0)),
                    Read(restored.T.EnsureCudaBfp8Buffer(0)));
                var expected = uninterrupted.GetCudaMix8Moments(0, 0);
                var actual = resumed.GetCudaMix8Moments(0, 0);
                Assert.Equal(Read(expected.Fast), Read(actual.Fast));
                Assert.Equal(Read(expected.Slow), Read(actual.Slow));
                Assert.Equal(
                    uninterrupted.CaptureState().ParameterStates[0].Confidence,
                    resumed.CaptureState().ParameterStates[0].Confidence);
            }
            finally
            {
                resumed?.DisposeCudaResources();
                uninterrupted.DisposeCudaResources();
                restored?.T.InvalidateCudaBuffers();
                source.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void FixedFiveTransferIsOneScalarRegardlessOfParameterCount()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            Parameter[] parameters = Enumerable.Range(0, 3)
                .Select(index => CreateParameter(
                    Values(257 + index * 32, 7 + index * 5, 0.4f),
                    [1, 257 + index * 32],
                    $"hidden.{index}",
                    Bfp8QuantizationDescriptor.Block(96)))
                .ToArray();
            var optimizer = new NekoMuon(parameters, FixedFiveOptions());
            try
            {
                for (int index = 0; index < parameters.Length; index++)
                {
                    parameters[index].T.SetCudaGradient(
                        Values(parameters[index].T.Numel, 19 + index, 0.06f),
                        0);
                }
                optimizer.step();
                optimizer.zero_grad();
                for (int index = 0; index < parameters.Length; index++)
                {
                    parameters[index].T.SetCudaGradient(
                        Values(parameters[index].T.Numel, 31 + index, 0.05f),
                        0);
                }
                NativeCudaTransferTelemetry before =
                    NativeCudaRuntime.TransferTelemetry;

                optimizer.step();

                NativeCudaTransferTelemetry transfer =
                    NativeCudaRuntime.TransferTelemetry - before;
                Assert.Equal(0, transfer.HostToDeviceBytes);
                Assert.Equal(sizeof(int), transfer.DeviceToHostBytes);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                foreach (Parameter parameter in parameters)
                    parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void FixedFiveOptimizerCommitPassesTrainingStepTransferGuard()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            Parameter[] parameters = Enumerable.Range(0, 3)
                .Select(index => CreateParameter(
                    Values(257 + index * 16, 17 + index, 0.35f),
                    [1, 257 + index * 16],
                    $"hidden.{index}",
                    Bfp8QuantizationDescriptor.Block(96)))
                .ToArray();
            var optimizer = new NekoMuon(parameters, FixedFiveOptions());
            using var execution = new ExecutionSession(new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda,
                CudaDevices = new DeviceSet(0),
                Precision = PrecisionPolicy.Mix8_32,
            });
            using IDisposable executionScope = execution.Enter();
            using var session = new TrainingSession(execution);
            var executor = new TrainingStepExecutor(session);
            try
            {
                optimizer.prepare();
                for (int index = 0; index < parameters.Length; index++)
                {
                    parameters[index].T.SetCudaGradient(
                        Values(parameters[index].T.Numel, 37 + index, 0.05f),
                        0);
                }
                var operations = new TrainingStepOperations(
                    () => { },
                    () => { },
                    () => { },
                    () => { },
                    () => { },
                    () => { },
                    () => { },
                    optimizer.step,
                    () => { });

                TrainingStepState committed = executor.Execute(operations);

                Assert.Equal(
                    TrainingStepPhase.MetricsCommitted,
                    committed.Phase);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                foreach (Parameter parameter in parameters)
                    parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void BlockDescriptorDispatchesOutsidePrecisionScope()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCudaWithoutPrecision([0], () =>
        {
            Parameter parameter = CreateParameter(
                Values(257, 11, 0.4f),
                [1, 257],
                "hidden",
                Bfp8QuantizationDescriptor.Block(64));
            var optimizer = new NekoMuon([parameter], FixedFiveOptions());
            try
            {
                Assert.Null(TensorExecutionContext.ActivePrecisionPolicy);
                parameter.T.SetCudaGradient(Values(257, 23, 0.05f), 0);
                optimizer.step();
                _ = optimizer.GetCudaMix8Moments(0, 0);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void FiniteGradientWhoseStatisticsOverflowIsRejected()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            Parameter parameter = CreateParameter(
                Values(257, 13, 0.2f),
                [1, 257],
                "hidden",
                Bfp8QuantizationDescriptor.Block(96));
            var optimizer = new NekoMuon([parameter], FixedFiveOptions());
            try
            {
                parameter.T.SetCudaGradient(
                    Enumerable.Repeat(1e30f, 257).ToArray(),
                    0);
                InvalidOperationException exception = Assert.Throws<
                    InvalidOperationException>(optimizer.step);
                Assert.Contains("Non-finite", exception.Message);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void MixedOptimizerAndTensorResourcesDisposeIdempotently()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            Parameter parameter = CreateParameter(
                Values(129, 11, 0.4f),
                [1, 129],
                "hidden",
                Bfp8QuantizationDescriptor.Block(64));
            var optimizer = new NekoMuon([parameter], Options());
            parameter.T.SetCudaGradient(Values(129, 23, 0.04f), 0);
            optimizer.step();
            NativeCudaAllocationTelemetry before =
                NativeCudaRuntime.AllocationTelemetry;

            optimizer.DisposeCudaResources();
            parameter.T.InvalidateCudaBuffers();
            optimizer.DisposeCudaResources();
            parameter.T.InvalidateCudaBuffers();

            NativeCudaAllocationTelemetry released =
                NativeCudaRuntime.AllocationTelemetry - before;
            Assert.True(released.FreeCount >= 8);
            Assert.True(released.FreeBytes > 0);
        });
    }

    private static NekoMuonOptions Options() => new()
    {
        LearningRate = 0.002f,
        BetaFast = 0.8f,
        BetaSlow = 0.95f,
        Rho = 0.7f,
        Epsilon = 1e-6f,
        MaxNewtonSchulzSteps = 5,
        NewtonSchulzInterval = 100,
        WeightDecay = 0.01f,
    };

    private static NekoMuonOptions FixedFiveOptions() => Options() with
    {
        NewtonSchulzInterval = 1,
        NewtonSchulzDepthMode = NekoMuonNewtonSchulzDepthMode.Fixed,
        NewtonSchulzDepth = 5f,
    };

    private static Parameter CreateParameter(
        float[] values,
        int[] shape,
        string name,
        Bfp8QuantizationDescriptor descriptor,
        int[]? devices = null)
    {
        var parameter = new Parameter(
            values,
            shape,
            name,
            WeightDecayPolicy.Apply);
        parameter.T.ConvertStorageInPlace(
            TensorDType.Bfp8,
            descriptor,
            preserveFloat32Master: true);
        foreach (int device in devices ?? [0])
            _ = parameter.T.EnsureCudaBfp8Buffer(device);
        parameter.T.to(new TorchDevice(
            TensorDevice.Cuda,
            devices?[0] ?? 0));
        return parameter;
    }

    private static void SetSynchronizedGradient(
        Parameter parameter,
        float[] values,
        int[] devices)
    {
        foreach (int device in devices)
            parameter.T.SetCudaGradient(values, device);
        parameter.T.MarkCudaGradientsSynchronized(devices);
    }

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index => index == length - 1
                ? scale * 2.75f
                : MathF.Sin((index + offset) * 0.173f) * scale)
            .ToArray();

    private static float[] Read(NativeCudaBuffer<float> buffer)
    {
        var result = new float[buffer.Length];
        buffer.CopyToCPU(result);
        return result;
    }

    private static Bfp8EncodedStorage Read(CudaBfp8BufferView view)
    {
        var payload = new sbyte[view.Payload.Length];
        var scales = new float[view.Scales.Length];
        view.Payload.CopyToCPU(payload);
        view.Scales.CopyToCPU(scales);
        return new Bfp8EncodedStorage(payload, scales, view.Descriptor);
    }

    private static void AssertEncoded(
        Bfp8EncodedStorage expected,
        Bfp8EncodedStorage actual)
    {
        Assert.Equal(expected.Descriptor, actual.Descriptor);
        Assert.Equal(expected.Payload.ToArray(), actual.Payload.ToArray());
        Assert.Equal(expected.Scales.ToArray(), actual.Scales.ToArray());
    }

    private static void AssertClose(
        IReadOnlyList<float> expected,
        IReadOnlyList<float> actual,
        float tolerance)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int index = 0; index < expected.Count; index++)
        {
            Assert.InRange(
                MathF.Abs(expected[index] - actual[index]),
                0f,
                tolerance);
        }
    }

    private static void AssertRelativeClose(
        double expected,
        double actual,
        double relativeTolerance)
    {
        double scale = Math.Max(1d, Math.Abs(expected));
        Assert.InRange(
            Math.Abs(actual - expected),
            0d,
            relativeTolerance * scale);
    }

    private static void WithCuda(int[] devices, Action action)
        => WithCudaPolicy(devices, PrecisionPolicy.Mix8_32, action);

    private static void WithCudaPolicy(
        int[] devices,
        PrecisionPolicy policy,
        Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = devices;
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable precision =
                TensorExecutionContext.PushPrecisionPolicy(
                    policy);
            action();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private static void WithCudaWithoutPrecision(
        int[] devices,
        Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = devices;
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            action();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private sealed class OptimizerTrainingOperations(IOptimizer optimizer)
        : ITrainingStepOperations
    {
        internal DeviceTransferSnapshot? GuardedSnapshot { get; private set; }

        public TrainingGradientExecutionMode GradientExecutionMode
            => TrainingGradientExecutionMode.Separate;

        public void Prepare() => optimizer.prepare();

        public void AcquireBatch()
        {
        }

        public void ClearGradients()
        {
        }

        public void Forward()
        {
        }

        public void Backward()
        {
        }

        public void ReduceGradients()
        {
        }

        public void ForwardBackwardReduced()
            => throw new InvalidOperationException();

        public void ClipGradients()
        {
        }

        public void ApplySchedule()
        {
        }

        public void CommitOptimizer() => optimizer.step();

        public void CommitMetrics()
            => GuardedSnapshot = DeviceTransferGuard.CurrentSnapshot;
    }
}
