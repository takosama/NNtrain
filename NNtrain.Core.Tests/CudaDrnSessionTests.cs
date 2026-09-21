using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaDrnSessionTests
{
    [Theory]
    [InlineData(1, false, TensorPrecisionMode.BFloat16)]
    [InlineData(2, false, TensorPrecisionMode.BFloat16)]
    [InlineData(2, true, TensorPrecisionMode.BFloat16)]
    [InlineData(1, false, TensorPrecisionMode.Mix8_32)]
    [InlineData(2, false, TensorPrecisionMode.Mix8_32)]
    [InlineData(2, true, TensorPrecisionMode.Mix8_32)]
    public void LowPrecisionDrnTrainsAndRetiresSession(int deviceCount,
        bool productionShape, TensorPrecisionMode precisionMode)
    {
        Assert.SkipWhen(productionShape && Environment.GetEnvironmentVariable("NNTRAIN_DRN_PRODUCTION_TEST") != "1",
            "Set NNTRAIN_DRN_PRODUCTION_TEST=1 to test the 56M-parameter production shape.");
        Assert.SkipWhen(Tensor.CudaDeviceCount < deviceCount, "Required CUDA devices are unavailable.");
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        int[] devices = Enumerable.Range(0, deviceCount).ToArray();
        var failures = new List<Exception>();
        ExecutionSession? session = null;
        IDisposable? scope = null;
        CudaDataParallelEngine? engine = null;
        NekoMuon? matrixOptimizer = null;
        AdamW? auxiliaryOptimizer = null;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = devices;
            session = new ExecutionSession(new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda,
                CudaDevices = new DeviceSet(devices),
                Precision = PrecisionPolicy.Parse(TensorPrecisionModeNames.Format(precisionMode)),
            }, devices.Select(index => CudaExecutionLaneFactory.Create(index)));
            scope = session.Enter();
            int vocabulary = productionShape ? 4096 : 64;
            int sequence = productionShape ? 1024 : 32;
            int batch = productionShape ? 16 : 4;
            using IDisposable policy = CudaDispatchPolicy.Push(CudaDispatchPolicy.Defaults with
            {
                DrnRetainedHistoryBudgetBytes = (long)Math.Max(1, batch / deviceCount) * sequence * 16 * 16 * sizeof(float),
            });
            var random = new CheckpointableRandom(1234);
            var model = new ForgetMemoryDRNGpt(vocabulary, sequence,
                productionShape ? 512 : 32, productionShape ? 1536 : 64,
                productionShape ? 32 : 2,
                keyWidth: 16, valueWidth: 16, dropout: 0.1f,
                random: random, dtype: precisionMode == TensorPrecisionMode.BFloat16
                    ? TensorDType.BFloat16 : TensorDType.Float32);
            random.BeginRuntime();
            model.AttachTrainingRandom(random);
            model.to(precisionMode);
            model.to(TensorDevice.Cuda);
            engine = new CudaDataParallelEngine(model, devices);
            matrixOptimizer = new NekoMuon(model.HiddenWeightParameters,
                new NekoMuonOptions { LearningRate = 0.003f, NewtonSchulzInterval = 5 });
            auxiliaryOptimizer = new AdamW(model.AuxiliaryParameters,
                new AdamWOptions { LearningRate = 0.001f, Beta2 = 0.95f,
                    UseBFloat16FirstMoment = precisionMode == TensorPrecisionMode.BFloat16,
                    UseBFloat16SecondMoment = precisionMode == TensorPrecisionMode.BFloat16 });
            engine.PrepareForTraining(batch);
            ((IOptimizer)matrixOptimizer).prepare();
            ((IOptimizer)auxiliaryOptimizer).prepare();
            int[] input = Enumerable.Range(0, batch * sequence).Select(i => i % (vocabulary - 3) + 1).ToArray();
            int[] targets = input.Select(i => (i + 1) % vocabulary).ToArray();
            for (int step = 0; step < 7; step++)
            {
                using IDisposable guard = DeviceTransferGuard.EnterTrainingStep(deviceCount);
                model.ZeroGrad();
                // An epoch can end with one sequence after two-GPU updates.
                int currentBatch = step == 6 ? 1 : batch;
                float loss = engine.ForwardBackward(
                    input[..(currentBatch * sequence)], targets[..(currentBatch * sequence)],
                    currentBatch, sequence, -1, step);
                Assert.True(float.IsFinite(loss));
                matrixOptimizer.Step();
                auxiliaryOptimizer.Step();
            }
            Assert.Null(engine.LastGraphFailure);
            Assert.True(engine.TrainingGraphTelemetry.CaptureCount > 0);
            Assert.True(engine.TrainingGraphTelemetry.ReplayCount > 0);
            Assert.Equal(0, engine.TrainingGraphTelemetry.FallbackCount);
        }
        catch (Exception exception)
        {
            if (engine?.LastGraphFailure is Exception graphFailure)
                failures.Add(graphFailure);
            failures.Add(exception);
        }
        finally
        {
            try { matrixOptimizer?.DisposeCudaResources(); }
            catch (Exception exception) { failures.Add(exception); }
            try { auxiliaryOptimizer?.DisposeCudaResources(); }
            catch (Exception exception) { failures.Add(exception); }
            foreach (IDisposable? resource in new IDisposable?[] { engine, scope, session })
            {
                try { resource?.Dispose(); }
                catch (Exception exception) { failures.Add(exception); }
            }
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
        if (failures.Count > 0)
            throw new AggregateException(failures);
    }
}
