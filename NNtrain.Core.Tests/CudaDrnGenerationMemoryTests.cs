using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaDrnGenerationMemoryTests
{
    private readonly ITestOutputHelper _output;
    public CudaDrnGenerationMemoryTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void ChunkedPrefillReleasesEarlierChunkActivations()
    {
        Assert.SkipWhen(!Tensor.IsCudaAvailable(), "CUDA unavailable.");
        TensorDevice previous = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            using var session = new ExecutionSession(new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda, CudaDevices = new DeviceSet([0]),
                Precision = PrecisionPolicy.Mix8_32,
            }, [CudaExecutionLaneFactory.Create(0)]);
            using var scope = session.Enter();
            var lane = (CudaExecutionLane)session.GetRequiredLane(ExecutionDeviceKind.Cuda, 0);
            var model = new ForgetMemoryDRNGpt(128, 512, 128, 384, 4, 16, 16,
                random: new Random(53), dtype: TensorDType.Float32);
            model.to(TensorPrecisionMode.Mix8_32, bfp8_block_size: 128);
            model.to(TensorDevice.Cuda);
            _ = model.GenerateTokenIds([1], 1, 0f, 1, null); // Prepare persistent BLAS/weight buffers.
            lane.SynchronizeComputeStream();
            long resident = lane.Memory.Telemetry.ActiveBytes;
            int[] prompt = Enumerable.Range(0, 1024).Select(i => i % 125 + 1).ToArray();
            long wholePrompt;
            using (var state = model.CreateRecurrentState())
            using (var inference = CudaInferenceScope.Begin(resetPool: true, clearPoolOnDispose: true))
            {
                _ = model.AdvanceToLastLogits(prompt, state); // Previous generation prefill.
                lane.SynchronizeComputeStream();
                wholePrompt = lane.Memory.Telemetry.ActiveBytes;
            }
            long bounded = 0;
            int[] generated = model.GenerateTokenIds(prompt, 1, 0f, 1, null, new Random(59), _ =>
            {
                lane.SynchronizeComputeStream();
                bounded = lane.Memory.Telemetry.ActiveBytes;
            });
            _output.WriteLine($"active allocations at first sample: resident={resident}, whole={wholePrompt}, bounded={bounded} bytes");
            Assert.Equal(prompt.Length + 1, generated.Length);
            Assert.True(bounded - resident < (wholePrompt - resident) / 2,
                $"Live activations were not bounded: {bounded - resident}/{wholePrompt - resident}.");
            Assert.True(model.IsTraining);
        }
        finally { Tensor.CudaDeviceIndices = previousDevices; Tensor.ExecutionDevice = previous; }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TrainingLongPromptGenerationAndTrainingAgainKeepResidentState(bool production)
    {
        Assert.SkipWhen(production && Environment.GetEnvironmentVariable("NNTRAIN_DRN_GENERATION_PRODUCTION_TEST") != "1",
            "Enable the bounded production-shape generation regression explicitly.");
        Assert.SkipWhen(Tensor.CudaDeviceCount < 2, "Two CUDA devices required.");
        TensorDevice previous = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0, 1];
            using var session = new ExecutionSession(new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda, CudaDevices = new DeviceSet([0, 1]),
                Precision = PrecisionPolicy.Mix8_32,
            }, new[] { 0, 1 }.Select(device => CudaExecutionLaneFactory.Create(device)));
            using var scope = session.Enter();
            using var policy = CudaDispatchPolicy.Push(CudaDispatchPolicy.Defaults);
            int vocabulary = production ? 11500 : 128;
            int context = production ? 512 : 32;
            int batch = production ? 64 : 4;
            var rng = new CheckpointableRandom(1234);
            var model = new ForgetMemoryDRNGpt(vocabulary, context, production ? 512 : 32,
                production ? 1536 : 64, production ? 32 : 2, 16, 16,
                random: rng, dropout: 0.1f, dtype: TensorDType.Float32);
            rng.BeginRuntime();
            model.AttachTrainingRandom(rng);
            model.to(TensorPrecisionMode.Mix8_32, bfp8_block_size: 128);
            model.to(TensorDevice.Cuda);
            using var engine = new CudaDataParallelEngine(model, [0, 1]);
            var matrix = new NekoMuon(model.HiddenWeightParameters,
                new NekoMuonOptions { LearningRate = 0.003f, NewtonSchulzInterval = 1,
                    NewtonSchulzDepth = 5, NewtonSchulzDepthMode = NekoMuonNewtonSchulzDepthMode.Fixed });
            var auxiliary = new AdamW(model.AuxiliaryParameters,
                new AdamWOptions { LearningRate = 0.001f, Beta2 = 0.95f });
            var optimizer = new CompositeOptimizer(matrix, auxiliary);
            try
            {
                engine.PrepareForTraining(batch);
                optimizer.prepare();
                int[] input = Enumerable.Range(0, batch * context).Select(i => i % (vocabulary - 3) + 1).ToArray();
                int[] targets = input.Select(i => (i + 1) % vocabulary).ToArray();
                void Step(int step)
                {
                    using var guard = DeviceTransferGuard.EnterTrainingStep(2);
                    optimizer.zero_grad();
                    float loss = engine.ForwardBackward(input, targets, batch, context, -1, step);
                    Assert.True(float.IsFinite(loss));
                    nn.utils.clip_grad_norm_(model.Parameters(), 1f);
                    optimizer.step();
                    _output.WriteLine($"training step {step}: loss={loss}");
                }
                Step(0); Step(1);
                Assert.True(engine.TrainingGraphTelemetry.ReplayCount > 0);
                var parameters = model.Parameters().ToArray();
                nint[] resident = parameters.SelectMany(p => new[] { 0, 1 }
                    .Select(device => p.T.EnsureCudaBfp8Buffer(device).Payload.NativePtr)).ToArray();
                int[] prompt = Enumerable.Range(0, production ? 8192 : 1024)
                    .Select(i => i % (vocabulary - 3) + 1).ToArray();
                engine.ReleaseCheckpointTransientMemory();
                Assert.Equal(0, engine.CachedTrainingShapePlanCount);
                long peak = 0;
                int tokens = production ? 200 : 8;
                var timer = System.Diagnostics.Stopwatch.StartNew();
                var generated = model.GenerateTokenIds(prompt, tokens, 0.8f, 40, null, new Random(71), _ =>
                {
                    var accelerator = ForgetMemoryV2Cuda.GetAccelerator(0);
                    peak = Math.Max(peak, accelerator.MemorySize - accelerator.GetFreeMemory());
                });
                _output.WriteLine($"prompt={prompt.Length}, generated={tokens}, generation={timer.Elapsed.TotalSeconds:F2}s, sampled VRAM={peak / 1048576d:F1} MiB");
                Assert.Equal(prompt.Length + tokens, generated.Length);
                engine.ReleaseCheckpointTransientMemory();
                Assert.True(model.IsTraining);
                Assert.Equal(resident, parameters.SelectMany(p => new[] { 0, 1 }
                    .Select(device => p.T.EnsureCudaBfp8Buffer(device).Payload.NativePtr)).ToArray());
                Step(2); Step(3);
                Assert.Null(engine.LastGraphFailure);
                Assert.Equal(0, engine.TrainingGraphTelemetry.FallbackCount);
                engine.ReleaseCheckpointTransientMemory();
                Assert.Throws<InvalidOperationException>(() => model.GenerateTokenIds(
                    prompt, 2, 0f, 1, null, new Random(71), _ => throw new InvalidOperationException("writer failed")));
                engine.ReleaseCheckpointTransientMemory();
                Step(4);
            }
            finally
            {
                try { matrix.DisposeCudaResources(); }
                finally { auxiliary.DisposeCudaResources(); }
            }
        }
        finally { Tensor.CudaDeviceIndices = previousDevices; Tensor.ExecutionDevice = previous; }
    }
}
