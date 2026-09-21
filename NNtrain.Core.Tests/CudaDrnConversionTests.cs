using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaDrnConversionTests
{
    [Theory]
    [InlineData(32, 129)]
    [InlineData(128, 257)]
    public void Mix8DrnLossConsumesBf16LogitsWithoutChangingPublicForward(int block, int vocabulary)
    {
        Assert.SkipWhen(!Tensor.IsCudaAvailable(), "CUDA unavailable.");
        WithCuda(() =>
        {
            var model = new ForgetMemoryDRNGpt(vocabulary, 3, 32, 64, 2, 16, 16,
                random: new Random(101), dtype: TensorDType.Float32);
            model.to(TensorPrecisionMode.Mix8_32, block);
            model.to(TensorDevice.Cuda);
            int[] input = [1, 2, 3];
            int[] targets = [2, -1, 4];
            var before = CudaBfp8GemmTelemetry.Snapshot;
            Tensor loss = model.ForwardLoss(input, targets, 1, 3);
            Assert.Equal(1, (CudaBfp8GemmTelemetry.Snapshot - before).DirectBFloat16LossHeadExecutions);
            Tensor logits = Assert.Single(loss.Node.Parents);
            Assert.Equal(TensorDType.BFloat16, logits.DType);
            Assert.Equal(TensorDType.Float32, loss.DType);
            float[] values = logits.Data.ToArray();
            double expected = 0;
            foreach (int row in new[] { 0, 2 })
            {
                double max = values.AsSpan(row * vocabulary, vocabulary).ToArray().Max();
                double sum = 0;
                for (int column = 0; column < vocabulary; column++)
                    sum += Math.Exp(values[row * vocabulary + column] - max);
                expected += Math.Log(sum) + max - values[row * vocabulary + targets[row]];
            }
            Assert.InRange(Math.Abs(loss.item() - expected / 2), 0, 1e-5);
            loss.BackwardAndRelease();
            foreach (var parameter in model.Parameters())
            {
                Assert.Equal(TensorDType.Bfp8, parameter.T.DType);
                Assert.All(parameter.T.Grad, gradient => Assert.True(float.IsFinite(gradient)));
            }
            using var noGrad = AutogradContext.NoGrad();
            using var inference = CudaInferenceScope.Begin(clearPoolOnDispose: true);
            Assert.Equal(TensorDType.Bfp8, model.Forward(input, 1, 3).DType);
            model.eval();
            Assert.True(float.IsFinite(model.ForwardLoss(input, targets, 1, 3).item()));
            model.train();
        });
    }

    [Theory]
    [InlineData(32, 127)]
    [InlineData(128, 512)]
    public void LayerNormParameterCacheMatchesUncachedValuesGradientsAndRefresh(int block, int width)
    {
        Assert.SkipWhen(!Tensor.IsCudaAvailable(), "CUDA unavailable.");
        WithCuda(() =>
        {
            var descriptor = Bfp8QuantizationDescriptor.Block(block);
            var input = Tensor.FromBfp8(Enumerable.Range(0, 3 * width).Select(i => MathF.Sin(i * 0.17f)).ToArray(), [3, width], descriptor);
            var gamma = Tensor.FromBfp8(Enumerable.Range(0, width).Select(i => 1 + i * 0.001f).ToArray(), [width], descriptor);
            var beta = Tensor.FromBfp8(new float[width], [width], descriptor);
            input.to(TensorDevice.Cuda); gamma.to(TensorDevice.Cuda); beta.to(TensorDevice.Cuda);
            (float[] Values, float[] Input, float[] Gamma, float[] Beta) Run(bool cached)
            {
                using var policy = CudaDispatchPolicy.Push(CudaDispatchPolicy.Defaults with
                { DisableBfp8LayerNormParameterCache = !cached });
                input.ZeroGrad(); gamma.ZeroGrad(); beta.ZeroGrad();
                var result = input.LayerNormLastDim(gamma, beta);
                var values = result.Data.ToArray();
                result.BackwardAndRelease(Enumerable.Range(0, input.Numel).Select(i => MathF.Cos(i * 0.11f)).ToArray());
                return (values, input.Grad.ToArray(), gamma.Grad.ToArray(), beta.Grad.ToArray());
            }
            for (int generation = 0; generation < 2; generation++)
            {
                var reference = Run(false);
                var cached = Run(true);
                Assert.Equal(reference.Values, cached.Values);
                Assert.Equal(reference.Input, cached.Input);
                Assert.Equal(reference.Gamma, cached.Gamma);
                Assert.Equal(reference.Beta, cached.Beta);
                var encoded = Bfp8QuantizationCodec.Default.Encode(Enumerable.Range(0, width)
                    .Select(i => 0.4f + MathF.Sin(i * 0.23f)).ToArray(), descriptor);
                var replica = gamma.EnsureCudaBfp8Buffer(0);
                replica.Payload.CopyFromCPU(encoded.Payload.Span);
                replica.Scales.CopyFromCPU(encoded.Scales.Span);
                gamma.MarkCudaBfp8DataReplicasSynchronized([0]);
            }
        });
    }

    private static void WithCuda(Action action)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        int[] devices = Tensor.CudaDeviceIndices.ToArray();
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
            action();
        }
        finally { Tensor.CudaDeviceIndices = devices; Tensor.ExecutionDevice = previous; }
    }

    [Fact]
    public void GraphReplayRefreshesLayerNormParametersAfterEveryOptimizerUpdate()
    {
        Assert.SkipWhen(!Tensor.IsCudaAvailable(), "CUDA unavailable.");
        (float[] Losses, float[][] Parameters) Run(bool graph)
        {
            float[] losses = new float[5];
            float[][] values = [];
            WithCuda(() =>
            {
                using var dispatch = CudaDispatchPolicy.Push(CudaDispatchPolicy.Defaults with { DisableCudaGraphs = !graph });
                var rng = new CheckpointableRandom(139);
                var model = new ForgetMemoryDRNGpt(64, 8, 32, 64, 2, 16, 16,
                    random: rng, dropout: 0f, dtype: TensorDType.Float32);
                rng.BeginRuntime(); model.AttachTrainingRandom(rng);
                model.to(TensorPrecisionMode.Mix8_32, 32);
                model.to(TensorDevice.Cuda);
                using var engine = new CudaDataParallelEngine(model, [0]);
                var optimizer = new AdamW(model.Parameters(), new AdamWOptions { LearningRate = 0.003f });
                try
                {
                    engine.PrepareForTraining(1);
                    ((IOptimizer)optimizer).prepare();
                    for (int step = 0; step < losses.Length; step++)
                    {
                        model.ZeroGrad();
                        losses[step] = engine.ForwardBackward([1, 2, 3, 4, 5, 6, 7, 8],
                            [2, 3, 4, 5, 6, 7, 8, 9], 1, 8, -1, step);
                        optimizer.Step();
                    }
                    if (graph)
                    {
                        Assert.True(engine.TrainingGraphTelemetry.ReplayCount >= 4);
                        Assert.Equal(0, engine.TrainingGraphTelemetry.FallbackCount);
                    }
                    values = model.Parameters().Select(p => p.T.Data.ToArray()).ToArray();
                }
                finally { optimizer.DisposeCudaResources(); }
            });
            return (losses, values);
        }
        var eager = Run(false);
        var replay = Run(true);
        for (int i = 0; i < eager.Losses.Length; i++)
            Assert.InRange(MathF.Abs(eager.Losses[i] - replay.Losses[i]), 0f, 1e-5f);
        for (int p = 0; p < eager.Parameters.Length; p++)
            for (int i = 0; i < eager.Parameters[p].Length; i++)
                Assert.InRange(MathF.Abs(eager.Parameters[p][i] - replay.Parameters[p][i]), 0f, 1e-5f);
    }
}
