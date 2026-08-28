using NNtrain;
using NNtrain.Runtime.Execution;
using NNtrain.Training.Execution;
using Xunit;

public sealed class CudaDataParallelTests
{
    [Fact]
    public void TransformerCudaGenerationReusesInferenceArenaSafely()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 4,
                dModel: 8,
                numHeads: 2,
                dHidden: 16,
                numLayers: 1,
                rng: new Random(29),
                dtype: TensorDType.BFloat16);

            for (int iteration = 0; iteration < 3; iteration++)
            {
                int[] generated = model.GenerateTokenIds(
                    [1, 2],
                    maxNewTokens: 6,
                    temperature: 0f,
                    stopTokenId: null,
                    random: new Random(31));
                Assert.Equal(8, generated.Length);
                Assert.All(generated, token => Assert.InRange(token, 0, 31));
            }
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void TransformerTwoGpuForwardBackwardIsFinite()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0, 1];
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 4,
                dModel: 8,
                numHeads: 2,
                dHidden: 16,
                numLayers: 1,
                rng: new Random(13),
                dtype: TensorDType.BFloat16);

            for (int iteration = 0; iteration < 3; iteration++)
            {
                model.ZeroGrad();
                float loss = CudaDataParallel.ForwardBackward(
                    model,
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    [2, 3, 4, 5, 6, 7, 8, 9],
                    batchSize: 2,
                    sequenceLength: 4);

                Assert.True(float.IsFinite(loss));
            }
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void StableTwoGpuShapeReusesCompiledShardPlan()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0, 1];
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 4,
                dModel: 8,
                numHeads: 2,
                dHidden: 16,
                numLayers: 1,
                rng: new Random(211),
                dropout: 0f,
                dtype: TensorDType.BFloat16);
            using var engine = new CudaDataParallelEngine(
                model,
                [0, 1],
                new CudaAdaptiveShardingOptions { Enabled = false });
            int[] input = [1, 2, 3, 4, 5, 6, 7, 8];
            int[] target = [2, 3, 4, 5, 6, 7, 8, 9];

            for (int step = 0; step < 3; step++)
            {
                model.ZeroGrad();
                Assert.True(float.IsFinite(engine.ForwardBackward(
                    input,
                    target,
                    batchSize: 2,
                    sequenceLength: 4)));
            }

            Assert.Equal(1, engine.TrainingShapePlanBuildCount);
            Assert.Equal(1, engine.CachedTrainingShapePlanCount);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void PureBfp8SingleGpuPublishesLocalGradientsWithoutHostFallback()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 4,
                dModel: 8,
                numHeads: 2,
                dHidden: 16,
                numLayers: 1,
                rng: new Random(109),
                dropout: 0f,
                dtype: TensorDType.Float32);
            model.to(TensorPrecisionMode.Bfp8);

            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            using var engine = new CudaDataParallelEngine(model, [0]);
            _ = engine.ForwardBackward(
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                batchSize: 1,
                sequenceLength: 4);
            model.zero_grad();

            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;
            float loss = engine.ForwardBackward(
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                batchSize: 1,
                sequenceLength: 4);
            NativeCudaTransferTelemetry transfer =
                NativeCudaRuntime.TransferTelemetry - before;

            Assert.True(float.IsFinite(loss));
            Parameter[] parameters = model.Parameters().ToArray();
            Assert.NotEmpty(parameters);
            foreach (Parameter parameter in parameters)
            {
                CudaGradientCoherenceSnapshot snapshot =
                    parameter.T.GetCudaGradientCoherenceSnapshot();
                Assert.Equal(CudaGradientCoherenceKind.Local, snapshot.Kind);
                Assert.Equal(0, snapshot.LocalDeviceIndex);
                Assert.False(snapshot.PendingStamp.IsValid);
                Assert.True(parameter.T.HasAuthoritativeCudaBfp8Gradient);
            }

            // H2D is input/target only. The autograd root seed is now passed
            // as a CUDA kernel argument and never uploaded as tensor data.
            // D2H is one graph-wide finite scalar, one norm scalar, and loss;
            // no parameter-sized gradient payload returns to the host.
            Assert.Equal(
                2 * 4 * sizeof(int),
                transfer.HostToDeviceBytes);
            Assert.Equal(
                sizeof(int) + sizeof(double) + sizeof(float),
                transfer.DeviceToHostBytes);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void ExplicitEngineTwoGpuGradientsMatchSingleGpu()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            const int vocabulary = 32;
            const int batch = 2;
            const int sequence = 4;
            int[] input = [1, 2, 3, 4, 5, 6, 7, 8];
            int[] target = [2, 3, 4, 5, 6, 7, 8, 9];

            (float Loss, float[][] Gradients) Run(int[] devices)
            {
                Tensor.ExecutionDevice = TensorDevice.Cuda;
                Tensor.CudaDeviceIndices = devices;
                var model = new ForgetMemoryV2Gpt(
                    vocabulary,
                    sequence,
                    modelWidth: 8,
                    hiddenWidth: 16,
                    numLayers: 1,
                    keyWidth: 4,
                    valueWidth: 4,
                    random: new Random(17),
                    dropout: 0f,
                    dtype: TensorDType.BFloat16);
                model.ZeroGrad();
                float loss;
                using (var engine = new CudaDataParallelEngine(
                    model,
                    new CudaAdaptiveShardingOptions { Enabled = false }))
                {
                    loss = engine.ForwardBackward(
                        input,
                        target,
                        batch,
                        sequence);
                }
                return (
                    loss,
                    model.Parameters()
                        .Select(parameter => parameter.T.Grad.ToArray())
                        .ToArray());
            }

            var single = Run([0]);
            var parallel = Run([0, 1]);
            Assert.InRange(MathF.Abs(single.Loss - parallel.Loss), 0f, 2e-3f);
            Assert.Equal(single.Gradients.Length, parallel.Gradients.Length);
            for (int parameter = 0;
                parameter < single.Gradients.Length;
                parameter++)
            {
                Assert.Equal(
                    single.Gradients[parameter].Length,
                    parallel.Gradients[parameter].Length);
                for (int index = 0;
                    index < single.Gradients[parameter].Length;
                    index++)
                {
                    float difference = MathF.Abs(
                        single.Gradients[parameter][index]
                        - parallel.Gradients[parameter][index]);
                    Assert.True(
                        difference <= 3e-3f,
                        $"Parameter {parameter}, index {index}: " +
                        $"single={single.Gradients[parameter][index]:R}, " +
                        $"parallel={parallel.Gradients[parameter][index]:R}, " +
                        $"difference={difference:R}.");
                }
            }
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void ConsecutiveExplicitEnginesReleaseOwnedCudaAllocations()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0, 1];
            var models = new List<LanguageModel>();
            foreach (TensorDType dtype in new[]
            {
                TensorDType.Float32,
                TensorDType.BFloat16,
            })
            {
                var model = new GptRinWikiJp(
                    vocabularySize: 32,
                    contextLength: 4,
                    dModel: 8,
                    numHeads: 2,
                    dHidden: 16,
                    numLayers: 1,
                    rng: new Random(83 + (int)dtype),
                    dropout: 0f,
                    dtype: dtype);
                models.Add(model);
                model.ZeroGrad();
                var execution = new ExecutionSession(new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet(0, 1),
                });
                var training = new TrainingSession(
                    execution,
                    ownsExecutionSession: true);
                CudaDataParallelEngine engine = training.OwnCudaDataParallel(
                    model,
                    [0, 1],
                    new CudaAdaptiveShardingOptions { Enabled = false });

                NativeCudaAllocationTelemetry beforeStep =
                    NativeCudaRuntime.AllocationTelemetry;
                float loss = engine.ForwardBackward(
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    [2, 3, 4, 5, 6, 7, 8, 9],
                    batchSize: 2,
                    sequenceLength: 4);
                NativeCudaAllocationTelemetry afterStep =
                    NativeCudaRuntime.AllocationTelemetry;

                training.Dispose();
                NativeCudaAllocationTelemetry afterDispose =
                    NativeCudaRuntime.AllocationTelemetry;

                Assert.True(float.IsFinite(loss));
                Assert.True(engine.IsDisposed);
                Assert.True(execution.IsDisposed);
                Assert.True(
                    afterStep.AllocationCount > beforeStep.AllocationCount);
                Assert.True(afterDispose.FreeCount > afterStep.FreeCount);
                Assert.True(afterDispose.FreeBytes > afterStep.FreeBytes);
            }
            GC.KeepAlive(models);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void TwoGpuAllReducePublishesExactGradientNormForClipping()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0, 1];
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 4,
                dModel: 8,
                numHeads: 2,
                dHidden: 16,
                numLayers: 1,
                rng: new Random(47),
                dropout: 0f,
                dtype: TensorDType.BFloat16);
            model.ZeroGrad();
            _ = CudaDataParallel.ForwardBackward(
                model,
                [1, 2, 3, 4, 5, 6, 7, 8],
                [2, 3, 4, 5, 6, 7, 8, 9],
                batchSize: 2,
                sequenceLength: 4);

            Parameter[] parameters = model.Parameters().ToArray();
            double expectedSquared = parameters
                .SelectMany(parameter => parameter.T.Grad)
                .Sum(value => (double)value * value);
            float actual = nn.utils.clip_grad_norm_(parameters, max_norm: 100f);

            Assert.InRange(
                Math.Abs(actual - Math.Sqrt(expectedSquared)),
                0d,
                1e-4d);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }
}
