using NNtrain;
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
    public void TwoGpuGradientsMatchSingleGpu()
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
                if (devices.Length == 1)
                {
                    Tensor logits = model.Forward(input, batch, sequence);
                    Tensor value = logits.CrossEntropyWithLogits(target);
                    loss = value.item();
                    value.Backward();
                }
                else
                {
                    loss = CudaDataParallel.ForwardBackward(
                        model,
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
                    Assert.InRange(
                        MathF.Abs(
                            single.Gradients[parameter][index]
                            - parallel.Gradients[parameter][index]),
                        0f,
                        3e-3f);
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
