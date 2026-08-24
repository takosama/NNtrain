using NNtrain;
using Xunit;

public sealed class CudaTrainingKernelTests
{
    [Fact]
    public void DenseLayerNormAndCrossEntropyMatchCpuForwardBackward()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previous = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            (float[] Output, float[] Input, float[] Weight, float[] Bias)
                Dense(TensorDevice device)
            {
                Tensor.ExecutionDevice = device;
                Tensor.CudaDeviceIndices = device == TensorDevice.Cuda
                    && Tensor.CudaDeviceCount >= 2
                        ? [0, 1]
                        : [0];
                var input = new Tensor(
                    [0.25f, -0.5f, 0.75f, 1f, -0.25f, 0.5f],
                    [2, 3],
                    dtype: TensorDType.BFloat16);
                var weight = new Tensor(
                    [0.2f, -0.3f, 0.4f, -0.5f, 0.6f, 0.1f,
                     0.7f, 0.2f, -0.4f, 0.3f, -0.1f, 0.8f],
                    [4, 3],
                    dtype: TensorDType.BFloat16);
                var bias = new Tensor(
                    [0.1f, -0.2f, 0.05f, 0.3f],
                    [4],
                    dtype: TensorDType.BFloat16);
                Tensor output = input.LinearLastDim(weight, bias, applyRelu: true);
                output.Backward(
                    [0.2f, -0.1f, 0.3f, 0.4f, -0.2f, 0.5f, 0.1f, -0.3f]);
                return (
                    output.Data.ToArray(),
                    input.Grad.ToArray(),
                    weight.Grad.ToArray(),
                    bias.Grad.ToArray());
            }

            var cpuDense = Dense(TensorDevice.Cpu);
            var cudaDense = Dense(TensorDevice.Cuda);
            AssertClose(cpuDense.Output, cudaDense.Output, 1e-5f);
            AssertClose(cpuDense.Input, cudaDense.Input, 1e-5f);
            AssertClose(cpuDense.Weight, cudaDense.Weight, 1e-5f);
            AssertClose(cpuDense.Bias, cudaDense.Bias, 1e-5f);

            (float[] Output, float[] Input, float[] Gamma, float[] Beta)
                Norm(TensorDevice device)
            {
                Tensor.ExecutionDevice = device;
                var input = new Tensor(
                    [0.25f, -0.5f, 0.75f, 1f, -0.25f, 0.5f],
                    [2, 3],
                    dtype: TensorDType.BFloat16);
                var gamma = new Tensor(
                    [1.1f, 0.9f, 1.2f],
                    [3],
                    dtype: TensorDType.BFloat16);
                var beta = new Tensor(
                    [0.1f, -0.2f, 0.05f],
                    [3],
                    dtype: TensorDType.BFloat16);
                Tensor output = input.LayerNormLastDim(gamma, beta);
                output.Backward([0.2f, -0.1f, 0.3f, 0.4f, -0.2f, 0.5f]);
                return (
                    output.Data.ToArray(),
                    input.Grad.ToArray(),
                    gamma.Grad.ToArray(),
                    beta.Grad.ToArray());
            }

            var cpuNorm = Norm(TensorDevice.Cpu);
            var cudaNorm = Norm(TensorDevice.Cuda);
            AssertClose(cpuNorm.Output, cudaNorm.Output, 1e-5f);
            AssertClose(cpuNorm.Input, cudaNorm.Input, 2e-5f);
            AssertClose(cpuNorm.Gamma, cudaNorm.Gamma, 1e-5f);
            AssertClose(cpuNorm.Beta, cudaNorm.Beta, 1e-5f);

            (float Loss, float[] Gradient) Loss(TensorDevice device)
            {
                Tensor.ExecutionDevice = device;
                var logits = new Tensor(
                    [0.25f, -0.5f, 0.75f, 1f, -0.25f, 0.5f],
                    [2, 3],
                    dtype: TensorDType.BFloat16);
                Tensor loss = logits.CrossEntropyWithLogits(
                    [2, 0],
                    labelSmoothing: 0.1f);
                loss.Backward([0.75f]);
                return (loss.item(), logits.Grad.ToArray());
            }

            var cpuLoss = Loss(TensorDevice.Cpu);
            var cudaLoss = Loss(TensorDevice.Cuda);
            Assert.InRange(MathF.Abs(cpuLoss.Loss - cudaLoss.Loss), 0f, 2e-5f);
            AssertClose(cpuLoss.Gradient, cudaLoss.Gradient, 2e-5f);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void AdamWAndNekoMuonCudaUpdatesMatchCpu()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previous = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            (float[] Data, OptimizerStateDictionary State) Adam(
                TensorDevice device)
            {
                Tensor.ExecutionDevice = device;
                var parameter = new Parameter(
                    [0.25f, -0.5f, 0.75f, 1f],
                    [2, 2],
                    "weight",
                    WeightDecayPolicy.Apply,
                    TensorDType.BFloat16);
                parameter.T.Backward([0.2f, -0.1f, 0.3f, -0.4f]);
                IOptimizer optimizer = optim.AdamW(
                    [parameter],
                    lr: 0.01f,
                    weight_decay: 0.02f,
                    bf16_first_moment: true,
                    bf16_second_moment: true);
                optimizer.step();
                return (parameter.T.Data.ToArray(), optimizer.state_dict());
            }

            var cpuAdam = Adam(TensorDevice.Cpu);
            var cudaAdam = Adam(TensorDevice.Cuda);
            AssertClose(cpuAdam.Data, cudaAdam.Data, 1e-5f);

            float[] Neko(TensorDevice device)
            {
                Tensor.ExecutionDevice = device;
                var parameter = new Parameter(
                    [0.25f, -0.5f, 0.75f, 1f, -0.2f, 0.4f],
                    [2, 3],
                    "hidden",
                    WeightDecayPolicy.Apply,
                    TensorDType.BFloat16);
                parameter.T.Backward([0.2f, -0.1f, 0.3f, -0.4f, 0.1f, 0.25f]);
                IOptimizer optimizer = optim.NekoMuon(
                    [parameter],
                    lr: 0.01f,
                    newton_schulz_steps: 2,
                    newton_schulz_interval: 1,
                    weight_decay: 0.02f);
                optimizer.step();
                return parameter.T.Data.ToArray();
            }

            AssertClose(Neko(TensorDevice.Cpu), Neko(TensorDevice.Cuda), 2e-4f);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void EmbeddingAndDropoutMatchCpuForwardBackward()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previous = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            (float[] Output, float[] Gradient) Embedding(TensorDevice device)
            {
                Tensor.ExecutionDevice = device;
                var table = new Tensor(
                    [0.1f, 0.2f, 0.3f, -0.4f, 0.5f, 0.6f,
                     0.7f, -0.8f, 0.9f, 1f, 1.1f, -1.2f],
                    [4, 3],
                    dtype: TensorDType.BFloat16);
                Tensor output = table.EmbeddingLookup([2, 1, 2], 3);
                output.Backward(
                    [0.1f, 0.2f, 0.3f, -0.4f, 0.5f, 0.6f,
                     0.7f, -0.8f, 0.9f]);
                return (output.Data.ToArray(), table.Grad.ToArray());
            }

            var cpuEmbedding = Embedding(TensorDevice.Cpu);
            var cudaEmbedding = Embedding(TensorDevice.Cuda);
            AssertClose(cpuEmbedding.Output, cudaEmbedding.Output, 1e-5f);
            AssertClose(cpuEmbedding.Gradient, cudaEmbedding.Gradient, 1e-5f);

            (float[] Output, float[] Input) Dropout(TensorDevice device)
            {
                Tensor.ExecutionDevice = device;
                var input = new Tensor(
                    [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f],
                    [2, 3],
                    dtype: TensorDType.BFloat16);
                Tensor output = input.Dropout(0.35f, new Random(41));
                output.Backward([0.2f, -0.1f, 0.3f, 0.4f, -0.2f, 0.5f]);
                return (output.Data.ToArray(), input.Grad.ToArray());
            }

            var cpuDropout = Dropout(TensorDevice.Cpu);
            var cudaDropout = Dropout(TensorDevice.Cuda);
            AssertClose(cpuDropout.Output, cudaDropout.Output, 1e-5f);
            AssertClose(cpuDropout.Input, cudaDropout.Input, 1e-5f);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
            Tensor.CudaDeviceIndices = previousIndices;
        }
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
}
