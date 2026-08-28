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
                Tensor.CudaDeviceIndices = device == TensorDevice.Cuda
                    && Tensor.CudaDeviceCount >= 2
                        ? [0, 1]
                        : [0];
                Tensor.ExecutionDevice = device;
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
                output.Backward(BFloat16(
                    [0.2f, -0.1f, 0.3f, 0.4f, -0.2f, 0.5f, 0.1f, -0.3f]));
                return (
                    output.Data.ToArray(),
                    input.Grad.ToArray(),
                    weight.Grad.ToArray(),
                    bias.Grad.ToArray());
            }

            var cpuDense = Dense(TensorDevice.Cpu);
            var cudaDense = Dense(TensorDevice.Cuda);
            AssertClose(cpuDense.Output, cudaDense.Output, 1e-5f);
            AssertClose(BFloat16(cpuDense.Input), cudaDense.Input, 1e-5f);
            AssertClose(BFloat16(cpuDense.Weight), cudaDense.Weight, 1e-5f);
            AssertClose(BFloat16(cpuDense.Bias), cudaDense.Bias, 1e-5f);

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
                output.Backward(BFloat16(
                    [0.2f, -0.1f, 0.3f, 0.4f, -0.2f, 0.5f]));
                return (
                    output.Data.ToArray(),
                    input.Grad.ToArray(),
                    gamma.Grad.ToArray(),
                    beta.Grad.ToArray());
            }

            var cpuNorm = Norm(TensorDevice.Cpu);
            var cudaNorm = Norm(TensorDevice.Cuda);
            AssertClose(cpuNorm.Output, cudaNorm.Output, 1e-5f);
            AssertClose(BFloat16(cpuNorm.Input), cudaNorm.Input, 2e-5f);
            AssertClose(BFloat16(cpuNorm.Gamma), cudaNorm.Gamma, 1e-5f);
            AssertClose(BFloat16(cpuNorm.Beta), cudaNorm.Beta, 1e-5f);

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
                loss.Backward(BFloat16([0.75f]));
                return (loss.item(), logits.Grad.ToArray());
            }

            var cpuLoss = Loss(TensorDevice.Cpu);
            var cudaLoss = Loss(TensorDevice.Cuda);
            Assert.InRange(MathF.Abs(cpuLoss.Loss - cudaLoss.Loss), 0f, 2e-5f);
            // Pure bfloat16 publishes the operand-quantized logits gradient
            // in BF16. Stable softmax/reduction math remains FP32.
            AssertClose(BFloat16(cpuLoss.Gradient), cudaLoss.Gradient, 1e-3f);
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousIndices;
            Tensor.ExecutionDevice = previous;
        }
    }

    [Fact]
    public void FusedBFloat16ResidualDropoutLayerNormMatchesCpu()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int rows = 1031;
        // Exercises the production-width one-warp-per-row fused kernel and a
        // 256-row parameter-reduction tail (1031 is intentionally uneven).
        const int columns = 512;
        float[] residualValues = Enumerable.Range(0, rows * columns)
            .Select(index => (index % 53 - 26) * 0.007f)
            .ToArray();
        float[] branchValues = Enumerable.Range(0, rows * columns)
            .Select(index => (index % 41 - 20) * 0.009f)
            .ToArray();
        float[] gammaValues = Enumerable.Range(0, columns)
            .Select(index => 0.8f + (index % 17) * 0.02f)
            .ToArray();
        float[] betaValues = Enumerable.Range(0, columns)
            .Select(index => (index % 13 - 6) * 0.01f)
            .ToArray();
        float[] seedGradient = Enumerable.Range(0, rows * columns)
            .Select(index => (index % 37 - 18) * 0.005f)
            .ToArray();

        TensorDevice previous = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            (float[] Output, float[] Residual, float[] Branch,
                float[] Gamma, float[] Beta) Run(TensorDevice device)
            {
                Tensor.CudaDeviceIndices = [0];
                Tensor.ExecutionDevice = device;
                var residual = new Tensor(
                    residualValues, [rows, columns],
                    dtype: TensorDType.BFloat16);
                var branch = new Tensor(
                    branchValues, [rows, columns],
                    dtype: TensorDType.BFloat16);
                var gamma = new Tensor(
                    gammaValues, [columns], dtype: TensorDType.BFloat16);
                var beta = new Tensor(
                    betaValues, [columns], dtype: TensorDType.BFloat16);
                Tensor output = residual.AddDropoutLayerNormLastDim(
                    branch, gamma, beta, 0.25f, new Random(71));
                output.Backward(BFloat16(seedGradient));
                return (
                    output.Data.ToArray(), residual.Grad.ToArray(),
                    branch.Grad.ToArray(), gamma.Grad.ToArray(),
                    beta.Grad.ToArray());
            }

            var cpu = Run(TensorDevice.Cpu);
            var cuda = Run(TensorDevice.Cuda);
            AssertClose(cpu.Output, cuda.Output, 2e-4f);
            AssertClose(BFloat16(cpu.Residual), cuda.Residual, 2e-4f);
            AssertClose(BFloat16(cpu.Branch), cuda.Branch, 3e-4f);
            AssertClose(BFloat16(cpu.Gamma), cuda.Gamma, 8e-4f);
            AssertClose(BFloat16(cpu.Beta), cuda.Beta, 8e-4f);
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousIndices;
            Tensor.ExecutionDevice = previous;
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
            Tensor.CudaDeviceIndices = previousIndices;
            Tensor.ExecutionDevice = previous;
        }
    }

    [Fact]
    public void NekoMuonBlockReducedFp32StatisticsMatchCpuForLargeTensor()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int rows = 256;
        const int columns = 256;
        float[] values = Enumerable.Range(0, rows * columns)
            .Select(index => (index % 47 - 23) * 0.001f)
            .ToArray();
        float[] firstGradient = Enumerable.Range(0, rows * columns)
            .Select(index => (index % 31 - 15) * 0.002f)
            .ToArray();
        float[] secondGradient = Enumerable.Range(0, rows * columns)
            .Select(index => (index % 43 - 21) * 0.0015f)
            .ToArray();
        TensorDevice previous = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            (float[] Data, float Confidence) Run(TensorDevice device)
            {
                Tensor.CudaDeviceIndices = [0];
                Tensor.ExecutionDevice = device;
                var parameter = new Parameter(
                    values,
                    [rows, columns],
                    "large_hidden",
                    WeightDecayPolicy.Exclude,
                    TensorDType.BFloat16);
                var optimizer = new NekoMuon(
                    [parameter],
                    new NekoMuonOptions
                    {
                        LearningRate = 0.01f,
                        WeightDecay = 0f,
                        NewtonSchulzInterval = 100,
                        MaxNewtonSchulzSteps = 2,
                    });
                parameter.T.Backward(firstGradient);
                optimizer.step();
                optimizer.zero_grad();
                parameter.T.Backward(secondGradient);
                optimizer.step();
                NekoMuonState state = optimizer.CaptureState();
                return (
                    parameter.T.Data.ToArray(),
                    state.ParameterStates[0].Confidence);
            }

            var cpu = Run(TensorDevice.Cpu);
            var cuda = Run(TensorDevice.Cuda);
            AssertClose(cpu.Data, cuda.Data, 2e-4f);
            Assert.InRange(
                MathF.Abs(cpu.Confidence - cuda.Confidence),
                0f,
                2e-5f);
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousIndices;
            Tensor.ExecutionDevice = previous;
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
            AssertClose(
                BFloat16(cpuEmbedding.Gradient),
                cudaEmbedding.Gradient,
                1e-5f);

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
            AssertClose(BFloat16(cpuDropout.Input), cudaDropout.Input, 1e-5f);
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousIndices;
            Tensor.ExecutionDevice = previous;
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
            float difference = MathF.Abs(expected[index] - actual[index]);
            Assert.True(
                difference <= tolerance,
                $"Index {index}: expected {expected[index]:R}, " +
                $"actual {actual[index]:R}, difference {difference:R}, " +
                $"tolerance {tolerance:R}.");
        }
    }

    private static float[] BFloat16(IEnumerable<float> values)
        => values.Select(TensorStorageCodec.RoundToBFloat16).ToArray();
}
