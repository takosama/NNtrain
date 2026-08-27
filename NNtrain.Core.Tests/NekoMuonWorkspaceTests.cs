using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class NekoMuonWorkspaceTests
{
    [Fact]
    public void ConstructorKeepsCpuWorkspacesLazyAndPlansLargestShapeOnly()
    {
        Parameter first = CreateParameter(32, [4, 8], "first");
        Parameter second = CreateParameter(16, [8, 2], "second");
        var optimizer = new NekoMuon(
            [first, second],
            new NekoMuonOptions
        {
            MaxNewtonSchulzSteps = 1,
            WeightDecay = 0f,
        });

        Assert.Equal(0, optimizer.MaterializedCpuWorkspaceCount);
        Assert.Equal(928L, optimizer.LegacyCudaScratchBytesPerDevice);
        Assert.Equal(384L, optimizer.SharedCudaScratchBytesPerDevice);
    }

    [Fact]
    public void SharedCudaScratchMatchesCpuAcrossParametersAndSteps()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        string? previousBatchSize = Environment.GetEnvironmentVariable(
            "NNTRAIN_NEKOMUON_BATCH_SIZE");
        string? previousDisable = Environment.GetEnvironmentVariable(
            "NNTRAIN_DISABLE_BATCHED_NEKOMUON");
        try
        {
            Environment.SetEnvironmentVariable(
                "NNTRAIN_NEKOMUON_BATCH_SIZE", "4");
            Environment.SetEnvironmentVariable(
                "NNTRAIN_DISABLE_BATCHED_NEKOMUON", null);
            (float[][] Data, NekoMuonState State, int CpuWorkspaces) Run(
                TensorDevice device)
            {
                Tensor.ExecutionDevice = device;
                Tensor.CudaDeviceIndices = [0];
                Parameter wide = new(
                    Values(60, 3),
                    [5, 12],
                    "wide",
                    WeightDecayPolicy.Exclude,
                    TensorDType.BFloat16);
                Parameter tall = new(
                    Values(60, 17),
                    [12, 5],
                    "tall",
                    WeightDecayPolicy.Apply,
                    TensorDType.BFloat16);
                var optimizer = new NekoMuon(
                    [wide, tall],
                    new NekoMuonOptions
                    {
                        LearningRate = 0.007f,
                        BetaFast = 0.8f,
                        BetaSlow = 0.93f,
                        Rho = 0.4f,
                        Epsilon = 1e-8f,
                        MaxNewtonSchulzSteps = 2,
                        NewtonSchulzInterval = 1,
                        NewtonSchulzDepthMode =
                            NekoMuonNewtonSchulzDepthMode.Minimum,
                        NewtonSchulzDepth = 1.5f,
                        WeightDecay = 0.01f,
                    });

                for (int step = 0; step < 2; step++)
                {
                    wide.T.Backward(Values(60, 31 + step * 7));
                    tall.T.Backward(Values(60, 53 + step * 11));
                    optimizer.Step();
                    optimizer.ZeroGrad();
                }

                return (
                    [wide.T.Data.ToArray(), tall.T.Data.ToArray()],
                    optimizer.CaptureState(),
                    optimizer.MaterializedCpuWorkspaceCount);
            }

            var cpu = Run(TensorDevice.Cpu);
            var cuda = Run(TensorDevice.Cuda);

            AssertClose(cpu.Data[0], cuda.Data[0], 3e-4f);
            AssertClose(cpu.Data[1], cuda.Data[1], 3e-4f);
            AssertClose(
                cpu.State.ParameterStates[0].FastMoment,
                cuda.State.ParameterStates[0].FastMoment,
                2e-5f);
            AssertClose(
                cpu.State.ParameterStates[1].SlowMoment,
                cuda.State.ParameterStates[1].SlowMoment,
                2e-5f);
            Assert.InRange(
                MathF.Abs(
                    cpu.State.ParameterStates[0].Confidence
                    - cuda.State.ParameterStates[0].Confidence),
                0f,
                2e-5f);
            Assert.Equal(0, cuda.CpuWorkspaces);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
            Environment.SetEnvironmentVariable(
                "NNTRAIN_NEKOMUON_BATCH_SIZE", previousBatchSize);
            Environment.SetEnvironmentVariable(
                "NNTRAIN_DISABLE_BATCHED_NEKOMUON", previousDisable);
        }
    }

    [Fact]
    public void EightItemCudaBatchIncludingTransposeMatchesCpu()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        string? previousBatchSize = Environment.GetEnvironmentVariable(
            "NNTRAIN_NEKOMUON_BATCH_SIZE");
        string? previousDisable = Environment.GetEnvironmentVariable(
            "NNTRAIN_DISABLE_BATCHED_NEKOMUON");
        try
        {
            Environment.SetEnvironmentVariable(
                "NNTRAIN_NEKOMUON_BATCH_SIZE", "8");
            Environment.SetEnvironmentVariable(
                "NNTRAIN_DISABLE_BATCHED_NEKOMUON", null);

            (float[][] Data, NekoMuonState State) Run(TensorDevice device)
            {
                Tensor.ExecutionDevice = device;
                Tensor.CudaDeviceIndices = [0];
                float[] initial = Values(48 * 64, 19);
                float[] gradient = Values(initial.Length, 47);
                Parameter[] parameters = Enumerable.Range(0, 8)
                    .Select(index => new Parameter(
                        initial.ToArray(),
                        index % 2 == 0 ? [48, 64] : [64, 48],
                        $"matrix-{index}",
                        WeightDecayPolicy.Exclude,
                        TensorDType.BFloat16))
                    .ToArray();
                var optimizer = new NekoMuon(
                    parameters,
                    new NekoMuonOptions
                    {
                        LearningRate = 0.004f,
                        BetaFast = 0.8f,
                        BetaSlow = 0.93f,
                        Rho = 0.4f,
                        Epsilon = 1e-8f,
                        MaxNewtonSchulzSteps = 2,
                        NewtonSchulzInterval = 1,
                        NewtonSchulzDepthMode =
                            NekoMuonNewtonSchulzDepthMode.Fixed,
                        NewtonSchulzDepth = 2f,
                        WeightDecay = 0f,
                    });

                Assert.Equal(8, optimizer.CudaBatchCapacity);
                for (int step = 0; step < 2; step++)
                {
                    foreach (Parameter parameter in parameters)
                        parameter.T.Backward(gradient);
                    optimizer.Step();
                    optimizer.ZeroGrad();
                }

                return (
                    parameters.Select(parameter =>
                        parameter.T.Data.ToArray()).ToArray(),
                    optimizer.CaptureState());
            }

            var cpu = Run(TensorDevice.Cpu);
            var cuda = Run(TensorDevice.Cuda);

            for (int index = 0; index < cpu.Data.Length; index++)
            {
                AssertClose(cpu.Data[index], cuda.Data[index], 8e-4f);
                AssertClose(
                    cpu.State.ParameterStates[index].FastMoment,
                    cuda.State.ParameterStates[index].FastMoment,
                    3e-5f);
                AssertClose(
                    cpu.State.ParameterStates[index].SlowMoment,
                    cuda.State.ParameterStates[index].SlowMoment,
                    3e-5f);
                Assert.InRange(
                    MathF.Abs(
                        cpu.State.ParameterStates[index].Confidence
                        - cuda.State.ParameterStates[index].Confidence),
                    0f,
                    3e-5f);
            }
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
            Environment.SetEnvironmentVariable(
                "NNTRAIN_NEKOMUON_BATCH_SIZE", previousBatchSize);
            Environment.SetEnvironmentVariable(
                "NNTRAIN_DISABLE_BATCHED_NEKOMUON", previousDisable);
        }
    }

    [Fact]
    public void ForceFullCudaDepthDoesNotOverwriteConfidence()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];

            (float Confidence, float Depth) Run(bool forceFull)
            {
                Parameter parameter = new(
                    Values(48, 23),
                    [6, 8],
                    "confidence",
                    WeightDecayPolicy.Exclude,
                    TensorDType.BFloat16);
                var optimizer = new NekoMuon(
                    [parameter],
                    new NekoMuonOptions
                    {
                        LearningRate = 0.004f,
                        BetaFast = 0.8f,
                        BetaSlow = 0.93f,
                        Rho = 0.4f,
                        Epsilon = 1e-8f,
                        MaxNewtonSchulzSteps = 2,
                        NewtonSchulzInterval = 1,
                        WeightDecay = 0f,
                    })
                {
                    ForceFullNewtonSchulz = forceFull,
                };
                parameter.T.Backward(Values(48, 71));

                optimizer.Step();

                NekoMuonDiagnostics diagnostics = optimizer.GetDiagnostics();
                return (
                    optimizer.CaptureState()
                        .ParameterStates[0]
                        .Confidence,
                    diagnostics.MeanNewtonSchulzDepth);
            }

            var adaptive = Run(forceFull: false);
            var forced = Run(forceFull: true);

            Assert.InRange(
                MathF.Abs(adaptive.Confidence - forced.Confidence),
                0f,
                1e-6f);
            Assert.True(adaptive.Depth < 2f);
            Assert.Equal(2f, forced.Depth);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private static Parameter CreateParameter(
        int length,
        int[] shape,
        string name)
        => new(
            Values(length, name.Length),
            shape,
            name,
            WeightDecayPolicy.Exclude);

    private static float[] Values(int length, int offset)
        => Enumerable.Range(0, length)
            .Select(index => MathF.Sin((index + offset) * 0.17f) * 0.25f)
            .ToArray();
}
