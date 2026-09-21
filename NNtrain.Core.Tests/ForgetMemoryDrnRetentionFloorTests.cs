using NNtrain;
using Xunit;

public sealed class ForgetMemoryDrnRetentionFloorTests : IDisposable
{
    private readonly TensorDevice _previousDevice = Tensor.ExecutionDevice;

    public ForgetMemoryDrnRetentionFloorTests() => Tensor.ExecutionDevice = TensorDevice.Cpu;

    public void Dispose() => Tensor.ExecutionDevice = _previousDevice;

    [Theory]
    [InlineData(0f)]
    [InlineData(0.5f)]
    [InlineData(0.99f)]
    public void ReadAndFinalStateFollowFlooredRetentionWithoutChangingDeltaWrite(float floor)
    {
        // At least three tokens are necessary: the first write starts from
        // zero and the second read cannot yet reveal retention of that write.
        float[] token = [0.8f, -0.6f, 0.3f, -0.4f, 0.2f];
        float[] values = Enumerable.Range(0, 4).SelectMany(_ => token).ToArray();
        var input = new Tensor(values, [1, 4, 5], dtype: TensorDType.Float32);
        Tensor trainingOutput = input.ForgetMemoryDRN(1, 1, floor);
        float[] state = [0f];
        using var noGrad = AutogradContext.NoGrad();
        Tensor continued = input.ForgetMemoryDRNContinue(1, 1, floor, state);

        (float[] expected, float finalState) = Reference(values, floor, initialState: 0f);
        AssertClose(expected, trainingOutput.Data, 1e-6f);
        AssertClose(expected, continued.Data, 1e-6f);
        Assert.InRange(MathF.Abs(finalState - state[0]), 0f, 1e-6f);
        Assert.Equal(0f, expected[0]);
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(0.5f)]
    [InlineData(0.99f)]
    public void ContinuationKeepsFloorWithExistingStateAndTokenBoundaries(float floor)
    {
        float[] values =
        [
            0.8f, -0.6f, 0.3f, -0.4f, 0.2f,
            -0.4f, 0.7f, -0.2f, 0.1f, -0.6f,
            0.6f, 0.4f, 0.5f, -1.2f, 0.7f,
        ];
        const float initialState = 0.37f;
        (float[] expected, float finalState) = Reference(values, floor, initialState);
        float[] arrayState = [initialState];
        using var ownedState = new ForgetMemoryRecurrentMemory(1);
        ownedState.HostForCpuMutation()[0] = initialState;
        ownedState.MarkHostMutated();
        using var noGrad = AutogradContext.NoGrad();
        for (int time = 0; time < 3; time++)
        {
            var input = new Tensor(values[(time * 5)..((time + 1) * 5)], [1, 1, 5], dtype: TensorDType.Float32);
            Tensor arrayOutput = input.ForgetMemoryDRNContinue(1, 1, floor, arrayState);
            Tensor ownedOutput = input.ForgetMemoryDRNContinue(1, 1, floor, ownedState);
            Assert.InRange(MathF.Abs(expected[time] - arrayOutput.Data[0]), 0f, 1e-6f);
            Assert.InRange(MathF.Abs(expected[time] - ownedOutput.Data[0]), 0f, 1e-6f);
        }
        Assert.InRange(MathF.Abs(finalState - arrayState[0]), 0f, 1e-6f);
        Assert.InRange(MathF.Abs(finalState - ownedState.HostForCpuMutation()[0]), 0f, 1e-6f);
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(0.5f)]
    [InlineData(0.99f)]
    public void GateGradientIncludesOneMinusFloorAndMatchesFiniteDifference(float floor)
    {
        float[] values = Enumerable.Range(0, 3)
            .SelectMany(_ => new float[] { 0.8f, 0.6f, 0.3f, 0f, 0.2f }).ToArray();
        var input = new Tensor(values, [1, 3, 5], dtype: TensorDType.Float32);
        Tensor result = input.ForgetMemoryDRN(1, 1, floor);
        result.Backward([0f, 0f, 1f]);
        double q = Normalize(Math.Tanh(values[0]));
        double k = Normalize(Math.Tanh(values[1]));
        double firstState = Sigmoid(values[4]) * Math.Tanh(values[2]) * k;
        float expected = (float)(q * firstState * (1d - floor) * 0.25d);
        Assert.InRange(MathF.Abs(expected - input.Grad[8]), 0f, 2e-7f);
        Assert.Equal(0f, input.Grad[3]); // No previous memory at first write.
        Assert.Equal(0f, input.Grad[13]); // Final write is never read.

        const float epsilon = 0.001f;
        using var noGrad = AutogradContext.NoGrad();
        values[8] = epsilon;
        float positive = new Tensor(values, [1, 3, 5], dtype: TensorDType.Float32)
            .ForgetMemoryDRN(1, 1, floor).Data[2];
        values[8] = -epsilon;
        float negative = new Tensor(values, [1, 3, 5], dtype: TensorDType.Float32)
            .ForgetMemoryDRN(1, 1, floor).Data[2];
        float numerical = (positive - negative) / (2f * epsilon);
        Assert.InRange(MathF.Abs(numerical - input.Grad[8]), 0f, 2e-5f);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(16)]
    [InlineData(32)]
    public void GptDistributesExistingDefaultFloorsAcrossLayers(int layerCount)
    {
        var model = new ForgetMemoryDRNGpt(16, 4, 4, 8, layerCount,
            keyWidth: 2, valueWidth: 2, random: new Random(29), dtype: TensorDType.Float32);
        Assert.Equal(0.5f, model.RetentionMinimum);
        Assert.Equal(0.99f, model.RetentionMaximum);
        for (int layer = 0; layer < layerCount; layer++)
        {
            float expected = layerCount == 1 ? 0.99f : 0.5f + 0.49f * layer / (layerCount - 1f);
            Assert.Equal(expected, model.Layers[layer].RetentionFloor, precision: 7);
            Assert.True(model.Layers[layer].UseDrn);
        }
    }

    private static (float[] Output, float FinalState) Reference(float[] projected, float floor, float initialState)
    {
        double state = initialState;
        float[] output = new float[projected.Length / 5];
        for (int time = 0; time < output.Length; time++)
        {
            int offset = time * 5;
            double q = Normalize(Math.Tanh(projected[offset]));
            double k = Normalize(Math.Tanh(projected[offset + 1]));
            double value = Math.Tanh(projected[offset + 2]);
            double retention = floor + (1d - floor) * Sigmoid(projected[offset + 3]);
            double beta = Sigmoid(projected[offset + 4]);
            output[time] = (float)(state * q);
            state = retention * state + beta * (value - state * k) * k;
        }
        return (output, (float)state);
    }

    private static double Normalize(double value) => value / Math.Sqrt(value * value + 1e-8d);
    private static double Sigmoid(double value) => 1d / (1d + Math.Exp(-value));

    private static void AssertClose(IReadOnlyList<float> expected, IReadOnlyList<float> actual, float tolerance)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int index = 0; index < expected.Count; index++)
            Assert.InRange(MathF.Abs(expected[index] - actual[index]), 0f, tolerance);
    }
}
