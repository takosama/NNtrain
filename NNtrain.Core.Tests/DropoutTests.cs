using NNtrain;
using Xunit;

public sealed class DropoutTests
{
    [Fact]
    public void TrainingDropoutScalesSurvivorsAndBackpropagatesTheSameMask()
    {
        var input = new Tensor(
            Enumerable.Repeat(1f, 64).ToArray(),
            [64]);

        Tensor output = input.Dropout(0.5f, new Random(17));
        output.Backward(Enumerable.Repeat(1f, 64).ToArray());

        Assert.Contains(0f, output.Data);
        Assert.Contains(2f, output.Data);
        Assert.Equal(output.Data, input.Grad);
    }

    [Fact]
    public void EvaluationModeDisablesDropoutRecursively()
    {
        var parent = new DropoutContainer(new Random(23));
        var input = new Tensor(Enumerable.Repeat(1f, 32).ToArray(), [32]);

        parent.Eval();
        Tensor evaluationOutput = parent.Forward(input);
        Assert.False(parent.IsTraining);
        parent.Train();
        Tensor trainingOutput = parent.Forward(input);

        Assert.True(parent.IsTraining);
        Assert.Same(input, evaluationOutput);
        Assert.NotSame(input, trainingOutput);
        Assert.Contains(0f, trainingOutput.Data);
    }

    [Fact]
    public void FusedResidualDropoutUsesTheSameMaskForForwardAndBackward()
    {
        var residual = new Tensor(
            Enumerable.Repeat(3f, 64).ToArray(),
            [64]);
        var branch = new Tensor(
            Enumerable.Repeat(1f, 64).ToArray(),
            [64]);

        Tensor output = residual.AddDropout(
            branch,
            0.5f,
            new Random(31));
        output.Backward(Enumerable.Repeat(1f, 64).ToArray());

        Assert.Contains(3f, output.Data);
        Assert.Contains(5f, output.Data);
        Assert.All(residual.Grad, gradient => Assert.Equal(1f, gradient));
        for (int index = 0; index < output.Numel; index++)
        {
            Assert.Equal(output.Data[index] - 3f, branch.Grad[index]);
        }
    }

    [Fact]
    public void ZeroProbabilityFusedResidualMatchesAddition()
    {
        var residual = new Tensor([1f, 2f, 3f, 4f], [4]);
        var branch = new Tensor([5f, 6f, 7f, 8f], [4]);

        Tensor output = residual.AddDropout(
            branch,
            0f,
            new Random(37));
        output.Backward([1f, 1f, 1f, 1f]);

        Assert.Equal(new[] { 6f, 8f, 10f, 12f }, output.Data);
        Assert.Equal(new[] { 1f, 1f, 1f, 1f }, residual.Grad);
        Assert.Equal(new[] { 1f, 1f, 1f, 1f }, branch.Grad);
    }

    [Fact]
    public void CounterMaskMatchesBetweenScalarAndSimdKernels()
    {
        bool previousSimd = Tensor.SimdEnabled;
        int previousParallelism = Tensor.MaxDegreeOfParallelism;
        try
        {
            float[] values = Enumerable.Range(1, 67)
                .Select(value => (float)value)
                .ToArray();

            Tensor.SimdEnabled = false;
            Tensor.MaxDegreeOfParallelism = 1;
            Tensor scalar = new Tensor(values, [67])
                .Dropout(0.3f, new Random(41));

            Tensor.SimdEnabled = true;
            Tensor simd = new Tensor(values, [67])
                .Dropout(0.3f, new Random(41));

            Assert.Equal(scalar.Data, simd.Data);
        }
        finally
        {
            Tensor.SimdEnabled = previousSimd;
            Tensor.MaxDegreeOfParallelism = previousParallelism;
        }
    }

    [Theory]
    [InlineData(-0.1f)]
    [InlineData(1f)]
    [InlineData(float.NaN)]
    public void RejectsInvalidProbability(float probability)
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new Dropout(probability));
    }

    private sealed class DropoutContainer : Module
    {
        private readonly Dropout _dropout;

        internal DropoutContainer(Random random)
        {
            _dropout = RegisterModule(new Dropout(0.5f, random));
        }

        internal Tensor Forward(Tensor input) => _dropout.Forward(input);
    }
}
