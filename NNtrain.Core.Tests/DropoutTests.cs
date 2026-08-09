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
