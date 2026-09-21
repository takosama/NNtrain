using NNtrain;
using Xunit;

public sealed class DpoLossTests
{
    [Theory]
    [InlineData(0f)]
    [InlineData(3f)]
    [InlineData(-3f)]
    [InlineData(1000f)]
    [InlineData(-1000f)]
    public void StableLossAndAnalyticGradient(float offset)
    {
        using var scope = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cpu, 0));
        Tensor c = new([2 + offset], [1]), r = new([3f], [1]);
        Tensor loss = Tensor.DpoLoss([c, r], [2, 3], [2, 3], .1f);
        double margin = -.2 * offset;
        double expected = Math.Max(-margin, 0) + Math.Log(1 + Math.Exp(-Math.Abs(margin)));
        Assert.InRange(Math.Abs(loss.item() - expected), 0, 1e-5);
        loss.Backward();
        double s = 1 / (1 + Math.Exp(margin));
        Assert.InRange(Math.Abs(c.Grad[0] - .2 * s), 0, 1e-6);
        Assert.InRange(Math.Abs(r.Grad[0] + .3 * s), 0, 1e-6);
    }

    [Fact]
    public void BatchedLossMatchesFiniteDifference()
    {
        using var scope = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cpu, 0));
        float[] values = [1.4f, 2.1f, 1.8f, 1.2f];
        int[] counts = [2, 5, 3, 1]; float[] reference = [1.5f, 2f, 1.7f, 1.3f];
        Tensor[] ce = values.Select(v => new Tensor([v], [1])).ToArray();
        Tensor loss = Tensor.DpoLoss(ce, counts, reference, .1f);
        loss.Backward();
        for (int i = 0; i < values.Length; i++)
        {
            float Eval(float delta) => Tensor.DpoLoss(values.Select((v, j) =>
                new Tensor([v + (i == j ? delta : 0)], [1])).ToArray(), counts, reference, .1f).item();
            float numerical = (Eval(.001f) - Eval(-.001f)) / .002f;
            Assert.InRange(Math.Abs(ce[i].Grad[0] - numerical), 0, 5e-5);
        }
    }
}
