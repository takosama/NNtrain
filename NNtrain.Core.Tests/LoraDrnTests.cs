using NNtrain;
using Xunit;

public sealed class LoraDrnTests
{
    [Fact]
    public void ZeroInitializedAdapterPreservesOutputAndOnlyAdaptersUpdate()
    {
        using var scope = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cpu, 0));
        var model = new ForgetMemoryDRNGpt(32, 8, 8, 16, 2, 2, 2, random: new Random(12), dtype: TensorDType.Float32);
        int[] input = [1, 2, 3, 4], target = [2, 3, 4, 5];
        float[] before;
        using (AutogradContext.NoGrad()) before = model.forward(input, 1, 4).Data.ToArray();
        var baseParameters = model.parameters().ToArray();
        var baseValues = baseParameters.Select(p => p.T.Data.ToArray()).ToArray();
        var adapter = model.AttachLora(2, 4);
        using (AutogradContext.NoGrad()) Assert.Equal(before, model.forward(input, 1, 4).Data);
        var optimizer = new AdamW(adapter.parameters(), new AdamWOptions { LearningRate = .01f, WeightDecay = 0 });
        var initial = adapter.state_dict();
        for (int i = 0; i < 3; i++)
        {
            model.ZeroGrad();
            model.forward_loss(input, target, 1, 4).BackwardAndRelease();
            optimizer.step();
        }
        Assert.Contains(initial.Parameters.Zip(adapter.state_dict().Parameters), p => !p.First.Values.SequenceEqual(p.Second.Values));
        for (int p = 0; p < baseParameters.Length; p++) Assert.Equal(baseValues[p], baseParameters[p].T.Data);
        adapter.SetEnabled(false);
        using (AutogradContext.NoGrad()) Assert.Equal(before, model.forward(input, 1, 4).Data);
        adapter.SetEnabled(true);
        Assert.Throws<InvalidOperationException>(() => model.AttachLora());
    }

    [Fact]
    public void V2AndUnknownTargetsAreRejected()
    {
        var v2 = new ForgetMemoryV2Gpt(32, 8, 8, 16, 1, 2, 2, dtype: TensorDType.Float32);
        Assert.Throws<NotSupportedException>(() => v2.AttachLora());
        var drn = new ForgetMemoryDRNGpt(32, 8, 8, 16, 1, 2, 2, dtype: TensorDType.Float32);
        Assert.Throws<ArgumentException>(() => drn.AttachLora(targets: ["q_proj"]));
        Assert.Throws<ArgumentOutOfRangeException>(() => drn.AttachLora(rank: 0));
    }
}
