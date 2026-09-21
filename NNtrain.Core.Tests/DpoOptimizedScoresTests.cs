using NNtrain;
using NNtrain.Runtime.Execution;
using NNtrain.Cuda.Execution;
using Xunit;

public sealed class DpoOptimizedScoresTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    [InlineData(true, true)]
    public void CompactFrozenScoresPreserveLossAndAdapterGradients(bool cuda, bool float32 = false)
    {
        Assert.SkipWhen(cuda && !Tensor.IsCudaAvailable(), "CUDA unavailable");
        using var session = new ExecutionSession(new ExecutionOptions {
            Device = cuda ? ExecutionDeviceKind.Cuda : ExecutionDeviceKind.Cpu,
            CudaDevices = new DeviceSet([0]), Precision = PrecisionPolicy.Parse(cuda && !float32 ? "mix16_32" : "float32")
        }, cuda ? [CudaExecutionLaneFactory.Create(0)] : []);
        using var scope = session.Enter();
        var model = new ForgetMemoryDRNGpt(32, 8, 8, 16, 2, 2, 2, random: new Random(12), dtype: TensorDType.Float32);
        model.to(cuda && !float32 ? TensorPrecisionMode.Mix16_32 : TensorPrecisionMode.Float32);
        var adapters = model.AttachLora(2, 4);
        if (cuda) model.to(TensorDevice.Cuda);
        int[] input = [1, 3, 4, 5, 1, 6, 7, 0];
        int[][] labels = [[-1, 4, 5, 2], [6, 7, -1, -1]];
        Tensor[] Legacy()
        {
            var logits = model.forward(input, 2, 4);
            return labels.Select((row, i) => logits.Slice(0, i * 4, 4).CrossEntropyWithLogits(row)).ToArray();
        }
        // Nonzero B makes gradients to both adapter matrices observable.
        var opt = new AdamW(adapters.parameters(), new AdamWOptions { LearningRate = .01f });
        try
        {
            Tensor.DpoLoss(Legacy(), [3, 2], [3, 3], .1f).BackwardAndRelease(); opt.step(); model.ZeroGrad();
            var legacy = Tensor.DpoLoss(Legacy(), [3, 2], [3, 3], .1f);
            float oldLoss = legacy.item(); legacy.BackwardAndRelease();
            float[][] gradients = adapters.parameters().Select(p => p.T.Grad.ToArray()).ToArray();
            model.ZeroGrad(); model.FreezeLoraBaseLinear();
            var compact = Tensor.DpoLoss(model.ForwardCompletionScores(input, labels, 4), [3, 2], [3, 3], .1f);
            Assert.InRange(Math.Abs(compact.item() - oldLoss), 0, cuda ? 1e-4f : 1e-6f);
            compact.BackwardAndRelease();
            foreach (var (old, p) in gradients.Zip(adapters.parameters()))
                foreach (var (a, b) in old.Zip(p.T.Grad)) Assert.InRange(Math.Abs(a - b), 0, cuda ? 2e-4f : 1e-6f);
        }
        finally { opt.DisposeCudaResources(); }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void FrozenLinearRetainsInputGradientWithoutAllocatingWeightGradients(bool cuda, bool relu)
    {
        Assert.SkipWhen(cuda && !Tensor.IsCudaAvailable(), "CUDA unavailable");
        using var session = new ExecutionSession(new ExecutionOptions {
            Device = cuda ? ExecutionDeviceKind.Cuda : ExecutionDeviceKind.Cpu, CudaDevices = new DeviceSet([0]),
            Precision = PrecisionPolicy.Parse(cuda ? "mix16_32" : "float32")
        }, cuda ? [CudaExecutionLaneFactory.Create(0)] : []);
        using var scope = session.Enter();
        var dtype = cuda ? TensorDType.BFloat16 : TensorDType.Float32;
        var input = new Tensor([.3f, -.2f, .6f, -.8f], [2, 2], dtype: dtype);
        var weight = new Tensor([.5f, .2f, -.4f, .7f], [2, 2], dtype: dtype);
        var bias = new Tensor([.1f, -.1f], [2], dtype: dtype);
        input.LinearLastDimFrozen(weight, bias, relu).BackwardAndRelease([1f, 1f, 1f, 1f]);
        float[] expected = input.Grad.ToArray();
        Assert.False(weight.HasGradientBuffer); Assert.False(bias.HasGradientBuffer);
        input.ZeroGrad();
        input.LinearLastDim(weight, bias, relu).BackwardAndRelease([1f, 1f, 1f, 1f]);
        Assert.Equal(expected, input.Grad.ToArray());
        float[] unscaled = input.Grad.ToArray();
        input.ScaleDpoGradient(2);
        Assert.Equal(unscaled.Select(x => x * 2), input.Grad.ToArray());
    }
}
