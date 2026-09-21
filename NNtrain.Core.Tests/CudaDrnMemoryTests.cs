using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaDrnMemoryTests
{
    [Fact]
    public void PersistentSessionBuffersCannotEscapeIntoLegacyPools()
    {
        Assert.SkipWhen(!Tensor.IsCudaAvailable(), "CUDA unavailable.");
        var device = ForgetMemoryV2Cuda.GetAccelerator(0);
        using (var session = new ExecutionSession(new ExecutionOptions
        {
            Device = ExecutionDeviceKind.Cuda, CudaDevices = new DeviceSet([0]),
        }, [CudaExecutionLaneFactory.Create(0)]))
        using (session.Enter())
        {
            var f = device.Allocate1D<float>(37);
            var b = device.Allocate1D<ushort>(37);
            var i = device.Allocate1D<int>(37);
            Assert.True(f.SessionGeneration > 0);
            Tensor.ReturnCudaFloatBuffer(device, f);
            Tensor.ReturnCudaBFloat16Buffer(device, b);
            Tensor.ReturnCudaIntBuffer(device, i);
            Assert.False(f.IsAlive); Assert.False(b.IsAlive); Assert.False(i.IsAlive);
        }
        using var rf = Tensor.RentCudaFloatBuffer(0, 37);
        using var rb = Tensor.RentCudaBFloat16Buffer(0, 37);
        using var ri = Tensor.RentCudaIntBuffer(0, 37);
        rf.MemSetToZero(); rb.MemSetToZero(); ri.MemSetToZero();
    }

    [Theory]
    [InlineData(0, 16, 1024, 0)]
    [InlineData(256, 16, 1024, 16)]
    [InlineData(256, 17, 1024, 15)]
    [InlineData(256, 32, 1024, 8)]
    [InlineData(256, 16, 2048, 8)]
    [InlineData(256, 1, 33, 32)]
    public void RetainedHistoryUsesActualShardShapeAndBoundedBudget(
        int mib, int batch, int sequence, int expected)
        => Assert.Equal(expected, ForgetMemoryV2Gpt.RetainedHistoryLayerCount(
            (long)mib * 1024 * 1024, batch, sequence, 16, 16, 32));

    [Theory]
    [InlineData(0f)]
    [InlineData(0.2f)]
    public void RetiringExclusiveLinearOutputsPreservesLossAndGradients(float dropout)
    {
        Assert.SkipWhen(!Tensor.IsCudaAvailable(), "CUDA is unavailable.");
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            (float Loss, float[][] Gradients, long ActiveBytes) Run(bool retire, bool recompute = false, long budget = 0)
            {
                using var session = new ExecutionSession(new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet([0]), Precision = PrecisionPolicy.Mix8_32,
                }, [CudaExecutionLaneFactory.Create(0)]);
                using IDisposable scope = session.Enter();
                using IDisposable policy = CudaDispatchPolicy.Push(CudaDispatchPolicy.Defaults with
                {
                    DisableExclusiveLinearOutputRetirement = !retire,
                    DisableDrnStateRecomputation = !recompute,
                    DrnRetainedHistoryBudgetBytes = budget,
                });
                var model = new ForgetMemoryDRNGpt(64, 33, 32, 64, 2,
                    keyWidth: 16, valueWidth: 16, dropout: dropout, random: new Random(765));
                model.to(TensorPrecisionMode.Mix8_32);
                model.to(TensorDevice.Cuda);
                int[] tokens = Enumerable.Range(0, 3 * 33).Select(i => 3 + i % 61).ToArray();
                Tensor loss = model.ForwardLoss(tokens, tokens.Select(i => (i + 1) % 64).ToArray(), 3, 33);
                var lane = (CudaExecutionLane)session.GetRequiredLane(ExecutionDeviceKind.Cuda, 0);
                long memory = lane.Memory.ActiveBytes;
                float lossValue = loss.item();
                loss.BackwardAndRelease();
                float[][] gradients = model.Parameters().Select(p => p.T.Grad.ToArray()).ToArray();
                return (lossValue, gradients, memory);
            }
            var saved = Run(false);
            var compact = Run(true);
            var partial = Run(true, recompute: true, budget: 3 * 33 * 16 * 16 * sizeof(float));
            Assert.Equal(saved.Loss, partial.Loss);
            Assert.True(partial.ActiveBytes < compact.ActiveBytes);
            Assert.Equal(saved.Loss, compact.Loss);
            Assert.True(compact.ActiveBytes < saved.ActiveBytes,
                $"Retirement did not reduce live allocations: {compact.ActiveBytes}/{saved.ActiveBytes}.");
            Assert.Equal(saved.Gradients.Length, compact.Gradients.Length);
            for (int p = 0; p < saved.Gradients.Length; p++)
            {
                Assert.Equal(saved.Gradients[p].Length, compact.Gradients[p].Length);
                for (int i = 0; i < saved.Gradients[p].Length; i++)
                {
                    Assert.InRange(MathF.Abs(saved.Gradients[p][i] - compact.Gradients[p][i]), 0f, 1e-5f);
                    Assert.InRange(MathF.Abs(saved.Gradients[p][i] - partial.Gradients[p][i]), 0f, 1e-5f);
                }
            }
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void RetirementRejectsReluAndValueReuseButPreservesBackward()
    {
        Assert.SkipWhen(!Tensor.IsCudaAvailable(), "CUDA is unavailable.");
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            using var session = new ExecutionSession(new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda, CudaDevices = new DeviceSet([0]),
                Precision = PrecisionPolicy.Mix8_32,
            }, [CudaExecutionLaneFactory.Create(0)]);
            using IDisposable scope = session.Enter();
            Tensor input = Tensor.FromBfp8([0.3f, -0.2f], [1, 2], Bfp8QuantizationDescriptor.Mix8_32);
            Tensor weight = Tensor.FromBfp8([0.2f, 0.7f], [1, 2], Bfp8QuantizationDescriptor.Mix8_32);
            Tensor bias = Tensor.FromBfp8([0.1f], [1], Bfp8QuantizationDescriptor.Mix8_32);
            input.to(TensorDevice.Cuda);
            weight.to(TensorDevice.Cuda);
            bias.to(TensorDevice.Cuda);
            Tensor relu = input.LinearLastDim(weight, bias, applyRelu: true);
            Assert.Throws<InvalidOperationException>(relu.RetireExclusiveCudaLinearOutputValues);
            relu.BackwardAndRelease([1f]);
            input.ZeroGrad(); weight.ZeroGrad(); bias.ZeroGrad();
            Tensor linear = input.LinearLastDim(weight, bias, applyRelu: false);
            linear.RetireExclusiveCudaLinearOutputValues();
            linear.RetireExclusiveCudaLinearOutputValues(); // Idempotent.
            Assert.Throws<InvalidOperationException>(() => linear.Data[0]);
            Assert.Throws<InvalidOperationException>(() => linear.EnsureCudaBfp8Buffer(0));
            linear.BackwardAndRelease([1f]);
            Assert.Equal(1f, bias.Grad[0]);
            Assert.Contains(input.Grad, value => value != 0f);
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }
}
