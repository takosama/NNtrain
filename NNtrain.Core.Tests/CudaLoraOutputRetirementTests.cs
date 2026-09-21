using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaLoraOutputRetirementTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void PrivateBaseOutputRetirementPreservesLoraAndInputGradients(bool frozenBase, bool relu)
    {
        Assert.SkipWhen(!Tensor.IsCudaAvailable(), "CUDA unavailable.");
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            (float[] Values, float[] InputGrad, float[] AGrad, float[] BGrad,
                float[] BaseGrad, float[] BiasGrad, long ActiveBytes) Run(bool retire)
            {
                using var session = new ExecutionSession(new ExecutionOptions {
                    Device = ExecutionDeviceKind.Cuda, CudaDevices = new DeviceSet([0]),
                    Precision = PrecisionPolicy.Mix8_32 }, [CudaExecutionLaneFactory.Create(0)]);
                using var scope = session.Enter();
                using var policy = CudaDispatchPolicy.Push(CudaDispatchPolicy.Defaults with {
                    DisableCudaGraphs = true, DisableExclusiveLinearOutputRetirement = !retire });
                const int rows = 64, inputWidth = 32, outputWidth = 96;
                var linear = new Linear(inputWidth, outputWidth, new Random(71), dtype: TensorDType.Float32);
                linear.to(TensorPrecisionMode.Mix8_32, 32);
                var adapter = linear.AttachLora(4, 8, new Random(73));
                using (var update = adapter.B.BeginUpdate())
                    for (int i = 0; i < update.Values.Length; i++)
                        update.Values[i] = .03f * MathF.Sin(i * .19f);
                linear.FrozenForLora = frozenBase;
                linear.to(TensorDevice.Cuda);
                var input = Tensor.FromBfp8(Enumerable.Range(0, rows * inputWidth)
                    .Select(i => MathF.Cos(i * .13f)).ToArray(), [rows, inputWidth],
                    Bfp8QuantizationDescriptor.Block(32));
                input.to(TensorDevice.Cuda);
                Tensor output = relu ? linear.ForwardBatchRelu(input) : linear.ForwardBatch(input);
                var lane = (CudaExecutionLane)session.GetRequiredLane(ExecutionDeviceKind.Cuda, 0);
                lane.SynchronizeComputeStream();
                long activeBytes = lane.Memory.ActiveBytes;
                float[] values = output.Data.ToArray();
                output.BackwardAndRelease(Enumerable.Range(0, output.Numel)
                    .Select(i => .5f + MathF.Sin(i * .07f)).ToArray());
                if (frozenBase)
                {
                    Assert.False(linear.W.T.HasGradientBuffer);
                    Assert.False(linear.B.T.HasGradientBuffer);
                }
                return (values, input.Grad.ToArray(), adapter.A.T.Grad.ToArray(), adapter.B.T.Grad.ToArray(),
                    frozenBase ? [] : linear.W.T.Grad.ToArray(), frozenBase ? [] : linear.B.T.Grad.ToArray(), activeBytes);
            }

            var baseline = Run(retire: false);
            var compact = Run(retire: true);
            Assert.Equal(baseline.Values, compact.Values);
            Assert.Equal(baseline.InputGrad, compact.InputGrad);
            Assert.Equal(baseline.AGrad, compact.AGrad);
            Assert.Equal(baseline.BGrad, compact.BGrad);
            Assert.Equal(baseline.BaseGrad, compact.BaseGrad);
            Assert.Equal(baseline.BiasGrad, compact.BiasGrad);
            Assert.Contains(compact.InputGrad, value => value != 0);
            Assert.Contains(compact.AGrad, value => value != 0);
            Assert.Contains(compact.BGrad, value => value != 0);
            const long twoPrivateOutputs = 2L * 64 * 96 * (32 + sizeof(float)) / 32;
            Assert.True(baseline.ActiveBytes - compact.ActiveBytes >= twoPrivateOutputs,
                $"Base and scaled adapter outputs were not both retired: {compact.ActiveBytes}/{baseline.ActiveBytes}.");
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(.2f)]
    public void FullDrnLoraRetirementPreservesOutputsAndAllGradients(float dropout)
    {
        Assert.SkipWhen(!Tensor.IsCudaAvailable(), "CUDA unavailable.");
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            (float[] Values, float Loss, float[][] Gradients, float[][] AdapterGradients, long ActiveBytes) Run(bool retire)
            {
                using var session = new ExecutionSession(new ExecutionOptions {
                    Device = ExecutionDeviceKind.Cuda, CudaDevices = new DeviceSet([0]),
                    Precision = PrecisionPolicy.Mix8_32 }, [CudaExecutionLaneFactory.Create(0)]);
                using var scope = session.Enter();
                using var policy = CudaDispatchPolicy.Push(CudaDispatchPolicy.Defaults with {
                    DisableCudaGraphs = true, DisableExclusiveLinearOutputRetirement = !retire });
                var model = new ForgetMemoryDRNGpt(64, 64, 32, 96, 2, 16, 16,
                    dropout: dropout, random: new Random(101), dtype: TensorDType.Float32);
                model.to(TensorPrecisionMode.Mix8_32, 32);
                var adapters = model.AttachLora(4, 8, seed: 103);
                foreach (var parameter in adapters.parameters().Where(parameter => parameter.Name == "lora_B"))
                {
                    using var update = parameter.BeginUpdate();
                    for (int i = 0; i < update.Values.Length; i++)
                        update.Values[i] = .03f * MathF.Sin(i * .19f);
                }
                model.FreezeLoraBaseLinear();
                model.to(TensorDevice.Cuda);
                int[] tokens = Enumerable.Range(0, 2 * 64).Select(i => 3 + i % 61).ToArray();
                Tensor output = model.forward(tokens, 2, 64);
                Tensor loss = output.CrossEntropyWithLogits(tokens.Select(token => (token + 1) % 64).ToArray());
                var lane = (CudaExecutionLane)session.GetRequiredLane(ExecutionDeviceKind.Cuda, 0);
                lane.SynchronizeComputeStream();
                long activeBytes = lane.Memory.ActiveBytes;
                float[] values = output.Data.ToArray();
                float lossValue = loss.item();
                loss.BackwardAndRelease();
                float[][] gradients = model.parameters().Select(parameter => parameter.T.HasGradientBuffer
                    ? parameter.T.Grad.ToArray() : Array.Empty<float>()).ToArray();
                float[][] adapterGradients = adapters.parameters().Select(parameter => parameter.T.Grad.ToArray()).ToArray();
                return (values, lossValue, gradients, adapterGradients, activeBytes);
            }

            var baseline = Run(retire: false);
            var compact = Run(retire: true);
            Assert.Equal(baseline.Values, compact.Values);
            Assert.Equal(baseline.Loss, compact.Loss);
            Assert.Equal(baseline.Gradients.Length, compact.Gradients.Length);
            for (int i = 0; i < baseline.Gradients.Length; i++)
                Assert.Equal(baseline.Gradients[i], compact.Gradients[i]);
            Assert.All(compact.AdapterGradients, gradient => Assert.Contains(gradient, value => value != 0f));
            // Per layer: base + scaled branches at widths 32/96/32, plus
            // the two private Add results at the memory and FFN residuals.
            const long minimumRetirementBytes = 2L * 128 * (2 * (32 + 96 + 32) + 2 * 32) * (32 + sizeof(float)) / 32;
            Assert.True(baseline.ActiveBytes - compact.ActiveBytes >= minimumRetirementBytes,
                $"The full DRN did not retire all private LoRA outputs: {compact.ActiveBytes}/{baseline.ActiveBytes}.");
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void PublicBinaryValuesRemainReusableAndOnlyPrivateProvenOutputsCanRetire()
    {
        Assert.SkipWhen(!Tensor.IsCudaAvailable(), "CUDA unavailable.");
        using var session = new ExecutionSession(new ExecutionOptions {
            Device = ExecutionDeviceKind.Cuda, CudaDevices = new DeviceSet([0]),
            Precision = PrecisionPolicy.Mix8_32 }, [CudaExecutionLaneFactory.Create(0)]);
        using var scope = session.Enter();
        using var policy = CudaDispatchPolicy.Push(CudaDispatchPolicy.Defaults with {
            DisableExclusiveLinearOutputRetirement = false });
        var descriptor = Bfp8QuantizationDescriptor.Block(32);
        var input = Tensor.FromBfp8([.25f, -.5f], [2], descriptor);
        var scale = Tensor.FromBfp8([2f], [1], descriptor);
        input.to(TensorDevice.Cuda);
        scale.to(TensorDevice.Cuda);
        Tensor relu = input.Relu();
        Assert.False(relu.SupportsExclusiveCudaOutputRetirement);
        Assert.Throws<InvalidOperationException>(relu.RetireExclusiveCudaOutputValues);
        relu.BackwardAndRelease([1f, 1f]);
        input.ZeroGrad();

        Tensor scaled = input * scale;
        Assert.Equal(scaled.Data.ToArray(), scaled.Data.ToArray());
        Assert.True(scaled.SupportsExclusiveCudaOutputRetirement);
        Assert.Throws<InvalidOperationException>(scaled.RetireExclusiveCudaLinearOutputValues);
        Tensor combined = scaled + input;
        Assert.True(combined.SupportsExclusiveCudaOutputRetirement);
        scaled.RetireExclusiveCudaOutputValues();
        scaled.RetireExclusiveCudaOutputValues();
        Assert.Throws<InvalidOperationException>(() => scaled.Data[0]);
        combined.BackwardAndRelease([1f, 1f]);
        Assert.Equal(new float[] { 3f, 3f }, input.Grad.ToArray());
    }
}
