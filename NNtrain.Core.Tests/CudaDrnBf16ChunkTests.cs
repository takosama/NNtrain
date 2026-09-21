using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaDrnBf16ChunkTests
{
    [Theory]
    [InlineData(2, 65, 16, 32)]
    [InlineData(2, 513, 32, 32)]
    [InlineData(21, 1024, 32, 32)]
    public void MixedBf16ResidentChunkMatchesOriginal(int batch, int sequence, int key, int value)
    {
        Assert.SkipWhen(!Tensor.IsCudaAvailable(), "CUDA unavailable.");
        using var session = new ExecutionSession(new ExecutionOptions {
            Device = ExecutionDeviceKind.Cuda, CudaDevices = new DeviceSet([0]),
            Precision = PrecisionPolicy.Mix16_32,
        }, [CudaExecutionLaneFactory.Create(0)]);
        using var scope = session.Enter();
        int width = 2 * key + 3 * value;
        var random = new Random(121);
        float[] input = Enumerable.Range(0, batch * sequence * width)
            .Select(_ => (float)(random.NextDouble() - .5) * .6f).ToArray();
        float[] dy = Enumerable.Range(0, batch * sequence * value)
            .Select(_ => (float)(random.NextDouble() - .5)).ToArray();
        (float[] Output, float[] Gradient) Run(bool chunk)
        {
            using var policy = CudaDispatchPolicy.Push(CudaDispatchPolicy.Defaults with {
                DisableDrnChunkBackward = !chunk });
            var p = new Tensor(input.ToArray(), [batch, sequence, width], dtype: TensorDType.BFloat16);
            p.to(TensorDevice.Cuda);
            using var forward = ForgetMemoryV2Cuda.ForwardResident(p, batch, sequence,
                width, key, value, .5f, true, false, true);
            Assert.Equal(chunk, forward.Chunked);
            var output = new Tensor(new float[dy.Length], [batch, sequence, value], dtype: TensorDType.BFloat16);
            output.to(TensorDevice.Cuda);
            output.EnsureCudaGradientBuffer(0).CopyFromCPU(dy);
            ForgetMemoryV2Cuda.BackwardResident(p, output, forward, batch, sequence,
                width, key, value, .5f, true, false, true);
            var encoded = new ushort[dy.Length];
            forward.OutputBFloat16!.CopyToCPU(encoded);
            forward.OutputBFloat16.Dispose();
            return (encoded.Select(TensorStorageCodec.DecodeBFloat16).ToArray(), p.Grad.ToArray());
        }
        var original = Run(false);
        var candidate = Run(true);
        Assert.Equal(original.Output, candidate.Output);
        for (int i = 0; i < original.Gradient.Length; i++)
            Assert.InRange(MathF.Abs(original.Gradient[i] - candidate.Gradient[i]), 0f, 6e-5f);
    }
}
