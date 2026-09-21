using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Interop;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaBfp8DirectElementwiseTests
{
    [Theory]
    [InlineData(32, 1, false)]
    [InlineData(32, 515, false)]
    [InlineData(128, 4099, false)]
    [InlineData(32, 515, true)]
    [InlineData(128, 4099, true)]
    public void FusedPathMatchesLegacyValuesAndGradients(int block, int length, bool graphRng)
    {
        Assert.SkipWhen(!Tensor.IsCudaAvailable(), "CUDA unavailable.");
        Assert.True(CudaNativeGateway.AbiVersion.Minor >= CudaAbiVersion.DirectBfp8ElementwiseMinor);
        TensorDevice previous = Tensor.ExecutionDevice;
        int[] devices = Tensor.CudaDeviceIndices.ToArray();
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
            var lane = (CudaExecutionLane)session.GetRequiredLane(ExecutionDeviceKind.Cuda, 0);
            using var rng = CudaGraphRngState.Create(lane, 13);
            for (int operation = 0; operation < 3; operation++)
            {
                (float[] Values, float[] LeftGrad, float[] RightGrad) Run(bool fused, ulong counter = 13)
                {
                    using IDisposable policy = CudaDispatchPolicy.Push(CudaDispatchPolicy.Defaults with
                    { DisableDirectBfp8Elementwise = !fused });
                    // Match the public operation's descriptor contract, including tails.
                    Tensor left = Tensor.FromBfp8(Enumerable.Range(0, length).Select(i => MathF.Sin(i * 0.17f)).ToArray(),
                        [length], Bfp8QuantizationDescriptor.Block(block));
                    Tensor right = Tensor.FromBfp8(Enumerable.Range(0, length).Select(i => MathF.Cos(i * 0.13f)).ToArray(),
                        [length], Bfp8QuantizationDescriptor.Block(block));
                    left.to(TensorDevice.Cuda); right.to(TensorDevice.Cuda);
                    rng.SetCounter(counter);
                    Tensor result;
                    using (graphRng ? CudaGraphDropoutCaptureScope.Begin(rng, 731) : null)
                        result = operation switch
                        {
                            0 => left.Dropout(0.2f, new Random(451)),
                            1 => left.AddDropout(right, 0.2f, new Random(451)),
                            _ => left + right,
                        };
                    float[] values = result.Data.ToArray();
                    result.BackwardAndRelease(Enumerable.Repeat(1f, length).ToArray());
                    float[] lg = left.Grad.ToArray();
                    float[] rg = operation == 0 ? [] : right.Grad.ToArray();
                    result.InvalidateCudaBuffers(); left.InvalidateCudaBuffers(); right.InvalidateCudaBuffers();
                    return (values, lg, rg);
                }
                var reference = Run(false);
                var fused = Run(true);
                Assert.Equal(reference.Values, fused.Values);
                Assert.Equal(reference.LeftGrad, fused.LeftGrad);
                Assert.Equal(reference.RightGrad, fused.RightGrad);
                if (graphRng && operation != 2)
                    Assert.False(fused.Values.SequenceEqual(Run(true, 14).Values));
            }
        }
        finally { Tensor.CudaDeviceIndices = devices; Tensor.ExecutionDevice = previous; }
    }
}
