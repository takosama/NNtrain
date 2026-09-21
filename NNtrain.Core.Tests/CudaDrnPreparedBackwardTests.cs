using NNtrain;
using NNtrain.Cuda.Interop;
using Xunit;

public sealed class CudaDrnPreparedBackwardTests
{
    [Theory]
    [InlineData(32, 32, 257)]
    [InlineData(17, 19, 129)]
    [InlineData(16, 128, 33)]
    public void PreparedBackwardPreservesExistingAndRecurrentGradients(int key, int value, int sequence)
    {
        Assert.SkipWhen(!Tensor.IsCudaAvailable(), "CUDA is unavailable.");
        Assert.True(CudaNativeGateway.AbiVersion.Minor >= CudaAbiVersion.DrnPreparedBackwardMinor);
        using var device = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cuda, 0));
        var accelerator = ForgetMemoryV2Cuda.GetAccelerator(0);
        const int batch = 8;
        int width = 2 * key + 3 * value;
        var rng = new Random(831);
        float[] Values(int count) => Enumerable.Range(0, count)
            .Select(_ => (float)(rng.NextDouble() - .5) * .6f).ToArray();
        using var projected = accelerator.Allocate1D(Values(batch * sequence * width)
            .Select(TensorStorageCodec.EncodeBFloat16).ToArray());
        using var output = accelerator.Allocate1D<ushort>(batch * sequence * value);
        using var history = accelerator.Allocate1D<float>(batch * sequence * key * value);
        using var state = accelerator.Allocate1D<float>(batch * key * value);
        state.MemSetToZero();
        CudaForgetMemoryNative.Forward(accelerator, 0, projected.NativePtr, 0, output.NativePtr,
            history.NativePtr, state.NativePtr, batch, sequence, width, key, value, .37f, 2, true);
        using var dy = accelerator.Allocate1D(Values(output.Length));
        float[] initialProjection = Values(projected.Length), initialState = Values(state.Length);
        using var oldProjection = accelerator.Allocate1D(initialProjection);
        using var newProjection = accelerator.Allocate1D(initialProjection);
        using var oldState = accelerator.Allocate1D(initialState);
        using var newState = accelerator.Allocate1D(initialState);
        using var oldPrevious = accelerator.Allocate1D<float>(state.Length);
        using var newPrevious = accelerator.Allocate1D<float>(state.Length);
        int length = CudaForgetMemoryNative.PreparedBackwardScratchLength(batch, sequence, key, value);
        Assert.True(length > 0);
        using var prepared = accelerator.Allocate1D<float>(length);
        CudaForgetMemoryNative.Backward(accelerator, 0, projected.NativePtr, oldProjection.NativePtr,
            dy.NativePtr, history.NativePtr, oldState.NativePtr, oldPrevious.NativePtr,
            batch, sequence, width, key, value, .37f, 2, true);
        CudaForgetMemoryNative.Backward(accelerator, 0, projected.NativePtr, newProjection.NativePtr,
            dy.NativePtr, history.NativePtr, newState.NativePtr, newPrevious.NativePtr,
            batch, sequence, width, key, value, .37f, 2, true, prepared.NativePtr);
        accelerator.Synchronize();
        Compare(oldProjection, newProjection);
        Compare(oldState, newState);
        Compare(oldPrevious, newPrevious);
    }

    private static void Compare(NativeCudaBuffer<float> expectedBuffer, NativeCudaBuffer<float> actualBuffer)
    {
        var expected = new float[expectedBuffer.Length];
        var actual = new float[actualBuffer.Length];
        expectedBuffer.CopyToCPU(expected); actualBuffer.CopyToCPU(actual);
        for (int i = 0; i < actual.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]));
            Assert.InRange(Math.Abs(actual[i] - expected[i]), 0f, 6e-5f);
        }
    }
}
