using NNtrain;
using NNtrain.Cuda.Execution;
using Xunit;

public sealed class CudaDrnTensorCoreLayoutTests
{
    [Theory]
    [InlineData(48, 32, 0)]
    [InlineData(48, 32, 17)]
    [InlineData(48, 32, 47)]
    [InlineData(16, 16, 0)]
    [InlineData(16, 32, 15)]
    [InlineData(32, 16, 17)]
    [InlineData(32, 32, 31)]
    [InlineData(32, 48, 5)]
    [InlineData(32, 128, 31)]
    public void DrnContinuationReadsExactQueryColumnFromNonzeroState(
        int keyWidth, int valueWidth, int queryIndex)
    {
        Assert.SkipWhen(!Tensor.IsCudaAvailable(), "CUDA is unavailable.");
        CudaKernelCapabilities capabilities =
            NativeCudaRuntime.GetKernelCapabilities(0);
        Assert.SkipWhen(!capabilities.Supports(
            CudaKernelFeature.TensorCores | CudaKernelFeature.BFloat16),
            "BF16 Tensor Cores are unavailable.");

        int projectionWidth = 2 * keyWidth + 3 * valueWidth;
        int matrixSize = keyWidth * valueWidth;
        int keyIndex = (queryIndex + 7) % keyWidth;
        var projectionBits = new ushort[projectionWidth];
        projectionBits[queryIndex] = 0x4100; // BF16 +8, positive Q basis.
        projectionBits[keyWidth + keyIndex] = 0xC100; // BF16 -8, distinct K.

        // The epsilon in DRN normalization cannot change this BF16 operand:
        // tanh(8) / sqrt(tanh(8)^2 + 1e-8) rounds to exactly +1.
        float query = MathF.Tanh(8f);
        Assert.Equal(1f, TensorStorageCodec.RoundToBFloat16(
            query / MathF.Sqrt(query * query + 1e-8f)));

        var initialState = new float[matrixSize];
        var expectedBits = new ushort[valueWidth];
        for (int row = 0; row < valueWidth; row++)
        {
            for (int key = 0; key < keyWidth; key++)
            {
                // Distinct, nonzero, exactly representable BF16 values in
                // every row/column expose both tile stride and Q/K swaps.
                ushort bits = (ushort)(0x3D00 + row * keyWidth + key);
                initialState[row * keyWidth + key] =
                    BitConverter.Int32BitsToSingle(bits << 16);
                if (key == queryIndex)
                    expectedBits[row] = bits;
            }
        }

        using IDisposable device = TensorExecutionContext.Push(
            new TorchDevice(TensorDevice.Cuda, 0));
        using IDisposable dispatch = CudaDispatchPolicy.Push(
            CudaDispatchPolicy.Current with
            {
                DisableTensorCoreForgetMemory = false,
            });
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(0);
        using NativeCudaBuffer<ushort> projected =
            accelerator.Allocate1D(projectionBits);
        using NativeCudaBuffer<ushort> output =
            accelerator.Allocate1D<ushort>(valueWidth);
        using NativeCudaBuffer<float> savedStates =
            accelerator.Allocate1D<float>(matrixSize);
        using NativeCudaBuffer<float> state =
            accelerator.Allocate1D(initialState);

        // Supply a nonzero resident state to the actual Tensor Core entry
        // point: a generic CUDA/CPU fallback must not satisfy this test.
        Assert.True(CudaForgetMemoryTensorCore.TryForward(
            accelerator, projected, output, savedStates, state,
            batch: 1, sequence: 1, projectionWidth, keyWidth, valueWidth,
            retentionFloor: 0.37f, memoryVariant: 2));
        accelerator.Synchronize();
        var actualBits = new ushort[valueWidth];
        output.CopyToCPU(actualBits);

        // DRN reads before writing. The basis query selects exactly one
        // original state column, independently of K, gates, beta and value.
        Assert.Equal<ushort>(expectedBits, actualBits);
    }
}
