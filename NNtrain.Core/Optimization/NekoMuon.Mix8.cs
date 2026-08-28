using NNtrain.Runtime.Execution;

namespace NNtrain;

public sealed partial class NekoMuon
{
    private static bool UsesMix8Parameter(Tensor tensor)
        => tensor.DType == TensorDType.Bfp8
            && tensor.Bfp8Quantization?.Granularity
                == Bfp8ScaleGranularity.Block;

    private static void ValidateMix8OptimizerContract()
    {
        PrecisionPolicy? policy = TensorExecutionContext.ActivePrecisionPolicy;
        if (policy is null)
            return;
        if (policy.Mode == PrecisionMode.Mix8_32
            && policy.Gradient == NumericFormat.Float32
            && policy.OptimizerState == NumericFormat.Float32
            && policy.MasterWeight == NumericFormat.Float32)
        {
            return;
        }

        throw new InvalidOperationException(
            "Block-scaled BFP8 parameter storage requires the mix8_32 " +
            "optimizer contract (FP32 gradient, optimizer state, and " +
            "master weight). The active precision policy is " +
            $"'{policy}'.");
    }

    private static void ThrowIfMix8PublicationNonFinite(
        IReadOnlyList<CudaOptimizerFiniteStatusReadback> readbacks,
        IReadOnlyList<int> devices,
        int step)
    {
        int nonFiniteDevice = -1;
        for (int deviceSlot = 0; deviceSlot < devices.Count; deviceSlot++)
        {
            int finite = readbacks[deviceSlot].ReadAfterSynchronization();
            if (finite != 0 && nonFiniteDevice < 0)
                nonFiniteDevice = devices[deviceSlot];
        }
        if (nonFiniteDevice >= 0)
        {
            throw new InvalidOperationException(
                $"Non-finite CUDA value detected while publishing " +
                $"mix8_32 NekoMuon parameters on device " +
                $"{nonFiniteDevice} at optimizer step {step}.");
        }
    }

    internal (NativeCudaBuffer<float> Fast, NativeCudaBuffer<float> Slow)
        GetCudaMix8Moments(int parameterIndex, int deviceIndex)
    {
        CudaOptimizerKernels.NekoMuonResidentState state =
            _cudaStates[parameterIndex]
            ?? throw new InvalidOperationException(
                "The NekoMuon parameter has no resident FP32 mixed state.");
        CudaOptimizerKernels.NekoMuonResidentState.NekoBuffers buffers =
            state.GetOrCreate(deviceIndex);
        return (buffers.Fast, buffers.Slow);
    }
}
