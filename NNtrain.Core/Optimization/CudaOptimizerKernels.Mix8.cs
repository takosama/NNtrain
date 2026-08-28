namespace NNtrain;

internal static partial class CudaOptimizerKernels
{
    internal static void AccumulateAdamWMix8FiniteStatus(
        Tensor parameter,
        int deviceIndex,
        AdamWResidentState state,
        NativeCudaBuffer<int> finiteStatus)
    {
        NativeCudaBuffer<float> gradient =
            parameter.EnsureCudaGradientBuffer(deviceIndex);
        AdamWResidentState.Buffers moments = state.GetOrCreate(deviceIndex);
        CudaOptimizerNative.AccumulateFiniteStatus(
            deviceIndex,
            gradient.NativePtr,
            gradient.Length,
            finiteStatus.NativePtr);
        CudaOptimizerNative.AccumulateFiniteStatus(
            deviceIndex,
            moments.First.NativePtr,
            moments.First.Length,
            finiteStatus.NativePtr);
        CudaOptimizerNative.AccumulateFiniteStatus(
            deviceIndex,
            moments.Second.NativePtr,
            moments.Second.Length,
            finiteStatus.NativePtr);
    }

    internal static void PublishMix8Master(
        Tensor parameter,
        int deviceIndex,
        NativeCudaBuffer<int> finiteStatus)
    {
        ArgumentNullException.ThrowIfNull(parameter);
        ArgumentNullException.ThrowIfNull(finiteStatus);
        if (parameter.DType != TensorDType.Bfp8
            || parameter.Bfp8Quantization?.Granularity
                != Bfp8ScaleGranularity.Block)
        {
            throw new InvalidOperationException(
                "mix8_32 publication requires block-scaled BFP8 storage.");
        }
        if (finiteStatus.Device.Index != deviceIndex
            || finiteStatus.Length != 1)
        {
            throw new ArgumentException(
                "The mix8_32 finite status must be one scalar on the " +
                "parameter device.",
                nameof(finiteStatus));
        }

        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<float> master =
            parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
        CudaBfp8BufferView encoded =
            parameter.EnsureCudaBfp8Buffer(deviceIndex);
        nint stream = accelerator.DefaultStream;
        CudaOptimizerNative.AccumulateFiniteStatus(
            deviceIndex,
            master.NativePtr,
            master.Length,
            finiteStatus.NativePtr);
        CudaBfp8Native.QuantizeFloat32(
            deviceIndex,
            master,
            encoded.Payload,
            encoded.Scales,
            encoded.Descriptor,
            stream);
    }
}
