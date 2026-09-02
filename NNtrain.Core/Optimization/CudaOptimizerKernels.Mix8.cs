using NNtrain.Cuda.Interop;

namespace NNtrain;

internal static partial class CudaOptimizerKernels
{
    internal static void ApplyNekoMuonUpdate(
        Tensor parameter,
        int deviceIndex,
        nint data,
        nint update,
        int length,
        float learningRate,
        float finalScale,
        float weightDecay,
        bool applyWeightDecay,
        NativeCudaBuffer<CudaMix8DiagnosticAccumulator>? diagnostics = null)
    {
        if (diagnostics is null)
        {
            CudaOptimizerNative.NekoApply(
                deviceIndex,
                data,
                update,
                length,
                learningRate,
                finalScale,
                weightDecay,
                applyWeightDecay);
            return;
        }

        if (diagnostics.Device.Index != deviceIndex
            || diagnostics.Length != 1)
        {
            throw new ArgumentException(
                "mix8 diagnostics must be one resident aggregate on the " +
                "parameter device.",
                nameof(diagnostics));
        }
        if (parameter.DType != TensorDType.Bfp8
            || parameter.Bfp8Quantization?.Granularity
                != Bfp8ScaleGranularity.Block)
        {
            throw new InvalidOperationException(
                "NekoMuon mix8 diagnostics require block-scaled BFP8 " +
                "parameter storage.");
        }

        CudaBfp8BufferView encoded =
            parameter.EnsureCudaBfp8Buffer(deviceIndex);
        CudaOptimizerNative.NekoApplyMix8Diagnostic(
            deviceIndex,
            data,
            update,
            encoded.Scales.NativePtr,
            encoded.Descriptor.GetEffectiveBlockSize(length),
            length,
            learningRate,
            finalScale,
            weightDecay,
            applyWeightDecay,
            diagnostics.NativePtr);
    }

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
        NativeCudaBuffer<int> finiteStatus,
        NativeCudaBuffer<CudaMix8DiagnosticAccumulator>? diagnostics = null)
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
        if (diagnostics is null)
        {
            CudaBfp8Native.QuantizeFloat32(
                deviceIndex,
                master,
                encoded.Payload,
                encoded.Scales,
                encoded.Descriptor,
                stream);
        }
        else
        {
            CudaBfp8Native.QuantizeFloat32Diagnostic(
                deviceIndex,
                master,
                encoded.Payload,
                encoded.Scales,
                encoded.Descriptor,
                diagnostics,
                stream);
        }
    }
}
