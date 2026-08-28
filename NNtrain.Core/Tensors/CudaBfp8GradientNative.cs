using NNtrain.Cuda.Interop;

namespace NNtrain;

/// <summary>
/// Native boundary for pure-BFP8 gradient publication and two-device
/// reduction. Payloads/scales never cross host memory; only the finite-status
/// and squared-norm scalars are read after the communication fence.
/// </summary>
internal static class CudaBfp8GradientNative
{
    internal static void Quantize(
        int deviceIndex,
        NativeCudaBuffer<float> source,
        CudaBfp8BufferView destination,
        NativeCudaBuffer<int> finiteStatus,
        nint stream,
        NativeCudaBuffer<double>? squaredSum = null)
    {
        ValidateTensorWide(destination, deviceIndex, source.Length);
        if (source.Device.Index != deviceIndex
            || finiteStatus.Device.Index != deviceIndex
            || finiteStatus.Length != 1
            || squaredSum is not null
                && (squaredSum.Device.Index != deviceIndex
                    || squaredSum.Length != 1))
        {
            throw new ArgumentException(
                "BFP8 gradient quantization buffers must share one device.");
        }
        ForgetMemoryV2Cuda.GetAccelerator(deviceIndex).Bind();
        int status = squaredSum is null
            ? CudaNativeGateway.Bfp8GradientQuantize(
                deviceIndex,
                source.NativePtr,
                destination.Payload.NativePtr,
                destination.Scales.NativePtr,
                source.Length,
                finiteStatus.NativePtr,
                stream)
            : CudaNativeGateway.Bfp8GradientQuantizeAccumulate(
                deviceIndex,
                source.NativePtr,
                destination.Payload.NativePtr,
                destination.Scales.NativePtr,
                source.Length,
                finiteStatus.NativePtr,
                squaredSum.NativePtr,
                stream);
        ThrowIfFailed(status,
            "BFP8 gradient quantize");
    }

    internal static void AccumulateSquaredSum(
        int deviceIndex,
        CudaBfp8BufferView gradient,
        NativeCudaBuffer<double> squaredSum,
        NativeCudaBuffer<int> finiteStatus,
        nint stream)
    {
        int length = checked((int)gradient.Payload.Length);
        ValidateTensorWide(gradient, deviceIndex, length);
        if (squaredSum.Device.Index != deviceIndex
            || squaredSum.Length != 1
            || finiteStatus.Device.Index != deviceIndex
            || finiteStatus.Length != 1)
        {
            throw new ArgumentException(
                "BFP8 norm accumulation buffers must share one device.");
        }
        ForgetMemoryV2Cuda.GetAccelerator(deviceIndex).Bind();
        ThrowIfFailed(
            CudaNativeGateway.Bfp8GradientSquaredSum(
                deviceIndex,
                gradient.Payload.NativePtr,
                gradient.Scales.NativePtr,
                length,
                squaredSum.NativePtr,
                finiteStatus.NativePtr,
                stream),
            "BFP8 gradient squared-norm reduction");
    }

    internal static void Scale(
        int deviceIndex,
        CudaBfp8BufferView gradient,
        float multiplier,
        nint stream)
    {
        int length = checked((int)gradient.Payload.Length);
        ValidateTensorWide(gradient, deviceIndex, length);
        if (!float.IsFinite(multiplier) || !(multiplier > 0f))
        {
            throw new ArgumentOutOfRangeException(
                nameof(multiplier),
                "BFP8 gradient scale multiplier must be finite and positive.");
        }
        ForgetMemoryV2Cuda.GetAccelerator(deviceIndex).Bind();
        ThrowIfFailed(
            CudaNativeGateway.Bfp8GradientScale(
                deviceIndex,
                gradient.Scales.NativePtr,
                multiplier,
                stream),
            "BFP8 gradient scale-only clip");
    }

    internal static void Reduce(
        int primaryDevice,
        int secondaryDevice,
        CudaBfp8BufferView local,
        CudaBfp8BufferView remote,
        NativeCudaBuffer<sbyte> remotePayloadStaging,
        NativeCudaBuffer<float> remoteScaleStaging,
        NativeCudaBuffer<float> reduced,
        CudaBfp8BufferView output,
        float reductionScale,
        NativeCudaBuffer<int> finiteStatus,
        NativeCudaBuffer<int> remoteFiniteStatus,
        NativeCudaBuffer<int> remoteStatusStaging,
        NativeCudaBuffer<double> squaredSum,
        nint communicationStream,
        nint localReady,
        nint remoteReady,
        nint reducedReady)
    {
        int length = checked((int)local.Payload.Length);
        ValidateTensorWide(local, primaryDevice, length);
        ValidateTensorWide(remote, secondaryDevice, length);
        ValidateTensorWide(output, primaryDevice, length);
        if (remotePayloadStaging.Device.Index != primaryDevice
            || remotePayloadStaging.Length != length
            || remoteScaleStaging.Device.Index != primaryDevice
            || remoteScaleStaging.Length != 1
            || reduced.Device.Index != primaryDevice
            || reduced.Length != length
            || finiteStatus.Device.Index != primaryDevice
            || finiteStatus.Length != 1
            || remoteFiniteStatus.Device.Index != secondaryDevice
            || remoteFiniteStatus.Length != 1
            || remoteStatusStaging.Device.Index != primaryDevice
            || remoteStatusStaging.Length != 1
            || squaredSum.Device.Index != primaryDevice
            || squaredSum.Length != 1)
        {
            throw new ArgumentException(
                "BFP8 reduction workspace must match the primary device.");
        }
        ForgetMemoryV2Cuda.GetAccelerator(primaryDevice).Bind();
        ThrowIfFailed(CudaNativeGateway.Bfp8GradientReduce(
            primaryDevice,
            secondaryDevice,
            local.Payload.NativePtr,
            local.Scales.NativePtr,
            remote.Payload.NativePtr,
            remote.Scales.NativePtr,
            remotePayloadStaging.NativePtr,
            remoteScaleStaging.NativePtr,
            reduced.NativePtr,
            output.Payload.NativePtr,
            output.Scales.NativePtr,
            length,
            reductionScale,
            finiteStatus.NativePtr,
            remoteFiniteStatus.NativePtr,
            remoteStatusStaging.NativePtr,
            squaredSum.NativePtr,
            communicationStream,
            localReady,
            remoteReady,
            reducedReady),
            "BFP8 gradient reduce");
    }

    internal static void Broadcast(
        int destinationDevice,
        int sourceDevice,
        CudaBfp8BufferView source,
        CudaBfp8BufferView destination,
        NativeCudaBuffer<float> destinationFloat,
        NativeCudaBuffer<int> destinationFiniteStatus,
        nint destinationStream,
        nint sourceReady)
    {
        int length = checked((int)source.Payload.Length);
        ValidateTensorWide(source, sourceDevice, length);
        ValidateTensorWide(destination, destinationDevice, length);
        if (destinationFloat.Device.Index != destinationDevice
            || destinationFloat.Length != length
            || destinationFiniteStatus.Device.Index != destinationDevice
            || destinationFiniteStatus.Length != 1)
        {
            throw new ArgumentException(
                "BFP8 broadcast destination workspace is invalid.");
        }
        ForgetMemoryV2Cuda.GetAccelerator(destinationDevice).Bind();
        ThrowIfFailed(CudaNativeGateway.Bfp8GradientBroadcast(
            destinationDevice,
            sourceDevice,
            source.Payload.NativePtr,
            source.Scales.NativePtr,
            destination.Payload.NativePtr,
            destination.Scales.NativePtr,
            destinationFloat.NativePtr,
            length,
            destinationFiniteStatus.NativePtr,
            destinationStream,
            sourceReady),
            "BFP8 gradient broadcast");
    }

    private static void ValidateTensorWide(
        CudaBfp8BufferView view,
        int deviceIndex,
        long length)
    {
        if (view.Descriptor != Bfp8QuantizationDescriptor.TensorWide
            || view.Payload.Device.Index != deviceIndex
            || view.Scales.Device.Index != deviceIndex
            || view.Payload.Length != length
            || view.Scales.Length != 1)
        {
            throw new ArgumentException(
                "Pure BFP8 gradients require one tensor-wide device scale.");
        }
    }

    private static void ThrowIfFailed(int status, string operation)
    {
        if (status != 0)
        {
            throw new InvalidOperationException(
                $"CUDA {operation} failed with status {status}.");
        }
    }

}
