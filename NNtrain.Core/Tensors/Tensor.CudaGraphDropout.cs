namespace NNtrain;

internal static partial class TensorCudaKernels
{
    internal static NativeCudaBuffer<float> DropoutForwardGraphResident(
        Tensor input,
        CudaGraphDropoutToken token,
        float probability)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        RequireGraphDevice(token, deviceIndex);
        NativeCudaBuffer<float> output = Tensor.RentCudaFloatBuffer(
            deviceIndex, input.Numel);
        token.RngState.EnqueueDropoutForwardFloat32(
            input.EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
            output.NativePtr,
            input.Numel,
            probability,
            token.OperationSeed);
        return output;
    }

    internal static NativeCudaBuffer<ushort>
        DropoutForwardBFloat16GraphResident(
            Tensor input,
            CudaGraphDropoutToken token,
            float probability)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        RequireGraphDevice(token, deviceIndex);
        NativeCudaBuffer<ushort> output = Tensor.RentCudaBFloat16Buffer(
            deviceIndex, input.Numel);
        token.RngState.EnqueueDropoutForwardBFloat16(
            input.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
            output.NativePtr,
            input.Numel,
            probability,
            token.OperationSeed);
        return output;
    }

    internal static NativeCudaBuffer<float> AddDropoutForwardGraphResident(
        Tensor residual,
        Tensor branch,
        CudaGraphDropoutToken token,
        float probability)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        RequireGraphDevice(token, deviceIndex);
        NativeCudaBuffer<float> output = Tensor.RentCudaFloatBuffer(
            deviceIndex, residual.Numel);
        token.RngState.EnqueueAddDropoutForwardFloat32(
            residual.EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
            branch.EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
            output.NativePtr,
            residual.Numel,
            probability,
            token.OperationSeed);
        return output;
    }

    internal static NativeCudaBuffer<ushort>
        AddDropoutForwardBFloat16GraphResident(
            Tensor residual,
            Tensor branch,
            CudaGraphDropoutToken token,
            float probability)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        RequireGraphDevice(token, deviceIndex);
        NativeCudaBuffer<ushort> output = Tensor.RentCudaBFloat16Buffer(
            deviceIndex, residual.Numel);
        token.RngState.EnqueueAddDropoutForwardBFloat16(
            residual.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
            branch.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
            output.NativePtr,
            residual.Numel,
            probability,
            token.OperationSeed);
        return output;
    }

    internal static void DropoutBackwardGraphResident(
        Tensor output,
        Tensor input,
        CudaGraphDropoutToken token,
        float probability)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        RequireGraphDevice(token, deviceIndex);
        token.RngState.EnqueueDropoutBackwardFloat32(
            output.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
            input.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
            output.Numel,
            probability,
            token.OperationSeed);
        input.MarkCudaGradientMutated(deviceIndex);
    }

    internal static void AddDropoutBackwardGraphResident(
        Tensor output,
        Tensor residual,
        Tensor branch,
        bool sameParent,
        CudaGraphDropoutToken token,
        float probability)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        RequireGraphDevice(token, deviceIndex);
        NativeCudaBuffer<float> residualGradient =
            residual.EnsureCudaGradientBuffer(deviceIndex);
        NativeCudaBuffer<float> branchGradient = sameParent
            ? residualGradient
            : branch.EnsureCudaGradientBuffer(deviceIndex);
        token.RngState.EnqueueAddDropoutBackwardFloat32(
            output.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
            residualGradient.NativePtr,
            branchGradient.NativePtr,
            output.Numel,
            probability,
            token.OperationSeed,
            sameParent);
        residual.MarkCudaGradientMutated(deviceIndex);
        if (!sameParent)
            branch.MarkCudaGradientMutated(deviceIndex);
    }

    internal static void DropoutForwardBFloat16Graph(
        NativeCudaBuffer<ushort> input,
        NativeCudaBuffer<ushort> output,
        int length,
        CudaGraphDropoutToken token,
        float probability)
    {
        RequireGraphDevice(token, input.Device.Index);
        token.RngState.EnqueueDropoutForwardBFloat16(
            input.NativePtr,
            output.NativePtr,
            length,
            probability,
            token.OperationSeed);
    }

    internal static void AddDropoutForwardBFloat16Graph(
        NativeCudaBuffer<ushort> residual,
        NativeCudaBuffer<ushort> branch,
        NativeCudaBuffer<ushort> output,
        int length,
        CudaGraphDropoutToken token,
        float probability)
    {
        RequireGraphDevice(token, residual.Device.Index);
        token.RngState.EnqueueAddDropoutForwardBFloat16(
            residual.NativePtr,
            branch.NativePtr,
            output.NativePtr,
            length,
            probability,
            token.OperationSeed);
    }

    private static void RequireGraphDevice(
        CudaGraphDropoutToken token,
        int deviceIndex)
    {
        if (token.RngState is null || token.DeviceIndex != deviceIndex)
        {
            throw new InvalidOperationException(
                "CUDA Graph dropout token belongs to a different device.");
        }
    }
}
