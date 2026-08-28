namespace NNtrain;

internal static partial class TensorCudaKernels
{
    internal static void DropoutBackwardBFloat16Resident(
        Tensor output,
        Tensor input,
        uint seed,
        uint dropThreshold,
        float scale)
        => DropoutBackwardBFloat16Core(
            output,
            input,
            (deviceIndex, outputGradient, inputGradient) =>
                CudaPureBFloat16GradientNative.DropoutBackward(
                    deviceIndex,
                    outputGradient.NativePtr,
                    inputGradient.NativePtr,
                    output.Numel,
                    seed,
                    dropThreshold,
                    scale,
                    ForgetMemoryV2Cuda.GetAccelerator(deviceIndex)
                        .DefaultStream));

    internal static void DropoutBackwardBFloat16GraphResident(
        Tensor output,
        Tensor input,
        CudaGraphDropoutToken token,
        float probability)
        => DropoutBackwardBFloat16Core(
            output,
            input,
            (deviceIndex, outputGradient, inputGradient) =>
            {
                if (token.DeviceIndex != deviceIndex)
                {
                    throw new InvalidOperationException(
                        "CUDA graph dropout token belongs to another device.");
                }
                token.RngState.EnqueueDropoutBackwardBFloat16Gradient(
                    outputGradient.NativePtr,
                    inputGradient.NativePtr,
                    output.Numel,
                    probability,
                    token.OperationSeed);
            });

    private static void DropoutBackwardBFloat16Core(
        Tensor output,
        Tensor input,
        Action<int, NativeCudaBuffer<ushort>, NativeCudaBuffer<ushort>> launch)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        using CudaBFloat16GradientSource outputGradientSource =
            CudaBFloat16GradientSource.Acquire(output, deviceIndex);
        using var targets =
            new CudaPureBFloat16GradientTargetSet(deviceIndex);
        CudaPureBFloat16GradientTarget inputGradientTarget =
            targets.Get(input);
        inputGradientTarget.EnsureZeroInitialized();
        launch(
            deviceIndex,
            outputGradientSource.Buffer,
            inputGradientTarget.Buffer);
        targets.CommitAll();
    }

    internal static void AddDropoutBackwardBFloat16Resident(
        Tensor output,
        Tensor residual,
        Tensor branch,
        bool sameParent,
        uint seed,
        uint dropThreshold,
        float scale)
        => AddDropoutBackwardBFloat16Core(
            output,
            residual,
            branch,
            sameParent,
            (deviceIndex,
             outputGradient,
             residualGradient,
             branchGradient) => CudaPureBFloat16GradientNative
                .AddDropoutBackward(
                deviceIndex,
                outputGradient.NativePtr,
                residualGradient.NativePtr,
                branchGradient.NativePtr,
                output.Numel,
                sameParent,
                seed,
                dropThreshold,
                scale,
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex)
                    .DefaultStream));

    internal static void AddDropoutBackwardBFloat16GraphResident(
        Tensor output,
        Tensor residual,
        Tensor branch,
        bool sameParent,
        CudaGraphDropoutToken token,
        float probability)
        => AddDropoutBackwardBFloat16Core(
            output,
            residual,
            branch,
            sameParent,
            (deviceIndex,
             outputGradient,
             residualGradient,
             branchGradient) =>
            {
                if (token.DeviceIndex != deviceIndex)
                {
                    throw new InvalidOperationException(
                        "CUDA graph dropout token belongs to another device.");
                }
                token.RngState.EnqueueAddDropoutBackwardBFloat16Gradient(
                    outputGradient.NativePtr,
                    residualGradient.NativePtr,
                    branchGradient.NativePtr,
                    output.Numel,
                    probability,
                    token.OperationSeed,
                    sameParent);
            });

    private static void AddDropoutBackwardBFloat16Core(
        Tensor output,
        Tensor residual,
        Tensor branch,
        bool sameParent,
        Action<
            int,
            NativeCudaBuffer<ushort>,
            NativeCudaBuffer<ushort>,
            NativeCudaBuffer<ushort>> launch)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        using CudaBFloat16GradientSource outputGradientSource =
            CudaBFloat16GradientSource.Acquire(output, deviceIndex);
        using var targets =
            new CudaPureBFloat16GradientTargetSet(deviceIndex);
        CudaPureBFloat16GradientTarget residualGradientTarget =
            targets.Get(residual);
        CudaPureBFloat16GradientTarget branchGradientTarget =
            targets.Get(branch);
        residualGradientTarget.EnsureZeroInitialized();
        if (!sameParent)
            branchGradientTarget.EnsureZeroInitialized();
        launch(
            deviceIndex,
            outputGradientSource.Buffer,
            residualGradientTarget.Buffer,
            branchGradientTarget.Buffer);
        targets.CommitAll();
    }
}
