using NNtrain.Cuda.Interop;

namespace NNtrain;

/// <summary>
/// Core-side checked launch surface for pure-BF16 gradient kernels. Every
/// pointer remains device resident; callers supply their execution lane's
/// stream so allocation, reduction, and publication stay ordered.
/// </summary>
internal static class CudaPureBFloat16GradientNative
{
    internal static void EmbeddingBackwardReduced(
        int deviceIndex,
        nint indices,
        nint outputGradient,
        nint tableGradient,
        nint workspace,
        int workspaceInts,
        int length,
        int width,
        nint stream)
    {
        Select(deviceIndex);
        Check(
            CudaNativeGateway.EmbeddingBackwardReducedBFloat16Gradient(
                deviceIndex,
                indices,
                outputGradient,
                tableGradient,
                workspace,
                workspaceInts,
                length,
                width,
                stream),
            "owner-reduced pure-BF16 embedding backward");
    }

    internal static void EmbeddingPositionsBackwardReduced(
        int deviceIndex,
        nint indices,
        nint outputGradient,
        nint tokenGradient,
        nint positionGradient,
        nint workspace,
        int workspaceInts,
        int length,
        int sequence,
        int width,
        nint stream)
    {
        Select(deviceIndex);
        Check(
            CudaNativeGateway
                .EmbeddingPositionsBackwardReducedBFloat16Gradient(
                    deviceIndex,
                    indices,
                    outputGradient,
                    tokenGradient,
                    positionGradient,
                    workspace,
                    workspaceInts,
                    length,
                    sequence,
                    width,
                    stream),
            "owner-reduced pure-BF16 embedding/position backward");
    }

    internal static void DropoutBackward(
        int deviceIndex,
        nint outputGradient,
        nint inputGradient,
        int length,
        uint seed,
        uint threshold,
        float scale,
        nint stream)
    {
        Select(deviceIndex);
        Check(
            CudaNativeGateway.DropoutBackwardBFloat16Gradient(
                deviceIndex,
                outputGradient,
                inputGradient,
                length,
                seed,
                threshold,
                scale,
                stream),
            "pure-BF16 dropout backward");
    }

    internal static void AddDropoutBackward(
        int deviceIndex,
        nint outputGradient,
        nint residualGradient,
        nint branchGradient,
        int length,
        bool sameParent,
        uint seed,
        uint threshold,
        float scale,
        nint stream)
    {
        Select(deviceIndex);
        Check(
            CudaNativeGateway.AddDropoutBackwardBFloat16Gradient(
                deviceIndex,
                outputGradient,
                residualGradient,
                branchGradient,
                length,
                sameParent,
                seed,
                threshold,
                scale,
                stream),
            "pure-BF16 residual-dropout backward");
    }

    internal static void LinearBiasBackward(
        int deviceIndex,
        nint outputGradient,
        nint biasGradient,
        int rows,
        int width,
        nint stream)
    {
        Select(deviceIndex);
        Check(
            CudaNativeGateway.LinearBiasBackwardBFloat16Gradient(
                deviceIndex,
                outputGradient,
                biasGradient,
                rows,
                width,
                stream),
            "pure-BF16 linear bias backward");
    }

    internal static void AccumulateSquaredSum(
        int deviceIndex,
        nint values,
        int length,
        nint result,
        nint stream)
    {
        Select(deviceIndex);
        Check(
            CudaNativeGateway.BFloat16GradientSquaredSum(
                deviceIndex,
                values,
                length,
                result,
                stream),
            "pure-BF16 gradient squared sum");
    }

    internal static void Scale(
        int deviceIndex,
        nint values,
        int length,
        float multiplier,
        nint stream)
    {
        Select(deviceIndex);
        Check(
            CudaNativeGateway.BFloat16GradientScale(
                deviceIndex,
                values,
                length,
                multiplier,
                stream),
            "pure-BF16 gradient scale");
    }

    private static void Select(int deviceIndex)
        => NativeCudaRuntime.BindDeviceAndComputeStream(deviceIndex);

    private static void Check(int status, string operation)
        => NativeCudaRuntime.Check(status, operation);
}
