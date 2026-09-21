using NNtrain.Cuda.Interop;

namespace NNtrain;

internal static class CudaForgetMemoryNative
{
    internal static void Forward(NativeCudaDevice accelerator, nint projected,
        nint projectedBFloat16, nint output, nint outputBFloat16, nint states,
        nint state, int batch, int sequence, int projectionWidth,
        int keyWidth, int valueWidth, float retentionFloor, int memoryVariant,
        bool bfloat16)
    {
        Prepare(accelerator);
        Check(CudaNativeGateway.ForgetMemoryForward(
            accelerator.Index,
            projected, projectedBFloat16, output,
            outputBFloat16, states, state, batch, sequence, projectionWidth,
            keyWidth, valueWidth, retentionFloor, memoryVariant,
            bfloat16 ? 1 : 0), "ForgetMemory forward");
    }

    internal static void Backward(NativeCudaDevice accelerator, nint projected,
        nint projectedBFloat16, nint projectedGradient, nint outputGradient,
        nint states, nint stateGradient, nint previousGradient, int batch,
        int sequence, int projectionWidth, int keyWidth, int valueWidth,
        float retentionFloor, int memoryVariant, bool bfloat16, nint prepared = 0)
    {
        Prepare(accelerator);
        if (prepared != 0)
        {
            if (memoryVariant != 2) throw new ArgumentException("Prepared backward requires DRN.");
            Check(CudaNativeGateway.DrnBackwardPrepared(accelerator.Index, projected,
                projectedBFloat16, projectedGradient, outputGradient, states, stateGradient,
                previousGradient, prepared, batch, sequence, projectionWidth, keyWidth,
                valueWidth, bfloat16 ? 1 : 0, retentionFloor), "DRN prepared backward");
            return;
        }
        Check(CudaNativeGateway.ForgetMemoryBackward(
            accelerator.Index,
            projected, projectedBFloat16,
            projectedGradient, outputGradient, states, stateGradient,
            previousGradient, batch, sequence, projectionWidth, keyWidth,
            valueWidth, retentionFloor, memoryVariant, bfloat16 ? 1 : 0),
            "ForgetMemory backward");
    }

    internal static int PreparedBackwardScratchLength(int batch, int sequence, int keyWidth, int valueWidth)
    {
        long length = (long)batch * sequence * (2 * keyWidth + 3 * valueWidth + 2);
        return keyWidth is >= 16 and <= 32 && (long)batch * valueWidth >= 128
            && length > 0 && length <= 16L * 1024 * 1024 / sizeof(float)
            && CudaNativeGateway.AbiVersion.Minor >= CudaAbiVersion.DrnRetentionFloorMinor
                ? (int)length : 0;
    }

    private static void Prepare(NativeCudaDevice accelerator)
        => accelerator.Bind();

    private static void Check(int status, string operation)
        => NativeCudaRuntime.Check(status, operation);

}
