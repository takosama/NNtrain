using NNtrain.Cuda.Interop;
using NNtrain.Cuda.Execution;

namespace NNtrain;

/// <summary>
/// Thin validation boundary for resident signed-Int8 embedding kernels. The
/// source payloads and scale sidecars remain encoded; only selected values are
/// decoded inside the CUDA kernels.
/// </summary>
internal static class CudaBfp8EmbeddingNative
{
    private const int TensorReductionValuesPerBlock = 1024;

    internal static int GetWorkspaceLength(
        int outputLength,
        int outputScaleCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputLength);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputScaleCount);
        return outputScaleCount == 1
            ? checked((int)((outputLength +
                (long)TensorReductionValuesPerBlock - 1) /
                TensorReductionValuesPerBlock))
            : 0;
    }

    internal static void EmbeddingForward(
        int deviceIndex,
        CudaBfp8BufferView table,
        NativeCudaBuffer<int> indices,
        int width,
        CudaBfp8OwnedBuffers output,
        NativeCudaBuffer<float>? workspace,
        nint stream)
    {
        ValidateCommon(
            deviceIndex, table, indices, width, output, workspace);
        int outputLength = checked(indices.Length * width);
        NativeCudaRuntime.Check(
            CudaNativeGateway.Bfp8EmbeddingForward(
                deviceIndex,
                table.Payload.NativePtr,
                table.Scales.NativePtr,
                table.Payload.Length,
                table.Descriptor.GetEffectiveBlockSize(table.Payload.Length),
                indices.NativePtr,
                indices.Length,
                width,
                output.Payload.NativePtr,
                output.Scales.NativePtr,
                output.Descriptor.GetEffectiveBlockSize(outputLength),
                output.Scales.Length,
                workspace?.NativePtr ?? nint.Zero,
                workspace?.Length ?? 0,
                stream),
            "CUDA BFP8 embedding lookup");
    }

    internal static void EmbeddingWithPositionsForward(
        int deviceIndex,
        CudaBfp8BufferView tokenTable,
        CudaBfp8BufferView positionTable,
        NativeCudaBuffer<int> indices,
        int sequenceLength,
        int width,
        CudaBfp8OwnedBuffers output,
        NativeCudaBuffer<float>? workspace,
        nint stream)
    {
        ValidateCommon(
            deviceIndex, tokenTable, indices, width, output, workspace);
        ValidateView(deviceIndex, positionTable, nameof(positionTable));
        if (positionTable.Payload.Length % width != 0)
        {
            throw new ArgumentException(
                "Position embedding table length must be divisible by width.",
                nameof(positionTable));
        }
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sequenceLength);
        if (sequenceLength > positionTable.Payload.Length / width)
        {
            throw new ArgumentOutOfRangeException(
                nameof(sequenceLength),
                "Sequence length exceeds the position embedding table.");
        }

        int outputLength = checked(indices.Length * width);
        NativeCudaRuntime.Check(
            CudaNativeGateway.Bfp8EmbeddingPositionsForward(
                deviceIndex,
                tokenTable.Payload.NativePtr,
                tokenTable.Scales.NativePtr,
                tokenTable.Payload.Length,
                tokenTable.Descriptor.GetEffectiveBlockSize(
                    tokenTable.Payload.Length),
                positionTable.Payload.NativePtr,
                positionTable.Scales.NativePtr,
                positionTable.Payload.Length,
                positionTable.Descriptor.GetEffectiveBlockSize(
                    positionTable.Payload.Length),
                indices.NativePtr,
                indices.Length,
                sequenceLength,
                width,
                output.Payload.NativePtr,
                output.Scales.NativePtr,
                output.Descriptor.GetEffectiveBlockSize(outputLength),
                output.Scales.Length,
                workspace?.NativePtr ?? nint.Zero,
                workspace?.Length ?? 0,
                stream),
            "CUDA BFP8 embedding and position lookup");
    }

    private static void ValidateCommon(
        int deviceIndex,
        CudaBfp8BufferView table,
        NativeCudaBuffer<int> indices,
        int width,
        CudaBfp8OwnedBuffers output,
        NativeCudaBuffer<float>? workspace)
    {
        ArgumentNullException.ThrowIfNull(indices);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ValidateView(deviceIndex, table, nameof(table));
        if (table.Payload.Length % width != 0)
        {
            throw new ArgumentException(
                "Embedding table length must be divisible by width.",
                nameof(table));
        }
        if (indices.Device.Index != deviceIndex || indices.Length == 0)
        {
            throw new ArgumentException(
                "Embedding indices must be non-empty and resident on the " +
                "requested CUDA device.",
                nameof(indices));
        }

        int outputLength = checked(indices.Length * width);
        if (output.Payload.Device.Index != deviceIndex ||
            output.Scales.Device.Index != deviceIndex ||
            output.Payload.Length != outputLength ||
            output.Scales.Length !=
                output.Descriptor.GetScaleCount(outputLength))
        {
            throw new ArgumentException(
                "BFP8 embedding output buffers do not match the selected " +
                "result descriptor.",
                nameof(output));
        }

        int expectedWorkspace = GetWorkspaceLength(
            outputLength, output.Scales.Length);
        if (expectedWorkspace == 0)
        {
            if (workspace is not null)
            {
                throw new ArgumentException(
                    "Block-scaled embedding output does not require a " +
                    "tensor reduction workspace.",
                    nameof(workspace));
            }
        }
        else if (workspace is null ||
            workspace.Device.Index != deviceIndex ||
            workspace.Length < expectedWorkspace)
        {
            throw new ArgumentException(
                "Tensor-scaled embedding output requires a resident partial " +
                "maximum workspace of the expected size.",
                nameof(workspace));
        }

        CudaKernelCapabilities capabilities =
            CudaBfp8Native.GetCapabilities(deviceIndex);
        if (!capabilities.Supports(CudaKernelFeature.Bfp8Quantization))
        {
            throw new NotSupportedException(
                $"CUDA device {deviceIndex} has no resident BFP8 embedding " +
                "capability. CPU fallback is forbidden.");
        }
    }

    private static void ValidateView(
        int deviceIndex,
        CudaBfp8BufferView view,
        string parameterName)
    {
        if (view.Payload is null ||
            view.Scales is null ||
            view.Descriptor is null ||
            view.Payload.Device.Index != deviceIndex ||
            view.Scales.Device.Index != deviceIndex ||
            view.Payload.Length <= 0 ||
            view.Scales.Length !=
                view.Descriptor.GetScaleCount(view.Payload.Length))
        {
            throw new ArgumentException(
                "BFP8 payload and scale sidecar must be resident on the " +
                "requested CUDA device and match their descriptor.",
                parameterName);
        }
    }
}
