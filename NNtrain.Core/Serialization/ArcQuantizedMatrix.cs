using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

/// <summary>
/// Immutable GGML K-quantized matrix.  The encoded payload is uploaded once
/// per Arc execution lane and remains quantized in VRAM for its lifetime.
/// </summary>
internal sealed class ArcQuantizedMatrix : IDisposable
{
    private readonly byte[] _payload;
    private ArcExecutionLane? _lane;
    private ArcBuffer? _devicePayload;

    internal ArcQuantizedMatrix(
        byte[] payload, uint ggmlType, int outputWidth, int inputWidth)
    {
        ArgumentNullException.ThrowIfNull(payload);
        if (ggmlType is not (Qwen2Gguf.Q4KType or Qwen2Gguf.Q6KType))
            throw new NotSupportedException($"GGML type {ggmlType} is not a K-quantized matrix.");
        if (outputWidth <= 0 || inputWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputWidth));
        if (inputWidth % 256 != 0)
            throw new NotSupportedException("K-quantized rows must contain complete 256-value blocks.");

        int blockBytes = ggmlType == Qwen2Gguf.Q4KType
            ? GgufQ4K.BlockBytes : GgufQ6K.BlockBytes;
        int expected = checked(outputWidth * (inputWidth / 256) * blockBytes);
        if (payload.Length != expected)
            throw new InvalidDataException(
                $"Quantized matrix payload has {payload.Length} bytes; expected {expected}.");

        _payload = payload;
        GgmlType = ggmlType;
        OutputWidth = outputWidth;
        InputWidth = inputWidth;
    }

    internal uint GgmlType { get; }
    internal int OutputWidth { get; }
    internal int InputWidth { get; }
    internal int StorageBytes => _payload.Length;

    internal Tensor Forward(Tensor input, Tensor bias)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(bias);
        if (Tensor.ExecutionDevice != TensorDevice.Arc)
            throw new NotSupportedException("Native GGUF K-quantized inference currently requires Intel Arc.");
        if (input.Shape[^1] != InputWidth)
            throw new ArgumentException("Quantized linear input width mismatch.", nameof(input));
        if (bias.Rank != 1 || bias.Numel != OutputWidth)
            throw new ArgumentException("Quantized linear bias width mismatch.", nameof(bias));

        ArcExecutionLane lane = Tensor.ArcLane;
        ArcBuffer encoded = DevicePayload(lane);
        return input.ArcQwenQuantizedLinear(
            encoded, bias, GgmlType, OutputWidth, InputWidth);
    }

    internal Tensor LookupEmbedding(int[] tokenIds, int batch, int sequence)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        if (Tensor.ExecutionDevice != TensorDevice.Arc)
            throw new NotSupportedException("Native GGUF K-quantized embedding requires Intel Arc.");
        if (batch <= 0 || sequence <= 0 || tokenIds.Length != checked(batch * sequence))
            throw new ArgumentException("Token count does not match batch * sequence.", nameof(tokenIds));
        for (int position = 0; position < tokenIds.Length; ++position)
            if ((uint)tokenIds[position] >= (uint)OutputWidth)
                throw new ArgumentOutOfRangeException(
                    nameof(tokenIds), tokenIds[position],
                    $"Embedding index at position {position} is outside the vocabulary.");

        ArcBuffer encoded = DevicePayload(Tensor.ArcLane);
        return Tensor.ArcQwenQuantizedEmbedding(
            encoded, tokenIds, batch, sequence, InputWidth, GgmlType);
    }

    private ArcBuffer DevicePayload(ArcExecutionLane lane)
    {
        if (_devicePayload is not null && ReferenceEquals(_lane, lane))
            return _devicePayload;
        _devicePayload?.Dispose();
        _devicePayload = lane.UploadRaw(_payload);
        _lane = lane;
        return _devicePayload;
    }

    public void Dispose()
    {
        _devicePayload?.Dispose();
        _devicePayload = null;
        _lane = null;
    }
}
