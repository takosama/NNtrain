namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// Fused no-grad token and position lookup for one inference sequence.
    /// The position offset permits incremental decoding after prefill.
    /// </summary>
    internal Tensor ArcInferenceEmbeddingWithPositions(
        Tensor positions, int[] ids, int positionOffset)
    {
        ArgumentNullException.ThrowIfNull(positions);
        ArgumentNullException.ThrowIfNull(ids);
        if (!ArcResident || AutogradContext.IsRecordingEnabled)
            throw new InvalidOperationException(
                "Arc inference embeddings require resident no-grad inference.");
        if (Rank != 2 || positions.Rank != 2 || _shape[1] != positions._shape[1])
            throw new ArgumentException("Token and position embedding widths must match.");
        if (ReferenceEquals(this, positions))
            throw new ArgumentException("Token and position tables must be distinct.", nameof(positions));
        if (ids.Length == 0)
            throw new ArgumentException("At least one token is required.", nameof(ids));
        if (positionOffset < 0 || (long)positionOffset + ids.Length > positions._shape[0])
            throw new ArgumentOutOfRangeException(nameof(positionOffset));
        for (int i = 0; i < ids.Length; i++)
            if ((uint)ids[i] >= (uint)_shape[0])
                throw new ArgumentOutOfRangeException(nameof(ids), ids[i],
                    $"Token at position {i} is outside the embedding table.");

        static int Format(TensorDType dtype) => dtype switch
        {
            TensorDType.Float32 => 0,
            TensorDType.BFloat16 => 1,
            TensorDType.Bfp8 => 2,
            _ => throw new NotSupportedException($"Packed Arc embedding does not support {dtype}."),
        };

        var lane = ArcLane;
        int width = _shape[1], length = checked(ids.Length * width);
        if (!lane.Options.InferencePackedEmbedding)
        {
            // Decode the complete tables for the reference path, but add the
            // selected rows before the one storage publication. Separate
            // EmbeddingLookup calls would round each operand prematurely.
            using var tokens = ArcUploadValues();
            using var positional = positions.ArcUploadValues();
            using var indices = lane.UploadRaw(ids);
            using var summed = lane.Allocate(length);
            lane.Run("arc_inference_embedding_packed", length, 0,
                tokens, tokens, positional, positional, indices, summed,
                ids.Length, width, positionOffset, 0, 1, 0, 1);
            return ArcDeviceResult(summed, [1, ids.Length, width], [this, positions]);
        }
        using var tokenTable = ArcMatrixOperand();
        using var positionTable = positions.ArcMatrixOperand();
        using var tokenIds = lane.UploadRaw(ids);
        using var output = lane.Allocate(length);
        lane.Run("arc_inference_embedding_packed", length, 0,
            tokenTable.Value, tokenTable.Scales ?? tokenTable.Value,
            positionTable.Value, positionTable.Scales ?? positionTable.Value,
            tokenIds, output, ids.Length, width, positionOffset,
            Format(tokenTable.DType), tokenTable.BlockSize,
            Format(positionTable.DType), positionTable.BlockSize);
        return ArcDeviceResult(output, [1, ids.Length, width], [this, positions]);
    }
}
