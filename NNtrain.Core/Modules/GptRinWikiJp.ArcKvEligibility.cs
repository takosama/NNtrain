namespace NNtrain;

public sealed partial class GptRinWikiJp
{
    /// <summary>
    /// Cache only when one-token execution preserves the quantization blocks
    /// of a full prefix. A block spanning multiple token rows can change the
    /// already-cached K/V whenever later tokens change that block's scale.
    /// </summary>
    internal bool CanUseArcKvCache()
        => Tensor.ArcResident && Tensor.ArcLane.Options.InferenceKvCache
            && ContextLength <= ArcAttentionKvCache.MaximumCapacity
            && HasArcKvCompatibleQuantization();

    /// <summary>
    /// Pure model-shape check, also usable before an Arc session is created.
    /// Conservatively checks every BFP8 parameter even when some activations
    /// currently publish BF16: tensor-parallel partial sums still publish
    /// BFP8, and toggling activation optimizations must remain correct.
    /// </summary>
    internal bool HasArcKvCompatibleQuantization()
    {
        int shards = ArcTensorParallelEnabled ? 2 : 1;
        var checkedBlockSizes = new HashSet<int>();
        foreach (Parameter parameter in Parameters())
        {
            if (parameter.T.DType != TensorDType.Bfp8) continue;
            Bfp8QuantizationDescriptor? quantization = parameter.T.Bfp8Quantization;
            if (quantization is not { Granularity: Bfp8ScaleGranularity.Block, BlockSize: > 0 })
                return false;
            int blockSize = quantization.BlockSize;
            if (!checkedBlockSizes.Add(blockSize)) continue;
            if (ModelWidth % shards != 0 || (ModelWidth / shards) % blockSize != 0)
                return false;
            foreach (TransformerBlock layer in _blocks)
            {
                int hiddenWidth = layer.Ffn.Fc1.W.T.Shape[0];
                if (hiddenWidth % shards != 0 || (hiddenWidth / shards) % blockSize != 0)
                    return false;
            }
        }
        return true;
    }
}
