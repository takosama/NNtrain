using NNtrain.Runtime.Execution;

namespace NNtrain;

public sealed partial class GptRinWikiJp
{
    private ArcTensorParallelWeights? _arcTensorParallelWeights;
    private bool _arcTensorParallelEnabled;
    private bool _arcTensorParallelFullWeightsReleased;

    private sealed record ArcParallelProjection(Tensor Weight, Tensor Bias);

    private sealed record ArcParallelLayer(
        ArcParallelProjection QkvFirst,
        ArcParallelProjection QkvSecond,
        Tensor WoFirst,
        Tensor WoSecond,
        ArcParallelProjection Fc1First,
        ArcParallelProjection Fc1Second,
        Tensor Fc2First,
        Tensor Fc2Second);

    private sealed class ArcTensorParallelWeights(
        Parameter[] sources,
        long[] versions,
        ArcParallelLayer[] layers)
    {
        internal Parameter[] Sources { get; } = sources;
        internal long[] Versions { get; } = versions;
        internal ArcParallelLayer[] Layers { get; } = layers;

        internal bool Matches(Parameter[] current)
        {
            if (current.Length != Sources.Length) return false;
            for (int i = 0; i < current.Length; i++)
                if (!ReferenceEquals(current[i], Sources[i])
                    || current[i].T.DataVersion != Versions[i]) return false;
            return true;
        }
    }

    internal bool CanUseArcTensorParallel(out string reason)
    {
        if (_blocks[0].Attn.NumHeads < 2 || _blocks[0].Attn.NumHeads % 2 != 0)
        {
            reason = "Arc tensor parallelism requires an even number of attention heads (at least two).";
            return false;
        }
        if (_blocks[0].Ffn.Fc1.W.T.Shape[0] % 2 != 0)
        {
            reason = "Arc tensor parallelism requires an even FFN hidden width.";
            return false;
        }
        ExecutionSession? session = ExecutionSession.Current;
        if (session is null || session.Options.Device != ExecutionDeviceKind.Arc)
        {
            reason = "Arc tensor parallelism requires an Arc inference session.";
            return false;
        }
        int[] devices = session.Options.ArcDevices?.ToArray()
            ?? [session.Options.ArcDeviceIndex];
        if (devices.Length != 2
            || devices.Any(device => !session.TryGetLane(ExecutionDeviceKind.Arc, device, out _)))
        {
            reason = "Arc tensor parallelism requires exactly two attached Arc lanes.";
            return false;
        }
        reason = string.Empty;
        return true;
    }

    private IDisposable PushArcTensorParallelPrimary()
    {
        if (!CanUseArcTensorParallel(out string reason))
            throw new NotSupportedException(reason);
        ExecutionSession session = ExecutionSession.Current!;
        int primary = session.Options.ArcDevices?[0]
            ?? session.Options.ArcDeviceIndex;
        return TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, primary));
    }

    private Tensor ForwardHiddenArcTensorParallel(int[] tokenIds, int sequenceLength,
        ArcAttentionKvCache[]? firstCaches = null, ArcAttentionKvCache[]? secondCaches = null,
        int position = 0)
    {
        if (!CanUseArcTensorParallel(out string reason))
            throw new NotSupportedException(reason);
        ExecutionSession session = ExecutionSession.Current!;
        int[] devices = session.Options.ArcDevices?.ToArray()
            ?? [session.Options.ArcDeviceIndex];
        if (tokenIds.Length != sequenceLength || sequenceLength < 1 || sequenceLength > ContextLength)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength));

        ArcTensorParallelWeights weights = GetArcTensorParallelWeights();
        int first = devices[0], second = devices[1];
        int localHeads = _blocks[0].Attn.NumHeads / 2;
        Tensor hidden;
        using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, first)))
            hidden = ArcInferenceEmbedding(tokenIds, position);

        for (int index = 0; index < _blocks.Length; index++)
        {
            TransformerBlock block = _blocks[index];
            ArcParallelLayer shards = weights.Layers[index];

            // The primary lane starts computing before the second lane's
            // projection is submitted. Both OpenCL queues can then run while
            // the second partial is read through host memory for the sum.
            Tensor secondInput;
            using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, first)))
                secondInput = hidden.ArcInferenceCopyToDevice(second);
            Tensor firstWoPartial;
            using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, first)))
            {
                Tensor qkv = hidden.LinearLastDim(
                    shards.QkvFirst.Weight, shards.QkvFirst.Bias, applyRelu: false);
                Tensor attended = ArcCachedAttention(qkv, localHeads, sequenceLength, firstCaches?[index], position);
                firstWoPartial = attended.ArcInferenceLinearPartial(shards.WoFirst);
            }
            float[] secondWoPartial;
            using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, second)))
            {
                Tensor qkv = secondInput.LinearLastDim(
                    shards.QkvSecond.Weight, shards.QkvSecond.Bias, applyRelu: false);
                Tensor attended = ArcCachedAttention(qkv, localHeads, sequenceLength, secondCaches?[index], position);
                secondWoPartial = attended.ArcInferenceLinearPartial(shards.WoSecond).Data.ToArray();
            }

            Tensor normed;
            using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, first)))
            {
                Tensor projected = firstWoPartial.ArcInferenceSumPartials(
                    secondWoPartial, block.Attn.Wo.B.T, block.Attn.Wo.B.T.DType);
                normed = block.Ln1.ForwardResidualDropout(hidden, projected, block.AttnDropout);
            }

            Tensor secondNormed;
            using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, first)))
                secondNormed = normed.ArcInferenceCopyToDevice(second);
            Tensor firstFc2Partial;
            using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, first)))
            {
                Tensor expanded = normed.LinearLastDim(
                    shards.Fc1First.Weight, shards.Fc1First.Bias, applyRelu: true);
                firstFc2Partial = expanded.ArcInferenceLinearPartial(shards.Fc2First);
            }
            float[] secondFc2Partial;
            using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, second)))
            {
                Tensor expanded = secondNormed.LinearLastDim(
                    shards.Fc1Second.Weight, shards.Fc1Second.Bias, applyRelu: true);
                secondFc2Partial = expanded.ArcInferenceLinearPartial(shards.Fc2Second).Data.ToArray();
            }
            using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, first)))
            {
                Tensor projected = firstFc2Partial.ArcInferenceSumPartials(
                    secondFc2Partial, block.Ffn.Fc2.B.T, block.Ffn.Fc2.B.T.DType);
                hidden = block.Ln2.ForwardResidualDropout(normed, projected, block.FfnDropout);
            }
        }
        using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, first)))
            return _finalNorm.Forward(hidden);
    }

    private ArcTensorParallelWeights GetArcTensorParallelWeights()
    {
        Parameter[] parameters = Parameters().ToArray();
        ArcTensorParallelWeights? cached = _arcTensorParallelWeights;
        if (cached is not null && cached.Matches(parameters))
        {
            ReleaseUnusedFullArcWeights();
            return cached;
        }
        if (cached is not null) ReleaseArcTensorParallelShards();
        int width = ModelWidth;
        int localWidth = width / 2;
        var layers = new ArcParallelLayer[_blocks.Length];
        for (int index = 0; index < layers.Length; index++)
        {
            TransformerBlock block = _blocks[index];
            Tensor qkvWeight = block.Attn.Qkv.W.T;
            Tensor qkvBias = block.Attn.Qkv.B.T;
            Tensor woWeight = block.Attn.Wo.W.T;
            Tensor fc1Weight = block.Ffn.Fc1.W.T;
            Tensor fc1Bias = block.Ffn.Fc1.B.T;
            Tensor fc2Weight = block.Ffn.Fc2.W.T;
            int hidden = fc1Weight.Shape[0], localHidden = hidden / 2;
            layers[index] = new(
                new(ShardQkv(qkvWeight, width, localWidth, 0),
                    ShardQkv(qkvBias, width, localWidth, 0)),
                new(ShardQkv(qkvWeight, width, localWidth, 1),
                    ShardQkv(qkvBias, width, localWidth, 1)),
                ShardColumns(woWeight, 0, localWidth),
                ShardColumns(woWeight, localWidth, localWidth),
                new(ShardRows(fc1Weight, 0, localHidden),
                    ShardRows(fc1Bias, 0, localHidden)),
                new(ShardRows(fc1Weight, localHidden, localHidden),
                    ShardRows(fc1Bias, localHidden, localHidden)),
                ShardColumns(fc2Weight, 0, localHidden),
                ShardColumns(fc2Weight, localHidden, localHidden));
        }
        cached = new(parameters, parameters.Select(p => p.T.DataVersion).ToArray(), layers);
        _arcTensorParallelWeights = cached;
        ReleaseUnusedFullArcWeights();
        return cached;
    }

    private void ReleaseUnusedFullArcWeights()
    {
        if (_arcTensorParallelFullWeightsReleased) return;
        foreach (TransformerBlock block in _blocks)
        {
            block.Attn.Qkv.W.T.ReleaseArcInferenceReplica();
            block.Attn.Qkv.B.T.ReleaseArcInferenceReplica();
            block.Attn.Wo.W.T.ReleaseArcInferenceReplica();
            block.Ffn.Fc1.W.T.ReleaseArcInferenceReplica();
            block.Ffn.Fc1.B.T.ReleaseArcInferenceReplica();
            block.Ffn.Fc2.W.T.ReleaseArcInferenceReplica();
        }
        _arcTensorParallelFullWeightsReleased = true;
    }

    private void ReleaseArcTensorParallelShards()
    {
        if (_arcTensorParallelWeights is null) return;
        foreach (ArcParallelLayer layer in _arcTensorParallelWeights.Layers)
        {
            layer.QkvFirst.Weight.ReleaseArcInferenceReplica();
            layer.QkvFirst.Bias.ReleaseArcInferenceReplica();
            layer.QkvSecond.Weight.ReleaseArcInferenceReplica();
            layer.QkvSecond.Bias.ReleaseArcInferenceReplica();
            layer.WoFirst.ReleaseArcInferenceReplica();
            layer.WoSecond.ReleaseArcInferenceReplica();
            layer.Fc1First.Weight.ReleaseArcInferenceReplica();
            layer.Fc1First.Bias.ReleaseArcInferenceReplica();
            layer.Fc1Second.Weight.ReleaseArcInferenceReplica();
            layer.Fc1Second.Bias.ReleaseArcInferenceReplica();
            layer.Fc2First.ReleaseArcInferenceReplica();
            layer.Fc2Second.ReleaseArcInferenceReplica();
        }
    }

    private static Tensor ShardQkv(Tensor source, int width, int localWidth, int shard)
    {
        int rowWidth = source.Rank == 2 ? source.Shape[1] : 1;
        var values = new float[checked(3 * localWidth * rowWidth)];
        IReadOnlyList<float> original = source.Data;
        for (int component = 0; component < 3; component++)
        {
            int from = checked((component * width + shard * localWidth) * rowWidth);
            int to = checked(component * localWidth * rowWidth);
            for (int i = 0; i < localWidth * rowWidth; i++)
                values[to + i] = original[from + i];
        }
        return NewShard(source, values, source.Rank == 2
            ? [3 * localWidth, rowWidth] : [3 * localWidth]);
    }

    private static Tensor ShardRows(Tensor source, int first, int count)
    {
        int rowWidth = source.Rank == 2 ? source.Shape[1] : 1;
        var values = new float[checked(count * rowWidth)];
        IReadOnlyList<float> original = source.Data;
        int start = checked(first * rowWidth);
        for (int i = 0; i < values.Length; i++) values[i] = original[start + i];
        return NewShard(source, values, source.Rank == 2 ? [count, rowWidth] : [count]);
    }

    private static Tensor ShardColumns(Tensor source, int first, int count)
    {
        if (source.Rank != 2) throw new ArgumentException("Column shards require a matrix.", nameof(source));
        int rows = source.Shape[0], columns = source.Shape[1];
        var values = new float[checked(rows * count)];
        IReadOnlyList<float> original = source.Data;
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < count; column++)
                values[row * count + column] = original[row * columns + first + column];
        return NewShard(source, values, [rows, count]);
    }

    private static Tensor NewShard(Tensor source, float[] values, int[] shape)
        => source.DType == TensorDType.Bfp8
            ? Tensor.FromBfp8(values, shape, source.Bfp8Quantization!)
            : new Tensor(values, shape, dtype: source.DType);
}
