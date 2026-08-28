using NNtrain.Cuda.Interop;

namespace NNtrain;

/// <summary>Base module for token language models.</summary>
public abstract class LanguageModel : Module
{
    private CheckpointableRandom? _trainingRandom;

    protected LanguageModel(TensorDType dtype = TensorDType.Float32)
        : base(dtype)
    {
    }

    public abstract int VocabularySize { get; }
    public abstract int ContextLength { get; }
    public abstract int ModelWidth { get; }
    public abstract IReadOnlyList<Parameter> HiddenWeightParameters { get; }
    public abstract IReadOnlyList<Parameter> AuxiliaryParameters { get; }

    internal void AttachTrainingRandom(CheckpointableRandom random)
    {
        ArgumentNullException.ThrowIfNull(random);
        if (_trainingRandom is not null
            && !ReferenceEquals(_trainingRandom, random))
        {
            throw new InvalidOperationException(
                "A different training random source is already attached.");
        }
        _trainingRandom = random;
    }

    internal TrainingRandomState? CaptureTrainingRandomState()
        => _trainingRandom?.CaptureRuntimeState();

    internal ulong TrainingRandomRootSeed
        => _trainingRandom?.RootSeed ?? 0x4e4e_5452_4752_4150UL;

    internal bool HasCheckpointableTrainingRandom => _trainingRandom is not null;

    internal void RestoreTrainingRandomState(TrainingRandomState? state)
    {
        if (state is null)
            return;
        if (_trainingRandom is null)
        {
            throw new InvalidDataException(
                "Checkpoint contains training random state, but the model " +
                "has no checkpointable random source.");
        }
        _trainingRandom.RestoreRuntimeState(state);
    }

    internal abstract Tensor Forward(
        int[] tokenIds,
        int batchSize,
        int sequenceLength);

    /// <summary>
    /// Computes the training loss while allowing an implementation to fuse
    /// its output projection with cross entropy. The default preserves the
    /// public forward/logit semantics for every existing model.
    /// </summary>
    internal virtual Tensor ForwardLoss(
        int[] tokenIds,
        int[] targetIds,
        int batchSize,
        int sequenceLength,
        int ignoreIndex = Tensor.DefaultCrossEntropyIgnoreIndex)
        => Forward(tokenIds, batchSize, sequenceLength)
            .CrossEntropyWithLogits(targetIds, ignoreIndex: ignoreIndex);

    public Tensor forward(
        int[] input_ids,
        int batch_size,
        int sequence_length)
        => Forward(input_ids, batch_size, sequence_length);

    public Tensor forward_loss(
        int[] input_ids,
        int[] target_ids,
        int batch_size,
        int sequence_length,
        int ignore_index = Tensor.DefaultCrossEntropyIgnoreIndex)
        => ForwardLoss(
            input_ids,
            target_ids,
            batch_size,
            sequence_length,
            ignore_index);

    internal abstract int[] GenerateTokenIds(
        IEnumerable<int> promptTokenIds,
        int maxNewTokens,
        float temperature,
        int topK,
        int? stopTokenId,
        Random? random);

    internal virtual int[] GenerateTokenIds(
        IEnumerable<int> promptTokenIds,
        int maxNewTokens,
        float temperature,
        int topK,
        int? stopTokenId,
        Random? random,
        Action<int>? onToken)
    {
        ArgumentNullException.ThrowIfNull(promptTokenIds);
        int[] prompt = promptTokenIds.ToArray();
        int[] generated = GenerateTokenIds(
            prompt,
            maxNewTokens,
            temperature,
            topK,
            stopTokenId,
            random);
        if (onToken is not null)
        {
            for (int index = prompt.Length; index < generated.Length; index++)
                onToken(generated[index]);
        }
        return generated;
    }

    internal abstract string Generate(
        string prompt,
        BpeTokenizer tokenizer,
        int maxNewTokens,
        float temperature,
        int topK,
        Random? random);

    /// <summary>
    /// Samples one logits row. CUDA greedy and top-K (1..64) reduce on-device
    /// and read only K value/index pairs; unrestricted topK=0 deliberately
    /// preserves the existing full-distribution host path.
    /// </summary>
    internal static int SampleLogits(
        Tensor logits,
        int offset,
        int count,
        float temperature,
        int topK,
        Random random)
    {
        ArgumentNullException.ThrowIfNull(logits);
        ArgumentNullException.ThrowIfNull(random);
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        if (offset > logits.Numel - count)
            throw new ArgumentOutOfRangeException(nameof(count));

        bool greedy = temperature == 0f || topK == 1;
        int candidateCount = greedy
            ? 1
            : topK == 0
                ? count
                : Math.Min(topK, count);
        if (logits.Device == TensorDevice.Cuda
            && (greedy || candidateCount <= 64))
        {
            CudaTopKCandidate[] candidates = logits
                .ReadCudaTopK(offset, count, candidateCount)
                .Candidates;
            if (greedy)
                return candidates[0].Index;
            return SampleCandidates(candidates, temperature, random);
        }

        IReadOnlyList<float> values = logits.Data;
        if (greedy)
            return ArgMax(values, offset, count);
        int[] indices = Enumerable.Range(0, count)
            .OrderByDescending(index => values[offset + index])
            .Take(candidateCount)
            .ToArray();
        var hostCandidates = new CudaTopKCandidate[candidateCount];
        for (int index = 0; index < candidateCount; index++)
        {
            hostCandidates[index] = new CudaTopKCandidate
            {
                Index = indices[index],
                Value = values[offset + indices[index]],
            };
        }
        return SampleCandidates(hostCandidates, temperature, random);
    }

    private static int SampleCandidates(
        IReadOnlyList<CudaTopKCandidate> candidates,
        float temperature,
        Random random)
    {
        float maximum = candidates[0].Value;
        var weights = new float[candidates.Count];
        float sum = 0f;
        for (int index = 0; index < candidates.Count; index++)
        {
            float value = candidates[index].Value;
            float weight;
            if (float.IsPositiveInfinity(maximum))
                weight = float.IsPositiveInfinity(value) ? 1f : 0f;
            else if (float.IsNegativeInfinity(maximum))
                weight = float.IsNegativeInfinity(value) ? 1f : 0f;
            else if (float.IsNaN(value))
                weight = 0f;
            else
                weight = MathF.Exp((value - maximum) / temperature);
            weights[index] = weight;
            sum += weight;
        }
        if (!(sum > 0f) || !float.IsFinite(sum))
            return candidates[0].Index;

        double threshold = random.NextDouble() * sum;
        float cumulative = 0f;
        for (int index = 0; index < candidates.Count; index++)
        {
            cumulative += weights[index];
            if (threshold <= cumulative)
                return candidates[index].Index;
        }
        return candidates[^1].Index;
    }

    private static int ArgMax(
        IReadOnlyList<float> values,
        int offset,
        int count)
    {
        int result = 0;
        float maximum = values[offset];
        for (int index = 1; index < count; index++)
        {
            float value = values[offset + index];
            if (value > maximum)
            {
                maximum = value;
                result = index;
            }
        }
        return result;
    }

    public int[] generate_token_ids(
        IEnumerable<int> prompt_token_ids,
        int max_new_tokens,
        float temperature,
        int top_k,
        int? stop_token_id,
        Random? random,
        Action<int>? on_token = null)
        => GenerateTokenIds(
            prompt_token_ids,
            max_new_tokens,
            temperature,
            top_k,
            stop_token_id,
            random,
            on_token);

    public string generate(
        string prompt,
        BpeTokenizer tokenizer,
        int max_new_tokens,
        float temperature,
        int top_k,
        Random? random)
        => Generate(
            prompt,
            tokenizer,
            max_new_tokens,
            temperature,
            top_k,
            random);
}
