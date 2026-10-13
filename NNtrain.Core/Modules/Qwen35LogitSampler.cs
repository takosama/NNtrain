namespace NNtrain;

/// <summary>CPU top-k and nucleus sampling independent of resident model state.</summary>
internal static class Qwen35LogitSampler
{
    internal static void ValidateSampling(float temperature, float topP, int topK)
    {
        if (!float.IsFinite(temperature) || temperature < 0f)
            throw new ArgumentOutOfRangeException(nameof(temperature));
        if (!float.IsFinite(topP) || topP <= 0f || topP > 1f)
            throw new ArgumentOutOfRangeException(nameof(topP));
        if (topK < 1)
            throw new ArgumentOutOfRangeException(nameof(topK));
    }

    /// <summary>
    /// Selects top-k before nucleus filtering, as required by Qwen sampling.
    /// The returned ID is reproducible when the caller supplies a seeded Random.
    /// </summary>
    internal static int SampleLogits(ReadOnlySpan<float> logits,
        float temperature, float topP, int topK, Random random)
    {
        ValidateSampling(temperature, topP, topK);
        ArgumentNullException.ThrowIfNull(random);
        if (logits.IsEmpty) throw new ArgumentException("Logits cannot be empty.", nameof(logits));

        int bestId = 0;
        float bestValue = float.NegativeInfinity;
        int count = Math.Min(topK, logits.Length);
        bool greedy = temperature == 0f || count == 1;
        var candidates = greedy ? null :
            new PriorityQueue<(int Id, float Value), (float Value, int NegativeId)>(count);
        for (int id = 0; id < logits.Length; id++)
        {
            float value = logits[id];
            if (!float.IsFinite(value))
                throw new ArithmeticException("Qwen3.5 produced non-finite logits.");
            if (value > bestValue) { bestValue = value; bestId = id; }
            if (greedy) continue;

            (float Value, int NegativeId) priority = (value, -id);
            if (candidates!.Count < count)
                candidates.Enqueue((id, value), priority);
            else if (candidates.TryPeek(out _, out var worst) && priority.CompareTo(worst) > 0)
            {
                candidates.Dequeue();
                candidates.Enqueue((id, value), priority);
            }
        }
        if (greedy) return bestId;

        (int Id, float Value)[] ordered = candidates!.UnorderedItems
            .Select(item => item.Element).ToArray();
        Array.Sort(ordered, (left, right) =>
        {
            int byValue = right.Value.CompareTo(left.Value);
            return byValue != 0 ? byValue : left.Id.CompareTo(right.Id);
        });
        var weights = new double[ordered.Length];
        double sum = 0;
        double maximum = ordered[0].Value;
        for (int i = 0; i < ordered.Length; i++)
        {
            double weight = Math.Exp(((double)ordered[i].Value - maximum) / temperature);
            weights[i] = weight;
            sum += weight;
        }
        double nucleusMass = 0;
        int nucleusCount = 0;
        do
        {
            nucleusMass += weights[nucleusCount++];
        } while (nucleusCount < ordered.Length && nucleusMass / sum < topP);

        double draw = random.NextDouble() * nucleusMass;
        double cumulative = 0;
        for (int i = 0; i < nucleusCount; i++)
        {
            cumulative += weights[i];
            if (draw < cumulative) return ordered[i].Id;
        }
        return ordered[nucleusCount - 1].Id;
    }
}
