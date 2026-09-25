using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcAttentionRowSlm2048Tests
{
    [Fact]
    public void SlmCandidateMatchesReferenceBitsForFreshSavedAndBackward()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var lane = new ArcExecutionLane();
        const int sequence = 2048, heads = 3, headWidth = 32, first = 2, count = 2;
        const int width = heads * headWidth, rows = count * sequence, elements = rows * sequence;
        foreach (int causal in new[] { 0, 1, 2 })
        {
            var scores = new float[elements];
            var gradients = new float[elements];
            for (int i = 0; i < elements; ++i)
            {
                scores[i] = (((i * 17) % 257) - 128) * 0.1875f;
                gradients[i] = MathF.ScaleB((((i * 23) % 109) - 54) * 0.03125f, i % 9 - 4);
            }
            if (causal != 0)
                for (int head = 0; head < count; ++head)
                for (int query = 0; query < sequence; ++query)
                for (int key = query + 1; key < sequence; ++key)
                {
                    int index = (head * sequence + query) * sequence + key;
                    scores[index] = BitConverter.Int32BitsToSingle(0x7f801234);
                    gradients[index] = BitConverter.Int32BitsToSingle(unchecked((int)0xff812345));
                }

            Snapshot Run(string forward, string savedForward, string backward)
            {
                using var probability = lane.Upload(scores);
                using var recomputed = lane.Upload(scores);
                using var derivative = lane.Upload(gradients);
                using var statistics = lane.Upload(Enumerable.Repeat(-0.125f,
                    (first + count + 1) * sequence * 2).ToArray());
                long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
                lane.Run(forward, rows * 64L, 64, probability, statistics,
                    sequence, width, heads, first, causal, 0);
                lane.Run(savedForward, rows * 64L, 64, recomputed, statistics,
                    sequence, width, heads, first, causal, 1);
                lane.Run(backward, rows * 64L, 64, probability, derivative,
                    sequence, width, heads, causal);
                Assert.Equal(uploads, lane.H2DBytes);
                Assert.Equal(downloads, lane.D2HBytes);

                var p = new float[elements];
                var savedP = new float[elements];
                var dp = new float[elements];
                var stats = new float[(first + count + 1) * sequence * 2];
                lane.Read(probability, p);
                lane.Read(recomputed, savedP);
                lane.Read(derivative, dp);
                lane.Read(statistics, stats);
                AssertBits(p, savedP, $"fresh/saved, causal={causal}");
                return new(p, savedP, dp, stats);
            }

            Snapshot expected = Run("attention_probabilities", "attention_probabilities",
                "attention_derivatives");
            Snapshot actual = Run("attention_probabilities_slm_2048",
                "attention_probabilities_saved_direct_2048", "attention_derivatives_slm_2048");
            AssertBits(expected.Probabilities, actual.Probabilities, $"probability, causal={causal}");
            AssertBits(expected.SavedProbabilities, actual.SavedProbabilities, $"saved probability, causal={causal}");
            AssertBits(expected.Derivatives, actual.Derivatives, $"derivative, causal={causal}");
            AssertBits(expected.Statistics, actual.Statistics, $"statistics, causal={causal}");
        }
    }

    private static void AssertBits(float[] expected, float[] actual, string description)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i)
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                Assert.Fail($"{description}, index {i}: 0x{BitConverter.SingleToInt32Bits(expected[i]):x8} vs "
                    + $"0x{BitConverter.SingleToInt32Bits(actual[i]):x8}");
    }

    private sealed record Snapshot(float[] Probabilities, float[] SavedProbabilities,
        float[] Derivatives, float[] Statistics);
}
