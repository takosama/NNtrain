using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMix816NativeExpAttentionTests
{
    // native_exp is approximate. The acceptance budget is fixed before the
    // device comparison and is comfortably below the BF16 output quantum.
    private const double AbsoluteTolerance = 2e-6;
    private const double RelativeTolerance = 1e-5;

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void NativeExpFreshAndSavedSoftmaxStayNormalizedAndMatchStableReference(int causal)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        ArcDeviceInfo device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16,
            "The mix8_16 T2048 path requires an SG16 XMX device.");

        const int sequence = 2048, width = 96, heads = 3, first = 2, count = 2;
        const int rows = count * sequence, elements = rows * sequence;
        const float sentinel = -13.5f;
        const int maskedNanBits = 0x7f801234;
        var scores = new float[elements];
        var scoreBits = new int[elements];
        for (int row = 0; row < rows; ++row)
        {
            int query = row % sequence;
            int lastValid = causal == 0 ? sequence - 1 : query;
            int peak = causal == 0 ? (row * 17 + 113) % sequence : query / 2;
            int baseIndex = row * sequence;
            for (int key = 0; key < sequence; ++key)
            {
                if (key > lastValid)
                {
                    scoreBits[baseIndex + key] = maskedNanBits;
                    continue;
                }
                float value = (row % 4) switch
                {
                    0 => 7.25f, // exactly equal logits
                    1 => key == peak ? 10000f : -10000f, // finite extreme spread
                    2 => (((row * 23 + key * 17) % 257) - 128) * .1875f,
                    _ => (((row * 11 + key * 13) % 31) - 15) * .0009765625f,
                };
                Assert.True(float.IsFinite(value),
                    $"Non-finite input, causal={causal}, row={row}, key={key}.");
                scores[baseIndex + key] = value;
                scoreBits[baseIndex + key] = BitConverter.SingleToInt32Bits(value);
            }
        }

        var initialStats = Enumerable.Repeat(sentinel,
            (first + count + 1) * sequence * 2).ToArray();
        using var lane = new ArcExecutionLane();
        // Upload the exact score bits so the signaling-NaN sentinel cannot
        // be quieted by a managed floating-point copy before dispatch.
        using var fresh = lane.UploadRaw(scoreBits);
        using var saved = lane.UploadRaw(scoreBits);
        using var statistics = lane.Upload(initialStats);
        lane.Run("attention_probabilities_native_exp_2048", rows * 64L, 64,
            fresh, statistics, sequence, width, heads, first, causal, 0);
        lane.Run("attention_probabilities_saved_native_exp_2048", rows * 64L, 64,
            saved, statistics, sequence, width, heads, first, causal, 1);
        lane.Synchronize();

        var freshValues = new float[elements];
        var savedValues = new float[elements];
        var stats = new float[initialStats.Length];
        lane.Read(fresh, freshValues);
        lane.Read(saved, savedValues);
        lane.Read(statistics, stats);
        double largestAbsoluteError = 0, largestRelativeError = 0;
        for (int row = 0; row < rows; ++row)
        {
            int query = row % sequence;
            int validCount = causal == 0 ? sequence : query + 1;
            int limit = causal == 2 ? Math.Min(sequence, (query / 64 + 1) * 64) : sequence;
            int baseIndex = row * sequence;
            int statIndex = 2 * ((first + row / sequence) * sequence + query);
            Assert.True(float.IsFinite(stats[statIndex]),
                $"Non-finite maximum, causal={causal}, row={row}.");
            Assert.True(float.IsFinite(stats[statIndex + 1]) && stats[statIndex + 1] > 0,
                $"Invalid reciprocal sum, causal={causal}, row={row}.");

            double maximum = double.NegativeInfinity;
            double scale = 1.0 / Math.Sqrt(width / heads);
            for (int key = 0; key < validCount; ++key)
                maximum = Math.Max(maximum, scores[baseIndex + key] * scale);
            double normalizer = 0;
            for (int key = 0; key < validCount; ++key)
                normalizer += Math.Exp(scores[baseIndex + key] * scale - maximum);

            double rowSum = 0;
            for (int key = 0; key < sequence; ++key)
            {
                int index = baseIndex + key;
                int freshBits = BitConverter.SingleToInt32Bits(freshValues[index]);
                int savedBits = BitConverter.SingleToInt32Bits(savedValues[index]);
                Assert.True(freshBits == savedBits,
                    $"Fresh/saved bits differ, causal={causal}, row={row}, key={key}: "
                    + $"0x{freshBits:x8} vs 0x{savedBits:x8}.");
                if (key >= limit)
                {
                    Assert.Equal(scoreBits[index], freshBits);
                    continue;
                }
                if (key >= validCount)
                {
                    Assert.Equal(0f, freshValues[index]);
                    continue;
                }
                double expected = Math.Exp(scores[index] * scale - maximum) / normalizer;
                double actual = freshValues[index];
                Assert.True(double.IsFinite(actual) && actual >= 0,
                    $"Invalid probability, causal={causal}, row={row}, key={key}: {actual}.");
                double absoluteError = Math.Abs(actual - expected);
                largestAbsoluteError = Math.Max(largestAbsoluteError, absoluteError);
                if (expected > 0)
                    largestRelativeError = Math.Max(largestRelativeError, absoluteError / expected);
                Assert.True(absoluteError <= AbsoluteTolerance + RelativeTolerance * expected,
                    $"Reference error, causal={causal}, row={row}, key={key}: "
                    + $"actual={actual:G9}, expected={expected:G9}, error={absoluteError:G4}.");
                rowSum += actual;
            }
            Assert.True(Math.Abs(rowSum - 1.0) <= 2e-5,
                $"Row sum, causal={causal}, row={row}: {rowSum:G9}.");
        }

        // The two selected heads straddle a batch boundary; all other head
        // statistics must retain their sentinel values.
        for (int h = 0; h < first + count + 1; ++h)
        {
            if (h >= first && h < first + count) continue;
            for (int i = 0; i < sequence * 2; ++i)
                Assert.Equal(sentinel, stats[h * sequence * 2 + i]);
        }
        TestContext.Current.TestOutputHelper?.WriteLine(
            $"causal={causal}: maximum absolute error={largestAbsoluteError:G6}, "
            + $"maximum relative error={largestRelativeError:G6}; fresh GPU="
            + $"{lane.KernelTimings.GetValueOrDefault("attention_probabilities_native_exp_2048"):F4} ms, "
            + $"saved GPU={lane.KernelTimings.GetValueOrDefault("attention_probabilities_saved_native_exp_2048"):F4} ms");
    }
}
