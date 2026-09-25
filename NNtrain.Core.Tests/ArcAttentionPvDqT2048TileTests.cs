using NNtrain;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

public sealed class ArcAttentionPvDqT2048TileTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void TileCandidatesPreserveFp32BitsAcrossBatchBoundaryAndAccumulation(bool backward, bool causal)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int sequence = 2048, width = 512, heads = 16, first = 15, count = 2, channels = 32;
        int outputComponents = backward ? 3 : 1;
        var matrix = new float[count * sequence * sequence];
        for (int head = 0; head < count; ++head)
        for (int query = 0; query < sequence; ++query)
        for (int key = 0; key < sequence; ++key)
        {
            // The production softmax publishes exact zero through the next
            // 64-key boundary. Check both the bounded and dense products.
            matrix[(head * sequence + query) * sequence + key] = causal && key > query
                ? 0f : ((query * 7 + key * 11 + head * 13) % 101 + 1) * .0001f;
        }
        var qkv = new float[2 * sequence * 3 * width];
        for (int i = 0; i < qkv.Length; ++i) qkv[i] = (i % 97 - 48) * .0003f;
        var initial = new float[2 * sequence * outputComponents * width];
        for (int i = 0; i < initial.Length; ++i) initial[i] = (i % 17 - 8) * .0001f;

        using var lane = new ArcExecutionLane();
        using ArcBuffer a = lane.Upload(matrix);
        using ArcBuffer b = lane.Upload(qkv);
        using ArcBuffer reference = lane.Upload(initial);
        using ArcBuffer candidate = lane.Upload(initial);
        string operation = backward ? "dq" : "pv";
        string baseline = $"attention_fp32_{operation}_d32_aligned";

        void Dispatch(string kernel, int tile, ArcBuffer destination)
            => lane.Run3D(kernel, 16, sequence / tile * 16L, count, 16, 16, 1,
                a, b, destination, sequence, channels, sequence,
                sequence, 1, sequence * sequence, 0, 0, 0,
                3 * width, 1, 0, sequence * 3 * width, channels, (backward ? 1 : 2) * width,
                outputComponents * width, 1, 0, sequence * outputComponents * width, channels, 0,
                heads, first, backward ? 1 : 0, causal ? 2 : 0);

        foreach (int tile in new[] { 32, 64 })
        {
            string kernel = $"attention_fp32_{operation}_d32_t2048_m{tile}k{tile}";
            lane.Write(candidate, initial);
            lane.Write(reference, initial);
            for (int repeat = 0; repeat < (backward ? 2 : 1); ++repeat)
            {
                Dispatch(baseline, 64, reference);
                Dispatch(kernel, tile, candidate);
                var expected = new float[initial.Length];
                var actual = new float[initial.Length];
                lane.Read(reference, expected);
                lane.Read(candidate, actual);
                for (int index = 0; index < actual.Length; ++index)
                    if (BitConverter.SingleToInt32Bits(expected[index]) != BitConverter.SingleToInt32Bits(actual[index]))
                        Assert.Fail($"{kernel} {operation} causal={causal} repeat={repeat} "
                            + $"element={index}: {expected[index]:R} versus {actual[index]:R}.");
            }
        }
    }
}
