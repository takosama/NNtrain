using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMix816AttentionReuse2048CandidateTests
{
    [Fact]
    public void K32AndSharedPanelTilesMatchProductionAttentionBitwise()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        ArcDeviceInfo device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16
            || !device.Extensions.Split(' ').Contains("cl_intel_subgroup_local_block_io"),
            "The candidates require an SG16 XMX device with SLM block I/O.");

        const int sequence = 2048, width = 64, heads = 2, first = 1, count = 1;
        const int scoreCount = sequence * sequence;
        var probability = new float[scoreCount];
        for (int row = 0; row < sequence; ++row)
        for (int key = 0; key <= row; ++key)
            probability[row * sequence + key] = ((row * 17 + key * 13) % 251 + 1) * 1e-5f;
        ushort[] qkvValues = Enumerable.Range(0, sequence * 3 * width)
            .Select(i => Bf16(MathF.Sin(i * .013f) * .09f)).ToArray();
        float[] dyValues = Enumerable.Range(0, sequence * width)
            .Select(i => MathF.Cos(i * .017f) * .012f).ToArray();
        float[] deltaValues = Enumerable.Range(0, heads * sequence)
            .Select(i => MathF.Sin(i * .011f) * .004f).ToArray();
        float[] dxSeed = Enumerable.Range(0, sequence * 3 * width)
            .Select(i => MathF.Sin(i * .009f) * .001f).ToArray();

        using var lane = new ArcExecutionLane();
        using var qkv = lane.UploadRaw(qkvValues);
        using var dy = lane.Upload(dyValues);
        using var delta = lane.Upload(deltaValues);
        using var pForward = lane.Upload(probability);
        using var pvBaseline = lane.Upload(new float[sequence * width]);
        using var pvCandidate = lane.Upload(new float[sequence * width]);
        lane.Run3D("attention_mix8_16_bf16_pv_d32_2048", 16,
            sequence / 64L * 16, count, 16, 16, 1,
            pForward, qkv, pvBaseline, sequence, width, heads, first);
        lane.Run3D("attention_mix8_16_bf16_pv_m128_d32_2048", 16,
            sequence / 128L * 16, count, 16, 16, 1,
            pForward, qkv, pvCandidate, sequence, width, heads, first);

        using var packedBaseline = lane.Upload(probability);
        using var packedCandidate = lane.Upload(probability);
        lane.Run3D("attention_mix8_16_bf16_dpds_d32_2048", sequence / 64L * 16,
            sequence / 64L * 16, count, 16, 16, 1,
            dy, qkv, packedBaseline, delta, sequence, width, heads, first);
        lane.Run3D("attention_mix8_16_bf16_dpds_k32_d32_2048", sequence / 64L * 16,
            sequence / 64L * 16, count, 16, 16, 1,
            dy, qkv, packedCandidate, delta, sequence, width, heads, first);

        using var dqBaseline = lane.Upload(dxSeed);
        using var dqCandidate = lane.Upload(dxSeed);
        lane.Run3D("attention_mix8_16_bf16_dq_packed_2048", 16,
            sequence / 64L * 16, count, 16, 16, 1,
            packedBaseline, qkv, dqBaseline, sequence, width, heads, first);
        lane.Run3D("attention_mix8_16_bf16_dq_packed_m128_2048", 16,
            sequence / 128L * 16, count, 16, 16, 1,
            packedBaseline, qkv, dqCandidate, sequence, width, heads, first);
        lane.Synchronize();

        var expectedPv = new float[sequence * width];
        var actualPv = new float[sequence * width];
        lane.Read(pvBaseline, expectedPv);
        lane.Read(pvCandidate, actualPv);
        var expectedPacked = new uint[scoreCount];
        var actualPacked = new uint[scoreCount];
        lane.ReadRaw(packedBaseline, expectedPacked);
        lane.ReadRaw(packedCandidate, actualPacked);
        var expectedDq = new float[dxSeed.Length];
        var actualDq = new float[dxSeed.Length];
        lane.Read(dqBaseline, expectedDq);
        lane.Read(dqCandidate, actualDq);

        for (int token = 0; token < sequence; ++token)
        for (int channel = 0; channel < 32; ++channel)
        {
            int pvIndex = token * width + first * 32 + channel;
            Assert.True(BitConverter.SingleToInt32Bits(expectedPv[pvIndex])
                == BitConverter.SingleToInt32Bits(actualPv[pvIndex]),
                $"PV differs at token {token}, channel {channel}.");
            int dqIndex = token * 3 * width + first * 32 + channel;
            Assert.True(BitConverter.SingleToInt32Bits(expectedDq[dqIndex])
                == BitConverter.SingleToInt32Bits(actualDq[dqIndex]),
                $"dQ differs at token {token}, channel {channel}.");
        }
        for (int i = 0; i < scoreCount; ++i)
            Assert.True(expectedPacked[i] == actualPacked[i],
                $"Fused dP/dS differs at score {i}: "
                + $"0x{expectedPacked[i]:x8} vs 0x{actualPacked[i]:x8}.");

        foreach (string kernel in new[] {
            "attention_mix8_16_bf16_pv_d32_2048",
            "attention_mix8_16_bf16_pv_m128_d32_2048",
            "attention_mix8_16_bf16_dpds_d32_2048",
            "attention_mix8_16_bf16_dpds_k32_d32_2048",
            "attention_mix8_16_bf16_dq_packed_2048",
            "attention_mix8_16_bf16_dq_packed_m128_2048",
        })
            TestContext.Current.TestOutputHelper?.WriteLine(
                $"{kernel}: {lane.KernelTimings.GetValueOrDefault(kernel):F4} GPU ms");
    }

    private static ushort Bf16(float value)
    {
        uint bits = unchecked((uint)BitConverter.SingleToInt32Bits(value));
        if ((bits & 0x7f800000u) != 0x7f800000u)
            bits += 0x7fffu + ((bits >> 16) & 1u);
        return (ushort)(bits >> 16);
    }
}
