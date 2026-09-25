using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMix816AttentionBPair2048CandidateTests
{
    [Fact]
    public void PackedBPanelPreservesPvAndDqBitsAtFullSequence()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        ArcDeviceInfo device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16
            || !device.Extensions.Split(' ').Contains("cl_intel_subgroup_local_block_io"),
            "The candidate requires an SG16 XMX device with SLM block I/O.");

        const int batch = 2, sequence = 2048, width = 64, heads = 2, first = 1, count = 2;
        // The selected heads are batch 0/head 1 and batch 1/head 0.
        var scores = new float[count * sequence * sequence];
        var packedScores = new uint[count * sequence * sequence];
        for(int group=0;group<count;group++)
        for(int row=0;row<sequence;row++)
        for(int key=0;key<=row;key++)
        {
            int index=(group*sequence+row)*sequence+key;
            scores[index]=((group*19+row*17+key*13)%251+1)*1e-5f;
            ushort probabilityBits=Bf16(scores[index]);
            ushort ds=Bf16(MathF.Sin((group*31+row-key)*.023f)*.0005f);
            packedScores[index]=(uint)probabilityBits|((uint)ds<<16);
        }
        ushort[] qkvValues=Enumerable.Range(0,batch*sequence*3*width)
            .Select(i=>Bf16(MathF.Sin(i*.013f)*.09f)).ToArray();
        float[] dxSeed=Enumerable.Range(0,batch*sequence*3*width)
            .Select(i=>MathF.Cos(i*.009f)*.001f).ToArray();

        using var lane=new ArcExecutionLane();
        using var qkv=lane.UploadRaw(qkvValues);
        using var p=lane.Upload(scores);
        using var packed=lane.UploadRaw(packedScores);
        using var pvBaseline=lane.Upload(new float[batch*sequence*width]);
        using var pvCandidate=lane.Upload(new float[batch*sequence*width]);
        using var dqBaseline=lane.Upload(dxSeed);
        using var dqCandidate=lane.Upload(dxSeed);
        using var dqCandidateAb=lane.Upload(dxSeed);
        lane.Run3D("attention_mix8_16_bf16_pv_d32_2048",16,
            sequence/64L*16,count,16,16,1,
            p,qkv,pvBaseline,sequence,width,heads,first);
        lane.Run3D("attention_mix8_16_bf16_pv_bpair_d32_2048",16,
            sequence/64L*16,count,16,16,1,
            p,qkv,pvCandidate,sequence,width,heads,first);
        lane.Run3D("attention_mix8_16_bf16_dq_packed_2048",16,
            sequence/64L*16,count,16,16,1,
            packed,qkv,dqBaseline,sequence,width,heads,first);
        lane.Run3D("attention_mix8_16_bf16_dq_packed_bpair_2048",16,
            sequence/64L*16,count,16,16,1,
            packed,qkv,dqCandidate,sequence,width,heads,first);
        lane.Run3D("attention_mix8_16_bf16_dq_packed_ab_2048",16,
            sequence/64L*16,count,16,16,1,
            packed,qkv,dqCandidateAb,sequence,width,heads,first);
        lane.Synchronize();

        var expectedPv=new float[batch*sequence*width];
        var actualPv=new float[batch*sequence*width];
        var expectedDq=new float[dxSeed.Length];
        var actualDq=new float[dxSeed.Length];
        var actualDqAb=new float[dxSeed.Length];
        lane.Read(pvBaseline,expectedPv);
        lane.Read(pvCandidate,actualPv);
        lane.Read(dqBaseline,expectedDq);
        lane.Read(dqCandidate,actualDq);
        lane.Read(dqCandidateAb,actualDqAb);
        for(int i=0;i<expectedPv.Length;i++)
            Assert.True(BitConverter.SingleToInt32Bits(expectedPv[i])
                ==BitConverter.SingleToInt32Bits(actualPv[i]),
                $"PV differs at output index {i}.");
        for(int i=0;i<expectedDq.Length;i++)
        {
            Assert.True(BitConverter.SingleToInt32Bits(expectedDq[i])
                ==BitConverter.SingleToInt32Bits(actualDq[i]),
                $"dQ B-pair differs at gradient index {i}.");
            Assert.True(BitConverter.SingleToInt32Bits(expectedDq[i])
                ==BitConverter.SingleToInt32Bits(actualDqAb[i]),
                $"dQ A+B compressed differs at gradient index {i}.");
        }
        foreach(string kernel in new[]{
            "attention_mix8_16_bf16_pv_d32_2048",
            "attention_mix8_16_bf16_pv_bpair_d32_2048",
            "attention_mix8_16_bf16_dq_packed_2048",
            "attention_mix8_16_bf16_dq_packed_bpair_2048",
            "attention_mix8_16_bf16_dq_packed_ab_2048",
        })
            TestContext.Current.TestOutputHelper?.WriteLine(
                $"{kernel}: {lane.KernelTimings.GetValueOrDefault(kernel):F4} GPU ms");
    }

    private static ushort Bf16(float value)
    {
        uint bits=unchecked((uint)BitConverter.SingleToInt32Bits(value));
        if((bits&0x7f800000u)!=0x7f800000u)
            bits+=0x7fffu+((bits>>16)&1u);
        return (ushort)(bits>>16);
    }
}
