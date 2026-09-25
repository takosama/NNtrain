// Decode the resident BFP8 QKV tensor exactly as DecodeArcReplica(..., true)
// while publishing the same BF16 Q/K DPAS panels as the subsequent combined
// QK pack. The FP32 V/Q/K view remains available to PV, dP, dQ and dK/dV;
// this is a bandwidth/launch candidate, not a changed attention precision.
#if defined(ARC_XMX) && ARC_SG == 16
float round_bf16(float x);
ushort xmx_bf16(float x);

__kernel void attention_bfp8_decode_qk_2048_candidate(
    __global const char* payload, __global const float* scales,
    __global float* decoded, __global uint* packedQ, __global uint* packedK,
    int batch, int seq, int width, int heads, int block) {
    // Pair boundaries coincide with BFP8 block32 and head D32 boundaries.
    // A single work item publishes each uint, avoiding adjacent halfword
    // writers sharing a device transaction.
    int i = get_global_id(0) * 2;
    int n = batch * seq * 3 * width;
    if (i >= n) return;
    char2 encoded = vload2(0, payload + i);
    float scale = scales[i / block];
    float x = round_bf16((float)encoded.s0 * scale);
    float y = round_bf16((float)encoded.s1 * scale);
    vstore2((float2)(x, y), 0, decoded + i);

    int tokenGlobal = i / (3 * width);
    int token = tokenGlobal % seq;
    int component = (i % (3 * width)) / width;
    if (component == 2) return;
    int feature = i % width;
    int head = feature / 32;
    int channel = feature % 32;
    int g = (tokenGlobal / seq) * heads + head;
    uint bits = (uint)xmx_bf16(x) | ((uint)xmx_bf16(y) << 16);
    if (component == 0) {
        int address = g * seq * 32
            + ((channel / 16) * (seq / 8) + token / 8) * 128
            + (token % 8) * 16 + channel % 16;
        packedQ[address / 2] = bits;
    } else {
        int pair = g * seq * 16
            + ((channel / 16) * (seq / 16) + token / 16) * 128
            + ((channel % 16) / 2) * 16 + token % 16;
        packedK[pair] = bits;
    }
}
#endif
