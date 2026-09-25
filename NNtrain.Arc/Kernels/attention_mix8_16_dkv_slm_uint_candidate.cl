// T2048/D32 causal dK/dV candidate for physical-BF16 QKV and packed BF16
// probability/score gradients. Keep the selected K32/Q32 tile, subgroup
// ownership, query order, and FP32 FMA order. Staging a score pair as its
// original uint reduces score SLM from 32*33*8 to 32*33*4 bytes. The pair is
// expanded only when the subgroup consumes it.
//
// Launch: global (16, (seq/32)*16, count), local (16,16,1).
// Arguments and complete-head/causal/D32 prerequisites match
// attention_mix8_16_bf16_dkv_packed_2048.
#if defined(ARC_XMX) && ARC_SG == 16 && defined(ARC_SLM_BLOCK_IO)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void attention_mix8_16_bf16_dkv_packed_slm_uint_2048(
    __global const ushort* qkv, __global const float* dy,
    __global const uint* packed, __global float* dx,
    int seq, int width, int heads, int first) {
    const int lx = get_local_id(0), ly = get_local_id(1), tid = ly * 16 + lx;
    const int g = get_group_id(2), h = first + g, kb = get_group_id(1) * 32;
    const int qb = (h / heads) * seq * 3 * width + (h % heads) * 32;
    const int yb = (h / heads) * seq * width + (h % heads) * 32;
    __local uint pairs[32][33];
    __local uint qy[32][64];
    float4 sums[2];
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int off = qb + (kb + ly + 16 * r) * 3 * width + width + lx;
        sums[r] = (float4)(dx[off], dx[off + 16],
            dx[off + width], dx[off + width + 16]);
    }
    for (int base = kb; base < seq; base += 32) {
        #pragma unroll
        for (int i = tid; i < 1024; i += 256) {
            const int query = i / 32, key = i % 32;
            const int qi = base + query;
            const int score = (g * seq + qi) * seq + kb + key;
            pairs[key][query] = packed[score];
            qy[query][key] = (uint)qkv[qb + qi * 3 * width + key] << 16;
            qy[query][key + 32] = as_uint(dy[yb + qi * width + key]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int inner = 0; inner < 32; ++inner) {
            const float4 xy = as_float4(intel_sub_group_block_read4(&qy[inner][0]));
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                const uint bits = pairs[ly + 16 * r][inner];
                const float p = as_float(bits << 16);
                const float ds = as_float(bits & 0xffff0000u);
                sums[r] = fma((float4)(ds, ds, p, p), xy, sums[r]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int off = qb + (kb + ly + 16 * r) * 3 * width + width + lx;
        dx[off] = sums[r].s0;
        dx[off + 16] = sums[r].s1;
        dx[off + width] = sums[r].s2;
        dx[off + width + 16] = sums[r].s3;
    }
}
#endif
