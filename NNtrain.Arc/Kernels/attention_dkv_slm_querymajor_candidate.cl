// D32 causal dK/dV probe for long sequences (in particular T=2048).
//
// The selected K32/Q32 block-SLM kernel stages P/dS as [key][query]. The
// producing lanes traverse keys, so that placement scatters a score row
// across SLM banks. These probes stage the same values as [query][key]; the
// consumer then broadcasts one key's score to the subgroup. Both variants
// have identical global reads, output ownership, initial accumulators and
// ascending-query FP32 FMA order. Only the SLM layout differs.
//
// Preconditions match attention_dkv_block_slm_causal: D=32, sequence a
// positive multiple of 32, width=heads*32, complete head tiles, SG16 and
// cl_intel_subgroup_local_block_io. Launch (16,seq/32*16,count) with local
// (16,16,1); arguments are (qkv,dy,p,ds,dx,seq,width,heads,first,causal).
#if defined(ARC_XMX) && ARC_SG == 16 && defined(ARC_SLM_BLOCK_IO) && defined(ARC_OPTIMIZATION_PROBES)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

#define ARC_DKV_SLM_QUERYMAJOR(NAME, SCORE_PITCH) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(16,16,1))) \
__kernel void NAME(__global const float* qkv, __global const float* dy, \
    __global const float* p, __global const float* ds, __global float* dx, \
    int seq, int width, int heads, int first, int runtimeCausal) { \
    const int lx = get_local_id(0), ly = get_local_id(1); \
    const int tid = ly * 16 + lx; \
    const int g = get_group_id(2), h = first + g; \
    const int kb = get_group_id(1) * 32; \
    const int qb = (h / heads) * seq * 3 * width + (h % heads) * 32; \
    const int yb = (h / heads) * seq * width + (h % heads) * 32; \
    __local float2 scores[32][SCORE_PITCH]; \
    __local uint qy[32][64]; \
    float4 sums[2]; \
    _Pragma("unroll") \
    for (int r = 0; r < 2; ++r) { \
        const int off = qb + (kb + ly + 16 * r) * 3 * width + width + lx; \
        sums[r] = (float4)(dx[off], dx[off + 16], \
            dx[off + width], dx[off + width + 16]); \
    } \
    for (int base = kb; base < seq; base += 32) { \
        _Pragma("unroll") \
        for (int i = tid; i < 1024; i += 256) { \
            const int query = i / 32, key = i % 32, qi = base + query; \
            const int score = (g * seq + qi) * seq + kb + key; \
            scores[query][key] = (float2)(p[score], ds[score]); \
            qy[query][key] = as_uint(qkv[qb + qi * 3 * width + key]); \
            qy[query][key + 32] = as_uint(dy[yb + qi * width + key]); \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        _Pragma("unroll") \
        for (int inner = 0; inner < 32; ++inner) { \
            const float4 xy = as_float4(intel_sub_group_block_read4(&qy[inner][0])); \
            _Pragma("unroll") \
            for (int r = 0; r < 2; ++r) { \
                const float2 weights = scores[inner][ly + 16 * r]; \
                sums[r] = fma(weights.yyxx, xy, sums[r]); \
            } \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    _Pragma("unroll") \
    for (int r = 0; r < 2; ++r) { \
        const int off = qb + (kb + ly + 16 * r) * 3 * width + width + lx; \
        dx[off] = sums[r].s0; \
        dx[off + 16] = sums[r].s1; \
        dx[off + width] = sums[r].s2; \
        dx[off + width + 16] = sums[r].s3; \
    } \
}

// Pitch 32 favors contiguous SLM writes; pitch 33 spreads successive query
// rows across banks. Measure both rather than assuming either is better.
ARC_DKV_SLM_QUERYMAJOR(attention_dkv_slm_qmajor32_causal, 32)
ARC_DKV_SLM_QUERYMAJOR(attention_dkv_slm_qmajor33_causal, 33)
#undef ARC_DKV_SLM_QUERYMAJOR
#endif
