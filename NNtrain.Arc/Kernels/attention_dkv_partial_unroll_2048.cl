// Experimental T2048/D32 causal dK/dV: the production K32/Q32 SLM kernel
// with only the inner-loop unroll factor changed. Each output still accumulates
// query contributions in ascending order with FP32 fma; no reduction or
// precision change is introduced. Host dispatch must enforce T2048/D32.
#if defined(ARC_XMX) && ARC_SG == 16 && defined(ARC_SLM_BLOCK_IO) && defined(ARC_OPTIMIZATION_PROBES)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

#define ARC_DKV_PARTIAL_UNROLL_2048(NAME, INNER_UNROLL) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(16,16,1))) \
__kernel void NAME(__global const float* qkv, __global const float* dy, \
    __global const float* p, __global const float* ds, __global float* dx, \
    int seq, int width, int heads, int first, int runtimeCausal) { \
    int lx = get_local_id(0), ly = get_local_id(1), tid = ly * 16 + lx; \
    int g = get_group_id(2), h = first + g, kb = get_group_id(1) * 32; \
    int qb = (h / heads) * seq * 3 * width + (h % heads) * 32; \
    int yb = (h / heads) * seq * width + (h % heads) * 32; \
    __local float2 pairs[32][33]; \
    __local uint qy[32][64]; \
    float4 sums[2]; \
    _Pragma("unroll") \
    for (int r = 0; r < 2; ++r) { \
        int off = qb + (kb + ly + 16 * r) * 3 * width + width + lx; \
        sums[r] = (float4)(dx[off], dx[off + 16], dx[off + width], dx[off + width + 16]); \
    } \
    for (int base = kb; base < seq; base += 32) { \
        _Pragma("unroll") \
        for (int i = tid; i < 1024; i += 256) { \
            int query = i / 32, key = i % 32; \
            int score = (g * seq + base + query) * seq + kb + key; \
            pairs[key][query] = (float2)(p[score], ds[score]); \
            int qi = base + query; \
            qy[query][key] = as_uint(qkv[qb + qi * 3 * width + key]); \
            qy[query][key + 32] = as_uint(dy[yb + qi * width + key]); \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        _Pragma(INNER_UNROLL) \
        for (int inner = 0; inner < 32; ++inner) { \
            float4 xy = as_float4(intel_sub_group_block_read4(&qy[inner][0])); \
            _Pragma("unroll") \
            for (int r = 0; r < 2; ++r) { \
                float2 weights = pairs[ly + 16 * r][inner]; \
                sums[r] = fma(weights.yyxx, xy, sums[r]); \
            } \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    _Pragma("unroll") \
    for (int r = 0; r < 2; ++r) { \
        int off = qb + (kb + ly + 16 * r) * 3 * width + width + lx; \
        dx[off] = sums[r].s0; \
        dx[off + 16] = sums[r].s1; \
        dx[off + width] = sums[r].s2; \
        dx[off + width + 16] = sums[r].s3; \
    } \
}

ARC_DKV_PARTIAL_UNROLL_2048(attention_dkv_block_slm_2048_unroll4_causal, "unroll 4")
ARC_DKV_PARTIAL_UNROLL_2048(attention_dkv_block_slm_2048_unroll8_causal, "unroll 8")
ARC_DKV_PARTIAL_UNROLL_2048(attention_dkv_block_slm_2048_unroll16_causal, "unroll 16")
#undef ARC_DKV_PARTIAL_UNROLL_2048
#endif
