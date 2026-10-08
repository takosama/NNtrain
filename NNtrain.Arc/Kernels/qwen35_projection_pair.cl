// Inference candidate: one SG16 computes two adjacent output channels and
// shares the activation loads. Each output keeps the existing single-output
// SG16 decode, FMA and reduction order. Packed weights are never expanded.
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

inline float8 q35p_iq2_octet(__global const uchar* block, int octet, float d) {
    int group = octet >> 2, part = octet & 3;
    int gridIndex = block[2 + octet] | (((block[66 + group] >> (part * 2)) & 3) << 8);
    float8 grid = convert_float8(as_uchar8(q35l_iq2s_grid[gridIndex]));
    int signs = block[34 + octet];
    int scale = (block[74 + group] >> ((part >> 1) * 4)) & 15;
    float multiplier = d * (0.5f + (float)scale) * 0.25f;
    grid = select(grid, -grid, ((int8)(signs) & (int8)(1,2,4,8,16,32,64,128)) != (int8)(0));
    return multiplier * grid;
}

inline float8 q35p_iq3_octet(__global const uchar* block, int octet, float d) {
    int group = octet >> 2, pair = (octet & 3) * 2;
    int high = block[66 + group];
    int grid0 = block[2 + octet * 2] | (((high >> pair) & 1) << 8);
    int grid1 = block[3 + octet * 2] | (((high >> (pair + 1)) & 1) << 8);
    float8 grid = convert_float8(as_uchar8((uint2)(q35l_iq3s_grid[grid0], q35l_iq3s_grid[grid1])));
    int signs = block[74 + octet];
    int scale = (block[106 + (group >> 1)] >> ((group & 1) * 4)) & 15;
    float multiplier = d * (float)(1 + 2 * scale);
    grid = select(grid, -grid, ((int8)(signs) & (int8)(1,2,4,8,16,32,64,128)) != (int8)(0));
    return multiplier * grid;
}

inline float8 q35p_iq2_octet_levels3(__global const uchar* block, int octet, float d) {
    // Exceptional coefficients retain the original NaN/Inf arithmetic. Finite
    // half coefficients cannot overflow any of the three FP32 grid products.
    if (!isfinite(d)) return q35p_iq2_octet(block, octet, d);
    int group = octet >> 2, part = octet & 3;
    int gridIndex = block[2 + octet] | (((block[66 + group] >> (part * 2)) & 3) << 8);
    uchar8 grid = as_uchar8(q35l_iq2s_grid[gridIndex]);
    int signs = block[34 + octet];
    int scale = (block[74 + group] >> ((part >> 1) * 4)) & 15;
    float multiplier = d * (0.5f + (float)scale) * 0.25f;
    uint v8 = as_uint(multiplier * 8.0f), v25 = as_uint(multiplier * 25.0f);
    uint v43 = as_uint(multiplier * 43.0f);
    uint8 bits = select((uint8)(v43), (uint8)(v25), convert_uint8(grid == (uchar8)(25)));
    bits = select(bits, (uint8)(v8), convert_uint8(grid == (uchar8)(8)));
    uint8 signBits = (((uint8)(signs) >> (uint8)(0,1,2,3,4,5,6,7)) & (uint8)(1)) << 31;
    return as_float8(bits ^ signBits);
}

#define Q35P_IQ_DOT(NAME, BYTES, DECODE) \
inline float2 NAME(__global const float* x, __global const uchar* weight0, \
    __global const uchar* weight1, int blocks, int lane) { \
    float8 sums0 = (float8)(0.0f), sums1 = (float8)(0.0f); \
    for (int bi = 0; bi < blocks; ++bi) { \
        __global const uchar* block0 = weight0 + bi * BYTES; \
        __global const uchar* block1 = weight1 + bi * BYTES; \
        float d0 = q35l_half_to_float((ushort)block0[0] | ((ushort)block0[1] << 8)); \
        float d1 = q35l_half_to_float((ushort)block1[0] | ((ushort)block1[1] << 8)); \
        _Pragma("unroll") \
        for (int halfIndex = 0; halfIndex < 2; ++halfIndex) { \
            int octet = halfIndex * 16 + lane; \
            float8 input = vload8(0, x + bi * 256 + octet * 8); \
            sums0 = fma(input, DECODE(block0, octet, d0), sums0); \
            sums1 = fma(input, DECODE(block1, octet, d1), sums1); \
        } \
    } \
    float partial0 = dot(sums0.lo, (float4)(1.0f)) + dot(sums0.hi, (float4)(1.0f)); \
    float partial1 = dot(sums1.lo, (float4)(1.0f)) + dot(sums1.hi, (float4)(1.0f)); \
    float total0 = sub_group_reduce_add(partial0); \
    float total1 = sub_group_reduce_add(partial1); \
    return (float2)(total0, total1); \
}
Q35P_IQ_DOT(q35p_iq2_pair_dot, 82, q35p_iq2_octet)
Q35P_IQ_DOT(q35p_iq3_pair_dot, 110, q35p_iq3_octet)
Q35P_IQ_DOT(q35p_iq2_levels3_pair_dot, 82, q35p_iq2_octet_levels3)
#undef Q35P_IQ_DOT

// Decode-only candidate: four independent accumulation chains share each
// float8 input load. Do not split K between accumulators: the block/octect FMA
// order and final float8 + SG16 reductions must remain identical to pair2.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void q35l_iq2_s_sg16_pair4(__global const float* x, __global const uchar* weight,
    __global const float* bias, __global float* y, int rows, int input_width, int output_width)
{
    int lane = get_sub_group_local_id(), groups = (output_width + 3) / 4;
    int flat = get_group_id(0) * 2 + get_sub_group_id();
    if (flat >= rows * groups) return;
    int row = flat / groups, column = (flat % groups) * 4, blocks = input_width >> 8;
    size_t stride = (size_t)blocks * 82;
    __global const uchar* w0 = weight + (size_t)column * stride;
    // Invalid tail columns reuse w0; their results are never stored.
    __global const uchar* w1 = column + 1 < output_width ? w0 + stride : w0;
    __global const uchar* w2 = column + 2 < output_width ? w0 + 2 * stride : w0;
    __global const uchar* w3 = column + 3 < output_width ? w0 + 3 * stride : w0;
    x += (size_t)row * input_width;
    float8 sum0 = (float8)(0.0f), sum1 = (float8)(0.0f);
    float8 sum2 = (float8)(0.0f), sum3 = (float8)(0.0f);
    for (int bi = 0; bi < blocks; ++bi)
    {
        __global const uchar* b0 = w0 + bi * 82;
        __global const uchar* b1 = w1 + bi * 82;
        __global const uchar* b2 = w2 + bi * 82;
        __global const uchar* b3 = w3 + bi * 82;
        float d0 = q35l_half_to_float((ushort)b0[0] | ((ushort)b0[1] << 8));
        float d1 = q35l_half_to_float((ushort)b1[0] | ((ushort)b1[1] << 8));
        float d2 = q35l_half_to_float((ushort)b2[0] | ((ushort)b2[1] << 8));
        float d3 = q35l_half_to_float((ushort)b3[0] | ((ushort)b3[1] << 8));
        #pragma unroll
        for (int halfIndex = 0; halfIndex < 2; ++halfIndex)
        {
            int octet = halfIndex * 16 + lane;
            float8 input = vload8(0, x + bi * 256 + octet * 8);
            sum0 = fma(input, q35p_iq2_octet(b0, octet, d0), sum0);
            sum1 = fma(input, q35p_iq2_octet(b1, octet, d1), sum1);
            sum2 = fma(input, q35p_iq2_octet(b2, octet, d2), sum2);
            sum3 = fma(input, q35p_iq2_octet(b3, octet, d3), sum3);
        }
    }
    float p0 = dot(sum0.lo, (float4)(1.0f)) + dot(sum0.hi, (float4)(1.0f));
    float p1 = dot(sum1.lo, (float4)(1.0f)) + dot(sum1.hi, (float4)(1.0f));
    float p2 = dot(sum2.lo, (float4)(1.0f)) + dot(sum2.hi, (float4)(1.0f));
    float p3 = dot(sum3.lo, (float4)(1.0f)) + dot(sum3.hi, (float4)(1.0f));
    float t0 = sub_group_reduce_add(p0), t1 = sub_group_reduce_add(p1);
    float t2 = sub_group_reduce_add(p2), t3 = sub_group_reduce_add(p3);
    if (lane == 0)
    {
        y[row * output_width + column] = bias[column] + t0;
        if (column + 1 < output_width) y[row * output_width + column + 1] = bias[column + 1] + t1;
        if (column + 2 < output_width) y[row * output_width + column + 2] = bias[column + 2] + t2;
        if (column + 3 < output_width) y[row * output_width + column + 3] = bias[column + 3] + t3;
    }
}

inline float4 q35p_q4_pair_weights(__global const uchar* block, int pair, int lane, float d, float dmin) {
    float em, en, om, on;
    q35l_q4_scale_min(block, pair * 2, d, dmin, &em, &en);
    q35l_q4_scale_min(block, pair * 2 + 1, d, dmin, &om, &on);
    uchar first = block[16 + pair * 32 + lane], second = block[32 + pair * 32 + lane];
    return (float4)(fma(em, (float)(first & 15), -en), fma(em, (float)(second & 15), -en),
        fma(om, (float)(first >> 4), -on), fma(om, (float)(second >> 4), -on));
}

inline float2 q35p_q4_pair_dot(__global const float* x, __global const uchar* weight0,
    __global const uchar* weight1, int blocks, int lane) {
    float partial0 = 0.0f, partial1 = 0.0f;
    for (int bi = 0; bi < blocks; ++bi) {
        __global const uchar* block0 = weight0 + bi * 144;
        __global const uchar* block1 = weight1 + bi * 144;
        float d0 = q35l_half_to_float((ushort)block0[0] | ((ushort)block0[1] << 8));
        float m0 = q35l_half_to_float((ushort)block0[2] | ((ushort)block0[3] << 8));
        float d1 = q35l_half_to_float((ushort)block1[0] | ((ushort)block1[1] << 8));
        float m1 = q35l_half_to_float((ushort)block1[2] | ((ushort)block1[3] << 8));
        #if Q35_Q4_UNROLL
        #pragma unroll
        #endif
        for (int pair = 0; pair < 4; ++pair) {
            int offset = bi * 256 + pair * 64 + lane;
            float4 input = (float4)(x[offset], x[offset + 16], x[offset + 32], x[offset + 48]);
            float4 w0 = q35p_q4_pair_weights(block0, pair, lane, d0, m0);
            float4 w1 = q35p_q4_pair_weights(block1, pair, lane, d1, m1);
            partial0 = fma(input.s0, w0.s0, partial0);
            partial0 = fma(input.s1, w0.s1, partial0);
            partial0 = fma(input.s2, w0.s2, partial0);
            partial0 = fma(input.s3, w0.s3, partial0);
            partial1 = fma(input.s0, w1.s0, partial1);
            partial1 = fma(input.s1, w1.s1, partial1);
            partial1 = fma(input.s2, w1.s2, partial1);
            partial1 = fma(input.s3, w1.s3, partial1);
        }
    }
    float total0 = sub_group_reduce_add(partial0);
    float total1 = sub_group_reduce_add(partial1);
    return (float2)(total0, total1);
}

#define Q35P_ARGUMENTS \
    __global const float* x, __global const uchar* weight, \
    __global const float* bias, __global float* y, int rows, int input_width, int output_width
#define Q35P_SETUP(BYTES) \
    int lane = get_sub_group_local_id(), pairCount = (output_width + 1) / 2; \
    int flatPair = get_group_id(0) * 2 + get_sub_group_id(); \
    if (flatPair >= rows * pairCount) return; \
    int row = flatPair / pairCount, output0 = (flatPair % pairCount) * 2, output1 = output0 + 1; \
    int blocks = input_width >> 8; \
    __global const uchar* weight0 = weight + (size_t)output0 * blocks * BYTES; \
    /* Odd output tails reuse a valid packed row; its second result is discarded. */ \
    __global const uchar* weight1 = output1 < output_width ? weight0 + (size_t)blocks * BYTES : weight0; \
    x += row * input_width;
#define Q35P_DEFINE(NAME, BYTES, DOT) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(32, 1, 1))) \
__kernel void NAME(Q35P_ARGUMENTS) { \
    Q35P_SETUP(BYTES) \
    float2 totals = DOT(x, weight0, weight1, blocks, lane); \
    if (lane == 0) { \
        y[row * output_width + output0] = bias[output0] + totals.s0; \
        if (output1 < output_width) y[row * output_width + output1] = bias[output1] + totals.s1; \
    } \
} \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(32, 1, 1))) \
__kernel void NAME##_lora(Q35P_ARGUMENTS, __global const float* z, __global const float* b, int rank, float scale) { \
    Q35P_SETUP(BYTES) \
    float2 totals = DOT(x, weight0, weight1, blocks, lane); \
    if (lane == 0) { \
        float base0 = bias[output0] + totals.s0, sum0 = 0.0f, sum1 = 0.0f; \
        for (int r = 0; r < rank; ++r) { \
            float value = z[row * rank + r]; \
            sum0 = fma(value, b[output0 * rank + r], sum0); \
            if (output1 < output_width) sum1 = fma(value, b[output1 * rank + r], sum1); \
        } \
        y[row * output_width + output0] = base0 + scale * sum0; \
        if (output1 < output_width) { \
            float base1 = bias[output1] + totals.s1; \
            y[row * output_width + output1] = base1 + scale * sum1; \
        } \
    } \
}
Q35P_DEFINE(q35l_iq2_s_sg16_pair, 82, q35p_iq2_pair_dot)
Q35P_DEFINE(q35l_iq3_s_sg16_pair, 110, q35p_iq3_pair_dot)
Q35P_DEFINE(q35l_q4_k_sg16_pair, 144, q35p_q4_pair_dot)
// The model enables only the base variant. Fused LoRA stays on its original
// projection dispatch while this decoder is validated independently.
Q35P_DEFINE(q35l_iq2_s_sg16_levels3, 82, q35p_iq2_levels3_pair_dot)
#undef Q35P_DEFINE
#undef Q35P_SETUP
#undef Q35P_ARGUMENTS
#endif
