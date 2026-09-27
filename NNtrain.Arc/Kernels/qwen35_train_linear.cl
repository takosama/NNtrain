// Training projections reuse decoded registers across independent token rows.
// Each token retains the inference SG16 kernel's FMA and reduction order.
// Packed weights remain resident in their original GGUF representation.
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

inline float8 q35t_iq2_octet(__global const uchar* block, int octet, float d) {
    int group = octet >> 2, part = octet & 3;
    int gridIndex = block[2 + octet] | (((block[66 + group] >> (part * 2)) & 3) << 8);
    float8 grid = convert_float8(as_uchar8(q35l_iq2s_grid[gridIndex]));
    int signs = block[34 + octet];
    int scale = (block[74 + group] >> ((part >> 1) * 4)) & 15;
    float multiplier = d * (0.5f + (float)scale) * 0.25f;
    grid = select(grid, -grid, ((int8)(signs) & (int8)(1,2,4,8,16,32,64,128)) != (int8)(0));
    return multiplier * grid;
}

inline float8 q35t_iq3_octet(__global const uchar* block, int octet, float d) {
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

#define Q35T_LINEAR_IQ_ROWS(NAME, BLOCK_BYTES, DECODE, ROWS) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(32, 1, 1))) \
__kernel void NAME(__global const float* x, __global const uchar* weight, \
    __global const float* bias, __global float* y, int rows, int input_width, int output_width) { \
    int lane = get_sub_group_local_id(), flat = get_group_id(0) * 2 + get_sub_group_id(); \
    int tiles = (rows + ROWS - 1) / ROWS; \
    if (flat >= tiles * output_width) return; \
    int row = (flat / output_width) * ROWS, output = flat % output_width, blocks = input_width >> 8; \
    __global const uchar* rowWeight = weight + (size_t)output * blocks * BLOCK_BYTES; \
    __global const float* x0 = x + (size_t)row * input_width; \
    float8 s0 = (float8)(0.0f), s1 = (float8)(0.0f), s2 = (float8)(0.0f), s3 = (float8)(0.0f); \
    for (int b = 0; b < blocks; ++b) { \
        __global const uchar* block = rowWeight + b * BLOCK_BYTES; \
        float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8)); \
        _Pragma("unroll") \
        for (int halfIndex = 0; halfIndex < 2; ++halfIndex) { \
            int octet = halfIndex * 16 + lane, offset = b * 256 + octet * 8; \
            float8 decoded = DECODE(block, octet, d); \
            s0 = fma(vload8(0, x0 + offset), decoded, s0); \
            if (row + 1 < rows) s1 = fma(vload8(0, x0 + input_width + offset), decoded, s1); \
            if (ROWS > 2 && row + 2 < rows) s2 = fma(vload8(0, x0 + 2 * input_width + offset), decoded, s2); \
            if (ROWS > 2 && row + 3 < rows) s3 = fma(vload8(0, x0 + 3 * input_width + offset), decoded, s3); \
        } \
    } \
    float total0 = sub_group_reduce_add(dot(s0.lo, (float4)(1.0f)) + dot(s0.hi, (float4)(1.0f))); \
    float total1 = sub_group_reduce_add(dot(s1.lo, (float4)(1.0f)) + dot(s1.hi, (float4)(1.0f))); \
    if (lane == 0) { \
        y[row * output_width + output] = bias[output] + total0; \
        if (row + 1 < rows) y[(row + 1) * output_width + output] = bias[output] + total1; \
    } \
    if (ROWS > 2) { \
        float total2 = sub_group_reduce_add(dot(s2.lo, (float4)(1.0f)) + dot(s2.hi, (float4)(1.0f))); \
        float total3 = sub_group_reduce_add(dot(s3.lo, (float4)(1.0f)) + dot(s3.hi, (float4)(1.0f))); \
        if (lane == 0) { \
            if (row + 2 < rows) y[(row + 2) * output_width + output] = bias[output] + total2; \
            if (row + 3 < rows) y[(row + 3) * output_width + output] = bias[output] + total3; \
        } \
    } \
}

Q35T_LINEAR_IQ_ROWS(q35t_linear_iq2_s_rows2, 82, q35t_iq2_octet, 2)
Q35T_LINEAR_IQ_ROWS(q35t_linear_iq2_s_rows4, 82, q35t_iq2_octet, 4)
Q35T_LINEAR_IQ_ROWS(q35t_linear_iq3_s_rows2, 110, q35t_iq3_octet, 2)
Q35T_LINEAR_IQ_ROWS(q35t_linear_iq3_s_rows4, 110, q35t_iq3_octet, 4)
#undef Q35T_LINEAR_IQ_ROWS

inline float q35t_q4_accumulate(__global const float* x, int offset, float4 weights, float sum) {
    sum = fma(x[offset], weights.s0, sum);
    sum = fma(x[offset + 16], weights.s1, sum);
    sum = fma(x[offset + 32], weights.s2, sum);
    return fma(x[offset + 48], weights.s3, sum);
}

#define Q35T_LINEAR_Q4_ROWS(NAME, ROWS) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(32, 1, 1))) \
__kernel void NAME(__global const float* x, __global const uchar* weight, \
    __global const float* bias, __global float* y, int rows, int input_width, int output_width) { \
    int lane = get_sub_group_local_id(), flat = get_group_id(0) * 2 + get_sub_group_id(); \
    int tiles = (rows + ROWS - 1) / ROWS; \
    if (flat >= tiles * output_width) return; \
    int row = (flat / output_width) * ROWS, output = flat % output_width, blocks = input_width >> 8; \
    __global const uchar* rowWeight = weight + (size_t)output * blocks * 144; \
    __global const float* x0 = x + (size_t)row * input_width; \
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f; \
    for (int b = 0; b < blocks; ++b) { \
        __global const uchar* block = rowWeight + b * 144; \
        float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8)); \
        float dmin = q35l_half_to_float((ushort)block[2] | ((ushort)block[3] << 8)); \
        for (int pair = 0; pair < 4; ++pair) { \
            float em, en, om, on; \
            q35l_q4_scale_min(block, pair * 2, d, dmin, &em, &en); \
            q35l_q4_scale_min(block, pair * 2 + 1, d, dmin, &om, &on); \
            uchar first = block[16 + pair * 32 + lane], second = block[32 + pair * 32 + lane]; \
            float4 decoded = (float4)(fma(em, (float)(first & 15), -en), fma(em, (float)(second & 15), -en), \
                fma(om, (float)(first >> 4), -on), fma(om, (float)(second >> 4), -on)); \
            int offset = b * 256 + pair * 64 + lane; \
            s0 = q35t_q4_accumulate(x0, offset, decoded, s0); \
            if (row + 1 < rows) s1 = q35t_q4_accumulate(x0 + input_width, offset, decoded, s1); \
            if (ROWS > 2 && row + 2 < rows) s2 = q35t_q4_accumulate(x0 + 2 * input_width, offset, decoded, s2); \
            if (ROWS > 2 && row + 3 < rows) s3 = q35t_q4_accumulate(x0 + 3 * input_width, offset, decoded, s3); \
        } \
    } \
    float total0 = sub_group_reduce_add(s0), total1 = sub_group_reduce_add(s1); \
    if (lane == 0) { \
        y[row * output_width + output] = bias[output] + total0; \
        if (row + 1 < rows) y[(row + 1) * output_width + output] = bias[output] + total1; \
    } \
    if (ROWS > 2) { \
        float total2 = sub_group_reduce_add(s2), total3 = sub_group_reduce_add(s3); \
        if (lane == 0) { \
            if (row + 2 < rows) y[(row + 2) * output_width + output] = bias[output] + total2; \
            if (row + 3 < rows) y[(row + 3) * output_width + output] = bias[output] + total3; \
        } \
    } \
}
Q35T_LINEAR_Q4_ROWS(q35t_linear_q4_k_rows2, 2)
Q35T_LINEAR_Q4_ROWS(q35t_linear_q4_k_rows4, 4)
#undef Q35T_LINEAR_Q4_ROWS
#endif
