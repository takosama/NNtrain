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

// Larger row tiles keep one independent accumulator per row. The macro lists
// expose fixed registers rather than dynamically indexing private arrays.
#define Q35T_ROW_LIST8(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)
#define Q35T_ROW_LIST16(M) Q35T_ROW_LIST8(M) M(8) M(9) M(10) M(11) M(12) M(13) M(14) M(15)
#define Q35T_IQ_DECLARE(N) float8 s##N = (float8)(0.0f);
#define Q35T_IQ_ACCUMULATE(N) \
    if (row + N < rows) s##N = fma(vload8(0, x0 + (size_t)N * input_width + offset), decoded, s##N);
#define Q35T_IQ_STORE(N) { \
    float total = sub_group_reduce_add(dot(s##N.lo, (float4)(1.0f)) + dot(s##N.hi, (float4)(1.0f))); \
    if (lane == 0 && row + N < rows) y[(size_t)(row + N) * output_width + output] = bias[output] + total; \
}

#define Q35T_LINEAR_IQ_WIDE_ROWS(NAME, BLOCK_BYTES, DECODE, ROWS, APPLY) \
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
    APPLY(Q35T_IQ_DECLARE) \
    for (int b = 0; b < blocks; ++b) { \
        __global const uchar* block = rowWeight + b * BLOCK_BYTES; \
        float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8)); \
        _Pragma("unroll") \
        for (int halfIndex = 0; halfIndex < 2; ++halfIndex) { \
            int octet = halfIndex * 16 + lane, offset = b * 256 + octet * 8; \
            float8 decoded = DECODE(block, octet, d); \
            APPLY(Q35T_IQ_ACCUMULATE) \
        } \
    } \
    APPLY(Q35T_IQ_STORE) \
}
Q35T_LINEAR_IQ_WIDE_ROWS(q35t_linear_iq2_s_rows8, 82, q35t_iq2_octet, 8, Q35T_ROW_LIST8)
Q35T_LINEAR_IQ_WIDE_ROWS(q35t_linear_iq2_s_rows16, 82, q35t_iq2_octet, 16, Q35T_ROW_LIST16)
Q35T_LINEAR_IQ_WIDE_ROWS(q35t_linear_iq3_s_rows8, 110, q35t_iq3_octet, 8, Q35T_ROW_LIST8)
Q35T_LINEAR_IQ_WIDE_ROWS(q35t_linear_iq3_s_rows16, 110, q35t_iq3_octet, 16, Q35T_ROW_LIST16)
#undef Q35T_LINEAR_IQ_WIDE_ROWS
#undef Q35T_IQ_DECLARE
#undef Q35T_IQ_ACCUMULATE
#undef Q35T_IQ_STORE

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

#define Q35T_Q4_DECLARE(N) float s##N = 0.0f;
#define Q35T_Q4_ACCUMULATE(N) \
    if (row + N < rows) s##N = q35t_q4_accumulate(x0 + (size_t)N * input_width, offset, decoded, s##N);
#define Q35T_Q4_STORE(N) { \
    float total = sub_group_reduce_add(s##N); \
    if (lane == 0 && row + N < rows) y[(size_t)(row + N) * output_width + output] = bias[output] + total; \
}
#define Q35T_LINEAR_Q4_WIDE_ROWS(NAME, ROWS, APPLY) \
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
    APPLY(Q35T_Q4_DECLARE) \
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
            APPLY(Q35T_Q4_ACCUMULATE) \
        } \
    } \
    APPLY(Q35T_Q4_STORE) \
}
Q35T_LINEAR_Q4_WIDE_ROWS(q35t_linear_q4_k_rows8, 8, Q35T_ROW_LIST8)
Q35T_LINEAR_Q4_WIDE_ROWS(q35t_linear_q4_k_rows16, 16, Q35T_ROW_LIST16)
#undef Q35T_LINEAR_Q4_WIDE_ROWS
#undef Q35T_Q4_DECLARE
#undef Q35T_Q4_ACCUMULATE
#undef Q35T_Q4_STORE

// The real 27B vocabulary head is Q5_K and supervises hundreds of rows at
// once. Reuse each 176-byte GGUF block's decoded weights across token rows.
// Every row keeps the original SG16 kernel's four-FMA order and reduction.
#define Q35T_Q5_ROWS4(M) M(0) M(1) M(2) M(3)
#define Q35T_Q5_ROWS8(M) Q35T_Q5_ROWS4(M) M(4) M(5) M(6) M(7)
#define Q35T_Q5_DECLARE(N) float s##N = 0.0f;
#define Q35T_Q5_ACCUMULATE(N) \
    if (row + N < rows) { \
        __global const float* xn = x + (size_t)(row + N) * input_width + inputBase; \
        s##N = fma(xn[0], w0, s##N); \
        s##N = fma(xn[16], w1, s##N); \
        s##N = fma(xn[32], w2, s##N); \
        s##N = fma(xn[48], w3, s##N); \
    }
#define Q35T_Q5_STORE(N) { \
    float total = sub_group_reduce_add(s##N); \
    if (lane == 0 && row + N < rows) \
        y[(size_t)(row + N) * output_width + output] = bias[output] + total; \
}
#define Q35T_Q5_FORWARD(NAME, ROWS, APPLY) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(32, 1, 1))) \
__kernel void NAME(__global const float* x, __global const uchar* weight, \
    __global const float* bias, __global float* y, \
    int rows, int input_width, int output_width) { \
    int lane = get_sub_group_local_id(); \
    int flat = get_group_id(0) * 2 + get_sub_group_id(); \
    int tiles = (rows + ROWS - 1) / ROWS; \
    if (flat >= tiles * output_width) return; \
    int row = (flat / output_width) * ROWS, output = flat % output_width; \
    int blocks = input_width >> 8; \
    __global const uchar* rowWeight = weight + (size_t)output * blocks * 176; \
    APPLY(Q35T_Q5_DECLARE) \
    for (int b = 0; b < blocks; ++b) { \
        __global const uchar* block = rowWeight + (size_t)b * 176; \
        float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8)); \
        float dmin = q35l_half_to_float((ushort)block[2] | ((ushort)block[3] << 8)); \
        uint high0 = block[16 + lane], high1 = block[32 + lane]; \
        _Pragma("unroll") \
        for (int pair = 0; pair < 4; ++pair) { \
            int evenGroup = pair * 2; \
            float evenMultiplier, evenMinimum, oddMultiplier, oddMinimum; \
            q35l_q5_fast_scale_min(block, evenGroup, d, dmin, \
                &evenMultiplier, &evenMinimum); \
            q35l_q5_fast_scale_min(block, evenGroup + 1, d, dmin, \
                &oddMultiplier, &oddMinimum); \
            int packedBase = 48 + pair * 32; \
            uint low0 = block[packedBase + lane], low1 = block[packedBase + lane + 16]; \
            uint q0 = (low0 & 15) | (((high0 >> evenGroup) & 1) << 4); \
            uint q1 = (low1 & 15) | (((high1 >> evenGroup) & 1) << 4); \
            uint q2 = (low0 >> 4) | (((high0 >> (evenGroup + 1)) & 1) << 4); \
            uint q3 = (low1 >> 4) | (((high1 >> (evenGroup + 1)) & 1) << 4); \
            float w0 = evenMultiplier * (float)q0 - evenMinimum; \
            float w1 = evenMultiplier * (float)q1 - evenMinimum; \
            float w2 = oddMultiplier * (float)q2 - oddMinimum; \
            float w3 = oddMultiplier * (float)q3 - oddMinimum; \
            int inputBase = b * 256 + pair * 64 + lane; \
            APPLY(Q35T_Q5_ACCUMULATE) \
        } \
    } \
    APPLY(Q35T_Q5_STORE) \
}
Q35T_Q5_FORWARD(q35t_linear_q5_k_rows4, 4, Q35T_Q5_ROWS4)
Q35T_Q5_FORWARD(q35t_linear_q5_k_rows8, 8, Q35T_Q5_ROWS8)
#undef Q35T_Q5_FORWARD
#undef Q35T_Q5_STORE
#undef Q35T_Q5_ACCUMULATE
#undef Q35T_Q5_DECLARE
#undef Q35T_Q5_ROWS8
#undef Q35T_Q5_ROWS4
#undef Q35T_ROW_LIST8
#undef Q35T_ROW_LIST16

// Opt-in training-only BF16/XMX IQ2_S projection. The GGUF payload stays
// packed in VRAM: each workgroup decodes one 32x32 weight tile into 2 KiB of
// local memory and reuses it across 64 sequence rows. BF16 rounding changes
// forward values versus the default FP32 FMA path; backward stays FP32.
inline ushort q35t_iq2_bf16_round(float x)
{
    uint bits = as_uint(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    else if ((bits & 0x007fffffu) != 0)
        bits |= 0x00400000u;
    return (ushort)(bits >> 16);
}

__kernel void q35t_iq2_input_bf16(__global const float* input,
    __global ushort* output, int count)
{
    int i = get_global_id(0);
    if (i < count) output[i] = q35t_iq2_bf16_round(input[i]);
}

inline int8 q35t_iq2_xmx_panel(const __local uint* panel, int lane)
{
    int8 values;
    for (int row = 0; row < 8; ++row)
        values[row] = as_int(panel[row * 16 + lane]);
    return values;
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 8, 1)))
__kernel void q35t_linear_iq2_s_xmx_bf16(__global const ushort* x,
    __global const uchar* packed, __global const float* bias, __global float* output,
    int rows, int input_width, int output_width)
{
    int row_base = get_group_id(1) * 64, col_base = get_group_id(0) * 32;
    int lane = get_local_id(0), subgroup = get_local_id(1), tid = subgroup * 16 + lane;
    __local ushort a[64][33];
    __local uint b[16 * 32];
    float8 accum[2] = {(float8)(0.0f), (float8)(0.0f)};
    for (int base = 0; base < input_width; base += 32)
    {
        for (int i = tid; i < 64 * 32; i += 128)
        {
            int row = row_base + i / 32, k = base + i % 32;
            a[i / 32][i % 32] = row < rows ? x[(size_t)row * input_width + k] : (ushort)0;
        }
        int col_inner = tid / 4, col = col_base + col_inner;
        int octet = base / 8 + tid % 4;
        ushort8 rounded = (ushort8)(0);
        if (col < output_width)
        {
            __global const uchar* block = packed
                + ((size_t)col * (input_width / 256) + octet / 32) * 82;
            float scale = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8));
            float8 decoded = q35t_iq2_octet(block, octet % 32, scale);
            rounded = (ushort8)(q35t_iq2_bf16_round(decoded.s0), q35t_iq2_bf16_round(decoded.s1),
                q35t_iq2_bf16_round(decoded.s2), q35t_iq2_bf16_round(decoded.s3),
                q35t_iq2_bf16_round(decoded.s4), q35t_iq2_bf16_round(decoded.s5),
                q35t_iq2_bf16_round(decoded.s6), q35t_iq2_bf16_round(decoded.s7));
        }
        int first_pair = (tid % 4) * 4;
        for (int pair = 0; pair < 4; ++pair)
        {
            int p = first_pair + pair;
            int dst = ((col_inner / 16) * 2 + p / 8) * 128
                + (p % 8) * 16 + col_inner % 16;
            b[dst] = (uint)rounded[2 * pair] | ((uint)rounded[2 * pair + 1] << 16);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int part = 0; part < 2; ++part)
        {
            short8 av;
            for (int r = 0; r < 8; ++r)
                av[r] = as_short(a[subgroup * 8 + r][part * 16 + lane]);
            for (int tile = 0; tile < 2; ++tile)
            {
                int8 bv = q35t_iq2_xmx_panel(b + (tile * 2 + part) * 128, lane);
                accum[tile] = intel_sub_group_bf16_bf16_matrix_mad_k16(av, bv, accum[tile]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int tile = 0; tile < 2; ++tile)
    {
        int col = col_base + tile * 16 + lane;
        for (int r = 0; r < 8; ++r)
        {
            int row = row_base + subgroup * 8 + r;
            if (row < rows && col < output_width)
                output[(size_t)row * output_width + col] = accum[tile][r] + bias[col];
        }
    }
}

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// FP16 improves significand precision over BF16 but has a narrower exponent
// range. It is a distinct opt-in route so training quality can be measured.
inline ushort q35t_iq2_f16_round(float x)
{
    return as_ushort(convert_half_rte(x));
}

__kernel void q35t_iq2_input_f16(__global const float* input,
    __global ushort* output, int count)
{
    int i = get_global_id(0);
    if (i < count) output[i] = q35t_iq2_f16_round(input[i]);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 8, 1)))
__kernel void q35t_linear_iq2_s_xmx_f16(__global const ushort* x,
    __global const uchar* packed, __global const float* bias, __global float* output,
    int rows, int input_width, int output_width)
{
    int row_base = get_group_id(1) * 64, col_base = get_group_id(0) * 32;
    int lane = get_local_id(0), subgroup = get_local_id(1), tid = subgroup * 16 + lane;
    __local ushort a[64][33];
    __local uint b[16 * 32];
    float8 accum[2] = {(float8)(0.0f), (float8)(0.0f)};
    for (int base = 0; base < input_width; base += 32)
    {
        for (int i = tid; i < 64 * 32; i += 128)
        {
            int row = row_base + i / 32, k = base + i % 32;
            a[i / 32][i % 32] = row < rows ? x[(size_t)row * input_width + k] : (ushort)0;
        }
        int col_inner = tid / 4, col = col_base + col_inner;
        int octet = base / 8 + tid % 4;
        ushort8 rounded = (ushort8)(0);
        if (col < output_width)
        {
            __global const uchar* block = packed
                + ((size_t)col * (input_width / 256) + octet / 32) * 82;
            float scale = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8));
            float8 decoded = q35t_iq2_octet(block, octet % 32, scale);
            rounded = (ushort8)(q35t_iq2_f16_round(decoded.s0), q35t_iq2_f16_round(decoded.s1),
                q35t_iq2_f16_round(decoded.s2), q35t_iq2_f16_round(decoded.s3),
                q35t_iq2_f16_round(decoded.s4), q35t_iq2_f16_round(decoded.s5),
                q35t_iq2_f16_round(decoded.s6), q35t_iq2_f16_round(decoded.s7));
        }
        int first_pair = (tid % 4) * 4;
        for (int pair = 0; pair < 4; ++pair)
        {
            int p = first_pair + pair;
            int dst = ((col_inner / 16) * 2 + p / 8) * 128
                + (p % 8) * 16 + col_inner % 16;
            b[dst] = (uint)rounded[2 * pair] | ((uint)rounded[2 * pair + 1] << 16);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int part = 0; part < 2; ++part)
        {
            short8 av;
            for (int r = 0; r < 8; ++r)
                av[r] = as_short(a[subgroup * 8 + r][part * 16 + lane]);
            for (int tile = 0; tile < 2; ++tile)
            {
                int8 bv = q35t_iq2_xmx_panel(b + (tile * 2 + part) * 128, lane);
                accum[tile] = intel_sub_group_f16_f16_matrix_mad_k16(av, bv, accum[tile]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int tile = 0; tile < 2; ++tile)
    {
        int col = col_base + tile * 16 + lane;
        for (int r = 0; r < 8; ++r)
        {
            int row = row_base + subgroup * 8 + r;
            if (row < rows && col < output_width)
                output[(size_t)row * output_width + col] = accum[tile][r] + bias[col];
        }
    }
}

// Training-shaped FP16 tile: 128 sequence rows and 64 output columns.
// Sharing one A tile across 64 columns halves repeated activation reads,
// while each packed IQ2 octet is decoded once for the whole 128-row tile.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void q35t_linear_iq2_s_xmx_f16_r128_n64(__global const ushort* x,
    __global const uchar* packed, __global const float* bias, __global float* output,
    int rows, int input_width, int output_width)
{
    int row_base = get_group_id(1) * 128, col_base = get_group_id(0) * 64;
    int lane = get_local_id(0), subgroup = get_local_id(1), tid = subgroup * 16 + lane;
    __local ushort a[128][33];
    __local uint b[32 * 32];
    float8 accum[4] = {(float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f)};
    for (int base = 0; base < input_width; base += 32)
    {
        for (int i = tid; i < 128 * 32; i += 256)
        {
            int row = row_base + i / 32, k = base + i % 32;
            a[i / 32][i % 32] = row < rows ? x[(size_t)row * input_width + k] : (ushort)0;
        }
        int col_inner = tid / 4, col = col_base + col_inner;
        int octet = base / 8 + tid % 4;
        ushort8 rounded = (ushort8)(0);
        if (col < output_width)
        {
            __global const uchar* block = packed
                + ((size_t)col * (input_width / 256) + octet / 32) * 82;
            float scale = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8));
            float8 decoded = q35t_iq2_octet(block, octet % 32, scale);
            rounded = (ushort8)(q35t_iq2_f16_round(decoded.s0), q35t_iq2_f16_round(decoded.s1),
                q35t_iq2_f16_round(decoded.s2), q35t_iq2_f16_round(decoded.s3),
                q35t_iq2_f16_round(decoded.s4), q35t_iq2_f16_round(decoded.s5),
                q35t_iq2_f16_round(decoded.s6), q35t_iq2_f16_round(decoded.s7));
        }
        int first_pair = (tid % 4) * 4;
        for (int pair = 0; pair < 4; ++pair)
        {
            int p = first_pair + pair;
            int dst = ((col_inner / 16) * 2 + p / 8) * 128
                + (p % 8) * 16 + col_inner % 16;
            b[dst] = (uint)rounded[2 * pair] | ((uint)rounded[2 * pair + 1] << 16);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int part = 0; part < 2; ++part)
        {
            short8 av;
            for (int r = 0; r < 8; ++r)
                av[r] = as_short(a[subgroup * 8 + r][part * 16 + lane]);
            for (int tile = 0; tile < 4; ++tile)
            {
                int8 bv = q35t_iq2_xmx_panel(b + (tile * 2 + part) * 128, lane);
                accum[tile] = intel_sub_group_f16_f16_matrix_mad_k16(av, bv, accum[tile]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int tile = 0; tile < 4; ++tile)
    {
        int col = col_base + tile * 16 + lane;
        for (int r = 0; r < 8; ++r)
        {
            int row = row_base + subgroup * 8 + r;
            if (row < rows && col < output_width)
                output[(size_t)row * output_width + col] = accum[tile][r] + bias[col];
        }
    }
}
#endif
#endif
