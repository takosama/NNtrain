// Prism packed formats. Layout/decoding follows the author's ggml-quants.c
// dequantize_row_pq2_0/dequantize_row_ptq1_0 and ggml-common.h block definitions.
// Encoded weights stay on the GPU; individual FP32 values exist only in registers.
inline float q35l_half_to_float(ushort encoded);

inline float q35p_pq2_0_value(__global const uchar* row, int index) {
    __global const uchar* block = row + (size_t)(index >> 7) * 34;
    int within = index & 127;
    float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8));
    int q = (block[2 + (within >> 2)] >> ((within & 3) * 2)) & 3;
    // Preserve the fourth code (+2), even though most model values are ternary.
    return (float)(q - 1) * d;
}

inline float q35p_ptq1_0_value(__global const uchar* row, int index) {
    __global const uchar* block = row + (size_t)(index >> 7) * 28;
    int within = index & 127;
    int byte_index, trit;
    // For 24 qs bytes, the author's {32,16,8} stage sequence emits a
    // 16-byte five-trit slab (80 values), then an 8-byte slab (40 values).
    if (within < 80) {
        byte_index = within & 15;
        trit = within >> 4;
    } else if (within < 120) {
        int tail = within - 80;
        byte_index = 16 + (tail & 7);
        trit = tail >> 3;
    } else {
        int tail = within - 120;
        byte_index = 24 + (tail & 1);
        trit = tail >> 1;
    }
    int power = trit == 0 ? 1 : trit == 1 ? 3 : trit == 2 ? 9 : trit == 3 ? 27 : 81;
    // The reference multiplies into uint8_t before extracting the trit.
    int q = ((int)block[byte_index] * power) & 255;
    int value = ((q * 3) >> 8) - 1;
    float d = q35l_half_to_float((ushort)block[26] | ((ushort)block[27] << 8));
    return (float)value * d;
}

inline float q35p_bf16_value(__global const uchar* row, int index) {
    size_t offset = (size_t)index * 2;
    uint bits = ((uint)row[offset] | ((uint)row[offset + 1] << 8)) << 16;
    return as_float(bits);
}

// The row-byte expression is evaluated in each kernel's input-width scope.
#define Q35P_REFERENCE(NAME, ROW_BYTES, VALUE) \
__kernel void NAME(__global const float* x, __global const uchar* weight, \
    __global const float* bias, __global float* y, int rows, int input_width, int output_width) { \
    size_t flat = get_global_id(0); \
    if (flat >= (size_t)rows * output_width) return; \
    int row = flat / output_width, output = flat % output_width; \
    __global const uchar* row_weight = weight + (size_t)output * (ROW_BYTES); \
    float sum = 0.0f; \
    for (int j = 0; j < input_width; ++j) \
        sum = fma(x[(size_t)row * input_width + j], VALUE(row_weight, j), sum); \
    y[flat] = bias[output] + sum; \
}

#define Q35P_COOPERATIVE(NAME, ROW_BYTES, VALUE) \
__attribute__((reqd_work_group_size(64, 1, 1))) \
__kernel void NAME(__global const float* x, __global const uchar* weight, \
    __global const float* bias, __global float* y, int rows, int input_width, int output_width) { \
    int output = get_group_id(0), row = get_group_id(1), lane = get_local_id(0); \
    if (output >= output_width || row >= rows) return; \
    __global const uchar* row_weight = weight + (size_t)output * (ROW_BYTES); \
    float sum = 0.0f; \
    for (int j = lane; j < input_width; j += 64) \
        sum = fma(x[(size_t)row * input_width + j], VALUE(row_weight, j), sum); \
    __local float sums[64]; \
    sums[lane] = sum; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    for (int stride = 32; stride > 0; stride >>= 1) { \
        if (lane < stride) sums[lane] += sums[lane + stride]; \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    if (lane == 0) y[(size_t)row * output_width + output] = bias[output] + sums[0]; \
}

#define Q35P_EMBEDDING(NAME, ROW_BYTES, VALUE) \
__kernel void NAME(__global const uchar* weight, __global const int* ids, \
    __global float* y, int input_width) { \
    int j = get_global_id(0); \
    if (j >= input_width) return; \
    __global const uchar* row_weight = weight + (size_t)ids[0] * (ROW_BYTES); \
    y[j] = VALUE(row_weight, j); \
}

Q35P_REFERENCE(q35l_pq2_0_reference, ((size_t)input_width >> 7) * 34, q35p_pq2_0_value)
Q35P_REFERENCE(q35l_ptq1_0_reference, ((size_t)input_width >> 7) * 28, q35p_ptq1_0_value)
Q35P_REFERENCE(q35l_bf16_reference, (size_t)input_width * 2, q35p_bf16_value)
Q35P_COOPERATIVE(q35l_pq2_0_coop64, ((size_t)input_width >> 7) * 34, q35p_pq2_0_value)
Q35P_COOPERATIVE(q35l_ptq1_0_coop64, ((size_t)input_width >> 7) * 28, q35p_ptq1_0_value)
Q35P_COOPERATIVE(q35l_bf16_coop64, (size_t)input_width * 2, q35p_bf16_value)
Q35P_EMBEDDING(q35l_pq2_0_embedding, ((size_t)input_width >> 7) * 34, q35p_pq2_0_value)
Q35P_EMBEDDING(q35l_ptq1_0_embedding, ((size_t)input_width >> 7) * 28, q35p_ptq1_0_value)
Q35P_EMBEDDING(q35l_bf16_embedding, (size_t)input_width * 2, q35p_bf16_value)

#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#ifndef Q35_PROJECTION_WG
#define Q35_PROJECTION_WG 32
#endif
#if Q35_PROJECTION_WG != 32 && Q35_PROJECTION_WG != 64 && Q35_PROJECTION_WG != 128
#error Q35_PROJECTION_WG must be 32, 64 or 128
#endif

#define Q35P_SUBGROUP(NAME, ROW_BYTES, VALUE) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(Q35_PROJECTION_WG, 1, 1))) \
__kernel void NAME(__global const float* x, __global const uchar* weight, \
    __global const float* bias, __global float* y, int rows, int input_width, int output_width) { \
    int lane = get_sub_group_local_id(); \
    size_t flat = get_group_id(0) * (Q35_PROJECTION_WG / 16) + get_sub_group_id(); \
    if (flat >= (size_t)rows * output_width) return; \
    int row = flat / output_width, output = flat % output_width; \
    __global const uchar* row_weight = weight + (size_t)output * (ROW_BYTES); \
    float sum = 0.0f; \
    for (int j = lane; j < input_width; j += 16) \
        sum = fma(x[(size_t)row * input_width + j], VALUE(row_weight, j), sum); \
    float total = sub_group_reduce_add(sum); \
    if (lane == 0) y[flat] = bias[output] + total; \
}

Q35P_SUBGROUP(q35l_pq2_0_sg16, ((size_t)input_width >> 7) * 34, q35p_pq2_0_value)
Q35P_SUBGROUP(q35l_bf16_sg16, (size_t)input_width * 2, q35p_bf16_value)
#undef Q35P_SUBGROUP

inline float q35p_ptq1_0_scaled(int packed, int power, float d) {
    int q = (packed * power) & 255;
    return (float)(((q * 3) >> 8) - 1) * d;
}

// One scale conversion per 128-value block; preserve the generic SG16 FMA order.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(Q35_PROJECTION_WG, 1, 1)))
__kernel void q35l_ptq1_0_sg16(__global const float* x, __global const uchar* weight,
    __global const float* bias, __global float* y, int rows, int input_width, int output_width) {
    int lane = get_sub_group_local_id();
    size_t flat = get_group_id(0) * (Q35_PROJECTION_WG / 16) + get_sub_group_id();
    if (flat >= (size_t)rows * output_width) return;
    int row = flat / output_width, output = flat % output_width, blocks = input_width >> 7;
    __global const uchar* row_weight = weight + (size_t)output * blocks * 28;
    x += (size_t)row * input_width;
    int tail_lane = lane & 7;
    int fifth_power = lane < 8 ? 1 : 3;
    int sixth_power = lane < 8 ? 9 : 27;
    int last_trit = tail_lane >> 1;
    int last_power = lane < 8 ? 81 : last_trit == 0 ? 1 : last_trit == 1 ? 3 : last_trit == 2 ? 9 : 27;
    float sum = 0.0f;
    for (int block_index = 0; block_index < blocks; ++block_index) {
        __global const uchar* block = row_weight + (size_t)block_index * 28;
        float d = q35l_half_to_float((ushort)block[26] | ((ushort)block[27] << 8));
        int first = block[lane], tail = block[16 + tail_lane];
        int last = lane < 8 ? tail : block[24 + (lane & 1)];
        int input_start = block_index * 128 + lane;
        // The first five steps use the same byte and successive powers of 3.
        sum = fma(x[input_start],      q35p_ptq1_0_scaled(first, 1, d), sum);
        sum = fma(x[input_start + 16], q35p_ptq1_0_scaled(first, 3, d), sum);
        sum = fma(x[input_start + 32], q35p_ptq1_0_scaled(first, 9, d), sum);
        sum = fma(x[input_start + 48], q35p_ptq1_0_scaled(first, 27, d), sum);
        sum = fma(x[input_start + 64], q35p_ptq1_0_scaled(first, 81, d), sum);
        // Steps 5/6 cover two trits from each of the eight remaining qs bytes;
        // step 7 ends that slab in lanes 0..7 and reads qh in lanes 8..15.
        sum = fma(x[input_start + 80], q35p_ptq1_0_scaled(tail, fifth_power, d), sum);
        sum = fma(x[input_start + 96], q35p_ptq1_0_scaled(tail, sixth_power, d), sum);
        sum = fma(x[input_start + 112], q35p_ptq1_0_scaled(last, last_power, d), sum);
    }
    float total = sub_group_reduce_add(sum);
    if (lane == 0) y[flat] = bias[output] + total;
}
#endif

#undef Q35P_REFERENCE
#undef Q35P_COOPERATIVE
#undef Q35P_EMBEDDING

// One WG256 owns one complete 1024-element Sylvester block. Input/output must
// be distinct when grouped permutation is enabled because it can cross blocks.
// Host requirements: width is a positive multiple of 1024; signs has width
// entries (upload +1 for identity); optional head dimensions exactly span width.
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void q35l_prism_hadamard(__global const float* input, __global const float* signs,
    __global float* output, int width, int rows, int inverse,
    int key_heads, int value_heads, int head_width) {
    int blocks = width >> 10;
    size_t group = get_group_id(0);
    int row = group / blocks, block = group % blocks, tid = get_local_id(0);
    if (row >= rows) return;
    int block_start = block * 1024;
    __local float values[1024];
    for (int within = tid; within < 1024; within += 256) {
        int feature = block_start + within, source = feature;
        if (!inverse && key_heads > 0) {
            // llama-graph.cpp: [head_width, key_heads, repetitions] ->
            // [head_width, repetitions, key_heads] before signs and rotation.
            int repetitions = value_heads / key_heads;
            int head = feature / head_width;
            source = feature % head_width
                + head_width * (head / repetitions + key_heads * (head % repetitions));
        }
        float value = input[(size_t)row * width + source];
        values[within] = inverse ? value : value * signs[feature];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 1; stride < 1024; stride <<= 1) {
        // Each work-item owns disjoint butterfly pairs during this stage.
        for (int pair = tid; pair < 512; pair += 256) {
            int first = (pair / stride) * (stride * 2) + pair % stride;
            int second = first + stride;
            float a = values[first], b = values[second];
            values[first] = a + b;
            values[second] = a - b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int within = tid; within < 1024; within += 256) {
        int feature = block_start + within;
        float value = values[within] * 0.03125f;
        output[(size_t)row * width + feature] = inverse ? value * signs[feature] : value;
    }
}
