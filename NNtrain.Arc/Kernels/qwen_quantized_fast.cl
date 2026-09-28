// Cooperative GGUF K-quantized GEMV. One 64-lane work-group owns an output
// channel; its lanes read contiguous portions of each encoded 256-value block.
// The four-row variant reuses the decoded weight across prefill rows.

inline float qwen_fast_half_to_float(ushort encoded) {
    uint sign = ((uint)encoded & 0x8000u) << 16;
    uint exponent = ((uint)encoded >> 10) & 0x1fu;
    uint mantissa = (uint)encoded & 0x03ffu;
    uint bits;
    if (exponent == 0u) {
        if (mantissa == 0u) bits = sign;
        else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0u) { mantissa <<= 1; ++shift; }
            bits = sign | ((uint)(127 - 14 - shift) << 23)
                | ((mantissa & 0x03ffu) << 13);
        }
    } else if (exponent == 31u) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    return as_float(bits);
}

inline float qwen_fast_q4_value(
    __global const uchar* block, int index, float d, float dmin) {
    int group = index >> 5;
    int within = index & 31;
    __global const uchar* scales = block + 4;
    int scale, minimum;
    if (group < 4) {
        scale = scales[group] & 63;
        minimum = scales[group + 4] & 63;
    } else {
        scale = (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
        minimum = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    }
    uchar packed = block[16 + (group >> 1) * 32 + within];
    int quant = (group & 1) ? packed >> 4 : packed & 15;
    return d * (float)scale * (float)quant - dmin * (float)minimum;
}

inline float qwen_fast_q6_value(
    __global const uchar* block, int index, float d, float unused_dmin) {
    int group128 = index & 128;
    int offset = index - group128;
    int lane = offset & 31;
    int quarter = offset >> 5;
    int qlBase = group128 >> 1;
    int qhBase = group128 >> 2;
    int scaleBase = group128 >> 4;
    int qlow;
    uchar qhigh = block[128 + qhBase + lane];
    if (quarter == 0)
        qlow = (block[qlBase + lane] & 15) | ((qhigh & 3) << 4);
    else if (quarter == 1)
        qlow = (block[qlBase + lane + 32] & 15) | (((qhigh >> 2) & 3) << 4);
    else if (quarter == 2)
        qlow = (block[qlBase + lane] >> 4) | (((qhigh >> 4) & 3) << 4);
    else
        qlow = (block[qlBase + lane + 32] >> 4) | (((qhigh >> 6) & 3) << 4);
    __global const char* scales = (__global const char*)(block + 192);
    int scale = (int)scales[scaleBase + (lane >> 4) + quarter * 2];
    return d * (float)scale * (float)(qlow - 32);
}

// ROW_TILE is a compile-time constant (1 for generation, 4 for prefill).
// Keeping separate entrypoints lets the compiler discard the unused row
// accumulators and input loads from single-token GEMV.
#define QWEN_DEFINE_FAST_LINEAR(KERNEL_NAME, BLOCK_BYTES, VALUE_FN, ROW_TILE) \
__kernel void KERNEL_NAME( \
    __global const float* x, __global const uchar* weight, \
    __global const float* bias, __global float* y, \
    int rows, int input_width, int output_width) { \
    int out = get_group_id(0); \
    int rowBase = get_group_id(1) * ROW_TILE; \
    int lane = get_local_id(0); \
    if (out >= output_width || rowBase >= rows) return; \
    int blocksPerRow = input_width >> 8; \
    __global const uchar* rowWeight = weight + out * blocksPerRow * BLOCK_BYTES; \
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f; \
    int has1 = ROW_TILE > 1 && rowBase + 1 < rows; \
    int has2 = ROW_TILE > 2 && rowBase + 2 < rows; \
    int has3 = ROW_TILE > 3 && rowBase + 3 < rows; \
    for (int blockIndex = 0; blockIndex < blocksPerRow; ++blockIndex) { \
        __global const uchar* block = rowWeight + blockIndex * BLOCK_BYTES; \
        float d = qwen_fast_half_to_float( \
            (ushort)block[BLOCK_BYTES == 144 ? 0 : 208] \
            | ((ushort)block[BLOCK_BYTES == 144 ? 1 : 209] << 8)); \
        float dmin = BLOCK_BYTES == 144 ? qwen_fast_half_to_float( \
            (ushort)block[2] | ((ushort)block[3] << 8)) : 0.0f; \
        for (int part = 0; part < 4; ++part) { \
            int within = part * 64 + lane; \
            int k = blockIndex * 256 + within; \
            float decoded = VALUE_FN(block, within, d, dmin); \
            acc0 = fma(x[rowBase * input_width + k], decoded, acc0); \
            if (has1) acc1 = fma(x[(rowBase + 1) * input_width + k], decoded, acc1); \
            if (has2) acc2 = fma(x[(rowBase + 2) * input_width + k], decoded, acc2); \
            if (has3) acc3 = fma(x[(rowBase + 3) * input_width + k], decoded, acc3); \
        } \
    } \
    __local float sums[4 * 64]; \
    sums[lane] = acc0; \
    if (has1) sums[64 + lane] = acc1; \
    if (has2) sums[128 + lane] = acc2; \
    if (has3) sums[192 + lane] = acc3; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    for (int stride = 32; stride > 0; stride >>= 1) { \
        if (lane < stride) { \
            sums[lane] += sums[lane + stride]; \
            if (has1) sums[64 + lane] += sums[64 + lane + stride]; \
            if (has2) sums[128 + lane] += sums[128 + lane + stride]; \
            if (has3) sums[192 + lane] += sums[192 + lane + stride]; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    if (lane == 0) { \
        y[rowBase * output_width + out] = bias[out] + sums[0]; \
        if (has1) y[(rowBase + 1) * output_width + out] = bias[out] + sums[64]; \
        if (has2) y[(rowBase + 2) * output_width + out] = bias[out] + sums[128]; \
        if (has3) y[(rowBase + 3) * output_width + out] = bias[out] + sums[192]; \
    } \
}

QWEN_DEFINE_FAST_LINEAR(qwen_linear_q4_k_fast_r1, 144, qwen_fast_q4_value, 1)
QWEN_DEFINE_FAST_LINEAR(qwen_linear_q4_k_fast_r4, 144, qwen_fast_q4_value, 4)
QWEN_DEFINE_FAST_LINEAR(qwen_linear_q6_k_fast_r1, 210, qwen_fast_q6_value, 1)
QWEN_DEFINE_FAST_LINEAR(qwen_linear_q6_k_fast_r4, 210, qwen_fast_q6_value, 4)

#undef QWEN_DEFINE_FAST_LINEAR
