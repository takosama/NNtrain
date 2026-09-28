// K-quantized GEMV: two independent SG16 outputs per WG32. Each Q4 lane
// owns two values in each 32-value quantization group. The low/high nibble
// groups share the same packed bytes, which are loaded only once per pair.
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

inline float qwen_sg_half_to_float(ushort encoded) {
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

inline void qwen_sg_q4_scale_min(
    __global const uchar* block, int group, float d, float dmin,
    float* multiplier, float* minimum) {
    __global const uchar* packed = block + 4;
    int scale, minValue;
    if (group < 4) {
        scale = packed[group] & 63;
        minValue = packed[group + 4] & 63;
    } else {
        scale = (packed[group + 4] & 15)
            | ((packed[group - 4] >> 6) << 4);
        minValue = (packed[group + 4] >> 4)
            | ((packed[group] >> 6) << 4);
    }
    *multiplier = d * (float)scale;
    *minimum = dmin * (float)minValue;
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void qwen_linear_q4_k_sg16_r1(
    __global const float* x, __global const uchar* weight,
    __global const float* bias, __global float* y,
    int rows, int input_width, int output_width) {
    int lane = get_sub_group_local_id();
    int flatOutput = get_group_id(0) * 2 + get_sub_group_id();
    if (flatOutput >= rows * output_width) return;
    int row = rows == 1 ? 0 : flatOutput / output_width;
    int output = flatOutput - row * output_width;
    x += row * input_width;

    int blocksPerRow = input_width >> 8;
    __global const uchar* rowWeight = weight + output * blocksPerRow * 144;
    float partial = 0.0f;
    for (int blockIndex = 0; blockIndex < blocksPerRow; ++blockIndex) {
        __global const uchar* block = rowWeight + blockIndex * 144;
        float d = qwen_sg_half_to_float(
            (ushort)block[0] | ((ushort)block[1] << 8));
        float dmin = qwen_sg_half_to_float(
            (ushort)block[2] | ((ushort)block[3] << 8));
        for (int pair = 0; pair < 4; ++pair) {
            int evenGroup = pair * 2;
            float evenMultiplier, evenMinimum, oddMultiplier, oddMinimum;
            qwen_sg_q4_scale_min(block, evenGroup, d, dmin,
                &evenMultiplier, &evenMinimum);
            qwen_sg_q4_scale_min(block, evenGroup + 1, d, dmin,
                &oddMultiplier, &oddMinimum);

            int packedBase = 16 + pair * 32;
            uchar first = block[packedBase + lane];
            uchar second = block[packedBase + lane + 16];
            int inputBase = blockIndex * 256 + pair * 64;
            float w0 = fma(evenMultiplier, (float)(first & 15), -evenMinimum);
            float w1 = fma(evenMultiplier, (float)(second & 15), -evenMinimum);
            float w2 = fma(oddMultiplier, (float)(first >> 4), -oddMinimum);
            float w3 = fma(oddMultiplier, (float)(second >> 4), -oddMinimum);
            partial = fma(x[inputBase + lane], w0, partial);
            partial = fma(x[inputBase + lane + 16], w1, partial);
            partial = fma(x[inputBase + lane + 32], w2, partial);
            partial = fma(x[inputBase + lane + 48], w3, partial);
        }
    }

    float total = sub_group_reduce_add(partial);
    if (lane == 0) y[flatOutput] = bias[output] + total;
}

// Q6_K stores four 32-value quarters in each 128-value half-block. A lane
// handles positions lane and lane+16 in every quarter. The same low/high
// packed bytes decode all four quarters, and each 16-value scale is uniform
// across the subgroup. Accumulation and the subgroup reduction stay FP32.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void qwen_linear_q6_k_sg16_r1(
    __global const float* x, __global const uchar* weight,
    __global const float* bias, __global float* y,
    int rows, int input_width, int output_width) {
    int lane = get_sub_group_local_id();
    int flatOutput = get_group_id(0) * 2 + get_sub_group_id();
    if (flatOutput >= rows * output_width) return;
    int row = rows == 1 ? 0 : flatOutput / output_width;
    int output = flatOutput - row * output_width;
    x += row * input_width;

    int blocksPerRow = input_width >> 8;
    __global const uchar* rowWeight = weight + output * blocksPerRow * 210;
    float partial = 0.0f;
    for (int blockIndex = 0; blockIndex < blocksPerRow; ++blockIndex) {
        __global const uchar* block = rowWeight + blockIndex * 210;
        float d = qwen_sg_half_to_float(
            (ushort)block[208] | ((ushort)block[209] << 8));
        for (int halfIndex = 0; halfIndex < 2; ++halfIndex) {
            int lowBase = halfIndex * 64 + lane;
            int highBase = 128 + halfIndex * 32 + lane;
            uchar low0 = block[lowBase];
            uchar low1 = block[lowBase + 16];
            uchar low2 = block[lowBase + 32];
            uchar low3 = block[lowBase + 48];
            uchar high0 = block[highBase];
            uchar high1 = block[highBase + 16];
            __global const char* scales =
                (__global const char*)(block + 192 + halfIndex * 8);
            int inputBase = blockIndex * 256 + halfIndex * 128 + lane;

            float w0 = (d * (float)scales[0])
                * (float)(((low0 & 15) | ((high0 & 3) << 4)) - 32);
            float w1 = (d * (float)scales[1])
                * (float)(((low1 & 15) | ((high1 & 3) << 4)) - 32);
            float w2 = (d * (float)scales[2])
                * (float)(((low2 & 15) | (((high0 >> 2) & 3) << 4)) - 32);
            float w3 = (d * (float)scales[3])
                * (float)(((low3 & 15) | (((high1 >> 2) & 3) << 4)) - 32);
            float w4 = (d * (float)scales[4])
                * (float)(((low0 >> 4) | (((high0 >> 4) & 3) << 4)) - 32);
            float w5 = (d * (float)scales[5])
                * (float)(((low1 >> 4) | (((high1 >> 4) & 3) << 4)) - 32);
            float w6 = (d * (float)scales[6])
                * (float)(((low2 >> 4) | (((high0 >> 6) & 3) << 4)) - 32);
            float w7 = (d * (float)scales[7])
                * (float)(((low3 >> 4) | (((high1 >> 6) & 3) << 4)) - 32);
            partial = fma(x[inputBase], w0, partial);
            partial = fma(x[inputBase + 16], w1, partial);
            partial = fma(x[inputBase + 32], w2, partial);
            partial = fma(x[inputBase + 48], w3, partial);
            partial = fma(x[inputBase + 64], w4, partial);
            partial = fma(x[inputBase + 80], w5, partial);
            partial = fma(x[inputBase + 96], w6, partial);
            partial = fma(x[inputBase + 112], w7, partial);
        }
    }

    float total = sub_group_reduce_add(partial);
    if (lane == 0) y[flatOutput] = bias[output] + total;
}
#endif
