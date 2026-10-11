// Lossless IQ2_S decode layout: exactly 82 bytes per original 256-weight block.
// One allocation holds three contiguous planes for B blocks:
//   codes:  ushort[32*B], low 8 grid-index bits | eight sign bits << 8
//   extras: uchar [16*B], two grid-index high-bit pairs | shared scale << 4
//   d:      ushort[B],   original IEEE half bits, unchanged
// The 64-byte code records align each SG16 half-octet load. Packing combines
// four byte gathers per octet into one ushort and one shared byte gather.

__kernel void q35l_iq2_s_pack82(__global const uchar* raw,
    __global uchar* packed, int blocks) {
    size_t id = get_global_id(0);
    int block_index = (int)(id >> 4), pair = (int)(id & 15);
    if (block_index >= blocks) return;
    __global const uchar* block = raw + (size_t)block_index * 82;
    int octet = pair * 2, group = pair >> 1, shift = (pair & 1) * 4;
    uint codes = (uint)block[2 + octet] | ((uint)block[34 + octet] << 8)
        | ((uint)block[3 + octet] << 16) | ((uint)block[35 + octet] << 24);
    ((__global uint*)packed)[(size_t)block_index * 16 + pair] = codes;
    packed[(size_t)blocks * 64 + (size_t)block_index * 16 + pair] =
        (uchar)(((block[66 + group] >> shift) & 15) | (((block[74 + group] >> shift) & 15) << 4));
    if (pair == 0)
        ((__global ushort*)(packed + (size_t)blocks * 80))[block_index] =
            (ushort)block[0] | ((ushort)block[1] << 8);
}

// Compatibility path for less common prefill implementations. The result is
// temporary queue-owned storage, never a second resident weight copy.
__kernel void q35l_iq2_s_unpack82(__global const uchar* packed,
    __global uchar* raw, int blocks) {
    size_t id = get_global_id(0);
    int block = (int)(id >> 3), group = (int)(id & 7);
    if (block >= blocks) return;
    __global const ushort* codes = (__global const ushort*)packed;
    __global const uchar* extras = packed + (size_t)blocks * 64;
    __global uchar* target = raw + (size_t)block * 82;
    for (int part = 0; part < 4; ++part) {
        int octet = group * 4 + part;
        ushort code = codes[(size_t)block * 32 + octet];
        target[2 + octet] = (uchar)code;
        target[34 + octet] = (uchar)(code >> 8);
    }
    int lo = extras[(size_t)block * 16 + group * 2];
    int hi = extras[(size_t)block * 16 + group * 2 + 1];
    target[66 + group] = (uchar)((lo & 15) | ((hi & 15) << 4));
    target[74 + group] = (uchar)((lo >> 4) | (hi & 240));
    if (group == 0) {
        ushort d = ((__global const ushort*)(packed + (size_t)blocks * 80))[block];
        target[0] = (uchar)d; target[1] = (uchar)(d >> 8);
    }
}

#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

inline float8 q35packed82_octet(__global const ushort* codes,
    __global const uchar* extras, size_t block_index, int octet, float d) {
    uint code = codes[block_index * 32 + octet];
    uint extra = extras[block_index * 16 + (octet >> 1)];
    int gridIndex = (int)(code & 255) | (int)(((extra >> ((octet & 1) * 2)) & 3) << 8);
    float8 grid = convert_float8(as_uchar8(q35l_iq2s_grid[gridIndex]));
    int signs = (int)(code >> 8);
    int scale = (int)(extra >> 4);
    // Keep the original parenthesization and signed-grid multiplication order.
    float multiplier = d * (0.5f + (float)scale) * 0.25f;
    grid = select(grid, -grid, ((int8)(signs) & (int8)(1,2,4,8,16,32,64,128)) != (int8)(0));
    return multiplier * grid;
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(Q35_PROJECTION_WG, 1, 1)))
__kernel void q35l_iq2_s_packed82_sg16(
    __global const float* x, __global const uchar* packed,
    __global const float* bias, __global float* y,
    int rows, int input_width, int output_width) {
    int lane = get_sub_group_local_id();
    int flat = get_group_id(0) * (Q35_PROJECTION_WG / 16) + get_sub_group_id();
    if (flat >= rows * output_width) return;
    int row = flat / output_width, output = flat % output_width, blocks = input_width >> 8;
    size_t totalBlocks = (size_t)blocks * output_width;
    __global const ushort* codes = (__global const ushort*)packed;
    __global const uchar* extras = packed + totalBlocks * 64;
    __global const ushort* ds = (__global const ushort*)(packed + totalBlocks * 80);
    size_t firstBlock = (size_t)output * blocks;
    x += (size_t)row * input_width;
    float8 sums = (float8)(0.0f);
    for (int b = 0; b < blocks; ++b) {
        size_t block_index = firstBlock + b;
        float d = q35l_half_to_float(ds[block_index]);
        #pragma unroll
        for (int halfIndex = 0; halfIndex < 2; ++halfIndex) {
            int octet = halfIndex * 16 + lane;
            float8 input = vload8(0, x + b * 256 + octet * 8);
            sums = fma(input, q35packed82_octet(codes, extras, block_index, octet, d), sums);
        }
    }
    float partial = dot(sums.lo, (float4)(1.0f)) + dot(sums.hi, (float4)(1.0f));
    float total = sub_group_reduce_add(partial);
    if (lane == 0) y[flat] = bias[output] + total;
}

// Match the production pair2 kernel: two independent float8 accumulator
// chains reuse each activation octet. No K split or arithmetic reassociation.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void q35l_iq2_s_packed82_sg16_pair(
    __global const float* x, __global const uchar* packed,
    __global const float* bias, __global float* y,
    int rows, int input_width, int output_width) {
    int lane = get_sub_group_local_id(), pairs = (output_width + 1) / 2;
    int flat = get_group_id(0) * 2 + get_sub_group_id();
    if (flat >= rows * pairs) return;
    int row = flat / pairs, output0 = flat % pairs * 2, output1 = output0 + 1;
    int blocks = input_width >> 8;
    size_t totalBlocks = (size_t)blocks * output_width;
    __global const ushort* codes = (__global const ushort*)packed;
    __global const uchar* extras = packed + totalBlocks * 64;
    __global const ushort* ds = (__global const ushort*)(packed + totalBlocks * 80);
    size_t first0 = (size_t)output0 * blocks;
    size_t first1 = output1 < output_width ? first0 + blocks : first0;
    x += (size_t)row * input_width;
    float8 sums0 = (float8)(0.0f), sums1 = (float8)(0.0f);
    for (int b = 0; b < blocks; ++b) {
        size_t block0 = first0 + b, block1 = first1 + b;
        float d0 = q35l_half_to_float(ds[block0]), d1 = q35l_half_to_float(ds[block1]);
        #pragma unroll
        for (int halfIndex = 0; halfIndex < 2; ++halfIndex) {
            int octet = halfIndex * 16 + lane;
            float8 input = vload8(0, x + b * 256 + octet * 8);
            sums0 = fma(input, q35packed82_octet(codes, extras, block0, octet, d0), sums0);
            sums1 = fma(input, q35packed82_octet(codes, extras, block1, octet, d1), sums1);
        }
    }
    float partial0 = dot(sums0.lo, (float4)(1.0f)) + dot(sums0.hi, (float4)(1.0f));
    float partial1 = dot(sums1.lo, (float4)(1.0f)) + dot(sums1.hi, (float4)(1.0f));
    float total0 = sub_group_reduce_add(partial0), total1 = sub_group_reduce_add(partial1);
    if (lane == 0) {
        y[(size_t)row * output_width + output0] = bias[output0] + total0;
        if (output1 < output_width) y[(size_t)row * output_width + output1] = bias[output1] + total1;
    }
}
// Fused adapters preserve the original serial low-rank sum and final addition.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(Q35_PROJECTION_WG, 1, 1)))
__kernel void q35l_iq2_s_packed82_sg16_lora(
    __global const float* x, __global const uchar* packed,
    __global const float* bias, __global float* y,
    int rows, int input_width, int output_width,
    __global const float* z, __global const float* adapter, int rank, float scale) {
    int lane = get_sub_group_local_id();
    int flat = get_group_id(0) * (Q35_PROJECTION_WG / 16) + get_sub_group_id();
    if (flat >= rows * output_width) return;
    int row = flat / output_width, output = flat % output_width, blocks = input_width >> 8;
    size_t totalBlocks = (size_t)blocks * output_width;
    __global const ushort* codes = (__global const ushort*)packed;
    __global const uchar* extras = packed + totalBlocks * 64;
    __global const ushort* ds = (__global const ushort*)(packed + totalBlocks * 80);
    size_t firstBlock = (size_t)output * blocks;
    x += (size_t)row * input_width;
    float8 sums = (float8)(0.0f);
    for (int b = 0; b < blocks; ++b) {
        size_t block_index = firstBlock + b;
        float d = q35l_half_to_float(ds[block_index]);
        #pragma unroll
        for (int halfIndex = 0; halfIndex < 2; ++halfIndex) {
            int octet = halfIndex * 16 + lane;
            float8 input = vload8(0, x + b * 256 + octet * 8);
            sums = fma(input, q35packed82_octet(codes, extras, block_index, octet, d), sums);
        }
    }
    float partial = dot(sums.lo, (float4)(1.0f)) + dot(sums.hi, (float4)(1.0f));
    float total = sub_group_reduce_add(partial);
    if (lane == 0) {
        float base = bias[output] + total, sum = 0.0f;
        for (int r = 0; r < rank; ++r)
            sum = fma(z[row * rank + r], adapter[output * rank + r], sum);
        y[flat] = base + scale * sum;
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void q35l_iq2_s_packed82_sg16_pair_lora(
    __global const float* x, __global const uchar* packed,
    __global const float* bias, __global float* y,
    int rows, int input_width, int output_width,
    __global const float* z, __global const float* adapter, int rank, float scale) {
    int lane = get_sub_group_local_id(), pairs = (output_width + 1) / 2;
    int flat = get_group_id(0) * 2 + get_sub_group_id();
    if (flat >= rows * pairs) return;
    int row = flat / pairs, output0 = flat % pairs * 2, output1 = output0 + 1;
    int blocks = input_width >> 8;
    size_t totalBlocks = (size_t)blocks * output_width;
    __global const ushort* codes = (__global const ushort*)packed;
    __global const uchar* extras = packed + totalBlocks * 64;
    __global const ushort* ds = (__global const ushort*)(packed + totalBlocks * 80);
    size_t first0 = (size_t)output0 * blocks;
    size_t first1 = output1 < output_width ? first0 + blocks : first0;
    x += (size_t)row * input_width;
    float8 sums0 = (float8)(0.0f), sums1 = (float8)(0.0f);
    for (int b = 0; b < blocks; ++b) {
        size_t block0 = first0 + b, block1 = first1 + b;
        float d0 = q35l_half_to_float(ds[block0]), d1 = q35l_half_to_float(ds[block1]);
        #pragma unroll
        for (int halfIndex = 0; halfIndex < 2; ++halfIndex) {
            int octet = halfIndex * 16 + lane;
            float8 input = vload8(0, x + b * 256 + octet * 8);
            sums0 = fma(input, q35packed82_octet(codes, extras, block0, octet, d0), sums0);
            sums1 = fma(input, q35packed82_octet(codes, extras, block1, octet, d1), sums1);
        }
    }
    float partial0 = dot(sums0.lo, (float4)(1.0f)) + dot(sums0.hi, (float4)(1.0f));
    float partial1 = dot(sums1.lo, (float4)(1.0f)) + dot(sums1.hi, (float4)(1.0f));
    float total0 = sub_group_reduce_add(partial0), total1 = sub_group_reduce_add(partial1);
    if (lane == 0) {
        float base0 = bias[output0] + total0, sum0 = 0.0f, sum1 = 0.0f;
        for (int r = 0; r < rank; ++r) {
            float value = z[row * rank + r];
            sum0 = fma(value, adapter[output0 * rank + r], sum0);
            if (output1 < output_width) sum1 = fma(value, adapter[output1 * rank + r], sum1);
        }
        y[(size_t)row * output_width + output0] = base0 + scale * sum0;
        if (output1 < output_width) {
            float base1 = bias[output1] + total1;
            y[(size_t)row * output_width + output1] = base1 + scale * sum1;
        }
    }
}
#endif
