// Lossless decode candidate: move sign application to a shared 128 KiB table.
// The original 82-byte GGUF blocks and FP32 accumulation order are preserved.
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

__kernel void q35l_iq2_s_build_signed_lookup(__global uint* table) {
    int id = get_global_id(0);
    if (id >= 16384) return;
    int index = id >> 4, signs = id & 15;
    uchar8 grid = as_uchar8(q35l_iq2s_grid[index]);
    int4 mask = ((int4)(signs) & (int4)(1,2,4,8)) != (int4)(0);
    int4 lo = convert_int4(grid.lo), hi = convert_int4(grid.hi);
    table[id * 2] = as_uint(convert_char4(select(lo, -lo, mask)));
    table[id * 2 + 1] = as_uint(convert_char4(select(hi, -hi, mask)));
}

inline float8 q35p_iq2_signed_octet(__global const uchar* block, int octet, float d,
    __global const uint* table) {
    int group = octet >> 2, part = octet & 3;
    int index = block[2 + octet] | (((block[66 + group] >> (part * 2)) & 3) << 8);
    int signs = block[34 + octet];
    int scale = (block[74 + group] >> ((part >> 1) * 4)) & 15;
    uint lo = table[(index * 16 + (signs & 15)) * 2];
    uint hi = table[(index * 16 + (signs >> 4)) * 2 + 1];
    float8 grid = convert_float8(as_char8((uint2)(lo, hi)));
    float multiplier = d * (0.5f + (float)scale) * 0.25f;
    return multiplier * grid;
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void q35l_iq2_s_sg16_pair_signed_lookup(__global const float* input,
    __global const uchar* weight, __global const float* bias, __global float* output,
    int rows, int input_width, int output_width, __global const uint* table) {
    int lane = get_sub_group_local_id(), pairs = (output_width + 1) / 2;
    int flat = get_group_id(0) * 2 + get_sub_group_id();
    if (flat >= rows * pairs) return;
    int row = flat / pairs, col = (flat % pairs) * 2, blocks = input_width >> 8;
    __global const uchar* w0 = weight + (size_t)col * blocks * 82;
    __global const uchar* w1 = col + 1 < output_width ? w0 + (size_t)blocks * 82 : w0;
    input += (size_t)row * input_width;
    float8 sum0 = (float8)(0.0f), sum1 = (float8)(0.0f);
    for (int bi = 0; bi < blocks; ++bi) {
        __global const uchar* b0 = w0 + bi * 82;
        __global const uchar* b1 = w1 + bi * 82;
        float d0 = q35l_half_to_float((ushort)b0[0] | ((ushort)b0[1] << 8));
        float d1 = q35l_half_to_float((ushort)b1[0] | ((ushort)b1[1] << 8));
        #pragma unroll
        for (int half_index = 0; half_index < 2; ++half_index) {
            int octet = half_index * 16 + lane;
            float8 x = vload8(0, input + bi * 256 + octet * 8);
            sum0 = fma(x, q35p_iq2_signed_octet(b0, octet, d0, table), sum0);
            sum1 = fma(x, q35p_iq2_signed_octet(b1, octet, d1, table), sum1);
        }
    }
    float p0 = dot(sum0.lo, (float4)(1.0f)) + dot(sum0.hi, (float4)(1.0f));
    float p1 = dot(sum1.lo, (float4)(1.0f)) + dot(sum1.hi, (float4)(1.0f));
    float t0 = sub_group_reduce_add(p0), t1 = sub_group_reduce_add(p1);
    if (lane == 0) {
        output[row * output_width + col] = bias[col] + t0;
        if (col + 1 < output_width) output[row * output_width + col + 1] = bias[col + 1] + t1;
    }
}
#endif
