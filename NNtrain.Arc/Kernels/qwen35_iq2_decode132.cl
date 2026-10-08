// Standalone decode experiment. Not used by any production model route.
// Each raw 82-byte IQ2_S block becomes 32 octet descriptors plus raw F16 d:
// descriptor bits 0..23 = eight signed 3-bit grid codes, 24..27 = scale.
#if defined(ARC_OPTIMIZATION_PROBES) && defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

__kernel void q35c_iq2_pack132(__global const uchar* raw, __global uint* packed, int blocks)
{
    int id = get_global_id(0), block_index = id >> 5, octet = id & 31;
    if (block_index >= blocks) return;
    __global const uchar* block = raw + (size_t)block_index * 82;
    int group = octet >> 2, part = octet & 3;
    int index = block[2 + octet] | (((block[66 + group] >> (part * 2)) & 3) << 8);
    uchar8 grid = as_uchar8(q35l_iq2s_grid[index]);
    uint signs = block[34 + octet], descriptor = 0;
    #pragma unroll
    for (int component = 0; component < 8; ++component)
    {
        uint magnitude = grid[component] == 8 ? 0u : grid[component] == 25 ? 1u : 2u;
        uint code = magnitude | (((signs >> component) & 1u) << 2);
        descriptor |= code << (component * 3);
    }
    descriptor |= (uint)((block[74 + group] >> ((part >> 1) * 4)) & 15) << 24;
    packed[(size_t)block_index * 33 + octet] = descriptor;
    if (octet == 0)
        packed[(size_t)block_index * 33 + 32] = (uint)block[0] | ((uint)block[1] << 8);
}

inline float8 q35c_iq2_octet132(uint descriptor, float d)
{
    int8 codes = convert_int8(((uint8)(descriptor) >> (uint8)(0,3,6,9,12,15,18,21)) & (uint8)(7));
    float8 grid = select((float8)(43.0f), (float8)(25.0f), (codes & (int8)(3)) == (int8)(1));
    grid = select(grid, (float8)(8.0f), (codes & (int8)(3)) == (int8)(0));
    grid = select(grid, -grid, (codes & (int8)(4)) != (int8)(0));
    float coefficient = d * (0.5f + (float)((descriptor >> 24) & 15u)) * 0.25f;
    return coefficient * grid;
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void q35c_iq2_pair132(__global const float* input, __global const uint* packed,
    __global const float* bias, __global float* output, int rows, int input_width, int output_width)
{
    int lane = get_sub_group_local_id(), pairs = (output_width + 1) / 2;
    int flat = get_group_id(0) * 2 + get_sub_group_id();
    if (flat >= rows * pairs) return;
    int row = flat / pairs, col = (flat % pairs) * 2, blocks = input_width >> 8;
    __global const uint* w0 = packed + (size_t)col * blocks * 33;
    __global const uint* w1 = col + 1 < output_width ? w0 + (size_t)blocks * 33 : w0;
    input += (size_t)row * input_width;
    float8 sum0 = (float8)(0.0f), sum1 = (float8)(0.0f);
    for (int block = 0; block < blocks; ++block)
    {
        __global const uint* b0 = w0 + block * 33;
        __global const uint* b1 = w1 + block * 33;
        float d0 = q35l_half_to_float((ushort)b0[32]), d1 = q35l_half_to_float((ushort)b1[32]);
        #pragma unroll
        for (int half_index = 0; half_index < 2; ++half_index)
        {
            int octet = half_index * 16 + lane;
            float8 x = vload8(0, input + block * 256 + octet * 8);
            sum0 = fma(x, q35c_iq2_octet132(b0[octet], d0), sum0);
            sum1 = fma(x, q35c_iq2_octet132(b1[octet], d1), sum1);
        }
    }
    float p0 = dot(sum0.lo, (float4)(1.0f)) + dot(sum0.hi, (float4)(1.0f));
    float p1 = dot(sum1.lo, (float4)(1.0f)) + dot(sum1.hi, (float4)(1.0f));
    float t0 = sub_group_reduce_add(p0), t1 = sub_group_reduce_add(p1);
    if (lane == 0)
    {
        output[(size_t)row * output_width + col] = bias[col] + t0;
        if (col + 1 < output_width) output[(size_t)row * output_width + col + 1] = bias[col + 1] + t1;
    }
}
#endif
