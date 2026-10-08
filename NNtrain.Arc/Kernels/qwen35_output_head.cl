// Q5_K vocabulary range. Keep each row's decode expressions, FMA order and
// SG16 reduction identical to q35l_q5_k_sg16; only row addresses differ.
// Included after qwen35_iq.cl for q35l_half_to_float / q35l_q5_fast_scale_min.
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(Q35_PROJECTION_WG, 1, 1)))
__kernel void q35l_q5_k_sg16_range(
    __global const float* x, __global const uchar* weight,
    __global const float* bias, __global float* y,
    int input_width, int output_count, int weight_first, int bias_first, int output_first) {
    int lane = get_sub_group_local_id();
    int output = get_group_id(0) * (Q35_PROJECTION_WG / 16) + get_sub_group_id();
    if (output >= output_count) return;

    int blocksPerRow = input_width >> 8;
    __global const uchar* rowWeight =
        weight + (size_t)(weight_first + output) * blocksPerRow * 176;
    float partial = 0.0f;
    for (int blockIndex = 0; blockIndex < blocksPerRow; ++blockIndex) {
        __global const uchar* block = rowWeight + (size_t)blockIndex * 176;
        float d = q35l_half_to_float(
            (ushort)block[0] | ((ushort)block[1] << 8));
        float dmin = q35l_half_to_float(
            (ushort)block[2] | ((ushort)block[3] << 8));

        // Each high byte supplies one bit for all eight 32-value groups.
        // Keep it in a register across the four low/high-nibble pairs.
        uint high0 = block[16 + lane];
        uint high1 = block[32 + lane];
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            int evenGroup = pair * 2;
            float evenMultiplier, evenMinimum, oddMultiplier, oddMinimum;
            q35l_q5_fast_scale_min(block, evenGroup, d, dmin,
                &evenMultiplier, &evenMinimum);
            q35l_q5_fast_scale_min(block, evenGroup + 1, d, dmin,
                &oddMultiplier, &oddMinimum);

            int packedBase = 48 + pair * 32;
            uint low0 = block[packedBase + lane];
            uint low1 = block[packedBase + lane + 16];
            uint q0 = (low0 & 15) | (((high0 >> evenGroup) & 1) << 4);
            uint q1 = (low1 & 15) | (((high1 >> evenGroup) & 1) << 4);
            uint q2 = (low0 >> 4) | (((high0 >> (evenGroup + 1)) & 1) << 4);
            uint q3 = (low1 >> 4) | (((high1 >> (evenGroup + 1)) & 1) << 4);

            // Match the current Q5 decoder's multiply/subtract expression.
            float w0 = evenMultiplier * (float)q0 - evenMinimum;
            float w1 = evenMultiplier * (float)q1 - evenMinimum;
            float w2 = oddMultiplier * (float)q2 - oddMinimum;
            float w3 = oddMultiplier * (float)q3 - oddMinimum;
            int inputBase = blockIndex * 256 + pair * 64 + lane;
            partial = fma(x[inputBase], w0, partial);
            partial = fma(x[inputBase + 16], w1, partial);
            partial = fma(x[inputBase + 32], w2, partial);
            partial = fma(x[inputBase + 48], w3, partial);
        }
    }

    float total = sub_group_reduce_add(partial);
    if (lane == 0) y[output_first + output] = bias[bias_first + output] + total;
}
#endif
