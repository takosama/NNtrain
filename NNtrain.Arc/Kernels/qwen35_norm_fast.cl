// Preserve q35a_rms_norm's 128-lane accumulation and binary reduction tree.
// Only the implementation of the reduction changes: two workgroup barriers
// replace eight, with the final four tree levels inside the first SG16.
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void q35a_rms_norm_sg16_exact(__global const float* input, __global const float* weight,
    __global float* output, int width, int input_stride, float epsilon)
{
    int row = get_group_id(0), tid = get_local_id(0);
    __local float sums[128];
    __local float inverse;
    float sum = 0.0f;
    for (int j = tid; j < width; j += 128)
    {
        float x = input[row * input_stride + j];
        sum += x * x;
    }
    sums[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_sub_group_id() == 0)
    {
        int lane = get_sub_group_local_id();
        // Identical pairs from strides 64, 32, 16 of the reference tree.
        float low0 = sums[lane] + sums[lane + 64];
        float low1 = sums[lane + 32] + sums[lane + 96];
        float high0 = sums[lane + 16] + sums[lane + 80];
        float high1 = sums[lane + 48] + sums[lane + 112];
        float reduced = (low0 + low1) + (high0 + high1);
        for (int stride = 8; stride > 0; stride >>= 1)
        {
            float other = intel_sub_group_shuffle(reduced, (lane + stride) & 15);
            if (lane < stride) reduced += other;
        }
        if (lane == 0) inverse = 1.0f / sqrt(reduced / width + epsilon);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int j = tid; j < width; j += 128)
        output[row * width + j] = input[row * input_stride + j] * inverse * weight[j];
}

#endif
