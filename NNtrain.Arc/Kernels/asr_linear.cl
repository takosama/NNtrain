// FP16 resident weights, FP32 activations and accumulation. OpenCL 1.2 vload_half
// converts storage without requiring native half arithmetic.
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void asr_linear(__global const float* input, __global const half* weight,
                __global const half* bias, __global float* output,
                int input_width, int output_width, int has_bias)
{
    int lane = get_local_id(0);
    size_t group = get_group_id(0);
    int column = group % output_width;
    size_t row = group / output_width;
    float sum = 0.0f;
    for (int i = lane; i < input_width; i += 64)
        sum += input[row * input_width + i] * vload_half((size_t)column * input_width + i, weight);
    __local float partial[64];
    partial[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1)
    {
        if (lane < stride) partial[lane] += partial[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) output[group] = partial[0] + (has_bias ? vload_half(column, bias) : 0.0f);
}
