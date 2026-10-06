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

// Global relative attention for Parakeet, recomputed for the complete prefix.
// These buffers never contain cached encoder state from another prefix.
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void asr_relative_scores(__global const float* query, __global const float* key,
    __global const float* position, __global const half* bias_u, __global const half* bias_v,
    __global float* scores, int length, int hidden, int heads)
{
    int lane = get_local_id(0);
    size_t group = get_group_id(0);
    int k = group % length, head = (group / length) % heads;
    int q = group / ((size_t)length * heads);
    int width = hidden / heads, offset = head * width;
    float sum = 0.0f;
    for (int d = lane; d < width; d += 64)
    {
        int i = offset + d;
        float value = query[(size_t)q * hidden + i];
        sum += (value + vload_half(i, bias_u)) * key[(size_t)k * hidden + i]
            + (value + vload_half(i, bias_v)) * position[(size_t)(length - 1 - q + k) * hidden + i];
    }
    __local float partial[64]; partial[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride; stride >>= 1)
    {
        if (lane < stride) partial[lane] += partial[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) scores[group] = partial[0] / sqrt((float)width);
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void asr_attention_softmax(__global float* scores, int length)
{
    int lane = get_local_id(0);
    size_t offset = get_group_id(0) * length;
    __local float partial[64];
    float maximum = -INFINITY;
    for (int k = lane; k < length; k += 64) maximum = fmax(maximum, scores[offset + k]);
    partial[lane] = maximum; barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride; stride >>= 1)
    {
        if (lane < stride) partial[lane] = fmax(partial[lane], partial[lane + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maximum = partial[0]; barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0.0f;
    for (int k = lane; k < length; k += 64)
    {
        float value = exp(scores[offset + k] - maximum);
        scores[offset + k] = value; sum += value;
    }
    partial[lane] = sum; barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride; stride >>= 1)
    {
        if (lane < stride) partial[lane] += partial[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum = partial[0];
    for (int k = lane; k < length; k += 64) scores[offset + k] /= sum;
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void asr_attention_context(__global const float* scores, __global const float* value,
    __global float* output, int length, int hidden, int heads)
{
    int lane = get_local_id(0), head = get_group_id(0) % heads;
    int q = get_group_id(0) / heads, width = hidden / heads;
    for (int d = lane; d < width; d += 64)
    {
        int i = head * width + d; float sum = 0.0f;
        for (int k = 0; k < length; k++)
            sum += scores[((size_t)q * heads + head) * length + k] * value[(size_t)k * hidden + i];
        output[(size_t)q * hidden + i] = sum;
    }
}

__kernel void asr_silu(__global float* values, int count)
{
    size_t i = get_global_id(0);
    if (i >= count) return;
    float value = values[i], exponential = exp(-fabs(value));
    float sigmoid = value >= 0.0f ? 1.0f / (1.0f + exponential) : exponential / (1.0f + exponential);
    values[i] = value * sigmoid;
}
