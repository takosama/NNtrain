// Qwen3.5 single-token inference. All activations and KV storage stay on device.
// Reduction kernels are launched with exactly 128 work-items per row/head.
__kernel void q35a_zero(__global float* values, int length)
{
    int i = get_global_id(0);
    if (i < length) values[i] = 0.0f;
}

inline float q35a_sigmoid(float x)
{
    if (x >= 0.0f) return 1.0f / (1.0f + exp(-x));
    float e = exp(x);
    return e / (1.0f + e);
}

__kernel void q35a_rms_norm(__global const float* input, __global const float* weight,
    __global float* output, int width, int input_stride, float epsilon)
{
    int row = get_group_id(0), tid = get_local_id(0);
    __local float sums[128];
    float sum = 0.0f;
    for (int j = tid; j < width; j += 128)
    {
        float x = input[row * input_stride + j];
        sum += x * x;
    }
    sums[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1)
    {
        if (tid < stride) sums[tid] += sums[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inverse = 1.0f / sqrt(sums[0] / width + epsilon);
    for (int j = tid; j < width; j += 128)
        output[row * width + j] = input[row * input_stride + j] * inverse * weight[j];
}

__kernel void q35a_silu_multiply(__global const float* gate, __global const float* up,
    __global float* output, int length)
{
    int i = get_global_id(0);
    if (i < length) output[i] = gate[i] * q35a_sigmoid(gate[i]) * up[i];
}

__kernel void q35a_add_in_place(__global float* destination, __global const float* source, int length)
{
    int i = get_global_id(0);
    if (i < length) destination[i] += source[i];
}

__kernel void q35a_rope(__global float* data, int heads, int width, int dimensions,
    int position, float theta)
{
    int pair = get_global_id(0), pairs = dimensions / 2;
    if (pair >= heads * pairs) return;
    int head = pair / pairs, j = pair % pairs;
    float angle = position * pow(theta, -2.0f * j / dimensions);
    float cosine = cos(angle), sine = sin(angle);
    int first = head * width + j, second = first + pairs;
    float a = data[first], b = data[second];
    data[first] = a * cosine - b * sine;
    data[second] = a * sine + b * cosine;
}

__kernel void q35a_scores(__global const float* query, __global const float* keys,
    __global float* scores, int heads, int kv_heads, int width, int sequence)
{
    int i = get_global_id(0);
    if (i >= heads * sequence) return;
    int head = i / sequence, time = i % sequence;
    int kv_head = head / (heads / kv_heads);
    int key_offset = (time * kv_heads + kv_head) * width;
    float dot = 0.0f;
    for (int j = 0; j < width; ++j) dot += query[head * width + j] * keys[key_offset + j];
    scores[i] = dot / sqrt((float)width);
}

__kernel void q35a_softmax(__global float* scores, int sequence)
{
    int head = get_group_id(0), tid = get_local_id(0), start = head * sequence;
    __local float reduction[128];
    float maximum = -INFINITY;
    for (int t = tid; t < sequence; t += 128) maximum = fmax(maximum, scores[start + t]);
    reduction[tid] = maximum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1)
    {
        if (tid < stride) reduction[tid] = fmax(reduction[tid], reduction[tid + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maximum = reduction[0];
    // Preserve the reduced maximum before reusing the local storage.
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0.0f;
    for (int t = tid; t < sequence; t += 128)
    {
        float p = exp(scores[start + t] - maximum);
        scores[start + t] = p;
        sum += p;
    }
    reduction[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1)
    {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum = reduction[0];
    for (int t = tid; t < sequence; t += 128) scores[start + t] /= sum;
}

__kernel void q35a_attend(__global const float* scores, __global const float* values,
    __global const float* q_and_gate, __global float* output,
    int heads, int kv_heads, int width, int sequence)
{
    int i = get_global_id(0);
    if (i >= heads * width) return;
    int head = i / width, j = i % width, kv_head = head / (heads / kv_heads);
    float sum = 0.0f;
    for (int t = 0; t < sequence; ++t)
        sum += scores[head * sequence + t] * values[(t * kv_heads + kv_head) * width + j];
    output[i] = sum * q35a_sigmoid(q_and_gate[(2 * head + 1) * width + j]);
}

// Both token selection and the all-logits finite check happen on the GPU.
// Equal maxima select the lowest token ID, matching the existing greedy loop.
__kernel void q35a_argmax(__global const float* logits, __global int* result, int count)
{
    int tid = get_local_id(0);
    __local float maxima[128];
    __local int indices[128], invalid[128];
    float maximum = -INFINITY;
    int index = 2147483647, bad = 0;
    for (int i = tid; i < count; i += 128)
    {
        float x = logits[i];
        if (!isfinite(x)) bad = 1;
        if (x > maximum || (x == maximum && i < index)) { maximum = x; index = i; }
    }
    maxima[tid] = maximum;
    indices[tid] = index;
    invalid[tid] = bad;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            float candidate = maxima[tid + stride];
            int candidate_index = indices[tid + stride];
            if (candidate > maxima[tid] || (candidate == maxima[tid] && candidate_index < indices[tid]))
            {
                maxima[tid] = candidate;
                indices[tid] = candidate_index;
            }
            invalid[tid] |= invalid[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) { result[0] = indices[0]; result[1] = invalid[0]; }
}
