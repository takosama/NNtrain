// Float32, one-token Gated DeltaNet. State layout matches Qwen35Math.DeltaStep.
inline float q35d_sigmoid(float x)
{
    if (x >= 0.0f) return 1.0f / (1.0f + exp(-x));
    float e = exp(x);
    return e / (1.0f + e);
}

inline float q35d_silu(float x) { return x * q35d_sigmoid(x); }

inline float q35d_softplus(float x)
{
    if (x > 20.0f) return x;
    if (x < -20.0f) return exp(x);
    return log(1.0f + exp(x));
}

__kernel void q35d_convolution(
    __global const float* qkv, __global const float* weights,
    __global float* history_state, __global float* mixed,
    int channels, int conv_kernel)
{
    int c = get_global_id(0);
    if (c >= channels) return;
    int history = conv_kernel - 1;
    int state_offset = c * history, weight_offset = c * conv_kernel;
    float sum = 0.0f;
    for (int tap = 0; tap < history; ++tap)
        sum += history_state[state_offset + tap] * weights[weight_offset + tap];
    sum += qkv[c] * weights[weight_offset + history];
    mixed[c] = q35d_silu(sum);
    for (int tap = 0; tap < history - 1; ++tap)
        history_state[state_offset + tap] = history_state[state_offset + tap + 1];
    if (history > 0) history_state[state_offset + history - 1] = qkv[c];
}

__kernel void q35d_normalize_qk(
    __global float* mixed, int key_heads, int width, float eps)
{
    int h = get_global_id(0);
    if (h >= key_heads) return;
    int q_offset = h * width, k_offset = key_heads * width + q_offset;
    float q_sum = 0.0f, k_sum = 0.0f;
    for (int i = 0; i < width; ++i)
    {
        float q = mixed[q_offset + i], k = mixed[k_offset + i];
        q_sum += q * q;
        k_sum += k * k;
    }
    float q_inverse = (1.0f / sqrt((float)width)) / sqrt(q_sum + eps);
    float k_inverse = 1.0f / sqrt(k_sum + eps);
    for (int i = 0; i < width; ++i)
    {
        mixed[q_offset + i] *= q_inverse;
        mixed[k_offset + i] *= k_inverse;
    }
}

__kernel void q35d_recurrent(
    __global const float* mixed, __global const float* alpha,
    __global const float* beta, __global const float* dt,
    __global const float* a, __global float* state,
    __global float* output, int key_heads, int value_heads, int width)
{
    // Each work item owns one value column, including all of its state keys.
    int item = get_global_id(0);
    if (item >= value_heads * width) return;
    int h = item / width, v = item % width;
    int key_size = key_heads * width;
    // GGUF tiles heads: [key0, key1, key0, key1], not contiguous groups.
    int q_offset = (h % key_heads) * width, k_offset = key_size + q_offset;
    int state_offset = h * width * width;
    float decay = exp(a[h] * q35d_softplus(alpha[h] + dt[h]));
    float update_rate = q35d_sigmoid(beta[h]);
    float predicted = 0.0f;
    for (int k = 0; k < width; ++k)
    {
        int index = state_offset + k * width + v;
        state[index] *= decay;
        predicted += mixed[k_offset + k] * state[index];
    }
    float delta = (mixed[2 * key_size + item] - predicted) * update_rate;
    float value = 0.0f;
    for (int k = 0; k < width; ++k)
    {
        int index = state_offset + k * width + v;
        state[index] += mixed[k_offset + k] * delta;
        value += mixed[q_offset + k] * state[index];
    }
    output[item] = value;
}

__kernel void q35d_gated_rmsnorm(
    __global const float* input, __global const float* norm,
    __global const float* gate, __global float* output,
    int heads, int width, float eps)
{
    int h = get_global_id(0);
    if (h >= heads) return;
    int offset = h * width;
    float sum = 0.0f;
    for (int i = 0; i < width; ++i)
    {
        float value = input[offset + i];
        sum += value * value;
    }
    float inverse = 1.0f / sqrt(sum / width + eps);
    for (int i = 0; i < width; ++i)
        output[offset + i] = input[offset + i] * inverse * norm[i] * q35d_silu(gate[offset + i]);
}
