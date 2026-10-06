// Experimental packed causal attention with one workgroup per query row/head.
// Each workgroup computes its row of scores and normalizes it without launching
// a second kernel. Scores/probabilities retain the existing packed FP32 layout.
// This kernel is optional until end-to-end numeric and speed checks finish.
inline int q35tr_score_start(int time, int head, int heads)
{
    return (time * (time + 1) / 2) * heads + head * (time + 1);
}

__kernel void q35t_attn_score_softmax_rowfused(
    __global const float* query, __global const float* key,
    __global float* probabilities, int sequence, int heads,
    int kv_heads, int width)
{
    int row = get_group_id(0), tid = get_local_id(0);
    if (row >= sequence * heads) return;
    int time = row / heads, head = row % heads;
    int kv = head / (heads / kv_heads);
    int valid = time + 1;
    int start = q35tr_score_start(time, head, heads);
    int query_base = row * width;
    __local float reduction[128];
    float maximum = -INFINITY;

    for (int source = tid; source < valid; source += 128)
    {
        int key_base = (source * kv_heads + kv) * width;
        float sum = 0.0f;
        for (int j = 0; j < width; ++j)
            sum += query[query_base + j] * key[key_base + j];
        float score = sum / sqrt((float)width);
        probabilities[start + source] = score;
        maximum = fmax(maximum, score);
    }
    reduction[tid] = maximum;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1)
    {
        if (tid < stride) reduction[tid] = fmax(reduction[tid], reduction[tid + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maximum = reduction[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0.0f;
    for (int source = tid; source < valid; source += 128)
    {
        float probability = exp(probabilities[start + source] - maximum);
        probabilities[start + source] = probability;
        sum += probability;
    }
    reduction[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1)
    {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float denominator = reduction[0];
    for (int source = tid; source < valid; source += 128)
        probabilities[start + source] /= denominator;
}

// The backward dot and the softmax Jacobian share the same query workgroup.
// Existing dQ/dK/dV kernels consume the packed dscore and probability arrays.
__kernel void q35t_attn_probability_softmax_backward_rowfused(
    __global const float* dcontext, __global const float* value,
    __global const float* probabilities, __global float* dscores,
    int sequence, int heads, int kv_heads, int width)
{
    int row = get_group_id(0), tid = get_local_id(0);
    if (row >= sequence * heads) return;
    int time = row / heads, head = row % heads;
    int kv = head / (heads / kv_heads);
    int valid = time + 1;
    int start = q35tr_score_start(time, head, heads);
    int context_base = row * width;
    __local float reduction[128];
    float dot = 0.0f;

    for (int source = tid; source < valid; source += 128)
    {
        int value_base = (source * kv_heads + kv) * width;
        float sum = 0.0f;
        for (int j = 0; j < width; ++j)
            sum += dcontext[context_base + j] * value[value_base + j];
        dscores[start + source] = sum;
        dot += probabilities[start + source] * sum;
    }
    reduction[tid] = dot;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1)
    {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dot = reduction[0];
    float scale = 1.0f / sqrt((float)width);
    for (int source = tid; source < valid; source += 128)
        dscores[start + source] = probabilities[start + source]
            * (dscores[start + source] - dot) * scale;
}

// Second opt-in stage: keep the row's normalized probabilities in the same
// workgroup through context aggregation and gating. The probability array is
// still materialized for backward, so this does not add recomputation.
__kernel void q35t_attn_score_softmax_output_rowfused(
    __global const float* query, __global const float* key,
    __global const float* value, __global const float* q_and_gate,
    __global float* probabilities, __global float* context,
    __global float* output, int sequence, int heads,
    int kv_heads, int width)
{
    int row = get_group_id(0), tid = get_local_id(0);
    if (row >= sequence * heads) return;
    int time = row / heads, head = row % heads;
    int kv = head / (heads / kv_heads);
    int valid = time + 1;
    int start = q35tr_score_start(time, head, heads);
    int query_base = row * width;
    __local float reduction[128];
    float maximum = -INFINITY;

    for (int source = tid; source < valid; source += 128)
    {
        int key_base = (source * kv_heads + kv) * width;
        float sum = 0.0f;
        for (int j = 0; j < width; ++j)
            sum += query[query_base + j] * key[key_base + j];
        float score = sum / sqrt((float)width);
        probabilities[start + source] = score;
        maximum = fmax(maximum, score);
    }
    reduction[tid] = maximum;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1)
    {
        if (tid < stride) reduction[tid] = fmax(reduction[tid], reduction[tid + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maximum = reduction[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0.0f;
    for (int source = tid; source < valid; source += 128)
    {
        float probability = exp(probabilities[start + source] - maximum);
        probabilities[start + source] = probability;
        sum += probability;
    }
    reduction[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1)
    {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float denominator = reduction[0];
    for (int source = tid; source < valid; source += 128)
        probabilities[start + source] /= denominator;
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int j = tid; j < width; j += 128)
    {
        float context_sum = 0.0f;
        for (int source = 0; source < valid; ++source)
            context_sum += probabilities[start + source]
                * value[(source * kv_heads + kv) * width + j];
        int output_index = row * width + j;
        context[output_index] = context_sum;
        output[output_index] = context_sum
            * q35a_sigmoid(q_and_gate[(2 * row + 1) * width + j]);
    }
}
