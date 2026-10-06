// Opt-in packed causal training attention: only [source <= query time] exists.
// FP32 dot/reduction order matches the dense kernels; no future masked entries
// are stored or read. Layout groups each time's heads with (time + 1) values.
inline int q35tp_score_start(int time, int head, int heads)
{
    return (time * (time + 1) / 2) * heads + head * (time + 1);
}

__kernel void q35t_attn_scores_packed(__global const float* query, __global const float* key,
    __global float* scores, int sequence, int heads, int kv_heads, int width)
{
    int i = get_global_id(0);
    if (i >= sequence * heads * sequence) return;
    int source = i % sequence, row = i / sequence, time = row / heads;
    int kv = (row % heads) / (heads / kv_heads);
    if (source > time) return;
    float sum = 0.0f;
    if (source <= time)
        for (int j = 0; j < width; ++j)
            sum += query[row * width + j] * key[(source * kv_heads + kv) * width + j];
    scores[q35tp_score_start(time, row % heads, heads) + source] = sum / sqrt((float)width);
}

// One 128-item workgroup per (query time, query head).

__kernel void q35t_attn_softmax_packed(__global float* scores, int sequence, int heads)
{
    int row = get_group_id(0), tid = get_local_id(0);
    int valid = row / heads + 1, start = q35tp_score_start(row / heads, row % heads, heads);
    __local float reduction[128];
    float maximum = -INFINITY;
    for (int s = tid; s < valid; s += 128) maximum = fmax(maximum, scores[start + s]);
    reduction[tid] = maximum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1)
    {
        if (tid < stride) reduction[tid] = fmax(reduction[tid], reduction[tid + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maximum = reduction[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0.0f;
    for (int s = tid; s < valid; s += 128)
    {
        float probability = s < valid ? exp(scores[start + s] - maximum) : 0.0f;
        scores[start + s] = probability;
        sum += probability;
    }
    reduction[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1)
    {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum = reduction[0];
    for (int s = tid; s < valid; s += 128) scores[start + s] /= sum;
}

__kernel void q35t_attn_output_packed(__global const float* probabilities, __global const float* value,
    __global const float* q_and_gate, __global float* context, __global float* output,
    int sequence, int heads, int kv_heads, int width)
{
    int i = get_global_id(0);
    if (i >= sequence * heads * width) return;
    int j = i % width, row = i / width, time = row / heads;
    int kv = (row % heads) / (heads / kv_heads);
    float sum = 0.0f;
    for (int s = 0; s <= time; ++s)
        sum += probabilities[q35tp_score_start(time, row % heads, heads) + s] * value[(s * kv_heads + kv) * width + j];
    context[i] = sum;
    output[i] = sum * q35a_sigmoid(q_and_gate[(2 * row + 1) * width + j]);
}

__kernel void q35t_attn_probability_backward_packed(__global const float* dcontext,
    __global const float* value, __global float* dscores,
    int sequence, int heads, int kv_heads, int width)
{
    int i = get_global_id(0);
    if (i >= sequence * heads * sequence) return;
    int source = i % sequence, row = i / sequence, time = row / heads;
    int kv = (row % heads) / (heads / kv_heads);
    if (source > time) return;
    float sum = 0.0f;
    if (source <= time)
        for (int j = 0; j < width; ++j)
            sum += dcontext[row * width + j] * value[(source * kv_heads + kv) * width + j];
    dscores[q35tp_score_start(time, row % heads, heads) + source] = sum;
}

__kernel void q35t_attn_softmax_backward_packed(__global const float* probabilities,
    __global float* dscores, int sequence, int heads, int width)
{
    int row = get_group_id(0), tid = get_local_id(0);
    int valid = row / heads + 1, start = q35tp_score_start(row / heads, row % heads, heads);
    __local float reduction[128];
    float dot = 0.0f;
    for (int s = tid; s < valid; s += 128)
        dot += probabilities[start + s] * dscores[start + s];
    reduction[tid] = dot;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1)
    {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dot = reduction[0];
    float scale = 1.0f / sqrt((float)width);
    for (int s = tid; s < valid; s += 128)
        dscores[start + s] = s < valid
            ? probabilities[start + s] * (dscores[start + s] - dot) * scale : 0.0f;
}

__kernel void q35t_attn_query_backward_packed(__global const float* dscores,
    __global const float* key, __global float* dquery,
    int sequence, int heads, int kv_heads, int width)
{
    int i = get_global_id(0);
    if (i >= sequence * heads * width) return;
    int j = i % width, row = i / width, time = row / heads;
    int kv = (row % heads) / (heads / kv_heads);
    float sum = 0.0f;
    for (int s = 0; s <= time; ++s)
        sum += dscores[q35tp_score_start(time, row % heads, heads) + s] * key[(s * kv_heads + kv) * width + j];
    dquery[i] = sum;
}

__kernel void q35t_attn_key_value_backward_packed(__global const float* dscores,
    __global const float* probabilities, __global const float* query,
    __global const float* dcontext, __global float* dkey, __global float* dvalue,
    int sequence, int heads, int kv_heads, int width)
{
    int i = get_global_id(0);
    if (i >= sequence * kv_heads * width) return;
    int j = i % width, row = i / width, source = row / kv_heads;
    int group = heads / kv_heads, first_head = (row % kv_heads) * group;
    float key_sum = 0.0f, value_sum = 0.0f;
    for (int t = source; t < sequence; ++t)
        for (int h = first_head; h < first_head + group; ++h)
        {
            int query_row = t * heads + h;
            int score_index = q35tp_score_start(t, h, heads) + source;
            key_sum += dscores[score_index] * query[query_row * width + j];
            value_sum += probabilities[score_index] * dcontext[query_row * width + j];
        }
    dkey[i] = key_sum;
    dvalue[i] += value_sum;
}

// Frozen normalization weights: only input gradients are accumulated.
