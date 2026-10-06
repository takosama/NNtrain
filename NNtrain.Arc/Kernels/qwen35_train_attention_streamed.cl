// Causal Qwen3.5 training attention with a reusable query-row score tile.
// The host owns the tile loop. first_row and row_count describe global query
// positions; scores and d_scores are laid out as
// [local_query_row, query_head, full_source_sequence]. Only causal entries
// are written or read. The normalized Q/K and the context remain full-length.
// These kernels intentionally keep the scalar FP32 reduction order of the
// existing dense attention path so the memory-saving path can be validated
// independently before introducing a faster matrix tile implementation.

inline int q35ts_score_index(int local_row, int head, int source,
    int heads, int sequence)
{
    return (local_row * heads + head) * sequence + source;
}

__kernel void q35t_attn_scores_streamed(__global const float* query,
    __global const float* key, __global float* scores,
    int sequence, int heads, int kv_heads, int width,
    int first_row, int row_count)
{
    int i = get_global_id(0);
    int active_sources = first_row + row_count;
    if (i >= row_count * heads * active_sources) return;
    int source = i % active_sources;
    int local_query = i / (heads * active_sources);
    int head = (i / active_sources) % heads;
    int time = first_row + local_query;
    if (source > time) return;
    int kv = head / (heads / kv_heads);
    int query_base = (time * heads + head) * width;
    int key_base = (source * kv_heads + kv) * width;
    float sum = 0.0f;
    for (int j = 0; j < width; ++j)
        sum += query[query_base + j] * key[key_base + j];
    scores[q35ts_score_index(local_query, head, source, heads, sequence)] =
        sum / sqrt((float)width);
}

// One 128-item workgroup per (local query row, query head).
__kernel void q35t_attn_softmax_streamed(__global float* scores,
    int sequence, int heads, int first_row, int row_count)
{
    int row = get_group_id(0), tid = get_local_id(0);
    if (row >= row_count * heads) return;
    int local_query = row / heads, head = row % heads;
    int valid = first_row + local_query + 1;
    int start = q35ts_score_index(local_query, head, 0, heads, sequence);
    __local float reduction[128];
    float maximum = -INFINITY;
    for (int s = tid; s < valid; s += 128)
        maximum = fmax(maximum, scores[start + s]);
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
        float probability = exp(scores[start + s] - maximum);
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
    for (int s = tid; s < valid; s += 128)
        scores[start + s] /= sum;
}

__kernel void q35t_attn_output_streamed(__global const float* probabilities,
    __global const float* value, __global const float* q_and_gate,
    __global float* context, __global float* output,
    int sequence, int heads, int kv_heads, int width,
    int first_row, int row_count)
{
    int i = get_global_id(0);
    if (i >= row_count * heads * width) return;
    int j = i % width, local_query = i / (heads * width);
    int head = (i / width) % heads, time = first_row + local_query;
    int kv = head / (heads / kv_heads);
    int row = time * heads + head;
    int score_start = q35ts_score_index(local_query, head, 0, heads, sequence);
    float sum = 0.0f;
    for (int s = 0; s <= time; ++s)
        sum += probabilities[score_start + s] * value[(s * kv_heads + kv) * width + j];
    int output_index = row * width + j;
    context[output_index] = sum;
    output[output_index] = sum * q35a_sigmoid(q_and_gate[(2 * row + 1) * width + j]);
}

__kernel void q35t_attn_probability_backward_streamed(
    __global const float* dcontext, __global const float* value,
    __global float* d_scores, int sequence, int heads, int kv_heads,
    int width, int first_row, int row_count)
{
    int i = get_global_id(0);
    int active_sources = first_row + row_count;
    if (i >= row_count * heads * active_sources) return;
    int source = i % active_sources;
    int local_query = i / (heads * active_sources);
    int head = (i / active_sources) % heads;
    int time = first_row + local_query;
    if (source > time) return;
    int kv = head / (heads / kv_heads);
    int context_base = (time * heads + head) * width;
    int value_base = (source * kv_heads + kv) * width;
    float sum = 0.0f;
    for (int j = 0; j < width; ++j)
        sum += dcontext[context_base + j] * value[value_base + j];
    d_scores[q35ts_score_index(local_query, head, source, heads, sequence)] = sum;
}

__kernel void q35t_attn_softmax_backward_streamed(
    __global const float* probabilities, __global float* d_scores,
    int sequence, int heads, int width, int first_row, int row_count)
{
    int row = get_group_id(0), tid = get_local_id(0);
    if (row >= row_count * heads) return;
    int local_query = row / heads, head = row % heads;
    int valid = first_row + local_query + 1;
    int start = q35ts_score_index(local_query, head, 0, heads, sequence);
    __local float reduction[128];
    float dot = 0.0f;
    for (int s = tid; s < valid; s += 128)
        dot += probabilities[start + s] * d_scores[start + s];
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
        d_scores[start + s] = probabilities[start + s]
            * (d_scores[start + s] - dot) * scale;
}

__kernel void q35t_attn_query_backward_streamed(
    __global const float* d_scores, __global const float* key,
    __global float* dquery, int sequence, int heads, int kv_heads,
    int width, int first_row, int row_count)
{
    int i = get_global_id(0);
    if (i >= row_count * heads * width) return;
    int j = i % width, local_query = i / (heads * width);
    int head = (i / width) % heads, time = first_row + local_query;
    int kv = head / (heads / kv_heads);
    int score_start = q35ts_score_index(local_query, head, 0, heads, sequence);
    float sum = 0.0f;
    for (int s = 0; s <= time; ++s)
        sum += d_scores[score_start + s] * key[(s * kv_heads + kv) * width + j];
    dquery[(time * heads + head) * width + j] = sum;
}

// Run tiles in increasing first_row order in one in-order queue. Each work
// item exclusively owns one dkey/dvalue component; no floating atomics occur.
// The first tile assigns dkey (including zero for future-only source rows),
// later tiles accumulate. dvalue always adds to its caller-owned gradient.
__kernel void q35t_attn_key_value_backward_streamed(
    __global const float* d_scores, __global const float* probabilities,
    __global const float* query, __global const float* dcontext,
    __global float* dkey, __global float* dvalue,
    int sequence, int heads, int kv_heads, int width,
    int first_row, int row_count, int accumulate_key)
{
    int i = get_global_id(0);
    if (i >= sequence * kv_heads * width) return;
    int j = i % width, row = i / width, source = row / kv_heads;
    int group = heads / kv_heads;
    int first_head = (row % kv_heads) * group;
    int end = min(sequence, first_row + row_count);
    if (source >= end)
    {
        if (!accumulate_key) dkey[i] = 0.0f;
        return;
    }
    // Carry the running sums across query tiles, rather than adding a tile
    // subtotal. This preserves the original time/head reduction order.
    float key_sum = accumulate_key ? dkey[i] : 0.0f;
    float value_sum = dvalue[i];
    for (int time = max(source, first_row); time < end; ++time)
        for (int head = first_head; head < first_head + group; ++head)
        {
            int query_row = time * heads + head;
            int score_index = q35ts_score_index(time - first_row, head, source,
                heads, sequence);
            key_sum += d_scores[score_index] * query[query_row * width + j];
            value_sum += probabilities[score_index] * dcontext[query_row * width + j];
        }
    dkey[i] = key_sum;
    dvalue[i] = value_sum;
}
