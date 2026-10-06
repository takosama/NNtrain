// Experimental bounded-score causal attention. One workgroup owns each query
// row/head inside a host-managed tile; all barriers remain within that row.
// Probabilities/dscores use [local_row, head, full_source_sequence] so the
// existing streamed dK/dV kernel can consume them without atomics.
inline int q35tsr_index(int local_row, int head, int source,
    int heads, int sequence)
{
    return (local_row * heads + head) * sequence + source;
}

inline void q35tsr_score_softmax_row(
    __global const float* query, __global const float* key,
    __global float* probabilities, __local float* reduction,
    int sequence, int heads, int kv_heads, int width,
    int first_row, int local_row, int head, int tid)
{
    int time = first_row + local_row;
    int kv = head / (heads / kv_heads);
    int valid = time + 1;
    int start = q35tsr_index(local_row, head, 0, heads, sequence);
    int query_base = (time * heads + head) * width;
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
}

__kernel void q35t_attn_score_softmax_streamed_rowfused(
    __global const float* query, __global const float* key,
    __global float* probabilities, int sequence, int heads,
    int kv_heads, int width, int first_row, int row_count)
{
    int row = get_group_id(0), tid = get_local_id(0);
    if (row >= row_count * heads) return;
    __local float reduction[128];
    q35tsr_score_softmax_row(query, key, probabilities, reduction,
        sequence, heads, kv_heads, width, first_row,
        row / heads, row % heads, tid);
}

__kernel void q35t_attn_score_softmax_output_streamed_rowfused(
    __global const float* query, __global const float* key,
    __global const float* value, __global const float* q_and_gate,
    __global float* probabilities, __global float* context,
    __global float* output, int sequence, int heads,
    int kv_heads, int width, int first_row, int row_count)
{
    int row = get_group_id(0), tid = get_local_id(0);
    if (row >= row_count * heads) return;
    int local_row = row / heads, head = row % heads;
    int time = first_row + local_row;
    int kv = head / (heads / kv_heads);
    int start = q35tsr_index(local_row, head, 0, heads, sequence);
    __local float reduction[128];
    q35tsr_score_softmax_row(query, key, probabilities, reduction,
        sequence, heads, kv_heads, width, first_row,
        local_row, head, tid);

    for (int j = tid; j < width; j += 128)
    {
        float sum = 0.0f;
        for (int source = 0; source <= time; ++source)
            sum += probabilities[start + source]
                * value[(source * kv_heads + kv) * width + j];
        int output_index = (time * heads + head) * width + j;
        context[output_index] = sum;
        output[output_index] = sum
            * q35a_sigmoid(q_and_gate[(2 * (time * heads + head) + 1) * width + j]);
    }
}

// Backward recomputes scores/probabilities once for this tile. It fuses dP,
// softmax backward and dQ while leaving dK/dV to the existing unique-writer
// streamed kernel. This avoids float atomics and retains the original order.
__kernel void q35t_attn_backward_streamed_rowfused(
    __global const float* query, __global const float* key,
    __global const float* value, __global const float* dcontext,
    __global float* probabilities, __global float* dscores,
    __global float* dquery, int sequence, int heads,
    int kv_heads, int width, int first_row, int row_count)
{
    int row = get_group_id(0), tid = get_local_id(0);
    if (row >= row_count * heads) return;
    int local_row = row / heads, head = row % heads;
    int time = first_row + local_row;
    int kv = head / (heads / kv_heads);
    int start = q35tsr_index(local_row, head, 0, heads, sequence);
    int context_base = (time * heads + head) * width;
    __local float reduction[128];
    q35tsr_score_softmax_row(query, key, probabilities, reduction,
        sequence, heads, kv_heads, width, first_row,
        local_row, head, tid);

    float dot = 0.0f;
    for (int source = tid; source <= time; source += 128)
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
    for (int source = tid; source <= time; source += 128)
        dscores[start + source] = probabilities[start + source]
            * (dscores[start + source] - dot) * scale;
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int j = tid; j < width; j += 128)
    {
        float sum = 0.0f;
        for (int source = 0; source <= time; ++source)
            sum += dscores[start + source]
                * key[(source * kv_heads + kv) * width + j];
        dquery[context_base + j] = sum;
    }
}
