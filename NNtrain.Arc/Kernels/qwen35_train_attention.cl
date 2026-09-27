// Full-sequence causal Qwen3.5 attention. Layout: [time, head, component],
// with Q/gate input interleaved as [time, head, Q-or-gate, component].
// Each backward destination has one writer; no float atomics are required.
__kernel void q35t_attn_rope(__global float* values, int sequence, int heads,
    int width, int dimensions, float theta, int inverse)
{
    int i = get_global_id(0), pairs = dimensions / 2;
    if (i >= sequence * heads * pairs) return;
    int row = i / pairs, j = i % pairs, position = row / heads;
    float angle = position * pow(theta, -2.0f * j / dimensions);
    float cosine = cos(angle), sine = sin(angle) * (inverse ? -1.0f : 1.0f);
    int first = row * width + j, second = first + pairs;
    float a = values[first], b = values[second];
    values[first] = a * cosine - b * sine;
    values[second] = a * sine + b * cosine;
}

__kernel void q35t_attn_scores(__global const float* query, __global const float* key,
    __global float* scores, int sequence, int heads, int kv_heads, int width)
{
    int i = get_global_id(0);
    if (i >= sequence * heads * sequence) return;
    int source = i % sequence, row = i / sequence, time = row / heads;
    int kv = (row % heads) / (heads / kv_heads);
    float sum = 0.0f;
    if (source <= time)
        for (int j = 0; j < width; ++j)
            sum += query[row * width + j] * key[(source * kv_heads + kv) * width + j];
    scores[i] = source <= time ? sum / sqrt((float)width) : -INFINITY;
}

// One 128-item workgroup per (query time, query head).
__kernel void q35t_attn_softmax(__global float* scores, int sequence, int heads)
{
    int row = get_group_id(0), tid = get_local_id(0);
    int valid = row / heads + 1, start = row * sequence;
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
    for (int s = tid; s < sequence; s += 128)
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
    for (int s = tid; s < sequence; s += 128) scores[start + s] /= sum;
}

__kernel void q35t_attn_output(__global const float* probabilities, __global const float* value,
    __global const float* q_and_gate, __global float* context, __global float* output,
    int sequence, int heads, int kv_heads, int width)
{
    int i = get_global_id(0);
    if (i >= sequence * heads * width) return;
    int j = i % width, row = i / width, time = row / heads;
    int kv = (row % heads) / (heads / kv_heads);
    float sum = 0.0f;
    for (int s = 0; s <= time; ++s)
        sum += probabilities[row * sequence + s] * value[(s * kv_heads + kv) * width + j];
    context[i] = sum;
    output[i] = sum * q35a_sigmoid(q_and_gate[(2 * row + 1) * width + j]);
}

__kernel void q35t_attn_gate_backward(__global const float* dy, __global const float* context,
    __global const float* q_and_gate, __global float* dcontext, __global float* dq_and_gate,
    int length, int width)
{
    int i = get_global_id(0);
    if (i >= length) return;
    int gate_index = (2 * (i / width) + 1) * width + i % width;
    float gate = q35a_sigmoid(q_and_gate[gate_index]);
    dcontext[i] = dy[i] * gate;
    dq_and_gate[gate_index] += dy[i] * context[i] * gate * (1.0f - gate);
}

__kernel void q35t_attn_probability_backward(__global const float* dcontext,
    __global const float* value, __global float* dscores,
    int sequence, int heads, int kv_heads, int width)
{
    int i = get_global_id(0);
    if (i >= sequence * heads * sequence) return;
    int source = i % sequence, row = i / sequence, time = row / heads;
    int kv = (row % heads) / (heads / kv_heads);
    float sum = 0.0f;
    if (source <= time)
        for (int j = 0; j < width; ++j)
            sum += dcontext[row * width + j] * value[(source * kv_heads + kv) * width + j];
    dscores[i] = sum;
}

__kernel void q35t_attn_softmax_backward(__global const float* probabilities,
    __global float* dscores, int sequence, int heads, int width)
{
    int row = get_group_id(0), tid = get_local_id(0);
    int valid = row / heads + 1, start = row * sequence;
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
    for (int s = tid; s < sequence; s += 128)
        dscores[start + s] = s < valid
            ? probabilities[start + s] * (dscores[start + s] - dot) * scale : 0.0f;
}

__kernel void q35t_attn_query_backward(__global const float* dscores,
    __global const float* key, __global float* dquery,
    int sequence, int heads, int kv_heads, int width)
{
    int i = get_global_id(0);
    if (i >= sequence * heads * width) return;
    int j = i % width, row = i / width, time = row / heads;
    int kv = (row % heads) / (heads / kv_heads);
    float sum = 0.0f;
    for (int s = 0; s <= time; ++s)
        sum += dscores[row * sequence + s] * key[(s * kv_heads + kv) * width + j];
    dquery[i] = sum;
}

__kernel void q35t_attn_key_value_backward(__global const float* dscores,
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
            int score_index = query_row * sequence + source;
            key_sum += dscores[score_index] * query[query_row * width + j];
            value_sum += probabilities[score_index] * dcontext[query_row * width + j];
        }
    dkey[i] = key_sum;
    dvalue[i] += value_sum;
}

// Frozen normalization weights: only input gradients are accumulated.
__kernel void q35t_attn_norm_backward(__global const float* input, __global const float* weight,
    __global const float* dy, __global float* dx, int width, int input_stride, float epsilon)
{
    int row = get_group_id(0), tid = get_local_id(0);
    __local float squares[128], products[128];
    float square = 0.0f, product = 0.0f;
    for (int j = tid; j < width; j += 128)
    {
        float x = input[row * input_stride + j];
        square += x * x;
        product += x * weight[j] * dy[row * width + j];
    }
    squares[tid] = square;
    products[tid] = product;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            squares[tid] += squares[tid + stride];
            products[tid] += products[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inverse = 1.0f / sqrt(squares[0] / width + epsilon);
    float correction = products[0] * inverse * inverse / width;
    for (int j = tid; j < width; j += 128)
    {
        int index = row * input_stride + j;
        dx[index] += inverse * (dy[row * width + j] * weight[j] - input[index] * correction);
    }
}
