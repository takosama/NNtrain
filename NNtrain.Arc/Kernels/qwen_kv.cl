// Token-major cache stores rotated keys and raw values in FP32. RoPE uses
// absolute positions, with the same NEOX pairing as the full Qwen GQA path.
inline void qwen_kv_rope_pair(
    __global const float* x, int base, int pair, int width, int position,
    float theta, float* first, float* second) {
    float angle = (float)position * pow(theta, -2.0f * (float)pair / (float)width);
    float c = cos(angle), s = sin(angle);
    float a = x[base + pair], b = x[base + pair + width / 2];
    *first = a * c - b * s;
    *second = b * c + a * s;
}

__kernel void qwen_kv_append(
    __global const float* key, __global const float* value,
    __global float* cached_key, __global float* cached_value,
    int tokens, int position, int kv_heads, int head_width, float theta) {
    int i = get_global_id(0), half_width = head_width / 2;
    if (i >= tokens * kv_heads * half_width) return;
    int pair = i % half_width;
    int head = (i / half_width) % kv_heads;
    int token = i / (half_width * kv_heads);
    int src = (token * kv_heads + head) * head_width;
    int dst = ((position + token) * kv_heads + head) * head_width;
    float first, second;
    qwen_kv_rope_pair(key, src, pair, head_width, position + token, theta, &first, &second);
    cached_key[dst + pair] = first;
    cached_key[dst + pair + half_width] = second;
    cached_value[dst + pair] = value[src + pair];
    cached_value[dst + pair + half_width] = value[src + pair + half_width];
}

inline float qwen_kv_score(
    __global const float* rotated_query, __global const float* key,
    int head, int key_position, int heads, int kv_heads, int head_width) {
    int kv_head = head / (heads / kv_heads);
    int qbase = head * head_width;
    int kbase = (key_position * kv_heads + kv_head) * head_width;
    float dot = 0.0f;
    for (int pair = 0; pair < head_width / 2; ++pair) {
        dot = fma(rotated_query[qbase + pair], key[kbase + pair], dot);
        dot = fma(rotated_query[qbase + pair + head_width / 2],
            key[kbase + pair + head_width / 2], dot);
    }
    return dot * rsqrt((float)head_width);
}

__kernel void qwen_kv_decode(
    __global const float* query, __global const float* key,
    __global const float* value, __global float* output,
    int position, int heads, int kv_heads, int head_width, float theta,
    __local float* scratch) {
    int head = get_group_id(0), lane = get_local_id(0);
    int step = get_local_size(0), length = position + 1;
    __local float* rotated_query = scratch;
    __local float* scores = scratch + head_width;
    // RoPE is identical for every key position. Compute it once per head,
    // cooperatively, instead of repeating pow/cos/sin inside every score dot.
    for (int pair = lane; pair < head_width / 2; pair += step) {
        float first, second;
        qwen_kv_rope_pair(query, head * head_width, pair, head_width,
            position, theta, &first, &second);
        rotated_query[pair] = first;
        rotated_query[pair + head_width / 2] = second;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int kv_head = head / (heads / kv_heads);
    for (int token = lane; token < length; token += step) {
        int kbase = (token * kv_heads + kv_head) * head_width;
        float dot = 0.0f;
        // Retain the reference first-half/second-half FMA order.
        for (int pair = 0; pair < head_width / 2; ++pair) {
            dot = fma(rotated_query[pair], key[kbase + pair], dot);
            dot = fma(rotated_query[pair + head_width / 2],
                key[kbase + pair + head_width / 2], dot);
        }
        scores[token] = dot * rsqrt((float)head_width);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0) {
        float maximum = -INFINITY, sum = 0.0f;
        for (int token = 0; token < length; ++token)
            maximum = fmax(maximum, scores[token]);
        for (int token = 0; token < length; ++token) {
            scores[token] = exp(scores[token] - maximum);
            sum += scores[token];
        }
        for (int token = 0; token < length; ++token) scores[token] /= sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int channel = lane; channel < head_width; channel += step) {
        float result = 0.0f;
        for (int token = 0; token < length; ++token)
            result = fma(scores[token], value[(token * kv_heads + kv_head) * head_width + channel], result);
        output[head * head_width + channel] = result;
    }
}

// The long-context score kernel keeps independent key positions parallel.
// This small first pass lets those work-items share one rotated query buffer.
__kernel void qwen_kv_rotate_query(
    __global const float* query, __global float* rotated_query,
    int position, int heads, int head_width, float theta) {
    int i = get_global_id(0), half_width = head_width / 2;
    if (i >= heads * half_width) return;
    int pair = i % half_width, base = (i / half_width) * head_width;
    float first, second;
    qwen_kv_rope_pair(query, base, pair, head_width, position, theta, &first, &second);
    rotated_query[base + pair] = first;
    rotated_query[base + pair + half_width] = second;
}

__kernel void qwen_kv_scores(
    __global const float* rotated_query, __global const float* key,
    __global float* scores, int length, int heads, int kv_heads,
    int head_width) {
    int i = get_global_id(0);
    if (i >= heads * length) return;
    scores[i] = qwen_kv_score(rotated_query, key, i / length, i % length,
        heads, kv_heads, head_width);
}

__kernel void qwen_kv_output(
    __global float* scores, __global const float* value,
    __global float* output, int length, int heads, int kv_heads, int head_width) {
    int head = get_group_id(0), lane = get_local_id(0), step = get_local_size(0);
    __global float* row = scores + head * length;
    if (lane == 0) {
        float maximum = -INFINITY, sum = 0.0f;
        for (int token = 0; token < length; ++token) maximum = fmax(maximum, row[token]);
        for (int token = 0; token < length; ++token) {
            row[token] = exp(row[token] - maximum);
            sum += row[token];
        }
        for (int token = 0; token < length; ++token) row[token] /= sum;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    int kv_head = head / (heads / kv_heads);
    for (int channel = lane; channel < head_width; channel += step) {
        float result = 0.0f;
        for (int token = 0; token < length; ++token)
            result = fma(row[token], value[(token * kv_heads + kv_head) * head_width + channel], result);
        output[head * head_width + channel] = result;
    }
}
