// One layer's persistent causal K/V cache is [head, capacity, headWidth] in
// FP32. Callers decode their physical FP32/BF16/BFP8 QKV to the established
// Arc matrix-operand FP32 view before invoking these kernels.

// Populate K/V after the ordinary full-prompt attention prefill. The cache
// never retains Q, and all writes stay on the owning Arc lane.
__kernel void attention_kv_prefill_fp32(
    __global const float* qkv, __global float* keys, __global float* values,
    int sequence, int capacity, int width, int heads)
{
    const int index = get_global_id(0);
    if (index >= sequence * width) return;
    const int token = index / width;
    const int feature = index % width;
    const int depth = width / heads;
    const int head = feature / depth;
    const int channel = feature % depth;
    const int destination = (head * capacity + token) * depth + channel;
    const int source = token * 3 * width + feature;
    keys[destination] = qkv[source + width];
    values[destination] = qkv[source + 2 * width];
}

// One 64-lane workgroup owns a head. First append this token's K/V, then
// compute its one causal query against all cached keys. The score buffer is
// local to the workgroup, so attention needs no T-by-T probability workspace.
// The FP32 dot products and PV accumulation visit channels/keys in ascending
// order; the softmax uses the existing 64-lane reduction shape. Most Arc
// attention paths keep normalized P in FP32. The opt-in mix8_16 XMX product
// path instead rounds P to BF16 before PV.
inline float attention_kv_round_bf16(float value)
{
    uint bits = as_uint(value);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    else if ((bits & 0x007fffffu) != 0)
        bits |= 0x00400000u;
    return as_float(bits & 0xffff0000u);
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void attention_kv_decode_causal_fp32(
    __global const float* qkv, __global float* keys, __global float* values,
    __global float* output, int position, int capacity, int width, int heads,
    int roundProbabilityBf16, int useNativeExp,
    __local float* scores)
{
    const int head = get_group_id(0), lane = get_local_id(0);
    const int depth = width / heads;
    const int headBase = head * depth;
    const int cacheBase = head * capacity * depth;
    const int current = cacheBase + position * depth;
    __local float reduction[64];

    for (int channel = lane; channel < depth; channel += 64) {
        keys[current + channel] = qkv[width + headBase + channel];
        values[current + channel] = qkv[2 * width + headBase + channel];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    const float scale = rsqrt((float)depth);
    float maximum = -INFINITY;
    for (int key = lane; key <= position; key += 64) {
        const int keyBase = cacheBase + key * depth;
        float dot = 0.0f;
        for (int channel = 0; channel < depth; ++channel)
            dot = fma(qkv[headBase + channel], keys[keyBase + channel], dot);
        // Keep the unscaled score, as in attention_probabilities. Scaling
        // inside exp may contract with the subtraction on the device.
        scores[key] = dot;
        maximum = fmax(maximum, dot * scale);
    }
    reduction[lane] = maximum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride)
            reduction[lane] = fmax(reduction[lane], reduction[lane + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maximum = reduction[0];

    float sum = 0.0f;
    for (int key = lane; key <= position; key += 64) {
        float argument = scores[key] * scale - maximum;
        float probability = useNativeExp ? native_exp(argument) : exp(argument);
        scores[key] = probability;
        sum += probability;
    }
    reduction[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride)
            reduction[lane] += reduction[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inverse = 1.0f / reduction[0];
    for (int key = lane; key <= position; key += 64) {
        float probability = scores[key] * inverse;
        scores[key] = roundProbabilityBf16
            ? attention_kv_round_bf16(probability) : probability;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int channel = lane; channel < depth; channel += 64) {
        float result = 0.0f;
        for (int key = 0; key <= position; ++key)
            result = fma(scores[key], values[cacheBase + key * depth + channel], result);
        output[headBase + channel] = result;
    }
}
