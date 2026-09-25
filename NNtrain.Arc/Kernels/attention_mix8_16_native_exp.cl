// mix8_16 speed policy: native_exp is an explicitly measured approximation.
// Keep the reference reduction tree, masking and FP32 probability workspace.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void attention_probabilities_native_exp_2048(
    __global float* scores, __global float* stats,
    int seq, int width, int heads, int first, int causal, int saved)
{
    const int row = get_group_id(0), query = row % seq, group = row / seq;
    const int lane = get_local_id(0), count = causal ? query + 1 : seq;
    const int limit = causal == 2 ? min(seq, (query / 64 + 1) * 64) : seq;
    const int stat = (first + group) * seq + query, base = row * seq;
    const float scale = rsqrt((float)(width / heads));
    __local float reduction[64];
    float maximum = -INFINITY;
    if (!saved) {
        for (int key = lane; key < count; key += 64)
            maximum = fmax(maximum, scores[base + key] * scale);
        reduction[lane] = maximum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = 32; stride; stride /= 2) {
            if (lane < stride) reduction[lane] = fmax(reduction[lane], reduction[lane + stride]);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        maximum = reduction[0];
    } else maximum = stats[2 * stat];
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0.0f;
    for (int key = lane; key < limit; key += 64) {
        float p = key < count ? native_exp(scores[base + key] * scale - maximum) : 0.0f;
        scores[base + key] = p;
        sum += p;
    }
    float inverse;
    if (!saved) {
        reduction[lane] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = 32; stride; stride /= 2) {
            if (lane < stride) reduction[lane] += reduction[lane + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        inverse = 1.0f / reduction[0];
        if (!lane) { stats[2 * stat] = maximum; stats[2 * stat + 1] = inverse; }
    } else inverse = stats[2 * stat + 1];
    for (int key = lane; key < limit; key += 64) scores[base + key] *= inverse;
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void attention_probabilities_saved_native_exp_2048(
    __global float* scores, __global float* stats,
    int seq, int width, int heads, int first, int causal, int saved)
{
    const int row = get_group_id(0), query = row % seq, group = row / seq;
    const int lane = get_local_id(0), count = causal ? query + 1 : seq;
    const int limit = causal == 2 ? min(seq, (query / 64 + 1) * 64) : seq;
    const int stat = (first + group) * seq + query, base = row * seq;
    const float scale = rsqrt((float)(width / heads));
    const float maximum = stats[2 * stat], inverse = stats[2 * stat + 1];
    for (int key = lane; key < limit; key += 64) {
        float p = key < count ? native_exp(scores[base + key] * scale - maximum) : 0.0f;
        scores[base + key] = p * inverse;
    }
}
