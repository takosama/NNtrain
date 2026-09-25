// T2048 attention softmax candidate.  Each work-item keeps the reference
// modulo-64 reduction stream while SLM retains the exponentials until the
// normalization constant is known.  This avoids an unnormalized FP32 global
// write and subsequent global read without holding 32 values in registers.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void attention_probabilities_slm_2048(
    __global float* scores, __global float* stats,
    int seq, int width, int heads, int first, int causal, int saved)
{
    const int row = get_group_id(0);
    const int query = row % seq;
    const int group = row / seq;
    const int lane = get_local_id(0);
    const int count = causal ? query + 1 : seq;
    const int limit = causal == 2 ? min(seq, (query / 64 + 1) * 64) : seq;
    const int stat = (first + group) * seq + query;
    const int base = row * seq;
    const float scale = rsqrt((float)(width / heads));
    __local float reduction[64];
    __local float exponentials[2048];

    // Match the reference max reduction, including its per-lane key order.
    float maximum = -INFINITY;
    if (!saved) {
        for (int key = lane; key < count; key += 64)
            maximum = fmax(maximum, scores[base + key] * scale);
        reduction[lane] = maximum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = 32; stride != 0; stride /= 2) {
            if (lane < stride)
                reduction[lane] = fmax(reduction[lane], reduction[lane + stride]);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        maximum = reduction[0];
        // All lanes must finish reading reduction[0] before lane 0 reuses
        // that location for the exponential sum below.
        barrier(CLK_LOCAL_MEM_FENCE);
    } else {
        maximum = stats[2 * stat];
    }

    // The saved-statistics path needs no reduction or SLM staging.
    if (saved) {
        const float inverse = stats[2 * stat + 1];
        for (int key = lane; key < limit; key += 64) {
            const float value = key < count
                ? exp(scores[base + key] * scale - maximum) : 0.0f;
            scores[base + key] = value * inverse;
        }
        return;
    }

    float sum = 0.0f;
    for (int key = lane; key < limit; key += 64) {
        const float value = key < count
            ? exp(scores[base + key] * scale - maximum) : 0.0f;
        exponentials[key] = value;
        sum += value;
    }
    reduction[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride != 0; stride /= 2) {
        if (lane < stride)
            reduction[lane] += reduction[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inverse = 1.0f / reduction[0];
    if (lane == 0) {
        stats[2 * stat] = maximum;
        stats[2 * stat + 1] = inverse;
    }
    for (int key = lane; key < limit; key += 64)
        scores[base + key] = exponentials[key] * inverse;
}

// Recomputation with saved row statistics can publish normalized P directly.
// Use a separate entry point so the driver reserves no SLM for this path.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void attention_probabilities_saved_direct_2048(
    __global float* scores, __global float* stats,
    int seq, int width, int heads, int first, int causal, int saved)
{
    const int row = get_group_id(0);
    const int query = row % seq;
    const int group = row / seq;
    const int lane = get_local_id(0);
    const int count = causal ? query + 1 : seq;
    const int limit = causal == 2 ? min(seq, (query / 64 + 1) * 64) : seq;
    const int stat = (first + group) * seq + query;
    const int base = row * seq;
    const float scale = rsqrt((float)(width / heads));
    const float maximum = stats[2 * stat];
    const float inverse = stats[2 * stat + 1];
    for (int key = lane; key < limit; key += 64) {
        const float value = key < count
            ? exp(scores[base + key] * scale - maximum) : 0.0f;
        scores[base + key] = value * inverse;
    }
}

// Cache both reduction operands in SLM.  The reference reads P and dP once
// for the reduction, then reads them again for the derivative.  This version
// performs one global read of each and leaves per-work-item register pressure
// independent of sequence length.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void attention_derivatives_slm_2048(
    __global const float* probabilities, __global float* derivatives,
    int seq, int width, int heads, int causal)
{
    const int row = get_group_id(0);
    const int query = row % seq;
    const int lane = get_local_id(0);
    const int count = causal ? query + 1 : seq;
    const int limit = causal == 2 ? min(seq, (query / 64 + 1) * 64) : seq;
    const int base = row * seq;
    __local float reduction[64];
    __local float cachedProbability[2048];
    __local float cachedDerivative[2048];
    float sum = 0.0f;

    for (int key = lane; key < count; key += 64) {
        const float p = probabilities[base + key];
        const float dp = derivatives[base + key];
        cachedProbability[key] = p;
        cachedDerivative[key] = dp;
        sum = fma(p, dp, sum);
    }
    reduction[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride != 0; stride /= 2) {
        if (lane < stride)
            reduction[lane] += reduction[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float delta = reduction[0];
    const float scale = rsqrt((float)(width / heads));
    for (int key = lane; key < limit; key += 64)
        derivatives[base + key] = key < count
            ? cachedProbability[key] * (cachedDerivative[key] - delta) * scale
            : 0.0f;
}
