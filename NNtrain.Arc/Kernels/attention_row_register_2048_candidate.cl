// T2048 candidate.  The 64 work-items keep the reference kernel's modulo-64
// per-key streams, FP32 arithmetic order, causal publication and stats ABI.
// SG16 performs the last four reduction levels after the same 32/16 levels
// are formed from SLM; each of the four subgroups computes the same result.
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

inline float attention_register_2048_reduce64(__local float* partials, int maximum)
{
    const int lane = get_sub_group_local_id();
    const float left = partials[lane];
    const float right = partials[lane + 32];
    float value = maximum ? fmax(left, right) : left + right;
    const float middleLeft = partials[lane + 16];
    const float middleRight = partials[lane + 48];
    value = maximum ? fmax(value, fmax(middleLeft, middleRight))
                    : value + (middleLeft + middleRight);
    #pragma unroll
    for (int stride = 8; stride != 0; stride /= 2) {
        const float other = intel_sub_group_shuffle(value, (lane + stride) & 15);
        if (lane < stride)
            value = maximum ? fmax(value, other) : value + other;
    }
    return intel_sub_group_shuffle(value, 0);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void attention_probabilities_register_2048(
    __global float* scores, __global float* stats,
    int seq, int width, int heads, int first, int causal, int saved)
{
    // Caller must use this entry point only for seq == 2048.
    const int row = get_group_id(0);
    const int query = row % seq;
    const int head = row / seq;
    const int lane = get_local_id(0);
    const int count = causal ? query + 1 : seq;
    const int limit = causal == 2 ? min(seq, (query / 64 + 1) * 64) : seq;
    const int stat = (first + head) * seq + query;
    const int rowBase = row * seq;
    const float scale = rsqrt((float)(width / heads));
    __local float partials[64];
    float raw[32];

    // Read each valid score once.  Do not touch masked signaling NaNs.
    #pragma unroll
    for (int slot = 0; slot < 32; ++slot) {
        const int key = lane + 64 * slot;
        raw[slot] = key < count ? scores[rowBase + key] : 0.0f;
    }

    float maximum = -INFINITY;
    if (!saved) {
        #pragma unroll
        for (int slot = 0; slot < 32; ++slot) {
            const int key = lane + 64 * slot;
            if (key < count)
                maximum = fmax(maximum, raw[slot] * scale);
        }
        partials[lane] = maximum;
        barrier(CLK_LOCAL_MEM_FENCE);
        maximum = attention_register_2048_reduce64(partials, 1);
    } else {
        maximum = stats[2 * stat];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0.0f;
    #pragma unroll
    for (int slot = 0; slot < 32; ++slot) {
        const int key = lane + 64 * slot;
        if (key < limit) {
            const float value = key < count ? exp(raw[slot] * scale - maximum) : 0.0f;
            raw[slot] = value;
            sum += value;
        }
    }

    float inverse;
    if (!saved) {
        partials[lane] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        const float total = attention_register_2048_reduce64(partials, 0);
        inverse = 1.0f / total;
        if (lane == 0) {
            stats[2 * stat] = maximum;
            stats[2 * stat + 1] = inverse;
        }
    } else {
        inverse = stats[2 * stat + 1];
    }

    #pragma unroll
    for (int slot = 0; slot < 32; ++slot) {
        const int key = lane + 64 * slot;
        if (key < limit)
            scores[rowBase + key] = raw[slot] * inverse;
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void attention_derivatives_register_2048(
    __global const float* probabilities, __global float* derivatives,
    int seq, int width, int heads, int causal)
{
    const int row = get_group_id(0);
    const int query = row % seq;
    const int lane = get_local_id(0);
    const int count = causal ? query + 1 : seq;
    const int limit = causal == 2 ? min(seq, (query / 64 + 1) * 64) : seq;
    const int rowBase = row * seq;
    __local float partials[64];
    float cachedProbability[32];
    float sum = 0.0f;

    // Retain P, but reload dP after the reduction.  This bounds private
    // storage to 32 floats/work-item instead of two 32-float arrays.
    #pragma unroll
    for (int slot = 0; slot < 32; ++slot) {
        const int key = lane + 64 * slot;
        cachedProbability[slot] = 0.0f;
        if (key < count) {
            const float probability = probabilities[rowBase + key];
            cachedProbability[slot] = probability;
            sum = fma(probability, derivatives[rowBase + key], sum);
        }
    }

    partials[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    const float delta = attention_register_2048_reduce64(partials, 0);
    const float scale = rsqrt((float)(width / heads));
    #pragma unroll
    for (int slot = 0; slot < 32; ++slot) {
        const int key = lane + 64 * slot;
        if (key < limit) {
            const int index = rowBase + key;
            derivatives[index] = key < count
                ? cachedProbability[slot] * (derivatives[index] - delta) * scale
                : 0.0f;
        }
    }
}

// Fixed-shape control: isolate runtime indexing and scale calculation from
// the register-reuse variants below.  Probe-only until whole-step acceptance.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void attention_derivatives_fixed_2048(
    __global const float* probabilities, __global float* derivatives,
    int seq, int width, int heads, int causal)
{
    // Dispatch only at T2048/D32.  Keep every floating-point operation in the
    // reference order while removing runtime integer division and scaling.
    const int row = get_group_id(0);
    const int query = row & 2047;
    const int lane = get_local_id(0);
    const int count = causal ? query + 1 : 2048;
    const int limit = causal == 2 ? ((query >> 6) + 1) << 6 : 2048;
    const int base = row << 11;
    __local float partials[64];
    float sum = 0.0f;
    for (int key = lane; key < count; key += 64)
        sum = fma(probabilities[base + key], derivatives[base + key], sum);
    partials[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride != 0; stride /= 2) {
        if (lane < stride)
            partials[lane] += partials[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float delta = partials[0];
    const float scale = rsqrt(32.0f);
    for (int key = lane; key < limit; key += 64) {
        const int index = base + key;
        derivatives[index] = key < count
            ? probabilities[index] * (derivatives[index] - delta) * scale : 0.0f;
    }
}

// Keep only the first P/dP pair from each reference modulo-64 lane stream.
// This preserves the reference FMA and reduction order without retaining a
// full T2048 row in registers or SLM.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void attention_derivatives_prefix_2048(
    __global const float* probabilities, __global float* derivatives,
    int seq, int width, int heads, int causal)
{
    const int row = get_group_id(0);
    const int query = row % seq;
    const int lane = get_local_id(0);
    const int count = causal ? query + 1 : seq;
    const int limit = causal == 2 ? min(seq, (query / 64 + 1) * 64) : seq;
    const int base = row * seq;
    const int first = base + lane;
    __local float partials[64];
    float firstProbability = 0.0f;
    float firstDerivative = 0.0f;
    float sum = 0.0f;

    if (lane < count) {
        firstProbability = probabilities[first];
        firstDerivative = derivatives[first];
        sum = fma(firstProbability, firstDerivative, sum);
    }
    for (int key = lane + 64; key < count; key += 64)
        sum = fma(probabilities[base + key], derivatives[base + key], sum);

    partials[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride != 0; stride /= 2) {
        if (lane < stride)
            partials[lane] += partials[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float delta = partials[0];
    const float scale = rsqrt((float)(width / heads));
    if (lane < limit)
        derivatives[first] = lane < count
            ? firstProbability * (firstDerivative - delta) * scale : 0.0f;
    for (int key = lane + 64; key < limit; key += 64) {
        const int index = base + key;
        derivatives[index] = key < count
            ? probabilities[index] * (derivatives[index] - delta) * scale : 0.0f;
    }
}

// A bounded four-pair variant probes the tradeoff between global rereads and
// private registers without the full-row cache used by the older candidate.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void attention_derivatives_prefix4_2048(
    __global const float* probabilities, __global float* derivatives,
    int seq, int width, int heads, int causal)
{
    const int row = get_group_id(0);
    const int query = row % seq;
    const int lane = get_local_id(0);
    const int count = causal ? query + 1 : seq;
    const int limit = causal == 2 ? min(seq, (query / 64 + 1) * 64) : seq;
    const int base = row * seq;
    __local float partials[64];
    float cachedProbability[4], cachedDerivative[4];
    float sum = 0.0f;

    #pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        const int key = lane + slot * 64;
        cachedProbability[slot] = 0.0f;
        cachedDerivative[slot] = 0.0f;
        if (key < count) {
            cachedProbability[slot] = probabilities[base + key];
            cachedDerivative[slot] = derivatives[base + key];
            sum = fma(cachedProbability[slot], cachedDerivative[slot], sum);
        }
    }
    for (int key = lane + 256; key < count; key += 64)
        sum = fma(probabilities[base + key], derivatives[base + key], sum);

    partials[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride != 0; stride /= 2) {
        if (lane < stride)
            partials[lane] += partials[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float delta = partials[0];
    const float scale = rsqrt((float)(width / heads));
    #pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        const int key = lane + slot * 64;
        if (key < limit)
            derivatives[base + key] = key < count
                ? cachedProbability[slot] * (cachedDerivative[slot] - delta) * scale : 0.0f;
    }
    for (int key = lane + 256; key < limit; key += 64) {
        const int index = base + key;
        derivatives[index] = key < count
            ? probabilities[index] * (derivatives[index] - delta) * scale : 0.0f;
    }
}

// Keep the four-pair register reuse above and replace only the workgroup
// reduction. The helper reproduces the reference's 32/16/8/4/2/1 add tree,
// while the subgroup shuffles need only the first workgroup barrier.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void attention_derivatives_prefix4_sg16_2048(
    __global const float* probabilities, __global float* derivatives,
    int seq, int width, int heads, int causal)
{
    const int row = get_group_id(0);
    const int query = row % seq;
    const int lane = get_local_id(0);
    const int count = causal ? query + 1 : seq;
    const int limit = causal == 2 ? min(seq, (query / 64 + 1) * 64) : seq;
    const int base = row * seq;
    __local float partials[64];
    float cachedProbability[4], cachedDerivative[4];
    float sum = 0.0f;

    #pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        const int key = lane + slot * 64;
        cachedProbability[slot] = 0.0f;
        cachedDerivative[slot] = 0.0f;
        if (key < count) {
            cachedProbability[slot] = probabilities[base + key];
            cachedDerivative[slot] = derivatives[base + key];
            sum = fma(cachedProbability[slot], cachedDerivative[slot], sum);
        }
    }
    for (int key = lane + 256; key < count; key += 64)
        sum = fma(probabilities[base + key], derivatives[base + key], sum);

    partials[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    const float delta = attention_register_2048_reduce64(partials, 0);
    const float scale = rsqrt((float)(width / heads));
    #pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        const int key = lane + slot * 64;
        if (key < limit)
            derivatives[base + key] = key < count
                ? cachedProbability[slot] * (cachedDerivative[slot] - delta) * scale : 0.0f;
    }
    for (int key = lane + 256; key < limit; key += 64) {
        const int index = base + key;
        derivatives[index] = key < count
            ? probabilities[index] * (derivatives[index] - delta) * scale : 0.0f;
    }
}

// Probe the next bounded register budget: eight P/dP pairs per lane. Keep the
// modulo-64 FMA stream and reference workgroup reduction unchanged.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void attention_derivatives_prefix8_2048(
    __global const float* probabilities, __global float* derivatives,
    int seq, int width, int heads, int causal)
{
    const int row = get_group_id(0);
    const int query = row % seq;
    const int lane = get_local_id(0);
    const int count = causal ? query + 1 : seq;
    const int limit = causal == 2 ? min(seq, (query / 64 + 1) * 64) : seq;
    const int base = row * seq;
    __local float partials[64];
    float cachedProbability[8], cachedDerivative[8];
    float sum = 0.0f;

    #pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        const int key = lane + slot * 64;
        cachedProbability[slot] = 0.0f;
        cachedDerivative[slot] = 0.0f;
        if (key < count) {
            cachedProbability[slot] = probabilities[base + key];
            cachedDerivative[slot] = derivatives[base + key];
            sum = fma(cachedProbability[slot], cachedDerivative[slot], sum);
        }
    }
    for (int key = lane + 512; key < count; key += 64)
        sum = fma(probabilities[base + key], derivatives[base + key], sum);

    partials[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride != 0; stride /= 2) {
        if (lane < stride)
            partials[lane] += partials[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float delta = partials[0];
    const float scale = rsqrt((float)(width / heads));
    #pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        const int key = lane + slot * 64;
        if (key < limit)
            derivatives[base + key] = key < count
                ? cachedProbability[slot] * (cachedDerivative[slot] - delta) * scale : 0.0f;
    }
    for (int key = lane + 512; key < limit; key += 64) {
        const int index = base + key;
        derivatives[index] = key < count
            ? probabilities[index] * (derivatives[index] - delta) * scale : 0.0f;
    }
}

// Probe a larger register budget without changing the FMA stream or
// workgroup reduction tree. The benchmark must check occupancy and spill cost.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void attention_derivatives_prefix16_2048(
    __global const float* probabilities, __global float* derivatives,
    int seq, int width, int heads, int causal)
{
    const int row = get_group_id(0);
    const int query = row % seq;
    const int lane = get_local_id(0);
    const int count = causal ? query + 1 : seq;
    const int limit = causal == 2 ? min(seq, (query / 64 + 1) * 64) : seq;
    const int base = row * seq;
    __local float partials[64];
    float cachedProbability[16], cachedDerivative[16];
    float sum = 0.0f;

    #pragma unroll
    for (int slot = 0; slot < 16; ++slot) {
        const int key = lane + slot * 64;
        cachedProbability[slot] = 0.0f;
        cachedDerivative[slot] = 0.0f;
        if (key < count) {
            cachedProbability[slot] = probabilities[base + key];
            cachedDerivative[slot] = derivatives[base + key];
            sum = fma(cachedProbability[slot], cachedDerivative[slot], sum);
        }
    }
    for (int key = lane + 1024; key < count; key += 64)
        sum = fma(probabilities[base + key], derivatives[base + key], sum);

    partials[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride != 0; stride /= 2) {
        if (lane < stride)
            partials[lane] += partials[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float delta = partials[0];
    const float scale = rsqrt((float)(width / heads));
    #pragma unroll
    for (int slot = 0; slot < 16; ++slot) {
        const int key = lane + slot * 64;
        if (key < limit)
            derivatives[base + key] = key < count
                ? cachedProbability[slot] * (cachedDerivative[slot] - delta) * scale : 0.0f;
    }
    for (int key = lane + 1024; key < limit; key += 64) {
        const int index = base + key;
        derivatives[index] = key < count
            ? probabilities[index] * (derivatives[index] - delta) * scale : 0.0f;
    }
}
#endif
