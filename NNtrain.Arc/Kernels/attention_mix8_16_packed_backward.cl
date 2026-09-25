// mix8_16 T2048/D32 backward experiment. After the FP32 dP reduction, each
// score slot contains two physical BF16 values: low=P, high=dS. This retains
// the FP32 softmax/dP reduction while halving the score bytes read by dK/dV.
#if defined(ARC_XMX) && ARC_SG == 16 && defined(ARC_SLM_BLOCK_IO)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

inline ushort attention_mix8_16_bf16(float value) {
    uint bits = as_uint(value);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    else if ((bits & 0x007fffffu) != 0u)
        bits |= 0x00400000u;
    return (ushort)(bits >> 16);
}

// Keep the established prefix8 modulo-64 FP32 FMA/reduction order. The final
// publication alone rounds P and dS to BF16 and packs them into P's buffer.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void attention_derivatives_prefix8_pack_bf16_2048(
    __global uint* packed, __global const float* derivatives,
    int seq, int width, int heads, int causal)
{
    __global const float* probabilities = (__global const float*)packed;
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
        if (key < limit) {
            const float probability = key < count ? cachedProbability[slot] : 0.0f;
            const float derivative = key < count
                ? probability * (cachedDerivative[slot] - delta) * scale : 0.0f;
            packed[base + key] = (uint)attention_mix8_16_bf16(probability)
                | ((uint)attention_mix8_16_bf16(derivative) << 16);
        }
    }
    for (int key = lane + 512; key < limit; key += 64) {
        const int index = base + key;
        const float probability = key < count ? probabilities[index] : 0.0f;
        const float derivative = key < count
            ? probability * (derivatives[index] - delta) * scale : 0.0f;
        packed[index] = (uint)attention_mix8_16_bf16(probability)
            | ((uint)attention_mix8_16_bf16(derivative) << 16);
    }
}

// The accepted BM64/K32 dQ tile with only the A loader changed. The packed
// high half is dS; all multiply-accumulate operations remain FP32 and ordered.
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void attention_dq_packed_bf16_2048(
    __global const uint* packed, __global const float* qkv, __global float* dx,
    int seq, int width, int heads, int first, int causalMode)
{
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int tid = ly * 16 + lx, g = get_group_id(2), h = first + g;
    const int rb = get_group_id(1) * 64;
    const int ap = g * seq * seq;
    const int bp = (h / heads) * seq * 3 * width + (h % heads) * 32 + width;
    const int cp = (h / heads) * seq * 3 * width + (h % heads) * 32;
    const int end = causalMode == 2 ? rb + 64 : seq;
    __local float at[64][33], bt[32][33];
    float2 sums[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = rb + ly + 16 * r;
        sums[r] = (float2)(dx[cp + row * 3 * width + lx],
            dx[cp + row * 3 * width + lx + 16]);
    }
    for (int base = 0; base < end; base += 32) {
        const int r0 = tid / 8, k0 = (tid % 8) * 4;
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            const int row = rb + r0 + 32 * part;
            const uint4 bits = vload4(0, packed + ap + row * seq + base + k0);
            const float4 av = as_float4(bits & (uint4)(0xffff0000u));
            vstore4(av, 0, &at[r0 + 32 * part][k0]);
        }
        const float4 bv = vload4(0, qkv + bp + (base + r0) * 3 * width + k0);
        vstore4(bv, 0, &bt[r0][k0]);
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int inner = 0; inner < 32; ++inner) {
            const float2 b = (float2)(bt[inner][lx], bt[inner][lx + 16]);
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                sums[r] = fma((float2)(at[ly + 16 * r][inner]), b, sums[r]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = rb + ly + 16 * r;
        dx[cp + row * 3 * width + lx] = sums[r].s0;
        dx[cp + row * 3 * width + lx + 16] = sums[r].s1;
    }
}

// The accepted causal K32/Q32 SLM block-I/O dK/dV tile. Packed P/dS is one
// global word per score instead of two; the ascending-query FP32 FMAs remain.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void attention_dkv_packed_bf16_2048_causal(
    __global const float* qkv, __global const float* dy,
    __global const uint* packed, __global float* dx,
    int seq, int width, int heads, int first)
{
    const int lx = get_local_id(0), ly = get_local_id(1), tid = ly * 16 + lx;
    const int g = get_group_id(2), h = first + g, kb = get_group_id(1) * 32;
    const int qb = (h / heads) * seq * 3 * width + (h % heads) * 32;
    const int yb = (h / heads) * seq * width + (h % heads) * 32;
    __local float2 pairs[32][33];
    __local uint qy[32][64];
    float4 sums[2];
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int off = qb + (kb + ly + 16 * r) * 3 * width + width + lx;
        sums[r] = (float4)(dx[off], dx[off + 16],
            dx[off + width], dx[off + width + 16]);
    }
    for (int base = kb; base < seq; base += 32) {
        #pragma unroll
        for (int i = tid; i < 1024; i += 256) {
            const int query = i / 32, key = i % 32;
            const int score = (g * seq + base + query) * seq + kb + key;
            const uint bits = packed[score];
            pairs[key][query] = (float2)(
                as_float((bits & 0xffffu) << 16),
                as_float(bits & 0xffff0000u));
            const int qi = base + query;
            qy[query][key] = as_uint(qkv[qb + qi * 3 * width + key]);
            qy[query][key + 32] = as_uint(dy[yb + qi * width + key]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int inner = 0; inner < 32; ++inner) {
            const float4 xy = as_float4(intel_sub_group_block_read4(&qy[inner][0]));
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                const float2 weights = pairs[ly + 16 * r][inner];
                sums[r] = fma(weights.yyxx, xy, sums[r]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int off = qb + (kb + ly + 16 * r) * 3 * width + width + lx;
        dx[off] = sums[r].s0;
        dx[off + 16] = sums[r].s1;
        dx[off + width] = sums[r].s2;
        dx[off + width + 16] = sums[r].s3;
    }
}
#endif
