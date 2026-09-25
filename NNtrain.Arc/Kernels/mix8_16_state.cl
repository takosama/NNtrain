// Physical BF16 optimizer state for mix8_16. FP32 is used only within each
// element's update arithmetic; all retained state occupies two bytes/value.
inline ushort mix8_16_pack(float value)
{
    uint bits = as_uint(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u)
        return (ushort)((bits >> 16) | 0x0040u);
    return (ushort)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

inline float mix8_16_unpack(ushort bits)
{
    return as_float(((uint)bits) << 16);
}

__kernel void pack_bf16_state(__global const float* source, __global ushort* target, int n)
{
    int i = get_global_id(0);
    if (i < n) target[i] = mix8_16_pack(source[i]);
}

__kernel void unpack_bf16_state(__global const ushort* source, __global float* target, int n)
{
    int i = get_global_id(0);
    if (i < n) target[i] = mix8_16_unpack(source[i]);
}

__kernel void add_bf16_state_to_float(__global float* target, __global const ushort* source, int n)
{
    int i = get_global_id(0);
    if (i < n) target[i] += mix8_16_unpack(source[i]);
}

__kernel void add_float_to_bf16_state(
    __global ushort* packed, __global const float* delta, int n)
{
    int i = get_global_id(0);
    if (i < n)
        packed[i] = mix8_16_pack(mix8_16_unpack(packed[i]) + delta[i]);
}

__kernel void add_bf16_to_bf16_state(
    __global ushort* target, __global const ushort* source, int n)
{
    int i = get_global_id(0);
    if (i < n)
        target[i] = mix8_16_pack(mix8_16_unpack(target[i]) + mix8_16_unpack(source[i]));
}

__kernel void resident_finite_bf16(__global const ushort* values, __global int* status, int n)
{
    int i = get_global_id(0);
    if (i < n && !isfinite(mix8_16_unpack(values[i]))) atomic_or(status, 1);
}

__kernel void reduce_sum_bf16(
    __global const ushort* values, __global float* out,
    int n, int squared, __local float* scratch)
{
    int i = get_global_id(0), lane = get_local_id(0);
    float value = i < n ? mix8_16_unpack(values[i]) : 0.0f;
    scratch[lane] = squared ? value * value : value;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = get_local_size(0) / 2; stride > 0; stride /= 2)
    {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) out[get_group_id(0)] = scratch[0];
}

__kernel void resident_clip_apply_bf16(
    __global ushort* gradient, __global const float* statistics, int n)
{
    int i = get_global_id(0);
    if (i < n)
        gradient[i] = mix8_16_pack(mix8_16_unpack(gradient[i]) * statistics[1]);
}

__kernel void resident_bfp8_from_bf16(
    __global const ushort* master, __global char* payload,
    __global float* scales, __global int* status, int n, int block)
{
    int b = get_global_id(0), start = b * block;
    if (start >= n) return;
    int end = min(n, start + block);
    float magnitude = 0.0f;
    int valid = 1;
    for (int i = start; i < end; i++)
    {
        float value = mix8_16_unpack(master[i]);
        valid &= isfinite(value);
        magnitude = fmax(magnitude, fabs(value));
    }
    if (!valid)
    {
        atomic_or(status, 1);
        scales[b] = NAN;
        for (int i = start; i < end; i++) payload[i] = 0;
        return;
    }
    float scale = magnitude == 0.0f ? 1.0f : magnitude / 127.0f;
    if (!(scale > 0.0f) || !isfinite(scale))
    {
        atomic_or(status, 1);
        scales[b] = NAN;
        for (int i = start; i < end; i++) payload[i] = 0;
        return;
    }
    scales[b] = scale;
    for (int i = start; i < end; i++)
        payload[i] = convert_char_sat_rte(clamp(mix8_16_unpack(master[i]) / scale, -127.0f, 127.0f));
}

#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
// Four lanes quantize a block of 32, matching the existing coalesced BFP8
// publication geometry while reading two-byte master values directly.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(256,1,1)))
__kernel void resident_bfp8_from_bf16_quad4_32(
    __global const ushort* master, __global char* payload,
    __global float* scales, __global int* status, int n, int block)
{
    int b = get_global_id(0) / 4, lane = get_global_id(0) % 4;
    int start = b * 32;
    float cached[8], maximum = 0.0f;
    int valid = 1;
    #pragma unroll
    for (int j = 0; j < 8; j++)
    {
        int i = start + lane + j * 4;
        float value = i < n ? mix8_16_unpack(master[i]) : 0.0f;
        cached[j] = value;
        maximum = fmax(maximum, fabs(value));
        valid &= isfinite(value);
    }
    #pragma unroll
    for (int stride = 2; stride; stride /= 2)
    {
        int other = get_sub_group_local_id() ^ stride;
        maximum = fmax(maximum, intel_sub_group_shuffle(maximum, other));
        valid &= intel_sub_group_shuffle(valid, other);
    }
    float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
    valid &= scale > 0.0f && isfinite(scale);
    if (lane == 0 && start < n)
    {
        scales[b] = valid ? scale : NAN;
        if (!valid) atomic_or(status, 1);
    }
    #pragma unroll
    for (int j = 0; j < 8; j++)
    {
        int i = start + lane + j * 4;
        if (i < n)
            payload[i] = valid
                ? convert_char_sat_rte(clamp(cached[j] / scale, -127.0f, 127.0f))
                : 0;
    }
}
#endif

__kernel void adam_bf16_packed(
    __global const ushort* gradient, __global ushort* firstMoment,
    __global ushort* secondMoment, __global ushort* master,
    int n, float beta1, float beta2, float scale, float epsilon, float decay)
{
    int i = get_global_id(0);
    if (i >= n) return;
    float g = mix8_16_unpack(gradient[i]);
    ushort mBits = mix8_16_pack(
        beta1 * mix8_16_unpack(firstMoment[i]) + (1.0f - beta1) * g);
    ushort vBits = mix8_16_pack(
        beta2 * mix8_16_unpack(secondMoment[i]) + (1.0f - beta2) * g * g);
    firstMoment[i] = mBits;
    secondMoment[i] = vBits;
    float m = mix8_16_unpack(mBits);
    float v = mix8_16_unpack(vBits);
    float w = mix8_16_unpack(master[i]);
    master[i] = mix8_16_pack(w * decay - scale * m / (sqrt(v) + epsilon));
}

__kernel void moments_bf16_packed(
    __global const ushort* gradient, __global ushort* fast,
    __global ushort* slow, __global float* fastHat, __global float* slowHat,
    int n, float betaFast, float betaSlow, float fastCorrection,
    float slowCorrection, int nesterov)
{
    int i = get_global_id(0);
    if (i >= n) return;
    float g = mix8_16_unpack(gradient[i]);
    ushort fBits = mix8_16_pack(
        betaFast * mix8_16_unpack(fast[i]) + (1.0f - betaFast) * g);
    ushort sBits = mix8_16_pack(
        betaSlow * mix8_16_unpack(slow[i]) + (1.0f - betaSlow) * g);
    fast[i] = fBits;
    slow[i] = sBits;
    float f = mix8_16_unpack(fBits);
    float s = mix8_16_unpack(sBits);
    fastHat[i] = mix8_16_unpack(mix8_16_pack(f / fastCorrection));
    slowHat[i] = mix8_16_unpack(mix8_16_pack(s / slowCorrection));
    if (nesterov)
    {
        ushort directionBits = mix8_16_pack(betaFast * f + (1.0f - betaFast) * g);
        float direction = mix8_16_unpack(directionBits);
        fastHat[i] = direction;
        slowHat[i] = direction;
        slow[i] = directionBits;
    }
}

__kernel void muon_weight_update_bf16_packed(
    __global ushort* master, __global const float* direction,
    int n, float weightScale, float directionScale)
{
    int i = get_global_id(0);
    if (i >= n) return;
    float weight = mix8_16_unpack(master[i]);
    master[i] = mix8_16_pack(weightScale * weight + directionScale * direction[i]);
}
