#ifdef ARC_OPTIMIZATION_PROBES
// Diagnostics only. Neither kernel is a replacement for attention_derivatives.
// The streaming kernel retains its causal reads, second sweep, masked writes,
// workgroup size and reduction barriers, but minimizes FP32 derivative math.
__attribute__((reqd_work_group_size(64,1,1)))
__kernel void attention_derivative_streaming_diagnostic_2048(
    __global const float* p, __global float* dp, __global uint* checksums,
    int seq, int width, int heads, int causal)
{
    int row = get_group_id(0), q = row % seq, lane = get_local_id(0);
    int count = causal ? q + 1 : seq;
    int limit = causal == 2 ? min(seq, (q / 64 + 1) * 64) : seq;
    __local uint partial[64];
    uint checksum = 0;
    for (int key = lane; key < count; key += 64) {
        int i = row * seq + key;
        checksum += as_uint(p[i]) ^ as_uint(dp[i]);
    }
    partial[lane] = checksum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) checksums[row] = partial[0];
    for (int key = lane; key < limit; key += 64) {
        int i = row * seq + key;
        dp[i] = key < count ? p[i] + dp[i] : 0.0f;
    }
}

// Runtime-loaded 256-element inputs remain device resident and cannot be
// constant-folded. Retain the FP32 FMA/reduction tree and second arithmetic
// sweep, but replace the large P/dP traffic with cache-hot table reads and
// one scalar sink per lane. This is an arithmetic/latency diagnostic bound.
__attribute__((reqd_work_group_size(64,1,1)))
__kernel void attention_derivative_reduction_diagnostic_2048(
    __global const float* smallP, __global const float* smallDp,
    __global float* sink, int seq, int width, int heads, int causal, int seed)
{
    int row = get_group_id(0), q = row % seq, lane = get_local_id(0);
    int count = causal ? q + 1 : seq;
    int limit = causal == 2 ? min(seq, (q / 64 + 1) * 64) : seq;
    int offset = (row * 31 + seed) & 255;
    __local float partial[64];
    float sum = 0.0f;
    for (int key = lane; key < count; key += 64) {
        int i = (offset + key) & 255;
        sum = fma(smallP[i], smallDp[i], sum);
    }
    partial[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float delta = partial[0], scale = rsqrt((float)(width / heads));
    float output = 0.0f;
    for (int key = lane; key < limit; key += 64) {
        if (key < count) {
            int i = (offset + key) & 255;
            output += smallP[i] * (smallDp[i] - delta) * scale;
        }
    }
    sink[row * 64 + lane] = output;
}
#endif
