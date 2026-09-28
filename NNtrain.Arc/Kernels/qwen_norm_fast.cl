// One work-group per row. Decode uses a single row, so both the RMS
// reduction and output loop must use its lanes instead of one work-item.
// The host enables this path only when no gradients are recorded.
__kernel void qwen_rmsnorm_fast(
    __global const float* x, __global const float* weight,
    __global float* y, __global float* inv_rms,
    int rows, int width, float eps) {
    int row = get_group_id(0), lane = get_local_id(0);
    if (row >= rows) return;
    int offset = row * width;
    float square = 0.0f;
    for (int column = lane; column < width; column += 64) {
        float value = x[offset + column];
        square = fma(value, value, square);
    }
    __local float sums[64];
    sums[lane] = square;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = rsqrt(sums[0] / (float)width + eps);
    if (lane == 0) inv_rms[row] = inv;
    for (int column = lane; column < width; column += 64)
        y[offset + column] = x[offset + column] * inv * weight[column];
}
