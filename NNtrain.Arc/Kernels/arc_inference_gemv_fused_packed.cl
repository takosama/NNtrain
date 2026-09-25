// One-token inference reads packed operands and writes either FP32 partials or
// the final BF16 activation. This can remove three decode/publish launches.
#if defined(ARC_XMX) && ARC_SG == 16
float round_bf16(float x);
ushort xmx_bf16(float x);

inline float arc_inference_fused_gemv_read(
    __global const uchar* data, __global const float* scales,
    int format, int blockSize, int offset, int matrixOperand)
{
    if (format == 2)
    {
        float decoded = (float)((__global const char*)data)[offset]
            * scales[offset / blockSize];
        return matrixOperand ? round_bf16(decoded) : decoded;
    }
    if (format == 1)
        return as_float((uint)((__global const ushort*)data)[offset] << 16);
    return ((__global const float*)data)[offset];
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 8, 1)))
__kernel void arc_inference_gemv_fused_packed(
    __global const uchar* input, __global const float* inputScales,
    __global const uchar* weight, __global const float* weightScales,
    __global const uchar* bias, __global const float* biasScales,
    __global uchar* output,
    int n, int k,
    int inputFormat, int inputBlockSize,
    int weightFormat, int weightBlockSize,
    int biasFormat, int biasBlockSize,
    int hasBias, int relu, int directBf16)
{
    int lane = get_local_id(0);
    int row = get_global_id(1);
    if (row >= n) return;

    float partial = 0.0f;
    for (int inner = lane; inner < k; inner += 16)
    {
        float x = arc_inference_fused_gemv_read(
            input, inputScales, inputFormat, inputBlockSize, inner, 1);
        float w = arc_inference_fused_gemv_read(
            weight, weightScales, weightFormat, weightBlockSize,
            row * k + inner, 1);
        partial = fma(x, w, partial);
    }

    float sum = sub_group_reduce_add(partial);
    if (lane == 0)
    {
        if (hasBias)
        {
            // Bias is an ordinary storage value, not a BF16 matrix operand.
            volatile float decodedBias = arc_inference_fused_gemv_read(
                bias, biasScales, biasFormat, biasBlockSize, row, 0);
            sum += decodedBias;
        }
        if (relu) sum = fmax(0.0f, sum);
        if (directBf16)
            ((__global ushort*)output)[row] = xmx_bf16(sum);
        else
            ((__global float*)output)[row] = sum;
    }
}
#endif
