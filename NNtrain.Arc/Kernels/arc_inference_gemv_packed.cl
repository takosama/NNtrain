// A single generated token is a matrix-vector product. Read the resident
// weight payload directly so BFP8/BF16 weights never become a full FP32 copy.
#if defined(ARC_XMX) && ARC_SG == 16
float round_bf16(float x);

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 8, 1)))
__kernel void arc_inference_gemv_packed(
    __global const float* input,
    __global const uchar* packedWeight,
    __global const float* scales,
    __global const float* bias,
    __global float* output,
    int n, int k, int weightFormat, int blockSize, int hasBias, int relu)
{
    int column = get_local_id(0);
    int row = get_global_id(1);
    if (row >= n) return;

    int base = row * k;
    float partial = 0.0f;
    for (int inner = column; inner < k; inner += 16)
    {
        int offset = base + inner;
        float weight;
        if (weightFormat == 2)
        {
            float decoded = (float)((__global const char*)packedWeight)[offset]
                * scales[offset / blockSize];
            // ArcUploadValues(matrixOperand:true) uses this same rounding.
            weight = round_bf16(decoded);
        }
        else if (weightFormat == 1)
            weight = as_float((uint)((__global const ushort*)packedWeight)[offset] << 16);
        else
            weight = ((__global const float*)packedWeight)[offset];

        partial = fma(input[inner], weight, partial);
    }

    float sum = sub_group_reduce_add(partial);
    if (column == 0)
    {
        if (hasBias) sum += bias[row];
        if (relu) sum = fmax(0.0f, sum);
        output[row] = sum;
    }
}
#endif
