// Inference lookup reads only the requested rows from the resident tables.
// Values are decoded as ordinary embedding values, without matrix BF16 rounding.
inline float arc_inference_embedding_value(
    __global const uchar* data, __global const float* scales,
    int format, int blockSize, int offset)
{
    if (format == 2)
        return (float)((__global const char*)data)[offset] * scales[offset / blockSize];
    if (format == 1)
        return as_float((uint)((__global const ushort*)data)[offset] << 16);
    return ((__global const float*)data)[offset];
}

__kernel void arc_inference_embedding_packed(
    __global const uchar* tokenTable, __global const float* tokenScales,
    __global const uchar* positionTable, __global const float* positionScales,
    __global const int* tokenIds, __global float* output,
    int tokens, int width, int positionOffset,
    int tokenFormat, int tokenBlockSize,
    int positionFormat, int positionBlockSize)
{
    int index = get_global_id(0);
    if (index >= tokens * width) return;
    int row = index / width;
    int column = index - row * width;
    int tokenIndex = tokenIds[row] * width + column;
    int positionIndex = (positionOffset + row) * width + column;
    volatile float token = arc_inference_embedding_value(
        tokenTable, tokenScales, tokenFormat, tokenBlockSize, tokenIndex);
    volatile float position = arc_inference_embedding_value(
        positionTable, positionScales, positionFormat, positionBlockSize, positionIndex);
    output[index] = token + position;
}
