// Experimental quantized transpose: one work item owns eight adjacent input
// components. Decode shared packed fields once for all eight components,
// then reuse that float8 across independent token rows. No dense weight cache.
// Each component retains the scalar transpose's output-channel FMA order and
// partial[row, split, input] layout; the existing split reducer is unchanged.
#if defined(ARC_QWEN35_TRAINING) && defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

inline float8 q35tx_iq2_decode8(__global const uchar* block, int octet) {
    int group = octet >> 2, part = octet & 3;
    int gridIndex = block[2 + octet] | (((block[66 + group] >> (part * 2)) & 3) << 8);
    float8 grid = convert_float8(as_uchar8(q35l_iq2s_grid[gridIndex]));
    int scale = (block[74 + group] >> ((part >> 1) * 4)) & 15;
    int signs = block[34 + octet];
    float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8));
    // Keep the reference multiplication order, including the final sign, to
    // retain subnormal-scale and signed-zero behavior exactly.
    float multiplier = (d * (0.5f + (float)scale)) * 0.25f;
    float8 sign = select((float8)(1.0f), (float8)(-1.0f),
        ((int8)(signs) & (int8)(1, 2, 4, 8, 16, 32, 64, 128)) != (int8)(0));
    return (multiplier * grid) * sign;
}

inline float8 q35tx_iq3_decode8(__global const uchar* block, int octet) {
    int group = octet >> 2, pair = (octet & 3) * 2;
    int high = block[66 + group];
    int grid0 = block[2 + octet * 2] | (((high >> pair) & 1) << 8);
    int grid1 = block[3 + octet * 2] | (((high >> (pair + 1)) & 1) << 8);
    float8 grid = convert_float8(as_uchar8((uint2)(q35l_iq3s_grid[grid0], q35l_iq3s_grid[grid1])));
    int scale = (block[106 + (group >> 1)] >> ((group & 1) * 4)) & 15;
    int signs = block[74 + octet];
    float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8));
    float multiplier = d * (float)(1 + 2 * scale);
    float8 sign = select((float8)(1.0f), (float8)(-1.0f),
        ((int8)(signs) & (int8)(1, 2, 4, 8, 16, 32, 64, 128)) != (int8)(0));
    return (multiplier * grid) * sign;
}

inline float8 q35tx_q4_decode8(__global const uchar* block, int octet) {
    int group = octet >> 2, within = (octet & 3) * 8;
    float d = qwen_half_to_float((ushort)block[0] | ((ushort)block[1] << 8));
    float dmin = qwen_half_to_float((ushort)block[2] | ((ushort)block[3] << 8));
    uchar scale, minimum;
    qwen_q4_scale_min(block + 4, group, &scale, &minimum);
    uchar8 packed = vload8(0, block + 16 + (group >> 1) * 32 + within);
    uchar8 quant = (group & 1) ? (packed >> (uchar8)(4)) : (packed & (uchar8)(15));
    // Match qwen_q4_k_value's multiply/multiply/subtract expression. An
    // explicit fma here could change the reference weight before the dot FMA.
    return (d * (float)scale) * convert_float8(quant) - dmin * (float)minimum;
}

inline float8 q35tx_q5_decode8(__global const uchar* block, int octet) {
    int group = octet >> 2, within = (octet & 3) * 8;
    float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8));
    float dmin = q35l_half_to_float((ushort)block[2] | ((ushort)block[3] << 8));
    float multiplier, minimum;
    q35l_q5_fast_scale_min(block, group, d, dmin, &multiplier, &minimum);
    uchar8 packed = vload8(0, block + 48 + (group >> 1) * 32 + within);
    uchar8 high = vload8(0, block + 16 + within);
    uchar8 quant = (group & 1) ? packed >> (uchar8)(4) : packed & (uchar8)(15);
    quant |= ((high >> (uchar8)(group)) & (uchar8)(1)) << (uchar8)(4);
    return multiplier * convert_float8(quant) - minimum;
}

#define Q35TX_VEC8_ROWS(NAME, DECODE, BLOCK_BYTES, ROWS) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(32, 1, 1))) \
__kernel void NAME(__global const float* dy, __global const uchar* weight, \
    __global float* partial, int rows, int input, int output, int splits, int tile) { \
    int i = get_global_id(0), columns = input >> 3, rowTiles = (rows + ROWS - 1) / ROWS; \
    if (i >= rowTiles * splits * columns) return; \
    int column = i % columns, split = (i / columns) % splits; \
    int row = (i / (columns * splits)) * ROWS, blocks = input >> 8; \
    int inputBlock = column >> 5, octet = column & 31; \
    float8 s0 = (float8)(0), s1 = (float8)(0), s2 = (float8)(0), s3 = (float8)(0); \
    float8 s4 = (float8)(0), s5 = (float8)(0), s6 = (float8)(0), s7 = (float8)(0); \
    float8 s8 = (float8)(0), s9 = (float8)(0), s10 = (float8)(0), s11 = (float8)(0); \
    float8 s12 = (float8)(0), s13 = (float8)(0), s14 = (float8)(0), s15 = (float8)(0); \
    int end = min(output, (split + 1) * tile); \
    for (int o = split * tile; o < end; ++o) { \
        __global const uchar* block = weight + ((size_t)o * blocks + inputBlock) * BLOCK_BYTES; \
        float8 w = DECODE(block, octet); \
        s0 = fma((float8)(dy[row * output + o]), w, s0); \
        if (row + 1 < rows) s1 = fma((float8)(dy[(row + 1) * output + o]), w, s1); \
        if (row + 2 < rows) s2 = fma((float8)(dy[(row + 2) * output + o]), w, s2); \
        if (row + 3 < rows) s3 = fma((float8)(dy[(row + 3) * output + o]), w, s3); \
        if (ROWS > 4 && row + 4 < rows) s4 = fma((float8)(dy[(row + 4) * output + o]), w, s4); \
        if (ROWS > 4 && row + 5 < rows) s5 = fma((float8)(dy[(row + 5) * output + o]), w, s5); \
        if (ROWS > 4 && row + 6 < rows) s6 = fma((float8)(dy[(row + 6) * output + o]), w, s6); \
        if (ROWS > 4 && row + 7 < rows) s7 = fma((float8)(dy[(row + 7) * output + o]), w, s7); \
        if (ROWS > 8 && row + 8 < rows) s8 = fma((float8)(dy[(row + 8) * output + o]), w, s8); \
        if (ROWS > 8 && row + 9 < rows) s9 = fma((float8)(dy[(row + 9) * output + o]), w, s9); \
        if (ROWS > 8 && row + 10 < rows) s10 = fma((float8)(dy[(row + 10) * output + o]), w, s10); \
        if (ROWS > 8 && row + 11 < rows) s11 = fma((float8)(dy[(row + 11) * output + o]), w, s11); \
        if (ROWS > 8 && row + 12 < rows) s12 = fma((float8)(dy[(row + 12) * output + o]), w, s12); \
        if (ROWS > 8 && row + 13 < rows) s13 = fma((float8)(dy[(row + 13) * output + o]), w, s13); \
        if (ROWS > 8 && row + 14 < rows) s14 = fma((float8)(dy[(row + 14) * output + o]), w, s14); \
        if (ROWS > 8 && row + 15 < rows) s15 = fma((float8)(dy[(row + 15) * output + o]), w, s15); \
    } \
    int offset = (row * splits + split) * input + column * 8, stride = splits * input; \
    vstore8(s0, 0, partial + offset); \
    if (row + 1 < rows) vstore8(s1, 0, partial + offset + stride); \
    if (row + 2 < rows) vstore8(s2, 0, partial + offset + 2 * stride); \
    if (row + 3 < rows) vstore8(s3, 0, partial + offset + 3 * stride); \
    if (ROWS > 4 && row + 4 < rows) vstore8(s4, 0, partial + offset + 4 * stride); \
    if (ROWS > 4 && row + 5 < rows) vstore8(s5, 0, partial + offset + 5 * stride); \
    if (ROWS > 4 && row + 6 < rows) vstore8(s6, 0, partial + offset + 6 * stride); \
    if (ROWS > 4 && row + 7 < rows) vstore8(s7, 0, partial + offset + 7 * stride); \
    if (ROWS > 8 && row + 8 < rows) vstore8(s8, 0, partial + offset + 8 * stride); \
    if (ROWS > 8 && row + 9 < rows) vstore8(s9, 0, partial + offset + 9 * stride); \
    if (ROWS > 8 && row + 10 < rows) vstore8(s10, 0, partial + offset + 10 * stride); \
    if (ROWS > 8 && row + 11 < rows) vstore8(s11, 0, partial + offset + 11 * stride); \
    if (ROWS > 8 && row + 12 < rows) vstore8(s12, 0, partial + offset + 12 * stride); \
    if (ROWS > 8 && row + 13 < rows) vstore8(s13, 0, partial + offset + 13 * stride); \
    if (ROWS > 8 && row + 14 < rows) vstore8(s14, 0, partial + offset + 14 * stride); \
    if (ROWS > 8 && row + 15 < rows) vstore8(s15, 0, partial + offset + 15 * stride); \
}

Q35TX_VEC8_ROWS(q35t_xpose_iq2_s_vec8_rows4, q35tx_iq2_decode8, 82, 4)
Q35TX_VEC8_ROWS(q35t_xpose_iq2_s_vec8_rows8, q35tx_iq2_decode8, 82, 8)
Q35TX_VEC8_ROWS(q35t_xpose_iq2_s_vec8_rows16, q35tx_iq2_decode8, 82, 16)
Q35TX_VEC8_ROWS(q35t_xpose_iq3_s_vec8_rows4, q35tx_iq3_decode8, 110, 4)
Q35TX_VEC8_ROWS(q35t_xpose_iq3_s_vec8_rows8, q35tx_iq3_decode8, 110, 8)
Q35TX_VEC8_ROWS(q35t_xpose_iq3_s_vec8_rows16, q35tx_iq3_decode8, 110, 16)
Q35TX_VEC8_ROWS(q35t_xpose_q4_k_vec8_rows4, q35tx_q4_decode8, 144, 4)
Q35TX_VEC8_ROWS(q35t_xpose_q4_k_vec8_rows8, q35tx_q4_decode8, 144, 8)
Q35TX_VEC8_ROWS(q35t_xpose_q4_k_vec8_rows16, q35tx_q4_decode8, 144, 16)
Q35TX_VEC8_ROWS(q35t_xpose_q5_k_vec8_rows4, q35tx_q5_decode8, 176, 4)
Q35TX_VEC8_ROWS(q35t_xpose_q5_k_vec8_rows8, q35tx_q5_decode8, 176, 8)
Q35TX_VEC8_ROWS(q35t_xpose_q5_k_vec8_rows16, q35tx_q5_decode8, 176, 16)
#undef Q35TX_VEC8_ROWS
#endif
