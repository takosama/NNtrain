// Full sequence embedding: one token-ID upload, one kernel, original packed weights.
#define Q35T_EMBED(NAME, BYTES, VALUE) \
__kernel void NAME(__global const uchar* weight, __global const int* ids, \
    __global float* result, int rows, int width) { \
    int i = get_global_id(0); if (i >= rows * width) return; \
    int row = i / width, column = i % width; \
    __global const uchar* block = weight + ((size_t)ids[row] * (width >> 8) + (column >> 8)) * BYTES; \
    result[i] = VALUE; \
}
Q35T_EMBED(q35t_embedding_q4_k, 144, qwen_q4_k_value(block, column & 255))
Q35T_EMBED(q35t_embedding_q6_k, 210, qwen_q6_k_value(block, column & 255))
#define Q35T_EMBED_SCALE q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8))
#define Q35T_EMBED_MIN q35l_half_to_float((ushort)block[2] | ((ushort)block[3] << 8))
Q35T_EMBED(q35t_embedding_q5_k, 176, q35l_q5_value(block, column & 255, Q35T_EMBED_SCALE, Q35T_EMBED_MIN))
Q35T_EMBED(q35t_embedding_iq2_s, 82, q35l_iq2_s_value(block, column & 255, Q35T_EMBED_SCALE, 0.0f))
Q35T_EMBED(q35t_embedding_iq3_s, 110, q35l_iq3_s_value(block, column & 255, Q35T_EMBED_SCALE, 0.0f))
#undef Q35T_EMBED
#undef Q35T_EMBED_SCALE
#undef Q35T_EMBED_MIN
