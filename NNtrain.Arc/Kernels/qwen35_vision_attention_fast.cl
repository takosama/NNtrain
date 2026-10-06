// Bidirectional vision attention with shared 16x16 operand tiles.
// Keep Qwen's token/3/width QKV storage and query/head/key score storage.
// The dot products retain the scalar kernels' increasing feature/key order.

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void q35v_attention_scores_tiled(__global const float * qkv,
    __global float * scores, int rows, int heads, int head_width) {
    int lx = get_local_id(0), ly = get_local_id(1);
    int query_tiles = (rows + 15) / 16;
    int head = get_group_id(1) / query_tiles;
    int query_base = (get_group_id(1) % query_tiles) * 16;
    int key_base = get_group_id(0) * 16;
    int query = query_base + ly, key = key_base + lx;
    int width = heads * head_width;
    __local float query_values[16][16];
    // The context read transposes the key tile across subgroup lanes. Padding
    // removes the otherwise identical local-memory bank for every lane.
    __local float key_values[16][17];
    float sum = 0.0f;
    for (int base = 0; base < head_width; base += 16) {
        int feature = base + lx;
        query_values[ly][lx] = query < rows && feature < head_width
            ? qkv[(size_t)query * (3 * width) + head * head_width + feature] : 0.0f;
        int key_row = key_base + ly;
        key_values[ly][lx] = key_row < rows && feature < head_width
            ? qkv[(size_t)key_row * (3 * width) + width + head * head_width + feature] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        int count = min(16, head_width - base);
        for (int feature_inner = 0; feature_inner < count; ++feature_inner)
            sum = fma(query_values[ly][feature_inner], key_values[lx][feature_inner], sum);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (query < rows && key < rows)
        scores[((size_t)query * heads + head) * rows + key] = sum * rsqrt((float)head_width);
}

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void q35v_attention_context_tiled(__global const float * qkv,
    __global const float * scores, __global float * output,
    int rows, int heads, int head_width) {
    int lx = get_local_id(0), ly = get_local_id(1);
    int query_tiles = (rows + 15) / 16;
    int head = get_group_id(1) / query_tiles;
    int query = (get_group_id(1) % query_tiles) * 16 + ly;
    int feature = get_group_id(0) * 16 + lx;
    int width = heads * head_width;
    __local float probabilities[16][16];
    __local float values[16][16];
    float sum = 0.0f;
    for (int key_base = 0; key_base < rows; key_base += 16) {
        int score_key = key_base + lx;
        probabilities[ly][lx] = query < rows && score_key < rows
            ? scores[((size_t)query * heads + head) * rows + score_key] : 0.0f;
        int value_key = key_base + ly;
        values[ly][lx] = value_key < rows && feature < head_width
            ? qkv[(size_t)value_key * (3 * width) + 2 * width + head * head_width + feature] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        int count = min(16, rows - key_base);
        for (int key_inner = 0; key_inner < count; ++key_inner)
            sum = fma(probabilities[ly][key_inner], values[key_inner][lx], sum);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (query < rows && feature < head_width)
        output[(size_t)query * width + head * head_width + feature] = sum;
}
