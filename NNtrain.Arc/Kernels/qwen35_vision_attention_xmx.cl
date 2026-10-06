// Full-resolution bidirectional vision attention with two-component F16
// operands and FP32 XMX accumulators. High+residual conversion retains the
// F32 operand's precision; every cross product is included. FP32 softmax is
// unchanged. The caller keeps this candidate separate from the exact path.
#if defined(ARC_XMX) && ARC_SG == 16 && defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#ifdef cl_intel_subgroups_short
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#endif

__kernel void q35v_attention_pack_q(__global const float * qkv,
    __global ushort * high, __global ushort * low,
    int rows, int heads, int head_width) {
    const int id = get_global_id(0), row_blocks = (rows + 7) / 8;
    const int k_blocks = (head_width + 15) / 16;
    const int per_head = row_blocks * k_blocks * 128;
    if (id >= heads * per_head) return;
    const int head = id / per_head, local_id = id % per_head;
    const int k = (local_id / (row_blocks * 128)) * 16 + local_id % 16;
    const int row = ((local_id / 128) % row_blocks) * 8 + (local_id % 128) / 16;
    const float value = row < rows && k < head_width
        ? qkv[(size_t)row * 3 * heads * head_width + head * head_width + k] : 0.0f;
    const half hi = convert_half_rte(value);
    high[id] = as_ushort(hi);
    low[id] = as_ushort(convert_half_rte(value - convert_float(hi)));
}

__kernel void q35v_attention_pack_k(__global const float * qkv,
    __global uint * high, __global uint * low,
    int rows, int heads, int head_width) {
    const int id = get_global_id(0), col_blocks = (rows + 15) / 16;
    const int k_blocks = (head_width + 15) / 16;
    const int per_head = col_blocks * k_blocks * 128;
    if (id >= heads * per_head) return;
    const int head = id / per_head, local_id = id % per_head;
    const int block = local_id / (col_blocks * 128);
    const int col = ((local_id / 128) % col_blocks) * 16 + local_id % 16;
    const int k = block * 16 + ((local_id % 128) / 16) * 2;
    const int width = heads * head_width;
    const float first = col < rows && k < head_width
        ? qkv[(size_t)col * 3 * width + width + head * head_width + k] : 0.0f;
    const float second = col < rows && k + 1 < head_width
        ? qkv[(size_t)col * 3 * width + width + head * head_width + k + 1] : 0.0f;
    const half first_hi = convert_half_rte(first), second_hi = convert_half_rte(second);
    const ushort first_lo = as_ushort(convert_half_rte(first - convert_float(first_hi)));
    const ushort second_lo = as_ushort(convert_half_rte(second - convert_float(second_hi)));
    high[id] = (uint)as_ushort(first_hi) | ((uint)as_ushort(second_hi) << 16);
    low[id] = (uint)first_lo | ((uint)second_lo << 16);
}

__kernel void q35v_attention_pack_v(__global const float * qkv,
    __global uint * high, __global uint * low,
    int rows, int heads, int head_width) {
    const int id = get_global_id(0), col_blocks = (head_width + 15) / 16;
    const int k_blocks = (rows + 15) / 16;
    const int per_head = col_blocks * k_blocks * 128;
    if (id >= heads * per_head) return;
    const int head = id / per_head, local_id = id % per_head;
    const int block = local_id / (col_blocks * 128);
    const int col = ((local_id / 128) % col_blocks) * 16 + local_id % 16;
    const int k = block * 16 + ((local_id % 128) / 16) * 2;
    const int width = heads * head_width;
    const float first = k < rows && col < head_width
        ? qkv[(size_t)k * 3 * width + 2 * width + head * head_width + col] : 0.0f;
    const float second = k + 1 < rows && col < head_width
        ? qkv[(size_t)(k + 1) * 3 * width + 2 * width + head * head_width + col] : 0.0f;
    const half first_hi = convert_half_rte(first), second_hi = convert_half_rte(second);
    const ushort first_lo = as_ushort(convert_half_rte(first - convert_float(first_hi)));
    const ushort second_lo = as_ushort(convert_half_rte(second - convert_float(second_hi)));
    high[id] = (uint)as_ushort(first_hi) | ((uint)as_ushort(second_hi) << 16);
    low[id] = (uint)first_lo | ((uint)second_lo << 16);
}

__kernel void q35v_attention_pack_p(__global const float * probabilities,
    __global ushort * high, __global ushort * low, int rows, int heads) {
    const int id = get_global_id(0), row_blocks = (rows + 7) / 8;
    const int k_blocks = (rows + 15) / 16;
    const int per_head = row_blocks * k_blocks * 128;
    if (id >= heads * per_head) return;
    const int head = id / per_head, local_id = id % per_head;
    const int k = (local_id / (row_blocks * 128)) * 16 + local_id % 16;
    const int row = ((local_id / 128) % row_blocks) * 8 + (local_id % 128) / 16;
    const float value = row < rows && k < rows
        ? probabilities[((size_t)row * heads + head) * rows + k] : 0.0f;
    const half hi = convert_half_rte(value);
    high[id] = as_ushort(hi);
    // Probabilities can be smaller than the F16 normal range. Scaling their
    // residual keeps the second component from underflowing to zero.
    low[id] = as_ushort(convert_half_rte((value - convert_float(hi)) * 32768.0f));
}

inline short8 q35v_attention_read_a(__global const ushort * values, size_t offset) {
#ifdef cl_intel_subgroups_short
    return as_short8(intel_sub_group_block_read_us8(values + offset));
#else
    const int lane = get_sub_group_local_id();
    short8 result;
    #pragma unroll
    for (int r = 0; r < 8; ++r) result[r] = as_short(values[offset + lane + r * 16]);
    return result;
#endif
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void q35v_attention_scores_xmx(__global const ushort * q_high,
    __global const ushort * q_low, __global const uint * k_high,
    __global const uint * k_low, __global float * scores,
    int rows, int heads, int head_width) {
    const int lane = get_local_id(0), subgroup = get_local_id(1);
    const int query_tiles = (rows + 127) / 128;
    const int head = get_group_id(1) / query_tiles;
    const int query = (get_group_id(1) % query_tiles) * 128 + subgroup * 8;
    const int col_base = get_group_id(0) * 64;
    const int row_blocks = (rows + 7) / 8, col_blocks = (rows + 15) / 16;
    const int k_blocks = (head_width + 15) / 16;
    const size_t a_base = (size_t)head * row_blocks * k_blocks * 128;
    const size_t b_base = (size_t)head * col_blocks * k_blocks * 128;
    float8 sums[4] = {(float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f)};
    for (int block = 0; block < k_blocks; ++block) {
        short8 hi = (short8)(0), lo = (short8)(0);
        if (query < rows) {
            const size_t offset = a_base + ((size_t)block * row_blocks + query / 8) * 128;
            hi = q35v_attention_read_a(q_high, offset);
            lo = q35v_attention_read_a(q_low, offset);
        }
        #pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
            const int col_block = col_base / 16 + tile;
            int8 wh = (int8)(0), wl = (int8)(0);
            if (col_block < col_blocks) {
                const size_t offset = b_base + ((size_t)block * col_blocks + col_block) * 128;
                wh = as_int8(intel_sub_group_block_read8(k_high + offset));
                wl = as_int8(intel_sub_group_block_read8(k_low + offset));
            }
            sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(hi, wh, sums[tile]);
            sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(lo, wh, sums[tile]);
            sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(hi, wl, sums[tile]);
            sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(lo, wl, sums[tile]);
        }
    }
    const float scale = rsqrt((float)head_width);
    #pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        const int col = col_base + tile * 16 + lane;
        #pragma unroll
        for (int r = 0; r < 8; ++r)
            if (query + r < rows && col < rows)
                scores[((size_t)(query + r) * heads + head) * rows + col] = sums[tile][r] * scale;
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void q35v_attention_context_xmx(__global const ushort * p_high,
    __global const ushort * p_low, __global const uint * v_high,
    __global const uint * v_low, __global float * output,
    int rows, int heads, int head_width) {
    const int lane = get_local_id(0), subgroup = get_local_id(1);
    const int query_tiles = (rows + 127) / 128;
    const int head = get_group_id(1) / query_tiles;
    const int query = (get_group_id(1) % query_tiles) * 128 + subgroup * 8;
    const int col_base = get_group_id(0) * 64;
    const int row_blocks = (rows + 7) / 8, col_blocks = (head_width + 15) / 16;
    const int k_blocks = (rows + 15) / 16;
    const size_t a_base = (size_t)head * row_blocks * k_blocks * 128;
    const size_t b_base = (size_t)head * col_blocks * k_blocks * 128;
    float8 sums[4] = {(float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f)};
    float8 low_sums[4] = {(float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f)};
    for (int block = 0; block < k_blocks; ++block) {
        short8 hi = (short8)(0), lo = (short8)(0);
        if (query < rows) {
            const size_t offset = a_base + ((size_t)block * row_blocks + query / 8) * 128;
            hi = q35v_attention_read_a(p_high, offset);
            lo = q35v_attention_read_a(p_low, offset);
        }
        #pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
            const int col_block = col_base / 16 + tile;
            int8 wh = (int8)(0), wl = (int8)(0);
            if (col_block < col_blocks) {
                const size_t offset = b_base + ((size_t)block * col_blocks + col_block) * 128;
                wh = as_int8(intel_sub_group_block_read8(v_high + offset));
                wl = as_int8(intel_sub_group_block_read8(v_low + offset));
            }
            sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(hi, wh, sums[tile]);
            low_sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(lo, wh, low_sums[tile]);
            sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(hi, wl, sums[tile]);
            low_sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(lo, wl, low_sums[tile]);
        }
    }
    const int width = heads * head_width;
    #pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        const int col = col_base + tile * 16 + lane;
        #pragma unroll
        for (int r = 0; r < 8; ++r)
            if (query + r < rows && col < head_width)
                output[(size_t)(query + r) * width + head * head_width + col]
                    = fma(low_sums[tile][r], 1.0f / 32768.0f, sums[tile][r]);
    }
}
#if defined(ARC_VISION_FLASH)
// One subgroup owns eight queries and all features of a head (up to 80).
// Online softmax visits every key in 32-key tiles; score/probability values
// and the normalization accumulator remain FP32 in registers. No quadratic
// score buffer is required. The two F16 components include all cross terms.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void q35v_attention_flash_xmx(__global const ushort * q_high,
    __global const ushort * q_low, __global const uint * k_high,
    __global const uint * k_low, __global const uint * v_high,
    __global const uint * v_low, __global float * output,
    int rows, int heads, int head_width) {
    const int lane = get_local_id(0), subgroup = get_local_id(1);
    const int query_tiles = (rows + 127) / 128;
    const int head = get_group_id(1) / query_tiles;
    const int query = (get_group_id(1) % query_tiles) * 128 + subgroup * 8;
    const int row_blocks = (rows + 7) / 8, col_blocks = (rows + 15) / 16;
    const int feature_blocks = (head_width + 15) / 16;
    const size_t q_base = (size_t)head * row_blocks * feature_blocks * 128;
    const size_t k_base = (size_t)head * col_blocks * feature_blocks * 128;
    const size_t v_base = (size_t)head * feature_blocks * col_blocks * 128;
    float8 maximum = (float8)(-INFINITY), denominator = (float8)(0.0f);
    float8 context[5] = {(float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f)};
    float8 low_context[5] = {(float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f)};
    const float scale = rsqrt((float)head_width);
    for (int key_base = 0; key_base < rows; key_base += 32) {
        float8 scores[2] = {(float8)(0.0f), (float8)(0.0f)};
        for (int block = 0; block < feature_blocks; ++block) {
            short8 hi = (short8)(0), lo = (short8)(0);
            if (query < rows) {
                const size_t offset = q_base + ((size_t)block * row_blocks + query / 8) * 128;
                hi = q35v_attention_read_a(q_high, offset);
                lo = q35v_attention_read_a(q_low, offset);
            }
            #pragma unroll
            for (int tile = 0; tile < 2; ++tile) {
                const int key_block = key_base / 16 + tile;
                int8 kh = (int8)(0), kl = (int8)(0);
                if (key_block < col_blocks) {
                    const size_t offset = k_base + ((size_t)block * col_blocks + key_block) * 128;
                    kh = as_int8(intel_sub_group_block_read8(k_high + offset));
                    kl = as_int8(intel_sub_group_block_read8(k_low + offset));
                }
                scores[tile] = intel_sub_group_f16_f16_matrix_mad_k16(hi, kh, scores[tile]);
                scores[tile] = intel_sub_group_f16_f16_matrix_mad_k16(lo, kh, scores[tile]);
                scores[tile] = intel_sub_group_f16_f16_matrix_mad_k16(hi, kl, scores[tile]);
                scores[tile] = intel_sub_group_f16_f16_matrix_mad_k16(lo, kl, scores[tile]);
            }
        }
        float8 old_scale;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            #pragma unroll
            for (int tile = 0; tile < 2; ++tile)
                scores[tile][r] = query + r < rows && key_base + tile * 16 + lane < rows
                    ? scores[tile][r] * scale : -INFINITY;
            float tile_max = fmax(scores[0][r], scores[1][r]);
            for (int stride = 8; stride > 0; stride >>= 1)
                tile_max = fmax(tile_max, intel_sub_group_shuffle(tile_max, lane ^ stride));
            float next_max = fmax(maximum[r], tile_max);
            if (query + r >= rows) next_max = 0.0f;
            old_scale[r] = exp(maximum[r] - next_max);
            scores[0][r] = exp(scores[0][r] - next_max);
            scores[1][r] = exp(scores[1][r] - next_max);
            float tile_sum = scores[0][r] + scores[1][r];
            for (int stride = 8; stride > 0; stride >>= 1)
                tile_sum += intel_sub_group_shuffle(tile_sum, lane ^ stride);
            denominator[r] = fma(denominator[r], old_scale[r], tile_sum);
            maximum[r] = next_max;
        }
        #pragma unroll
        for (int feature = 0; feature < 5; ++feature) {
            if (feature < feature_blocks) {
                context[feature] *= old_scale;
                low_context[feature] *= old_scale;
            }
        }
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            short8 ph, pl;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                const half hi = convert_half_rte(scores[part][r]);
                ph[r] = as_short(hi);
                pl[r] = as_short(convert_half_rte((scores[part][r] - convert_float(hi)) * 32768.0f));
            }
            const int key_block = key_base / 16 + part;
            #pragma unroll
            for (int feature = 0; feature < 5; ++feature) {
                if (feature >= feature_blocks) continue;
                int8 vh = (int8)(0), vl = (int8)(0);
                if (key_block < col_blocks) {
                    const size_t offset = v_base + ((size_t)key_block * feature_blocks + feature) * 128;
                    vh = as_int8(intel_sub_group_block_read8(v_high + offset));
                    vl = as_int8(intel_sub_group_block_read8(v_low + offset));
                }
                context[feature] = intel_sub_group_f16_f16_matrix_mad_k16(ph, vh, context[feature]);
                context[feature] = intel_sub_group_f16_f16_matrix_mad_k16(ph, vl, context[feature]);
                low_context[feature] = intel_sub_group_f16_f16_matrix_mad_k16(pl, vh, low_context[feature]);
                low_context[feature] = intel_sub_group_f16_f16_matrix_mad_k16(pl, vl, low_context[feature]);
            }
        }
    }
    #pragma unroll
    for (int feature = 0; feature < 5; ++feature) {
        if (feature >= feature_blocks) continue;
        const int col = feature * 16 + lane;
        #pragma unroll
        for (int r = 0; r < 8; ++r)
            if (query + r < rows && col < head_width)
                output[(size_t)(query + r) * heads * head_width + head * head_width + col]
                    = fma(low_context[feature][r], 1.0f / 32768.0f, context[feature][r]) / denominator[r];
    }
}
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void q35v_attention_context_xmx_direct(__global const float * probabilities,
    __global const uint * v_high,
    __global const uint * v_low, __global float * output,
    int rows, int heads, int head_width) {
    const int lane = get_local_id(0), subgroup = get_local_id(1);
    const int query_tiles = (rows + 127) / 128;
    const int head = get_group_id(1) / query_tiles;
    const int query = (get_group_id(1) % query_tiles) * 128 + subgroup * 8;
    const int col_base = get_group_id(0) * 64;
    const int row_blocks = (rows + 7) / 8, col_blocks = (head_width + 15) / 16;
    const int k_blocks = (rows + 15) / 16;
    const size_t a_base = (size_t)head * row_blocks * k_blocks * 128;
    const size_t b_base = (size_t)head * col_blocks * k_blocks * 128;
    float8 sums[4] = {(float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f)};
    float8 low_sums[4] = {(float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f)};
    for (int block = 0; block < k_blocks; ++block) {
        short8 hi = (short8)(0), lo = (short8)(0);
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int key = block * 16 + lane;
            const float value = query + r < rows && key < rows
                ? probabilities[((size_t)(query + r) * heads + head) * rows + key] : 0.0f;
            const half high_value = convert_half_rte(value);
            hi[r] = as_short(high_value);
            lo[r] = as_short(convert_half_rte((value - convert_float(high_value)) * 32768.0f));
        }
        #pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
            const int col_block = col_base / 16 + tile;
            int8 wh = (int8)(0), wl = (int8)(0);
            if (col_block < col_blocks) {
                const size_t offset = b_base + ((size_t)block * col_blocks + col_block) * 128;
                wh = as_int8(intel_sub_group_block_read8(v_high + offset));
                wl = as_int8(intel_sub_group_block_read8(v_low + offset));
            }
            sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(hi, wh, sums[tile]);
            low_sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(lo, wh, low_sums[tile]);
            sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(hi, wl, sums[tile]);
            low_sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(lo, wl, low_sums[tile]);
        }
    }
    const int width = heads * head_width;
    #pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        const int col = col_base + tile * 16 + lane;
        #pragma unroll
        for (int r = 0; r < 8; ++r)
            if (query + r < rows && col < head_width)
                output[(size_t)(query + r) * width + head * head_width + col]
                    = fma(low_sums[tile][r], 1.0f / 32768.0f, sums[tile][r]);
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void q35v_attention_context_xmx_direct80(__global const float * probabilities,
    __global const uint * v_high,
    __global const uint * v_low, __global float * output,
    int rows, int heads, int head_width) {
    const int lane = get_local_id(0), subgroup = get_local_id(1);
    const int query_tiles = (rows + 127) / 128;
    const int head = get_group_id(1) / query_tiles;
    const int query = (get_group_id(1) % query_tiles) * 128 + subgroup * 8;
    const int col_base = 0;
    const int row_blocks = (rows + 7) / 8, col_blocks = (head_width + 15) / 16;
    const int k_blocks = (rows + 15) / 16;
    const size_t a_base = (size_t)head * row_blocks * k_blocks * 128;
    const size_t b_base = (size_t)head * col_blocks * k_blocks * 128;
    float8 sums[5] = {(float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f)};
    float8 low_sums[5] = {(float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f)};
    for (int block = 0; block < k_blocks; ++block) {
        short8 hi = (short8)(0), lo = (short8)(0);
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int key = block * 16 + lane;
            const float value = query + r < rows && key < rows
                ? probabilities[((size_t)(query + r) * heads + head) * rows + key] : 0.0f;
            const half high_value = convert_half_rte(value);
            hi[r] = as_short(high_value);
            lo[r] = as_short(convert_half_rte((value - convert_float(high_value)) * 32768.0f));
        }
        #pragma unroll
        for (int tile = 0; tile < 5; ++tile) {
            const int col_block = tile;
            if (col_block >= col_blocks) continue;
            int8 wh = (int8)(0), wl = (int8)(0);
            if (col_block < col_blocks) {
                const size_t offset = b_base + ((size_t)block * col_blocks + col_block) * 128;
                wh = as_int8(intel_sub_group_block_read8(v_high + offset));
                wl = as_int8(intel_sub_group_block_read8(v_low + offset));
            }
            sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(hi, wh, sums[tile]);
            low_sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(lo, wh, low_sums[tile]);
            sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(hi, wl, sums[tile]);
            low_sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(lo, wl, low_sums[tile]);
        }
    }
    const int width = heads * head_width;
    #pragma unroll
    for (int tile = 0; tile < 5; ++tile) {
        const int col = col_base + tile * 16 + lane;
        #pragma unroll
        for (int r = 0; r < 8; ++r)
            if (query + r < rows && col < head_width)
                output[(size_t)(query + r) * width + head * head_width + col]
                    = fma(low_sums[tile][r], 1.0f / 32768.0f, sums[tile][r]);
    }
}
#endif
