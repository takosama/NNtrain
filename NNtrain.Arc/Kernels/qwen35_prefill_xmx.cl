// Inference-only XMX prefill. Decode one quantized 64x32 weight panel for
// 128 prompt rows; retain both high and residual F16 components. GGUF weights
// stay encoded and no full dequantized model allocation is introduced.
#if defined(ARC_XMX) && ARC_SG == 16 && defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#ifdef ARC_SLM_BLOCK_IO
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable
#endif

inline float8 q35p_xmx_q4_octet(__global const uchar* block, int octet, float d) {
    const int start = octet * 8, group = start >> 5, within = start & 31;
    float multiplier, minimum;
    q35l_q4_scale_min(block, group, d,
        q35l_half_to_float((ushort)block[2] | ((ushort)block[3] << 8)),
        &multiplier, &minimum);
    uchar8 packed = vload8(0, block + 16 + (group >> 1) * 32 + within);
    float8 quant = convert_float8((group & 1) ? packed >> 4 : packed & (uchar8)(15));
    return fma((float8)(multiplier), quant, (float8)(-minimum));
}

inline int8 q35p_xmx_panel(const __local uint* panel, int lane) {
#ifdef ARC_SLM_BLOCK_IO
    return as_int8(intel_sub_group_block_read8(panel));
#else
    int8 result;
    #pragma unroll
    for (int i = 0; i < 8; ++i) result[i] = as_int(panel[i * 16 + lane]);
    return result;
#endif
}

// A separate input conversion avoids repeating activation rounding for every
// output tile. Layout stays row-major for bounded temporary storage.
__kernel void q35l_prefill_xmx_input_f16x2(__global const float* input,
    __global ushort* high, __global ushort* low, __global int* range_status, int count) {
    const int i = get_global_id(0);
    if (i < count) {
        const float value = input[i];
        if (!isfinite(value) || fabs(value) > 65504.0f) {
            atomic_or(range_status, 1); high[i] = 0; low[i] = 0; return;
        }
        const half hi = convert_half_rte(value);
        high[i] = as_ushort(hi);
        low[i] = as_ushort(convert_half_rte(value - convert_float(hi)));
    }
}

// A uniform device flag selects exact FP32 work when a valid activation lies
// outside F16's range. No CPU readback or queue fence is needed. The original
// resident payload and activation buffer remain available for that fallback.
#define Q35P_XMX_FALLBACK(NAME, BYTES, DECODE) \
inline void NAME(__global const float* x, __global const uchar* packed, \
    __global const float* bias, __global float* output, int rows, int input_width, \
    int output_width, int row_base, int col_base) { \
    const int lane = get_sub_group_local_id(), sg = get_sub_group_id(); \
    for (int item = sg; item < 128 * 64; item += 16) { \
        const int row = row_base + item / 64, col = col_base + item % 64; \
        if (row >= rows || col >= output_width) continue; \
        const int blocks = input_width / 256; \
        __global const uchar* row_weight = packed + (size_t)col * blocks * BYTES; \
        __global const float* row_x = x + (size_t)row * input_width; \
        float8 sum = (float8)(0.0f); \
        for (int b = 0; b < blocks; ++b) { \
            __global const uchar* block = row_weight + b * BYTES; \
            const float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8)); \
            for (int part = 0; part < 2; ++part) { \
                const int octet = part * 16 + lane; \
                sum = fma(vload8(0, row_x + b * 256 + octet * 8), DECODE(block, octet, d), sum); \
            } \
        } \
        const float total = sub_group_reduce_add(dot(sum.lo, (float4)(1.0f)) + dot(sum.hi, (float4)(1.0f))); \
        if (lane == 0) output[(size_t)row * output_width + col] = bias[col] + total; \
    } \
}
Q35P_XMX_FALLBACK(q35p_xmx_fallback_iq2, 82, q35l_prefill_iq2_octet)
Q35P_XMX_FALLBACK(q35p_xmx_fallback_iq3, 110, q35l_prefill_iq3_octet)
#undef Q35P_XMX_FALLBACK

inline void q35p_xmx_fallback_q4(__global const float* x, __global const uchar* packed,
    __global const float* bias, __global float* output, int rows, int input_width,
    int output_width, int row_base, int col_base) {
    const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
    for (int item = sg; item < 128 * 64; item += 16) {
        const int row = row_base + item / 64, col = col_base + item % 64;
        if (row >= rows || col >= output_width) continue;
        const int blocks = input_width / 256;
        __global const uchar* row_weight = packed + (size_t)col * blocks * 144;
        __global const float* row_x = x + (size_t)row * input_width;
        float sum = 0.0f;
        for (int b = 0; b < blocks; ++b) {
            __global const uchar* block = row_weight + b * 144;
            const float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8));
            const float dmin = q35l_half_to_float((ushort)block[2] | ((ushort)block[3] << 8));
            for (int pair = 0; pair < 4; ++pair) {
                float em, en, om, on;
                q35l_q4_scale_min(block, pair * 2, d, dmin, &em, &en);
                q35l_q4_scale_min(block, pair * 2 + 1, d, dmin, &om, &on);
                const uchar first = block[16 + pair * 32 + lane], second = block[32 + pair * 32 + lane];
                const int offset = b * 256 + pair * 64 + lane;
                sum = fma(row_x[offset], fma(em, (float)(first & 15), -en), sum);
                sum = fma(row_x[offset + 16], fma(em, (float)(second & 15), -en), sum);
                sum = fma(row_x[offset + 32], fma(om, (float)(first >> 4), -on), sum);
                sum = fma(row_x[offset + 48], fma(om, (float)(second >> 4), -on), sum);
            }
        }
        const float total = sub_group_reduce_add(sum);
        if (lane == 0) output[(size_t)row * output_width + col] = bias[col] + total;
    }
}

#define Q35P_XMX_LINEAR(NAME, BYTES, DECODE, FALLBACK) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(16, 16, 1))) \
__kernel void NAME(__global const ushort* xhigh, __global const ushort* xlow, \
    __global const uchar* packed, __global const float* bias, __global float* output, \
    int rows, int input_width, int output_width, __global const float* input, __global const int* range_status) { \
    const int row_base = get_group_id(1) * 128, col_base = get_group_id(0) * 64; \
    if (range_status[0] != 0) { FALLBACK(input, packed, bias, output, rows, input_width, output_width, row_base, col_base); return; } \
    const int lane = get_local_id(0), subgroup = get_local_id(1), tid = subgroup * 16 + lane; \
    __local ushort ahigh[128][33], alow[128][33]; \
    __local uint bhigh[64 * 16], blow[64 * 16]; \
    float8 accum[4] = {(float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f)}; \
    for (int base = 0; base < input_width; base += 32) { \
        for (int i = tid; i < 128 * 32; i += 256) { \
            const int row = row_base + i / 32, k = base + i % 32; \
            ahigh[i / 32][i % 32] = row < rows ? xhigh[(size_t)row * input_width + k] : (ushort)0; \
            alow[i / 32][i % 32] = row < rows ? xlow[(size_t)row * input_width + k] : (ushort)0; \
        } \
        const int col_inner = tid / 4, col = col_base + col_inner; \
        const int octet = base / 8 + tid % 4; \
        float8 decoded = (float8)(0.0f); \
        if (col < output_width) { \
            __global const uchar* block = packed + ((size_t)col * (input_width / 256) + octet / 32) * BYTES; \
            const float scale = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8)); \
            decoded = DECODE(block, octet % 32, scale); \
        } \
        half8 whigh = convert_half8_rte(decoded); \
        half8 wlow = convert_half8_rte(decoded - convert_float8(whigh)); \
        ushort8 hi = as_ushort8(whigh), lo = as_ushort8(wlow); \
        const int first_pair = (tid % 4) * 4; \
        _Pragma("unroll") \
        for (int pair = 0; pair < 4; ++pair) { \
            const int p = first_pair + pair; \
            const int dst = ((col_inner / 16) * 2 + p / 8) * 128 + (p % 8) * 16 + col_inner % 16; \
            bhigh[dst] = (uint)hi[2 * pair] | ((uint)hi[2 * pair + 1] << 16); \
            blow[dst] = (uint)lo[2 * pair] | ((uint)lo[2 * pair + 1] << 16); \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        _Pragma("unroll") \
        for (int part = 0; part < 2; ++part) { \
            short8 avh, avl; \
            _Pragma("unroll") \
            for (int r = 0; r < 8; ++r) { \
                avh[r] = as_short(ahigh[subgroup * 8 + r][part * 16 + lane]); \
                avl[r] = as_short(alow[subgroup * 8 + r][part * 16 + lane]); \
            } \
            _Pragma("unroll") \
            for (int tile = 0; tile < 4; ++tile) { \
                const int8 bvh = q35p_xmx_panel(bhigh + (tile * 2 + part) * 128, lane); \
                const int8 bvl = q35p_xmx_panel(blow + (tile * 2 + part) * 128, lane); \
                accum[tile] = intel_sub_group_f16_f16_matrix_mad_k16(avh, bvh, accum[tile]); \
                accum[tile] = intel_sub_group_f16_f16_matrix_mad_k16(avl, bvh, accum[tile]); \
                accum[tile] = intel_sub_group_f16_f16_matrix_mad_k16(avh, bvl, accum[tile]); \
                accum[tile] = intel_sub_group_f16_f16_matrix_mad_k16(avl, bvl, accum[tile]); \
            } \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    _Pragma("unroll") \
    for (int tile = 0; tile < 4; ++tile) { \
        const int col = col_base + tile * 16 + lane; \
        _Pragma("unroll") \
        for (int r = 0; r < 8; ++r) { \
            const int row = row_base + subgroup * 8 + r; \
            if (row < rows && col < output_width) \
                output[(size_t)row * output_width + col] = accum[tile][r] + bias[col]; \
        } \
    } \
}

Q35P_XMX_LINEAR(q35l_prefill_xmx_iq2_s_f16x2, 82, q35l_prefill_iq2_octet, q35p_xmx_fallback_iq2)
Q35P_XMX_LINEAR(q35l_prefill_xmx_iq3_s_f16x2, 110, q35l_prefill_iq3_octet, q35p_xmx_fallback_iq3)
Q35P_XMX_LINEAR(q35l_prefill_xmx_q4_k_f16x2, 144, q35p_xmx_q4_octet, q35p_xmx_fallback_q4)
#undef Q35P_XMX_LINEAR

inline float8 q35p_iq2_integer_grid(__global const uchar* block, int octet) {
    const int group = octet >> 2, part = octet & 3;
    const int index = block[2 + octet] | (((block[66 + group] >> (part * 2)) & 3) << 8);
    float8 grid = convert_float8(as_uchar8(q35l_iq2s_grid[index]));
    const int signs = block[34 + octet];
    return select(grid, -grid, ((int8)(signs) & (int8)(1, 2, 4, 8, 16, 32, 64, 128)) != (int8)(0));
}
inline float q35p_iq2_integer_scale(__global const uchar* block, int octet, float d) {
    const int scale = (block[74 + (octet >> 2)] >> (((octet & 3) >> 1) * 4)) & 15;
    return d * (0.5f + (float)scale) * 0.25f;
}
inline float8 q35p_iq3_integer_grid(__global const uchar* block, int octet) {
    const int group = octet >> 2, pair = (octet & 3) * 2, high = block[66 + group];
    const int grid0 = block[2 + octet * 2] | (((high >> pair) & 1) << 8);
    const int grid1 = block[3 + octet * 2] | (((high >> (pair + 1)) & 1) << 8);
    float8 grid = convert_float8(as_uchar8((uint2)(q35l_iq3s_grid[grid0], q35l_iq3s_grid[grid1])));
    const int signs = block[74 + octet];
    return select(grid, -grid, ((int8)(signs) & (int8)(1, 2, 4, 8, 16, 32, 64, 128)) != (int8)(0));
}
inline float q35p_iq3_integer_scale(__global const uchar* block, int octet, float d) {
    const int group = octet >> 2;
    const int scale = (block[106 + (group >> 1)] >> ((group & 1) * 4)) & 15;
    return d * (float)(1 + 2 * scale);
}

// IQ lookup grids are small exact integers. Keep their per-16 coefficients
// in FP32 and apply them after two XMX products, avoiding rounded coefficient
// multiplication inside the F16 matrix panel and halving XMX issue count.
#define Q35P_XMX_FACTORED(NAME, BYTES, GRID, SCALE, FALLBACK) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(16, 16, 1))) \
__kernel void NAME(__global const ushort* xhigh, __global const ushort* xlow, \
    __global const uchar* packed, __global const float* bias, __global float* output, \
    int rows, int input_width, int output_width, __global const float* input, __global const int* range_status) { \
    const int row_base = get_group_id(1) * 128, col_base = get_group_id(0) * 64; \
    if (range_status[0] != 0) { FALLBACK(input, packed, bias, output, rows, input_width, output_width, row_base, col_base); return; } \
    const int lane = get_local_id(0), subgroup = get_local_id(1), tid = subgroup * 16 + lane; \
    __local ushort ahigh[128][33], alow[128][33]; \
    __local uint grids[64 * 16]; \
    __local float scales[2][64]; \
    float8 accum[4] = {(float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f)}; \
    for (int base = 0; base < input_width; base += 32) { \
        for (int i = tid; i < 128 * 32; i += 256) { \
            const int row = row_base + i / 32, k = base + i % 32; \
            ahigh[i / 32][i % 32] = row < rows ? xhigh[(size_t)row * input_width + k] : (ushort)0; \
            alow[i / 32][i % 32] = row < rows ? xlow[(size_t)row * input_width + k] : (ushort)0; \
        } \
        const int col_inner = tid / 4, col = col_base + col_inner; \
        const int octet = base / 8 + tid % 4; \
        float8 decoded = (float8)(0.0f); \
        float scale = 0.0f; \
        if (col < output_width) { \
            __global const uchar* block = packed + ((size_t)col * (input_width / 256) + octet / 32) * BYTES; \
            const float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8)); \
            decoded = GRID(block, octet % 32); scale = SCALE(block, octet % 32, d); \
        } \
        if ((tid & 1) == 0) scales[(tid & 3) / 2][col_inner] = scale; \
        const ushort8 values = as_ushort8(convert_half8_rte(decoded)); \
        const int first_pair = (tid % 4) * 4; \
        _Pragma("unroll") \
        for (int pair = 0; pair < 4; ++pair) { \
            const int p = first_pair + pair; \
            const int dst = ((col_inner / 16) * 2 + p / 8) * 128 + (p % 8) * 16 + col_inner % 16; \
            grids[dst] = (uint)values[2 * pair] | ((uint)values[2 * pair + 1] << 16); \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        _Pragma("unroll") \
        for (int part = 0; part < 2; ++part) { \
            short8 avh, avl; \
            _Pragma("unroll") \
            for (int r = 0; r < 8; ++r) { \
                avh[r] = as_short(ahigh[subgroup * 8 + r][part * 16 + lane]); \
                avl[r] = as_short(alow[subgroup * 8 + r][part * 16 + lane]); \
            } \
            _Pragma("unroll") \
            for (int tile = 0; tile < 4; ++tile) { \
                const int8 b = q35p_xmx_panel(grids + (tile * 2 + part) * 128, lane); \
                float8 partial = intel_sub_group_f16_f16_matrix_mad_k16(avh, b, (float8)(0.0f)); \
                partial = intel_sub_group_f16_f16_matrix_mad_k16(avl, b, partial); \
                accum[tile] = fma(partial, (float8)(scales[part][tile * 16 + lane]), accum[tile]); \
            } \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    _Pragma("unroll") \
    for (int tile = 0; tile < 4; ++tile) { \
        const int col = col_base + tile * 16 + lane; \
        _Pragma("unroll") \
        for (int r = 0; r < 8; ++r) { \
            const int row = row_base + subgroup * 8 + r; \
            if (row < rows && col < output_width) output[(size_t)row * output_width + col] = accum[tile][r] + bias[col]; \
        } \
    } \
}
Q35P_XMX_FACTORED(q35l_prefill_xmx_iq2_s_factored, 82, q35p_iq2_integer_grid, q35p_iq2_integer_scale, q35p_xmx_fallback_iq2)
Q35P_XMX_FACTORED(q35l_prefill_xmx_iq3_s_factored, 110, q35p_iq3_integer_grid, q35p_iq3_integer_scale, q35p_xmx_fallback_iq3)
#undef Q35P_XMX_FACTORED

#ifdef cl_intel_subgroups_short
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#endif

__kernel void q35l_prefill_xmx_pack_input_f16x2(__global const float* input,
    __global ushort* high, __global ushort* low, __global int* range_status, int rows, int input_width) {
    const int id = get_global_id(0), row_blocks = (rows + 7) / 8;
    const int k_blocks = input_width / 16;
    if (id >= row_blocks * k_blocks * 128) return;
    const int k = (id / (row_blocks * 128)) * 16 + id % 16;
    const int row = ((id / 128) % row_blocks) * 8 + (id % 128) / 16;
    const float value = row < rows ? input[(size_t)row * input_width + k] : 0.0f;
    if (!isfinite(value) || fabs(value) > 65504.0f) {
        atomic_or(range_status, 1); high[id] = 0; low[id] = 0; return;
    }
    const half hi = convert_half_rte(value);
    high[id] = as_ushort(hi);
    low[id] = as_ushort(convert_half_rte(value - convert_float(hi)));
}

// Original high/residual rounding, with no normalization or added quantization.
// Panel order is [ceil(rows/8)][input_width/16][8 rows][16 K values], so one
// row tile consumes adjacent K16 panels instead of striding across all rows.
__kernel void q35l_prefill_xmx_pack_input_f16x2_rowmajor(__global const float* input,
    __global ushort* high, __global ushort* low, __global int* range_status, int rows, int input_width) {
    const int id = get_global_id(0), row_blocks = (rows + 7) / 8;
    const int k_blocks = input_width / 16;
    if (id >= row_blocks * k_blocks * 128) return;
    const int k = ((id / 128) % k_blocks) * 16 + id % 16;
    const int row = (id / (k_blocks * 128)) * 8 + (id % 128) / 16;
    const float value = row < rows ? input[(size_t)row * input_width + k] : 0.0f;
    if (!isfinite(value) || fabs(value) > 65504.0f) {
        atomic_or(range_status, 1); high[id] = 0; low[id] = 0; return;
    }
    const half hi = convert_half_rte(value);
    high[id] = as_ushort(hi);
    low[id] = as_ushort(convert_half_rte(value - convert_float(hi)));
}

#if defined(ARC_OPTIMIZATION_PROBES) || defined(ARC_Q35_GGUF_BSLM)
// Retain the original GGUF storage and decode B only once per workgroup.
// The packed global A panel avoids the much larger A SLM allocation.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void q35l_prefill_xmx_iq2_s_gguf_bslm(__global const ushort* xhigh, __global const ushort* xlow,
    __global const uchar* packed, __global const float* bias, __global float* output,
    int rows, int input_width, int output_width, __global const float* input, __global const int* range_status) {
    const int lane = get_local_id(0), subgroup = get_local_id(1), tid = subgroup * 16 + lane;
    const int row_base = get_group_id(1) * 128, row = row_base + subgroup * 8;
    const int col_base = get_group_id(0) * 64, row_blocks = (rows + 7) / 8;
    if (range_status[0] != 0) {
        q35p_xmx_fallback_iq2(input, packed, bias, output, rows, input_width, output_width, row_base, col_base);
        return;
    }
    __local uint grids[1024];
    __local float coefficients[2][64];
    float8 accum[4] = {(float8)(0), (float8)(0), (float8)(0), (float8)(0)};
    for (int kblock = 0; kblock < input_width / 32; ++kblock) {
        const int col_inner = tid / 4, col = col_base + col_inner;
        const int octet = (kblock % 8) * 4 + tid % 4;
        ushort8 decoded = (ushort8)(0);
        float coefficient = 0.0f;
        if (col < output_width) {
            __global const uchar* block = packed + ((size_t)col * (input_width / 256) + kblock / 8) * 82;
            const float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8));
            decoded = as_ushort8(convert_half8_rte(q35p_iq2_integer_grid(block, octet)));
            coefficient = q35p_iq2_integer_scale(block, octet, d);
        }
        if ((tid & 1) == 0) coefficients[(tid & 3) / 2][col_inner] = coefficient;
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int kpair = (tid % 4) * 4 + pair;
            const int dst = ((col_inner / 16) * 2 + kpair / 8) * 128 + (kpair % 8) * 16 + col_inner % 16;
            grids[dst] = (uint)decoded[pair * 2] | ((uint)decoded[pair * 2 + 1] << 16);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int half_index = 0; half_index < 2; ++half_index) {
            short8 ah = (short8)(0), al = (short8)(0);
            if (row < rows) {
                const size_t offset = ((size_t)(kblock * 2 + half_index) * row_blocks + row / 8) * 128;
#ifdef cl_intel_subgroups_short
                ah = as_short8(intel_sub_group_block_read_us8(xhigh + offset));
                al = as_short8(intel_sub_group_block_read_us8(xlow + offset));
#else
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    ah[r] = as_short(xhigh[offset + r * 16 + lane]);
                    al[r] = as_short(xlow[offset + r * 16 + lane]);
                }
#endif
            }
            #pragma unroll
            for (int tile = 0; tile < 4; ++tile) {
                const int8 operand = q35p_xmx_panel(grids + (tile * 2 + half_index) * 128, lane);
                float8 partial = intel_sub_group_f16_f16_matrix_mad_k16(ah, operand, (float8)(0));
                partial = intel_sub_group_f16_f16_matrix_mad_k16(al, operand, partial);
                accum[tile] = fma(partial, (float8)(coefficients[half_index][tile * 16 + lane]), accum[tile]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        const int col = col_base + tile * 16 + lane;
        #pragma unroll
        for (int r = 0; r < 8; ++r) if (row + r < rows && col < output_width)
            output[(size_t)(row + r) * output_width + col] = accum[tile][r] + bias[col];
    }
}
// Wider B-only SLM stages retain the original BSLM arithmetic exactly:
// signed grid and coefficient remain separate, and every K16 step performs
// high MMA, residual MMA, then coefficient FMA in increasing K order.
// Only the number of K16 panels decoded between barriers changes. K64 uses
// 8 KiB grid + 1 KiB coefficients; K128 uses 16 KiB + 2 KiB. Both preserve
// the original packed A layout and its range fallback.
inline short8 q35p_bslm_staged_input(__global const ushort* panel, size_t offset, int lane) {
#ifdef cl_intel_subgroups_short
    return as_short8(intel_sub_group_block_read_us8(panel + offset));
#else
    short8 value;
    #pragma unroll
    for (int r = 0; r < 8; ++r) value[r] = as_short(panel[offset + r * 16 + lane]);
    return value;
#endif
}

#define Q35P_GGUF_BSLM_STAGED(NAME, KSTEP, COALESCED, ROW_MAJOR, UNROLL_PARTS) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(16,16,1))) \
__kernel void NAME(__global const ushort* xhigh, __global const ushort* xlow, \
    __global const uchar* packed, __global const float* bias, __global float* output, \
    int rows, int input_width, int output_width, __global const float* input, __global const int* range_status) { \
    const int lane = get_local_id(0), subgroup = get_local_id(1), tid = subgroup * 16 + lane; \
    const int row_base = get_group_id(1) * 128, row = row_base + subgroup * 8; \
    const int col_base = get_group_id(0) * 64, row_blocks = (rows + 7) / 8; \
    if (range_status[0] != 0) { \
        q35p_xmx_fallback_iq2(input, packed, bias, output, rows, input_width, output_width, row_base, col_base); \
        return; \
    } \
    __local uint grids[32 * KSTEP]; \
    __local float coefficients[KSTEP / 16][64]; \
    float8 accum[4] = {(float8)(0), (float8)(0), (float8)(0), (float8)(0)}; \
    for (int base = 0; base < input_width; base += KSTEP) { \
        for (int item = tid; item < 64 * (KSTEP / 8); item += 256) { \
            const int col_inner = COALESCED ? item % 64 : item / (KSTEP / 8); \
            const int col = col_base + col_inner; \
            const int stage_octet = COALESCED ? item / 64 : item % (KSTEP / 8); \
            const int octet = (base % 256) / 8 + stage_octet; \
            ushort8 decoded = (ushort8)(0); \
            float coefficient = 0.0f; \
            if (col < output_width) { \
                __global const uchar* block = packed + ((size_t)col * (input_width / 256) + base / 256) * 82; \
                const float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8)); \
                decoded = as_ushort8(convert_half8_rte(q35p_iq2_integer_grid(block, octet))); \
                coefficient = q35p_iq2_integer_scale(block, octet, d); \
            } \
            if ((stage_octet & 1) == 0) coefficients[stage_octet / 2][col_inner] = coefficient; \
            _Pragma("unroll") \
            for (int pair = 0; pair < 4; ++pair) { \
                const int kpair = stage_octet * 4 + pair; \
                const int dst = ((col_inner / 16) * (KSTEP / 16) + kpair / 8) * 128 \
                    + (kpair % 8) * 16 + col_inner % 16; \
                grids[dst] = (uint)decoded[pair * 2] | ((uint)decoded[pair * 2 + 1] << 16); \
            } \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        _Pragma(UNROLL_PARTS) \
        for (int part = 0; part < KSTEP / 16; ++part) { \
            short8 ah = (short8)(0), al = (short8)(0); \
            if (row < rows) { \
                const size_t offset = ROW_MAJOR \
                    ? ((size_t)(row / 8) * (input_width / 16) + base / 16 + part) * 128 \
                    : ((size_t)(base / 16 + part) * row_blocks + row / 8) * 128; \
                ah = q35p_bslm_staged_input(xhigh, offset, lane); \
                al = q35p_bslm_staged_input(xlow, offset, lane); \
            } \
            _Pragma("unroll") \
            for (int tile = 0; tile < 4; ++tile) { \
                const int8 operand = q35p_xmx_panel(grids + (tile * (KSTEP / 16) + part) * 128, lane); \
                float8 partial = intel_sub_group_f16_f16_matrix_mad_k16(ah, operand, (float8)(0)); \
                partial = intel_sub_group_f16_f16_matrix_mad_k16(al, operand, partial); \
                accum[tile] = fma(partial, (float8)(coefficients[part][tile * 16 + lane]), accum[tile]); \
            } \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    _Pragma("unroll") \
    for (int tile = 0; tile < 4; ++tile) { \
        const int col = col_base + tile * 16 + lane; \
        _Pragma("unroll") \
        for (int r = 0; r < 8; ++r) if (row + r < rows && col < output_width) \
            output[(size_t)(row + r) * output_width + col] = accum[tile][r] + bias[col]; \
    } \
}
Q35P_GGUF_BSLM_STAGED(q35l_prefill_xmx_iq2_s_gguf_bslm_k64, 64, 0, 0, "unroll")
Q35P_GGUF_BSLM_STAGED(q35l_prefill_xmx_iq2_s_gguf_bslm_k128, 128, 0, 0, "unroll")
// Adjacent decode lanes write adjacent columns in each SLM operand. This
// changes only which thread decodes an octet, leaving all values and the
// ordered K16 arithmetic above identical to the corresponding stage width.
Q35P_GGUF_BSLM_STAGED(q35l_prefill_xmx_iq2_s_gguf_bslm_k32c, 32, 1, 0, "unroll")
Q35P_GGUF_BSLM_STAGED(q35l_prefill_xmx_iq2_s_gguf_bslm_k64c, 64, 1, 0, "unroll")
Q35P_GGUF_BSLM_STAGED(q35l_prefill_xmx_iq2_s_gguf_bslm_k128c, 128, 1, 0, "unroll")
// Layout-only variants use the rowmajor pack above. The no-unroll variant
// independently limits live A/B operand ranges without changing reduction order.
Q35P_GGUF_BSLM_STAGED(q35l_prefill_xmx_iq2_s_gguf_bslm_k32r, 32, 0, 1, "unroll")
Q35P_GGUF_BSLM_STAGED(q35l_prefill_xmx_iq2_s_gguf_bslm_k64r, 64, 0, 1, "unroll")
Q35P_GGUF_BSLM_STAGED(q35l_prefill_xmx_iq2_s_gguf_bslm_k64n, 64, 0, 0, "unroll 1")
#undef Q35P_GGUF_BSLM_STAGED
#endif // ARC_OPTIMIZATION_PROBES || ARC_Q35_GGUF_BSLM

// Decode each original octet once into a temporary XMX panel. The temporary
// belongs to one projection call and is released before the next matrix;
// residency of the quantized 27B base model is unchanged.
#define Q35P_XMX_PACK(NAME, BYTES, DECODE) \
__kernel void NAME(__global const uchar* input, __global uint* high, __global uint* low, \
    int input_width, int output_width) { \
    const int id = get_global_id(0), octets = input_width / 8; \
    if (id >= output_width * octets) return; \
    const int col = id / octets, octet = id % octets, col_blocks = (output_width + 15) / 16; \
    __global const uchar* block = input + ((size_t)col * (input_width / 256) + octet / 32) * BYTES; \
    const float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8)); \
    const float8 values = DECODE(block, octet % 32, d); \
    const half8 hi = convert_half8_rte(values), lo = convert_half8_rte(values - convert_float8(hi)); \
    const ushort8 high_bits = as_ushort8(hi), low_bits = as_ushort8(lo); \
    _Pragma("unroll") \
    for (int pair = 0; pair < 4; ++pair) { \
        const int kpair = octet * 4 + pair; \
        const size_t dst = ((size_t)(kpair / 8) * col_blocks + col / 16) * 128 + (kpair % 8) * 16 + col % 16; \
        high[dst] = (uint)high_bits[pair * 2] | ((uint)high_bits[pair * 2 + 1] << 16); \
        low[dst] = (uint)low_bits[pair * 2] | ((uint)low_bits[pair * 2 + 1] << 16); \
    } \
}
Q35P_XMX_PACK(q35l_prefill_xmx_pack_iq2_s_f16x2, 82, q35l_prefill_iq2_octet)
Q35P_XMX_PACK(q35l_prefill_xmx_pack_iq3_s_f16x2, 110, q35l_prefill_iq3_octet)
Q35P_XMX_PACK(q35l_prefill_xmx_pack_q4_k_f16x2, 144, q35p_xmx_q4_octet)
#undef Q35P_XMX_PACK

#define Q35P_XMX_PACK_FACTORED(NAME, BYTES, GRID, SCALE) \
__kernel void NAME(__global const uchar* input, __global uint* grid, \
    __global float* scales, int input_width, int output_width) { \
    const int id = get_global_id(0), octets = input_width / 8; \
    if (id >= output_width * octets) return; \
    const int col = id / octets, octet = id % octets, col_blocks = (output_width + 15) / 16; \
    __global const uchar* block = input + ((size_t)col * (input_width / 256) + octet / 32) * BYTES; \
    const float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8)); \
    const ushort8 values = as_ushort8(convert_half8_rte(GRID(block, octet % 32))); \
    if ((octet & 1) == 0) scales[((size_t)(octet / 2) * col_blocks + col / 16) * 16 + col % 16] = SCALE(block, octet % 32, d); \
    _Pragma("unroll") \
    for (int pair = 0; pair < 4; ++pair) { \
        const int kpair = octet * 4 + pair; \
        const size_t dst = ((size_t)(kpair / 8) * col_blocks + col / 16) * 128 + (kpair % 8) * 16 + col % 16; \
        grid[dst] = (uint)values[pair * 2] | ((uint)values[pair * 2 + 1] << 16); \
    } \
}
Q35P_XMX_PACK_FACTORED(q35l_prefill_xmx_pack_iq2_s_factored, 82, q35p_iq2_integer_grid, q35p_iq2_integer_scale)
Q35P_XMX_PACK_FACTORED(q35l_prefill_xmx_pack_iq3_s_factored, 110, q35p_iq3_integer_grid, q35p_iq3_integer_scale)
#undef Q35P_XMX_PACK_FACTORED

// IQ grids contain signed exact integers in [-43,43]. An int8 panel halves
// weight traffic; converting the register operand to F16 loses no information.
#ifdef ARC_OPTIMIZATION_PROBES
#define Q35P_XMX_PACK_FACTORED_I8(NAME, BYTES, GRID, SCALE) \
__kernel void NAME(__global const uchar* input, __global uint* grid, \
    __global float* scales, int input_width, int output_width) { \
    const int id = get_global_id(0), octets = input_width / 8; \
    if (id >= output_width * octets) return; \
    const int col = id / octets, octet = id % octets, col_blocks = (output_width + 15) / 16; \
    __global const uchar* block = input + ((size_t)col * (input_width / 256) + octet / 32) * BYTES; \
    const float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8)); \
    const char8 values = convert_char8(GRID(block, octet % 32)); \
    if ((octet & 1) == 0) scales[((size_t)(octet / 2) * col_blocks + col / 16) * 16 + col % 16] = SCALE(block, octet % 32, d); \
    const int quad = octet * 2; \
    const size_t dst = ((size_t)(quad / 4) * col_blocks + col / 16) * 64 + (quad % 4) * 16 + col % 16; \
    grid[dst] = as_uint(values.lo); grid[dst + 16] = as_uint(values.hi); \
}
Q35P_XMX_PACK_FACTORED_I8(q35l_prefill_xmx_pack_iq2_s_factored_i8, 82, q35p_iq2_integer_grid, q35p_iq2_integer_scale)
Q35P_XMX_PACK_FACTORED_I8(q35l_prefill_xmx_pack_iq3_s_factored_i8, 110, q35p_iq3_integer_grid, q35p_iq3_integer_scale)
#undef Q35P_XMX_PACK_FACTORED_I8
#endif

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void q35l_prefill_xmx_packed_factored(__global const ushort* xhigh,
    __global const ushort* xlow, __global const uint* grid, __global const float* scales,
    __global const float* bias, __global float* output, int rows, int input_width, int output_width,
    __global const float* input, __global const uchar* packed, __global const int* range_status, int quantization) {
    const int lane = get_local_id(0), subgroup = get_local_id(1);
    const int row = get_group_id(1) * 128 + subgroup * 8, col_base = get_group_id(0) * 64;
    if (range_status[0] != 0) {
        const int row_base = get_group_id(1) * 128;
        if (quantization == 0) q35p_xmx_fallback_iq2(input, packed, bias, output, rows, input_width, output_width, row_base, col_base);
        else q35p_xmx_fallback_iq3(input, packed, bias, output, rows, input_width, output_width, row_base, col_base);
        return;
    }
    const int row_blocks = (rows + 7) / 8, col_blocks = (output_width + 15) / 16;
    float8 accum[4] = {(float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f)};
    for (int block = 0; block < input_width / 16; ++block) {
        short8 ah = (short8)(0), al = (short8)(0);
        if (row < rows) {
            const size_t offset = ((size_t)block * row_blocks + row / 8) * 128;
#ifdef cl_intel_subgroups_short
            ah = as_short8(intel_sub_group_block_read_us8(xhigh + offset));
            al = as_short8(intel_sub_group_block_read_us8(xlow + offset));
#else
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                ah[r] = as_short(xhigh[offset + r * 16 + lane]);
                al[r] = as_short(xlow[offset + r * 16 + lane]);
            }
#endif
        }
        #pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
            const int col_block = col_base / 16 + tile;
            int8 b = (int8)(0); float coefficient = 0.0f;
            if (col_block < col_blocks) {
                const size_t offset = ((size_t)block * col_blocks + col_block) * 128;
                b = as_int8(intel_sub_group_block_read8(grid + offset));
                if (col_block * 16 + lane < output_width)
                    coefficient = scales[((size_t)block * col_blocks + col_block) * 16 + lane];
            }
            float8 partial = intel_sub_group_f16_f16_matrix_mad_k16(ah, b, (float8)(0.0f));
            partial = intel_sub_group_f16_f16_matrix_mad_k16(al, b, partial);
            accum[tile] = fma(partial, (float8)(coefficient), accum[tile]);
        }
    }
    #pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        const int col = col_base + tile * 16 + lane;
        #pragma unroll
        for (int r = 0; r < 8; ++r) if (row + r < rows && col < output_width)
            output[(size_t)(row + r) * output_width + col] = accum[tile][r] + bias[col];
    }
}

#ifdef ARC_OPTIMIZATION_PROBES
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void q35l_prefill_xmx_packed_factored_i8(__global const ushort* xhigh,
    __global const ushort* xlow, __global const uint* grid, __global const float* scales,
    __global const float* bias, __global float* output, int rows, int input_width, int output_width,
    __global const float* input, __global const uchar* packed, __global const int* range_status, int quantization) {
    const int lane = get_local_id(0), subgroup = get_local_id(1);
    const int row = get_group_id(1) * 128 + subgroup * 8, col_base = get_group_id(0) * 64;
    if (range_status[0] != 0) {
        const int row_base = get_group_id(1) * 128;
        if (quantization == 0) q35p_xmx_fallback_iq2(input, packed, bias, output, rows, input_width, output_width, row_base, col_base);
        else q35p_xmx_fallback_iq3(input, packed, bias, output, rows, input_width, output_width, row_base, col_base);
        return;
    }
    const int row_blocks = (rows + 7) / 8, col_blocks = (output_width + 15) / 16;
    float8 accum[4] = {(float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f)};
    for (int block = 0; block < input_width / 16; ++block) {
        short8 ah = (short8)(0), al = (short8)(0);
        if (row < rows) {
            const size_t offset = ((size_t)block * row_blocks + row / 8) * 128;
#ifdef cl_intel_subgroups_short
            ah = as_short8(intel_sub_group_block_read_us8(xhigh + offset));
            al = as_short8(intel_sub_group_block_read_us8(xlow + offset));
#else
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                ah[r] = as_short(xhigh[offset + r * 16 + lane]);
                al[r] = as_short(xlow[offset + r * 16 + lane]);
            }
#endif
        }
        #pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
            const int col_block = col_base / 16 + tile;
            int8 b = (int8)(0); float coefficient = 0.0f;
            if (col_block < col_blocks) {
                const size_t offset = ((size_t)block * col_blocks + col_block) * 64;
                b = as_int8(convert_half16_rte(as_char16(intel_sub_group_block_read4(grid + offset))));
                if (col_block * 16 + lane < output_width)
                    coefficient = scales[((size_t)block * col_blocks + col_block) * 16 + lane];
            }
            float8 partial = intel_sub_group_f16_f16_matrix_mad_k16(ah, b, (float8)(0.0f));
            partial = intel_sub_group_f16_f16_matrix_mad_k16(al, b, partial);
            accum[tile] = fma(partial, (float8)(coefficient), accum[tile]);
        }
    }
    #pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        const int col = col_base + tile * 16 + lane;
        #pragma unroll
        for (int r = 0; r < 8; ++r) if (row + r < rows && col < output_width)
            output[(size_t)(row + r) * output_width + col] = accum[tile][r] + bias[col];
    }
}
#endif // ARC_OPTIMIZATION_PROBES

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void q35l_prefill_xmx_packed_f16x2(__global const ushort* xhigh,
    __global const ushort* xlow, __global const uint* whigh, __global const uint* wlow,
    __global const float* bias, __global float* output, int rows, int input_width, int output_width,
    __global const float* input, __global const uchar* packed, __global const int* range_status, int quantization) {
    const int lane = get_local_id(0), subgroup = get_local_id(1);
    const int row = get_group_id(1) * 128 + subgroup * 8, col_base = get_group_id(0) * 64;
    if (range_status[0] != 0) {
        const int row_base = get_group_id(1) * 128;
        if (quantization == 0) q35p_xmx_fallback_iq2(input, packed, bias, output, rows, input_width, output_width, row_base, col_base);
        else if (quantization == 1) q35p_xmx_fallback_iq3(input, packed, bias, output, rows, input_width, output_width, row_base, col_base);
        else q35p_xmx_fallback_q4(input, packed, bias, output, rows, input_width, output_width, row_base, col_base);
        return;
    }
    const int row_blocks = (rows + 7) / 8, col_blocks = (output_width + 15) / 16;
    float8 accum[4] = {(float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f)};
    for (int block = 0; block < input_width / 16; ++block) {
        short8 ah = (short8)(0), al = (short8)(0);
        if (row < rows) {
            const size_t offset = ((size_t)block * row_blocks + row / 8) * 128;
#ifdef cl_intel_subgroups_short
            ah = as_short8(intel_sub_group_block_read_us8(xhigh + offset));
            al = as_short8(intel_sub_group_block_read_us8(xlow + offset));
#else
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                ah[r] = as_short(xhigh[offset + r * 16 + lane]);
                al[r] = as_short(xlow[offset + r * 16 + lane]);
            }
#endif
        }
        #pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
            const int col_block = col_base / 16 + tile;
            int8 bh = (int8)(0), bl = (int8)(0);
            if (col_block < col_blocks) {
                const size_t offset = ((size_t)block * col_blocks + col_block) * 128;
                bh = as_int8(intel_sub_group_block_read8(whigh + offset));
                bl = as_int8(intel_sub_group_block_read8(wlow + offset));
            }
            accum[tile] = intel_sub_group_f16_f16_matrix_mad_k16(ah, bh, accum[tile]);
            accum[tile] = intel_sub_group_f16_f16_matrix_mad_k16(al, bh, accum[tile]);
            accum[tile] = intel_sub_group_f16_f16_matrix_mad_k16(ah, bl, accum[tile]);
            accum[tile] = intel_sub_group_f16_f16_matrix_mad_k16(al, bl, accum[tile]);
        }
    }
    #pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        const int col = col_base + tile * 16 + lane;
        #pragma unroll
        for (int r = 0; r < 8; ++r) if (row + r < rows && col < output_width)
            output[(size_t)(row + r) * output_width + col] = accum[tile][r] + bias[col];
    }
}

#if defined(ARC_OPTIMIZATION_PROBES) || defined(ARC_Q35_RESIDENT_IQ2)
// Exact IQ2 recoding: 3-bit signed code per weight, original scale nibbles
// and F16 block coefficient. 106 bytes/256 weights instead of GGUF's82.
// K32 panels hold three uint planes across16 output columns; no projection
// needs to expand a resident grid into a temporary full-width F16 matrix.
__kernel void q35l_prefill_xmx_pack_iq2_s_grid3(__global const uchar* packed,
    __global uint* grid, __global uchar* scales, __global ushort* block_d,
    int input_width, int output_width) {
    const int id = get_global_id(0), kblocks = input_width / 32;
    if (id >= output_width * kblocks) return;
    const int col = id / kblocks, kb = id % kblocks, col_blocks = (output_width + 15) / 16;
    __global const uchar* block = packed + ((size_t)col * (input_width / 256) + kb / 8) *82;
    uint3 words = (uint3)(0);
    #pragma unroll
    for (int octet_part = 0; octet_part <4; ++octet_part) {
        float8 values = q35p_iq2_integer_grid(block, (kb %8) *4 + octet_part);
        #pragma unroll
        for (int element = 0; element <8; ++element) {
            const int magnitude = (int)fabs(values[element]);
            const uint code = (magnitude ==8 ?0u : magnitude ==25 ?1u :2u) | (values[element] <0 ?4u :0u);
            const int bit = (octet_part *8 + element) *3, word = bit /32, shift = bit %32;
            words[word] |= code << shift;
            if (shift >29) words[word +1] |= code >> (32 - shift);
        }
    }
    const size_t panel = ((size_t)kb * col_blocks + col /16) *48 + col %16;
    grid[panel] = words.x; grid[panel +16] = words.y; grid[panel +32] = words.z;
    scales[((size_t)kb * col_blocks + col /16) *16 + col %16] = block[74 + kb %8];
    if ((kb %8) ==0)
        block_d[((size_t)(kb /8) * col_blocks + col /16) *16 + col %16] = (ushort)block[0] | ((ushort)block[1] <<8);
}

inline ushort q35p_grid3_half(uint3 words, int element) {
    const int bit = element *3, word = bit /32, shift = bit %32;
    uint code = words[word] >> shift;
    if (shift >29) code |= words[word +1] << (32 - shift);
    code &=7u;
    const ushort magnitude = (code &3u) ==0 ? (ushort)0x4800 : (code &3u) ==1 ? (ushort)0x4e40 : (ushort)0x5160;
    return magnitude | (ushort)((code &4u) <<13);
}
inline int8 q35p_grid3_operand(uint3 words, int half_index) {
    ushort16 bits;
    #pragma unroll
    for (int element =0; element <16; ++element) bits[element] = q35p_grid3_half(words, half_index *16 + element);
    return as_int8(bits);
}
inline float8 q35p_grid3_decode_octet(__global const uint* grid, __global const uchar* scales,
    __global const ushort* block_d, int input_width, int col_blocks, int col, int block_index, int octet) {
    const int kb = block_index *8 + octet /4;
    const size_t panel = ((size_t)kb * col_blocks + col /16) *48 + col %16;
    const uint3 words = (uint3)(grid[panel], grid[panel +16], grid[panel +32]);
    const int scale = (scales[((size_t)kb * col_blocks + col /16) *16 + col %16] >> ((octet %4) /2 *4)) &15;
    const float d = q35l_half_to_float(block_d[((size_t)block_index * col_blocks + col /16) *16 + col %16]);
    const float coefficient = d * (0.5f + (float)scale) *0.25f;
    ushort8 bits;
    #pragma unroll
    for (int element =0; element <8; ++element) bits[element] = q35p_grid3_half(words, (octet %4) *8 + element);
    return coefficient * convert_float8(as_half8(bits));
}
inline float q35p_grid3_dot(__global const float* input, __global const uint* grid,
    __global const uchar* scales, __global const ushort* block_d, int input_width, int col_blocks, int col) {
    const int lane = get_sub_group_local_id();
    float8 sums = (float8)(0.0f);
    for (int block =0; block <input_width /256; ++block) {
        #pragma unroll
        for (int half_index =0; half_index <2; ++half_index) {
            const int octet = half_index *16 + lane;
            const float8 decoded = q35p_grid3_decode_octet(grid, scales, block_d, input_width, col_blocks, col, block, octet);
            sums = fma(vload8(0, input + block *256 + octet *8), decoded, sums);
        }
    }
    return sub_group_reduce_add(dot(sums.lo, (float4)(1.0f)) + dot(sums.hi, (float4)(1.0f)));
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(Q35_PROJECTION_WG,1,1)))
__kernel void q35l_iq2_s_grid3_sg16(__global const float* input, __global const uint* grid,
    __global const uchar* scales, __global const ushort* block_d, __global const float* bias,
    __global float* output, int rows, int input_width, int output_width) {
    const int flat = get_group_id(0) *(Q35_PROJECTION_WG /16) + get_sub_group_id();
    if (flat >=rows *output_width) return;
    const int row = flat /output_width, col = flat %output_width;
    const float total = q35p_grid3_dot(input + (size_t)row *input_width, grid, scales, block_d,
        input_width, (output_width +15) /16, col);
    if (get_sub_group_local_id() ==0) output[flat] = bias[col] + total;
}
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(Q35_PROJECTION_WG,1,1)))
__kernel void q35l_iq2_s_grid3_sg16_lora(__global const float* input, __global const uint* grid,
    __global const uchar* scales, __global const ushort* block_d, __global const float* bias,
    __global float* output, int rows, int input_width, int output_width,
    __global const float* z, __global const float* b, int rank, float scale) {
    const int flat = get_group_id(0) *(Q35_PROJECTION_WG /16) + get_sub_group_id();
    if (flat >=rows *output_width) return;
    const int row = flat /output_width, col = flat %output_width;
    const float total = q35p_grid3_dot(input + (size_t)row *input_width, grid, scales, block_d,
        input_width, (output_width +15) /16, col);
    if (get_sub_group_local_id() ==0) {
        const float base = bias[col] + total;
        float sum =0.0f;
        for (int r =0; r <rank; ++r) sum = fma(z[row *rank +r], b[col *rank +r], sum);
        output[flat] = base + scale *sum;
    }
}

inline ushort8 q35p_grid3_octet_bits(uint3 words, int part) {
    const uint packed =part ==0 ?words.x :part ==1 ?(words.x >>24) | (words.y <<8)
        :part ==2 ?(words.y >>16) | (words.z <<16) :words.z >>8;
    const ushort8 code =convert_ushort8(((uint8)(packed) >> (uint8)(0,3,6,9,12,15,18,21)) & (uint8)(7));
    ushort8 magnitude =select((ushort8)(0x5160), (ushort8)(0x4e40), as_ushort8((code & (ushort8)(3)) ==(ushort8)(1)));
    magnitude =select(magnitude, (ushort8)(0x4800), as_ushort8((code & (ushort8)(3)) ==(ushort8)(0)));
    return magnitude | ((code & (ushort8)(4)) <<13);
}
inline float8 q35p_grid3_decode_octet_fast(__global const uint* grid, __global const uchar* scales,
    __global const ushort* block_d, int col_blocks, int col, int block_index, int octet) {
    const int kb =block_index *8 +octet /4;
    const size_t panel =((size_t)kb *col_blocks +col /16) *48 +col %16;
    const uint3 words =(uint3)(grid[panel], grid[panel +16], grid[panel +32]);
    const int scale =(scales[((size_t)kb *col_blocks +col /16) *16 +col %16] >> ((octet %4) /2 *4)) &15;
    const float d =q35l_half_to_float(block_d[((size_t)block_index *col_blocks +col /16) *16 +col %16]);
    const float coefficient =d *(0.5f +(float)scale) *0.25f;
    return coefficient *convert_float8(as_half8(q35p_grid3_octet_bits(words, octet %4)));
}
inline float q35p_grid3_dot_fast(__global const float* input, __global const uint* grid,
    __global const uchar* scales, __global const ushort* block_d, int input_width, int col_blocks, int col) {
    const int lane = get_sub_group_local_id();
    float8 sums = (float8)(0.0f);
    for (int block =0; block <input_width /256; ++block) {
        #pragma unroll
        for (int half_index =0; half_index <2; ++half_index) {
            const int octet = half_index *16 + lane;
            const float8 decoded = q35p_grid3_decode_octet_fast(grid, scales, block_d, col_blocks, col, block, octet);
            sums = fma(vload8(0, input + block *256 + octet *8), decoded, sums);
        }
    }
    return sub_group_reduce_add(dot(sums.lo, (float4)(1.0f)) + dot(sums.hi, (float4)(1.0f)));
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(Q35_PROJECTION_WG,1,1)))
__kernel void q35l_iq2_s_grid3_sg16_fast(__global const float* input, __global const uint* grid,
    __global const uchar* scales, __global const ushort* block_d, __global const float* bias,
    __global float* output, int rows, int input_width, int output_width) {
    const int flat = get_group_id(0) *(Q35_PROJECTION_WG /16) + get_sub_group_id();
    if (flat >=rows *output_width) return;
    const int row = flat /output_width, col = flat %output_width;
    const float total = q35p_grid3_dot_fast(input + (size_t)row *input_width, grid, scales, block_d,
        input_width, (output_width +15) /16, col);
    if (get_sub_group_local_id() ==0) output[flat] = bias[col] + total;
}
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(Q35_PROJECTION_WG,1,1)))
__kernel void q35l_iq2_s_grid3_sg16_fast_lora(__global const float* input, __global const uint* grid,
    __global const uchar* scales, __global const ushort* block_d, __global const float* bias,
    __global float* output, int rows, int input_width, int output_width,
    __global const float* z, __global const float* b, int rank, float scale) {
    const int flat = get_group_id(0) *(Q35_PROJECTION_WG /16) + get_sub_group_id();
    if (flat >=rows *output_width) return;
    const int row = flat /output_width, col = flat %output_width;
    const float total = q35p_grid3_dot_fast(input + (size_t)row *input_width, grid, scales, block_d,
        input_width, (output_width +15) /16, col);
    if (get_sub_group_local_id() ==0) {
        const float base = bias[col] + total;
        float sum =0.0f;
        for (int r =0; r <rank; ++r) sum = fma(z[row *rank +r], b[col *rank +r], sum);
        output[flat] = base + scale *sum;
    }
}

inline float2 q35p_grid3_pair_dot(__global const float* input, __global const uint* grid,
    __global const uchar* scales, __global const ushort* block_d, int input_width, int col_blocks, int col, int output_width) {
    const int lane =get_sub_group_local_id();
    float8 sums0 =(float8)(0), sums1 =(float8)(0);
    for (int block =0; block <input_width /256; ++block) {
        #pragma unroll
        for (int half_index =0; half_index <2; ++half_index) {
            const int octet =half_index *16 +lane;
            const float8 x =vload8(0, input +block *256 +octet *8);
            const float8 w0 =q35p_grid3_decode_octet_fast(grid, scales, block_d, col_blocks, col, block, octet);
            const float8 w1 =col +1 <output_width
                ?q35p_grid3_decode_octet_fast(grid, scales, block_d, col_blocks, col +1, block, octet) :(float8)(0);
            sums0 =fma(x, w0, sums0); sums1 =fma(x, w1, sums1);
        }
    }
    const float total0 =sub_group_reduce_add(dot(sums0.lo, (float4)(1.0f)) +dot(sums0.hi, (float4)(1.0f)));
    const float total1 =sub_group_reduce_add(dot(sums1.lo, (float4)(1.0f)) +dot(sums1.hi, (float4)(1.0f)));
    return (float2)(total0, total1);
}
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(32,1,1)))
__kernel void q35l_iq2_s_grid3_sg16_pair(__global const float* input, __global const uint* grid,
    __global const uchar* scales, __global const ushort* block_d, __global const float* bias,
    __global float* output, int rows, int input_width, int output_width) {
    const int flat =get_group_id(0) *2 +get_sub_group_id(), pairs =(output_width +1) /2;
    if (flat >=rows *pairs) return;
    const int row =flat /pairs, col =flat %pairs *2;
    const float2 total =q35p_grid3_pair_dot(input +(size_t)row *input_width, grid, scales, block_d,
        input_width, (output_width +15) /16, col, output_width);
    if (get_sub_group_local_id() ==0) {
        output[(size_t)row *output_width +col] =bias[col] +total.x;
        if (col +1 <output_width) output[(size_t)row *output_width +col +1] =bias[col +1] +total.y;
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(32,1,1)))
__kernel void q35l_iq2_s_grid3_sg16_pair_lora(__global const float* input, __global const uint* grid,
    __global const uchar* scales, __global const ushort* block_d, __global const float* bias,
    __global float* output, int rows, int input_width, int output_width,
    __global const float* z, __global const float* b, int rank, float scale) {
    const int flat =get_group_id(0) *2 +get_sub_group_id(), pairs =(output_width +1) /2;
    if (flat >=rows *pairs) return;
    const int row =flat /pairs, col =flat %pairs *2;
    const float2 total =q35p_grid3_pair_dot(input +(size_t)row *input_width, grid, scales, block_d,
        input_width, (output_width +15) /16, col, output_width);
    if (get_sub_group_local_id() ==0) {
        float sum0 =0.0f, sum1 =0.0f;
        for (int r =0; r <rank; ++r) {
            const float value =z[row *rank +r];
            sum0 =fma(value, b[col *rank +r], sum0);
            if (col +1 <output_width) sum1 =fma(value, b[(col +1) *rank +r], sum1);
        }
        output[(size_t)row *output_width +col] =(bias[col] +total.x) +scale *sum0;
        if (col +1 <output_width) output[(size_t)row *output_width +col +1] =(bias[col +1] +total.y) +scale *sum1;
    }
}

#ifdef ARC_OPTIMIZATION_PROBES
// Coalesce resident weight reads across16 output columns, then transpose the
//16 original K-lane partials in SLM before the unchanged SG16 reduction.
#define Q35P_GRID3_COALESCED_BODY \
    const int lane =get_local_id(0), k_lane =get_local_id(1); \
    const int col_base =get_group_id(0) *16, row =get_group_id(1), col =col_base +lane; \
    const int col_blocks =(output_width +15) /16; \
    float8 sums =(float8)(0); \
    if (col <output_width) { \
        for (int block =0; block <input_width /256; ++block) { \
            _Pragma("unroll") \
            for (int half_index =0; half_index <2; ++half_index) { \
                const int octet =half_index *16 +k_lane; \
                const float8 decoded =q35p_grid3_decode_octet(grid, scales, block_d, input_width, col_blocks, col, block, octet); \
                sums =fma(vload8(0, input +(size_t)row *input_width +block *256 +octet *8), decoded, sums); \
            } \
        } \
    } \
    __local float partials[16][17]; \
    partials[lane][k_lane] =dot(sums.lo, (float4)(1.0f)) +dot(sums.hi, (float4)(1.0f)); \
    barrier(CLK_LOCAL_MEM_FENCE); \
    const float total =sub_group_reduce_add(partials[k_lane][lane]); \
    const int output_col =col_base +k_lane;

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void q35l_iq2_s_grid3_coalesced(__global const float* input, __global const uint* grid,
    __global const uchar* scales, __global const ushort* block_d, __global const float* bias,
    __global float* output, int rows, int input_width, int output_width) {
    Q35P_GRID3_COALESCED_BODY
    if (lane ==0 && output_col <output_width) output[(size_t)row *output_width +output_col] =bias[output_col] +total;
}
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void q35l_iq2_s_grid3_coalesced_lora(__global const float* input, __global const uint* grid,
    __global const uchar* scales, __global const ushort* block_d, __global const float* bias,
    __global float* output, int rows, int input_width, int output_width,
    __global const float* z, __global const float* b, int rank, float scale) {
    Q35P_GRID3_COALESCED_BODY
    if (lane ==0 && output_col <output_width) {
        const float base =bias[output_col] +total;
        float sum =0.0f;
        for (int r =0; r <rank; ++r) sum =fma(z[row *rank +r], b[output_col *rank +r], sum);
        output[(size_t)row *output_width +output_col] =base +scale *sum;
    }
}
#undef Q35P_GRID3_COALESCED_BODY

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void q35l_prefill_xmx_iq2_s_grid3(__global const ushort* xhigh, __global const ushort* xlow,
    __global const uint* grid, __global const uchar* scales, __global const ushort* block_d,
    __global const float* bias, __global float* output, int rows, int input_width, int output_width,
    __global const float* input, __global const int* range_status) {
    const int lane =get_local_id(0), subgroup =get_local_id(1);
    const int row =get_group_id(1) *128 +subgroup *8, col_base =get_group_id(0) *64;
    const int row_blocks =(rows +7) /8, col_blocks =(output_width +15) /16;
    if (range_status[0] !=0) {
        const int row_base =get_group_id(1) *128;
        for (int item =subgroup; item <128 *64; item +=16) {
            const int r =row_base +item /64, col =col_base +item %64;
            if (r <rows && col <output_width) {
                const float total =q35p_grid3_dot(input +(size_t)r *input_width, grid, scales, block_d, input_width, col_blocks, col);
                if (lane ==0) output[(size_t)r *output_width +col] =bias[col] +total;
            }
        }
        return;
    }
    float8 accum[4] ={(float8)(0), (float8)(0), (float8)(0), (float8)(0)};
    for (int block =0; block <input_width /32; ++block) {
        #pragma unroll
        for (int half_index =0; half_index <2; ++half_index) {
            short8 ah =(short8)(0), al =(short8)(0);
            if (row <rows) {
                const size_t offset =((size_t)(block *2 +half_index) *row_blocks +row /8) *128;
#ifdef cl_intel_subgroups_short
                ah =as_short8(intel_sub_group_block_read_us8(xhigh +offset));
                al =as_short8(intel_sub_group_block_read_us8(xlow +offset));
#else
                #pragma unroll
                for (int r =0; r <8; ++r) {
                    ah[r] =as_short(xhigh[offset +r *16 +lane]);
                    al[r] =as_short(xlow[offset +r *16 +lane]);
                }
#endif
            }
            #pragma unroll
            for (int tile =0; tile <4; ++tile) {
                const int col_block =col_base /16 +tile;
                int8 operand =(int8)(0); float coefficient =0.0f;
                if (col_block <col_blocks) {
                    const size_t panel =((size_t)block *col_blocks +col_block) *48;
                    const uint2 first =intel_sub_group_block_read2(grid +panel);
                    const uint3 words =(uint3)(first.x, first.y, intel_sub_group_block_read(grid +panel +32));
                    operand =q35p_grid3_operand(words, half_index);
                    if (col_block *16 +lane <output_width) {
                        const int scale =(scales[((size_t)block *col_blocks +col_block) *16 +lane] >> (half_index *4)) &15;
                        const float d =q35l_half_to_float(block_d[((size_t)(block /8) *col_blocks +col_block) *16 +lane]);
                        coefficient =d *(0.5f +(float)scale) *0.25f;
                    }
                }
                float8 partial =intel_sub_group_f16_f16_matrix_mad_k16(ah, operand, (float8)(0));
                partial =intel_sub_group_f16_f16_matrix_mad_k16(al, operand, partial);
                accum[tile] =fma(partial, (float8)(coefficient), accum[tile]);
            }
        }
    }
    #pragma unroll
    for (int tile =0; tile <4; ++tile) {
        const int col =col_base +tile *16 +lane;
        #pragma unroll
        for (int r =0; r <8; ++r) if (row +r <rows && col <output_width)
            output[(size_t)(row +r) *output_width +col] =accum[tile][r] +bias[col];
    }
}
#endif // ARC_OPTIMIZATION_PROBES
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void q35l_prefill_xmx_iq2_s_grid3_bslm(__global const ushort* xhigh, __global const ushort* xlow,
    __global const uint* grid, __global const uchar* scales, __global const ushort* block_d,
    __global const float* bias, __global float* output, int rows, int input_width, int output_width,
    __global const float* input, __global const int* range_status) {
    const int lane =get_local_id(0), subgroup =get_local_id(1), tid =subgroup *16 +lane;
    const int row =get_group_id(1) *128 +subgroup *8, col_base =get_group_id(0) *64;
    const int row_blocks =(rows +7) /8, col_blocks =(output_width +15) /16;
    if (range_status[0] !=0) {
        const int row_base =get_group_id(1) *128;
        for (int item =subgroup; item <128 *64; item +=16) {
            const int r =row_base +item /64, col =col_base +item %64;
            if (r <rows && col <output_width) {
                const float total =q35p_grid3_dot(input +(size_t)r *input_width, grid, scales, block_d, input_width, col_blocks, col);
                if (lane ==0) output[(size_t)r *output_width +col] =bias[col] +total;
            }
        }
        return;
    }
    __local uint grids[1024];
    __local float coefficients[2][64];
    float8 accum[4] ={(float8)(0), (float8)(0), (float8)(0), (float8)(0)};
    for (int block =0; block <input_width /32; ++block) {
        const int col_inner =tid /4, col =col_base +col_inner, octet =tid %4;
        ushort8 decoded =(ushort8)(0); float coefficient =0.0f;
        if (col <output_width) {
            const size_t panel =((size_t)block *col_blocks +col /16) *48 +col %16;
            const uint3 words =(uint3)(grid[panel], grid[panel +16], grid[panel +32]);
            #pragma unroll
            for (int element =0; element <8; ++element) decoded[element] =q35p_grid3_half(words, octet *8 +element);
            const int scale =(scales[((size_t)block *col_blocks +col /16) *16 +col %16] >> (octet /2 *4)) &15;
            const float d =q35l_half_to_float(block_d[((size_t)(block /8) *col_blocks +col /16) *16 +col %16]);
            coefficient =d *(0.5f +(float)scale) *0.25f;
        }
        if ((tid &1) ==0) coefficients[(tid &3) /2][col_inner] =coefficient;
        #pragma unroll
        for (int pair =0; pair <4; ++pair) {
            const int kpair =octet *4 +pair;
            const int dst =((col_inner /16) *2 +kpair /8) *128 +(kpair %8) *16 +col_inner %16;
            grids[dst] =(uint)decoded[pair *2] | ((uint)decoded[pair *2 +1] <<16);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int half_index =0; half_index <2; ++half_index) {
            short8 ah =(short8)(0), al =(short8)(0);
            if (row <rows) {
                const size_t offset =((size_t)(block *2 +half_index) *row_blocks +row /8) *128;
#ifdef cl_intel_subgroups_short
                ah =as_short8(intel_sub_group_block_read_us8(xhigh +offset));
                al =as_short8(intel_sub_group_block_read_us8(xlow +offset));
#else
                #pragma unroll
                for (int r =0; r <8; ++r) {
                    ah[r] =as_short(xhigh[offset +r *16 +lane]);
                    al[r] =as_short(xlow[offset +r *16 +lane]);
                }
#endif
            }
            #pragma unroll
            for (int tile =0; tile <4; ++tile) {
                const int8 operand =q35p_xmx_panel(grids +(tile *2 +half_index) *128, lane);
                float8 partial =intel_sub_group_f16_f16_matrix_mad_k16(ah, operand, (float8)(0));
                partial =intel_sub_group_f16_f16_matrix_mad_k16(al, operand, partial);
                accum[tile] =fma(partial, (float8)(coefficients[half_index][tile *16 +lane]), accum[tile]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int tile =0; tile <4; ++tile) {
        const int col =col_base +tile *16 +lane;
        #pragma unroll
        for (int r =0; r <8; ++r) if (row +r <rows && col <output_width)
            output[(size_t)(row +r) *output_width +col] =accum[tile][r] +bias[col];
    }
}
#ifdef ARC_OPTIMIZATION_PROBES
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void q35l_prefill_xmx_packed_factored_bslm(__global const ushort* xhigh, __global const ushort* xlow,
    __global const uint* grid, __global const float* scales,
    __global const float* bias, __global float* output, int rows, int input_width, int output_width,
    __global const float* input, __global const uchar* packed, __global const int* range_status, int quantization) {
    const int lane =get_local_id(0), subgroup =get_local_id(1), tid =subgroup *16 +lane;
    const int row =get_group_id(1) *128 +subgroup *8, col_base =get_group_id(0) *64;
    const int row_blocks =(rows +7) /8, col_blocks =(output_width +15) /16;
    if (range_status[0] !=0) {
        const int row_base =get_group_id(1) *128;
        if (quantization ==0) q35p_xmx_fallback_iq2(input, packed, bias, output, rows, input_width, output_width, row_base, col_base);
        else q35p_xmx_fallback_iq3(input, packed, bias, output, rows, input_width, output_width, row_base, col_base);
        return;
    }
    __local uint grids[1024];
    __local float coefficients[2][64];
    float8 accum[4] ={(float8)(0), (float8)(0), (float8)(0), (float8)(0)};
    for (int block =0; block <input_width /32; ++block) {
        for (int i =tid; i <1024; i +=256) {
            const int col_block =col_base /16 +i /256, half_index =(i %256) /128;
            grids[i] =col_block <col_blocks
                ?grid[((size_t)(block *2 +half_index) *col_blocks +col_block) *128 +i %128] :0;
        }
        if (tid <128) {
            const int part =tid /64, col =col_base +tid %64;
            coefficients[part][tid %64] =col <output_width
                ?scales[((size_t)(block *2 +part) *col_blocks +col /16) *16 +col %16] :0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int half_index =0; half_index <2; ++half_index) {
            short8 ah =(short8)(0), al =(short8)(0);
            if (row <rows) {
                const size_t offset =((size_t)(block *2 +half_index) *row_blocks +row /8) *128;
                ah =as_short8(intel_sub_group_block_read_us8(xhigh +offset));
                al =as_short8(intel_sub_group_block_read_us8(xlow +offset));
            }
            #pragma unroll
            for (int tile =0; tile <4; ++tile) {
                const int8 operand =q35p_xmx_panel(grids +(tile *2 +half_index) *128, lane);
                float8 partial =intel_sub_group_f16_f16_matrix_mad_k16(ah, operand, (float8)(0));
                partial =intel_sub_group_f16_f16_matrix_mad_k16(al, operand, partial);
                accum[tile] =fma(partial, (float8)(coefficients[half_index][tile *16 +lane]), accum[tile]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int tile =0; tile <4; ++tile) {
        const int col =col_base +tile *16 +lane;
        #pragma unroll
        for (int r =0; r <8; ++r) if (row +r <rows && col <output_width)
            output[(size_t)(row +r) *output_width +col] =accum[tile][r] +bias[col];
    }
}
#endif // ARC_OPTIMIZATION_PROBES
#endif // ARC_OPTIMIZATION_PROBES || ARC_Q35_RESIDENT_IQ2
#endif
