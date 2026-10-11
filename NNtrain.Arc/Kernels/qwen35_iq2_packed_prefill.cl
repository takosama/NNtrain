// Packed82-native IQ2_S prefill. Storage is the same lossless three-plane
// layout as qwen35_iq2_packed_decode.cl. Only weight addresses/metadata loads
// differ from the existing rows and GGUF BSLM kernels; arithmetic order stays
// unchanged, including the FP32 out-of-F16-range fallback.
// Resources are concatenated alphabetically; XMX helpers are defined later.
#if defined(ARC_XMX) && ARC_SG == 16
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
inline int8 q35p_xmx_panel(const __local uint* panel, int lane);
inline short8 q35p_bslm_staged_input(__global const ushort* panel, size_t offset, int lane);
#define Q35PK_LINEAR_IQ_ROWS(NAME, ROWS) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(32, 1, 1))) \
__kernel void NAME(__global const float* x, __global const uchar* weight, \
    __global const float* bias, __global float* y, int rows, int input_width, int output_width) { \
    int lane = get_sub_group_local_id(), flat = get_group_id(0) * 2 + get_sub_group_id(); \
    int tiles = (rows + ROWS - 1) / ROWS; \
    if (flat >= tiles * output_width) return; \
    int row = (flat / output_width) * ROWS, output = flat % output_width, blocks = input_width >> 8; \
    const size_t totalBlocks = (size_t)blocks * output_width; \
    __global const ushort* codes = (__global const ushort*)weight; \
    __global const uchar* extras = weight + totalBlocks * 64; \
    __global const ushort* ds = (__global const ushort*)(weight + totalBlocks * 80); \
    const size_t firstBlock = (size_t)output * blocks; \
    __global const float* x0 = x + (size_t)row * input_width; \
    float8 s0 = (float8)(0.0f), s1 = (float8)(0.0f), s2 = (float8)(0.0f), s3 = (float8)(0.0f); \
    for (int b = 0; b < blocks; ++b) { \
        const size_t block_index = firstBlock + b; \
        float d = q35l_half_to_float(ds[block_index]); \
        _Pragma("unroll") \
        for (int halfIndex = 0; halfIndex < 2; ++halfIndex) { \
            int octet = halfIndex * 16 + lane, offset = b * 256 + octet * 8; \
            float8 decoded = q35packed82_octet(codes, extras, block_index, octet, d); \
            s0 = fma(vload8(0, x0 + offset), decoded, s0); \
            if (row + 1 < rows) s1 = fma(vload8(0, x0 + input_width + offset), decoded, s1); \
            if (ROWS > 2 && row + 2 < rows) s2 = fma(vload8(0, x0 + 2 * input_width + offset), decoded, s2); \
            if (ROWS > 2 && row + 3 < rows) s3 = fma(vload8(0, x0 + 3 * input_width + offset), decoded, s3); \
        } \
    } \
    float total0 = sub_group_reduce_add(dot(s0.lo, (float4)(1.0f)) + dot(s0.hi, (float4)(1.0f))); \
    float total1 = sub_group_reduce_add(dot(s1.lo, (float4)(1.0f)) + dot(s1.hi, (float4)(1.0f))); \
    if (lane == 0) { \
        y[row * output_width + output] = bias[output] + total0; \
        if (row + 1 < rows) y[(row + 1) * output_width + output] = bias[output] + total1; \
    } \
    if (ROWS > 2) { \
        float total2 = sub_group_reduce_add(dot(s2.lo, (float4)(1.0f)) + dot(s2.hi, (float4)(1.0f))); \
        float total3 = sub_group_reduce_add(dot(s3.lo, (float4)(1.0f)) + dot(s3.hi, (float4)(1.0f))); \
        if (lane == 0) { \
            if (row + 2 < rows) y[(row + 2) * output_width + output] = bias[output] + total2; \
            if (row + 3 < rows) y[(row + 3) * output_width + output] = bias[output] + total3; \
        } \
    } \
}

Q35PK_LINEAR_IQ_ROWS(q35l_prefill_linear_iq2_s_rows2_packed82, 2)
Q35PK_LINEAR_IQ_ROWS(q35l_prefill_linear_iq2_s_rows4_packed82, 4)
#undef Q35PK_LINEAR_IQ_ROWS

// Larger row tiles keep one independent accumulator per row. The macro lists
// expose fixed registers rather than dynamically indexing private arrays.
#define Q35PK_ROW_LIST8(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)
#define Q35PK_ROW_LIST16(M) Q35PK_ROW_LIST8(M) M(8) M(9) M(10) M(11) M(12) M(13) M(14) M(15)
#define Q35PK_IQ_DECLARE(N) float8 s##N = (float8)(0.0f);
#define Q35PK_IQ_ACCUMULATE(N) \
    if (row + N < rows) s##N = fma(vload8(0, x0 + (size_t)N * input_width + offset), decoded, s##N);
#define Q35PK_IQ_STORE(N) { \
    float total = sub_group_reduce_add(dot(s##N.lo, (float4)(1.0f)) + dot(s##N.hi, (float4)(1.0f))); \
    if (lane == 0 && row + N < rows) y[(size_t)(row + N) * output_width + output] = bias[output] + total; \
}

#define Q35PK_LINEAR_IQ_WIDE_ROWS(NAME, ROWS, APPLY) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(32, 1, 1))) \
__kernel void NAME(__global const float* x, __global const uchar* weight, \
    __global const float* bias, __global float* y, int rows, int input_width, int output_width) { \
    int lane = get_sub_group_local_id(), flat = get_group_id(0) * 2 + get_sub_group_id(); \
    int tiles = (rows + ROWS - 1) / ROWS; \
    if (flat >= tiles * output_width) return; \
    int row = (flat / output_width) * ROWS, output = flat % output_width, blocks = input_width >> 8; \
    const size_t totalBlocks = (size_t)blocks * output_width; \
    __global const ushort* codes = (__global const ushort*)weight; \
    __global const uchar* extras = weight + totalBlocks * 64; \
    __global const ushort* ds = (__global const ushort*)(weight + totalBlocks * 80); \
    const size_t firstBlock = (size_t)output * blocks; \
    __global const float* x0 = x + (size_t)row * input_width; \
    APPLY(Q35PK_IQ_DECLARE) \
    for (int b = 0; b < blocks; ++b) { \
        const size_t block_index = firstBlock + b; \
        float d = q35l_half_to_float(ds[block_index]); \
        _Pragma("unroll") \
        for (int halfIndex = 0; halfIndex < 2; ++halfIndex) { \
            int octet = halfIndex * 16 + lane, offset = b * 256 + octet * 8; \
            float8 decoded = q35packed82_octet(codes, extras, block_index, octet, d); \
            APPLY(Q35PK_IQ_ACCUMULATE) \
        } \
    } \
    APPLY(Q35PK_IQ_STORE) \
}
Q35PK_LINEAR_IQ_WIDE_ROWS(q35l_prefill_linear_iq2_s_rows8_packed82, 8, Q35PK_ROW_LIST8)
Q35PK_LINEAR_IQ_WIDE_ROWS(q35l_prefill_linear_iq2_s_rows16_packed82, 16, Q35PK_ROW_LIST16)
#undef Q35PK_LINEAR_IQ_WIDE_ROWS
#undef Q35PK_IQ_DECLARE
#undef Q35PK_IQ_ACCUMULATE
#undef Q35PK_IQ_STORE

#undef Q35PK_ROW_LIST8
#undef Q35PK_ROW_LIST16

#if defined(cl_khr_fp16) && (defined(ARC_OPTIMIZATION_PROBES) || defined(ARC_Q35_GGUF_BSLM))
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
inline float8 q35packed82_integer_grid(__global const ushort* codes,
    __global const uchar* extras, size_t block_index, int octet) {
    const uint code = codes[block_index * 32 + octet];
    const uint extra = extras[block_index * 16 + (octet >> 1)];
    const int index = (int)(code & 255) | (int)(((extra >> ((octet & 1) * 2)) & 3) << 8);
    float8 grid = convert_float8(as_uchar8(q35l_iq2s_grid[index]));
    const int signs = (int)(code >> 8);
    return select(grid, -grid, ((int8)(signs) & (int8)(1, 2, 4, 8, 16, 32, 64, 128)) != (int8)(0));
}
inline float q35packed82_integer_scale(__global const uchar* extras,
    size_t block_index, int octet, float d) {
    const int scale = extras[block_index * 16 + (octet >> 1)] >> 4;
    return d * (0.5f + (float)scale) * 0.25f;
}

inline void q35packed82_xmx_fallback(__global const float* x, __global const uchar* packed,
    __global const float* bias, __global float* output, int rows, int input_width,
    int output_width, int row_base, int col_base) {
    const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
    for (int item = sg; item < 128 * 64; item += 16) {
        const int row = row_base + item / 64, col = col_base + item % 64;
        if (row >= rows || col >= output_width) continue;
        const int blocks = input_width / 256;
        const size_t totalBlocks = (size_t)blocks * output_width;
        __global const ushort* codes = (__global const ushort*)packed;
        __global const uchar* extras = packed + totalBlocks * 64;
        __global const ushort* ds = (__global const ushort*)(packed + totalBlocks * 80);
        const size_t firstBlock = (size_t)col * blocks;
        __global const float* row_x = x + (size_t)row * input_width;
        float8 sum = (float8)(0.0f);
        for (int b = 0; b < blocks; ++b) {
            const size_t block_index = firstBlock + b;
            const float d = q35l_half_to_float(ds[block_index]);
            for (int part = 0; part < 2; ++part) {
                const int octet = part * 16 + lane;
                sum = fma(vload8(0, row_x + b * 256 + octet * 8), q35packed82_octet(codes, extras, block_index, octet, d), sum);
            }
        }
        const float total = sub_group_reduce_add(dot(sum.lo, (float4)(1.0f)) + dot(sum.hi, (float4)(1.0f)));
        if (lane == 0) output[(size_t)row * output_width + col] = bias[col] + total;
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void q35l_prefill_xmx_iq2_s_gguf_bslm_packed82(__global const ushort* xhigh, __global const ushort* xlow,
    __global const uchar* packed, __global const float* bias, __global float* output,
    int rows, int input_width, int output_width, __global const float* input, __global const int* range_status) {
    const int lane = get_local_id(0), subgroup = get_local_id(1), tid = subgroup * 16 + lane;
    const int row_base = get_group_id(1) * 128, row = row_base + subgroup * 8;
    const int col_base = get_group_id(0) * 64, row_blocks = (rows + 7) / 8;
    const int blocks = input_width / 256;
    const size_t totalBlocks = (size_t)blocks * output_width;
    __global const ushort* codes = (__global const ushort*)packed;
    __global const uchar* extras = packed + totalBlocks * 64;
    __global const ushort* ds = (__global const ushort*)(packed + totalBlocks * 80);
    if (range_status[0] != 0) {
        q35packed82_xmx_fallback(input, packed, bias, output, rows, input_width, output_width, row_base, col_base);
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
            const size_t block_index = (size_t)col * blocks + kblock / 8;
            const float d = q35l_half_to_float(ds[block_index]);
            decoded = as_ushort8(convert_half8_rte(q35packed82_integer_grid(codes, extras, block_index, octet)));
            coefficient = q35packed82_integer_scale(extras, block_index, octet, d);
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

#define Q35PK_BSLM_STAGED(NAME, KSTEP, COALESCED, ROW_MAJOR, UNROLL_PARTS) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(16,16,1))) \
__kernel void NAME(__global const ushort* xhigh, __global const ushort* xlow, \
    __global const uchar* packed, __global const float* bias, __global float* output, \
    int rows, int input_width, int output_width, __global const float* input, __global const int* range_status) { \
    const int lane = get_local_id(0), subgroup = get_local_id(1), tid = subgroup * 16 + lane; \
    const int row_base = get_group_id(1) * 128, row = row_base + subgroup * 8; \
    const int col_base = get_group_id(0) * 64, row_blocks = (rows + 7) / 8; \
    const int blocks = input_width / 256; \
    const size_t totalBlocks = (size_t)blocks * output_width; \
    __global const ushort* codes = (__global const ushort*)packed; \
    __global const uchar* extras = packed + totalBlocks * 64; \
    __global const ushort* ds = (__global const ushort*)(packed + totalBlocks * 80); \
    if (range_status[0] != 0) { \
        q35packed82_xmx_fallback(input, packed, bias, output, rows, input_width, output_width, row_base, col_base); \
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
                const size_t block_index = (size_t)col * blocks + base / 256; \
                const float d = q35l_half_to_float(ds[block_index]); \
                decoded = as_ushort8(convert_half8_rte(q35packed82_integer_grid(codes, extras, block_index, octet))); \
                coefficient = q35packed82_integer_scale(extras, block_index, octet, d); \
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
Q35PK_BSLM_STAGED(q35l_prefill_xmx_iq2_s_gguf_bslm_k32r_packed82, 32, 0, 1, "unroll")
#undef Q35PK_BSLM_STAGED

#endif
#endif
