// Inference-only alternatives to the 128x64/K32 XMX prefill panel.
// M64/K64 uses exactly 32 KiB of SLM so two workgroups can share a 64 KiB
// slice. M128/K64 is retained as an independent comparison (48 KiB SLM).
// The original quantized payload stays resident. All four high/residual
// products and their K16 accumulation order match the two-component kernel.
#if defined(ARC_XMX) && ARC_SG == 16 && defined(cl_khr_fp16) && defined(ARC_OPTIMIZATION_PROBES)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#ifdef ARC_SLM_BLOCK_IO
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable
#endif

inline int8 q35pt_xmx_panel_scalar(const __local uint* panel, int lane) {
    int8 result;
    #pragma unroll
    for (int i = 0; i < 8; ++i) result[i] = as_int(panel[i * 16 + lane]);
    return result;
}

inline int8 q35pt_xmx_panel_block(const __local uint* panel, int lane) {
#ifdef ARC_SLM_BLOCK_IO
    return as_int8(intel_sub_group_block_read8(panel));
#else
    return q35pt_xmx_panel_scalar(panel, lane);
#endif
}

inline short8 q35pt_xmx_a_rowmajor(const __local uint* panel, int bm,
    int subgroup, int part, int lane) {
    const __local short* halves = (const __local short*)panel;
    short8 result;
    #pragma unroll
    for (int r = 0; r < 8; ++r) result[r] = halves[(subgroup * 8 + r) * 64 + part * 16 + lane];
    return result;
}

// A DPAS lane owns eight rows at one K column. Pack two adjacent rows in
// each uint so one local block_read4 obtains the entire short8 operand.
inline short8 q35pt_xmx_a_block(const __local uint* panel, int bm,
    int subgroup, int part, int lane) {
    const __local uint* start = panel + (part * (bm / 8) + subgroup) * 64;
#ifdef ARC_SLM_BLOCK_IO
    return as_short8(intel_sub_group_block_read4(start));
#else
    uint4 result;
    #pragma unroll
    for (int r = 0; r < 4; ++r) result[r] = start[r * 16 + lane];
    return as_short8(result);
#endif
}

// The original fallback owns 128 rows. These alternatives must instead own
// precisely their workgroup's BM rows, including short final row/column tiles.
#define Q35PT_XMX_FALLBACK(NAME, BYTES, DECODE) \
inline void NAME(__global const float* x, __global const uchar* packed, \
    __global const float* bias, __global float* output, int rows, int input_width, \
    int output_width, int row_base, int col_base, int bm) { \
    const int lane = get_sub_group_local_id(), sg = get_sub_group_id(); \
    for (int item = sg; item < bm * 64; item += bm / 8) { \
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
Q35PT_XMX_FALLBACK(q35pt_xmx_fallback_iq2, 82, q35l_prefill_iq2_octet)
Q35PT_XMX_FALLBACK(q35pt_xmx_fallback_iq3, 110, q35l_prefill_iq3_octet)
#undef Q35PT_XMX_FALLBACK

inline void q35pt_xmx_fallback_q4(__global const float* x, __global const uchar* packed,
    __global const float* bias, __global float* output, int rows, int input_width,
    int output_width, int row_base, int col_base, int bm) {
    const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
    for (int item = sg; item < bm * 64; item += bm / 8) {
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

#define Q35PT_XMX_LINEAR(NAME, BM, PACK_A, LOAD_A, PANEL, BYTES, DECODE, FALLBACK) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(16, BM / 8, 1))) \
__kernel void NAME(__global const ushort* xhigh, __global const ushort* xlow, \
    __global const uchar* packed, __global const float* bias, __global float* output, \
    int rows, int input_width, int output_width, __global const float* input, __global const int* range_status) { \
    const int row_base = get_group_id(1) * BM, col_base = get_group_id(0) * 64; \
    if (range_status[0] != 0) { FALLBACK(input, packed, bias, output, rows, input_width, output_width, row_base, col_base, BM); return; } \
    const int lane = get_local_id(0), subgroup = get_local_id(1), tid = subgroup * 16 + lane; \
    __local uint ahigh[BM * 32], alow[BM * 32]; \
    __local uint bhigh[64 * 32], blow[64 * 32]; \
    float8 accum[4] = {(float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f)}; \
    for (int base = 0; base < input_width; base += 64) { \
        for (int i = tid; i < BM * 64; i += 16 * (BM / 8)) { \
            const int row = row_base + i / 64, k = base + i % 64; \
            const int rr = i / 64, kk = i % 64; \
            const int destination = PACK_A ? ((kk / 16) * (BM / 8) + rr / 8) * 128 \
                + ((rr % 8) / 2) * 32 + (kk % 16) * 2 + rr % 2 : i; \
            ((__local ushort*)ahigh)[destination] = row < rows ? xhigh[(size_t)row * input_width + k] : (ushort)0; \
            ((__local ushort*)alow)[destination] = row < rows ? xlow[(size_t)row * input_width + k] : (ushort)0; \
        } \
        for (int i = tid; i < 64 * 8; i += 16 * (BM / 8)) { \
            const int col_inner = i / 8, col = col_base + col_inner; \
            const int octet = base / 8 + i % 8; \
            float8 decoded = (float8)(0.0f); \
            if (col < output_width) { \
                __global const uchar* block = packed + ((size_t)col * (input_width / 256) + octet / 32) * BYTES; \
                const float scale = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8)); \
                decoded = DECODE(block, octet % 32, scale); \
            } \
            const half8 whigh = convert_half8_rte(decoded); \
            const half8 wlow = convert_half8_rte(decoded - convert_float8(whigh)); \
            const ushort8 hi = as_ushort8(whigh), lo = as_ushort8(wlow); \
            const int first_pair = (i % 8) * 4; \
            _Pragma("unroll") \
            for (int pair = 0; pair < 4; ++pair) { \
                const int p = first_pair + pair; \
                const int dst = ((col_inner / 16) * 4 + p / 8) * 128 + (p % 8) * 16 + col_inner % 16; \
                bhigh[dst] = (uint)hi[2 * pair] | ((uint)hi[2 * pair + 1] << 16); \
                blow[dst] = (uint)lo[2 * pair] | ((uint)lo[2 * pair + 1] << 16); \
            } \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        _Pragma("unroll") \
        for (int part = 0; part < 4; ++part) { \
            const short8 avh = LOAD_A(ahigh, BM, subgroup, part, lane); \
            const short8 avl = LOAD_A(alow, BM, subgroup, part, lane); \
            _Pragma("unroll") \
            for (int tile = 0; tile < 4; ++tile) { \
                const int8 bvh = PANEL(bhigh + (tile * 4 + part) * 128, lane); \
                const int8 bvl = PANEL(blow + (tile * 4 + part) * 128, lane); \
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

#define Q35PT_XMX_VARIANTS(QUANT, BYTES, DECODE, FALLBACK) \
Q35PT_XMX_LINEAR(q35l_prefill_xmx_##QUANT##_f16x2_m64_k64, 64, 0, q35pt_xmx_a_rowmajor, q35pt_xmx_panel_scalar, BYTES, DECODE, FALLBACK) \
Q35PT_XMX_LINEAR(q35l_prefill_xmx_##QUANT##_f16x2_m64_k64_slm, 64, 0, q35pt_xmx_a_rowmajor, q35pt_xmx_panel_block, BYTES, DECODE, FALLBACK) \
Q35PT_XMX_LINEAR(q35l_prefill_xmx_##QUANT##_f16x2_k64, 128, 0, q35pt_xmx_a_rowmajor, q35pt_xmx_panel_scalar, BYTES, DECODE, FALLBACK) \
Q35PT_XMX_LINEAR(q35l_prefill_xmx_##QUANT##_f16x2_k64_slm, 128, 0, q35pt_xmx_a_rowmajor, q35pt_xmx_panel_block, BYTES, DECODE, FALLBACK) \
Q35PT_XMX_LINEAR(q35l_prefill_xmx_##QUANT##_f16x2_m64_k64_ab_slm, 64, 1, q35pt_xmx_a_block, q35pt_xmx_panel_block, BYTES, DECODE, FALLBACK) \
Q35PT_XMX_LINEAR(q35l_prefill_xmx_##QUANT##_f16x2_k64_ab_slm, 128, 1, q35pt_xmx_a_block, q35pt_xmx_panel_block, BYTES, DECODE, FALLBACK)
Q35PT_XMX_VARIANTS(iq2_s, 82, q35l_prefill_iq2_octet, q35pt_xmx_fallback_iq2)
Q35PT_XMX_VARIANTS(iq3_s, 110, q35l_prefill_iq3_octet, q35pt_xmx_fallback_iq3)
Q35PT_XMX_VARIANTS(q4_k, 144, q35p_xmx_q4_octet, q35pt_xmx_fallback_q4)
#undef Q35PT_XMX_VARIANTS
#undef Q35PT_XMX_LINEAR

// Packed compact grids permit independent M/N reuse choices without any SLM
// or barrier. Only exact integer grids enter the F16 products; original
// per-K16 coefficients remain FP32 and are applied in the same order as the
// compact reference kernel. Inputs and packed grids use its unchanged ABI.
#define Q35PT_PACKED_FALLBACK(NAME, BYTES, DECODE) \
inline void NAME(__global const float* x, __global const uchar* packed, \
    __global const float* bias, __global float* output, int rows, int input_width, \
    int output_width, int row_base, int col_base, int bm, int bn) { \
    const int lane = get_sub_group_local_id(), sg = get_sub_group_id(); \
    for (int item = sg; item < bm * bn; item += bm / 8) { \
        const int row = row_base + item / bn, col = col_base + item % bn; \
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
Q35PT_PACKED_FALLBACK(q35pt_packed_fallback_iq2, 82, q35l_prefill_iq2_octet)
Q35PT_PACKED_FALLBACK(q35pt_packed_fallback_iq3, 110, q35l_prefill_iq3_octet)
#undef Q35PT_PACKED_FALLBACK

inline short8 q35pt_global_a_panel(__global const ushort* panel, size_t offset, int lane) {
#ifdef cl_intel_subgroups_short
    return as_short8(intel_sub_group_block_read_us8(panel + offset));
#else
    short8 result;
    #pragma unroll
    for (int r = 0; r < 8; ++r) result[r] = as_short(panel[offset + r * 16 + lane]);
    return result;
#endif
}

#define Q35PT_PACKED_FACTORED(NAME, BM, BN) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(16, BM / 8, 1))) \
__kernel void NAME(__global const ushort* xhigh, __global const ushort* xlow, \
    __global const uint* grid, __global const float* scales, __global const float* bias, \
    __global float* output, int rows, int input_width, int output_width, \
    __global const float* input, __global const uchar* packed, __global const int* range_status, int quantization) { \
    const int lane = get_local_id(0), subgroup = get_local_id(1); \
    const int row_base = get_group_id(1) * BM, row = row_base + subgroup * 8; \
    const int col_base = get_group_id(0) * BN; \
    if (range_status[0] != 0) { \
        if (quantization == 0) q35pt_packed_fallback_iq2(input, packed, bias, output, rows, input_width, output_width, row_base, col_base, BM, BN); \
        else q35pt_packed_fallback_iq3(input, packed, bias, output, rows, input_width, output_width, row_base, col_base, BM, BN); \
        return; \
    } \
    const int row_blocks = (rows + 7) / 8, col_blocks = (output_width + 15) / 16; \
    float8 accum[BN / 16]; \
    _Pragma("unroll") \
    for (int tile = 0; tile < BN / 16; ++tile) accum[tile] = (float8)(0.0f); \
    for (int block = 0; block < input_width / 16; ++block) { \
        short8 ah = (short8)(0), al = (short8)(0); \
        if (row < rows) { \
            const size_t offset = ((size_t)block * row_blocks + row / 8) * 128; \
            ah = q35pt_global_a_panel(xhigh, offset, lane); \
            al = q35pt_global_a_panel(xlow, offset, lane); \
        } \
        _Pragma("unroll") \
        for (int tile = 0; tile < BN / 16; ++tile) { \
            const int col_block = col_base / 16 + tile; \
            int8 b = (int8)(0); float coefficient = 0.0f; \
            if (col_block < col_blocks) { \
                const size_t offset = ((size_t)block * col_blocks + col_block) * 128; \
                b = as_int8(intel_sub_group_block_read8(grid + offset)); \
                if (col_block * 16 + lane < output_width) \
                    coefficient = scales[((size_t)block * col_blocks + col_block) * 16 + lane]; \
            } \
            float8 partial = intel_sub_group_f16_f16_matrix_mad_k16(ah, b, (float8)(0.0f)); \
            partial = intel_sub_group_f16_f16_matrix_mad_k16(al, b, partial); \
            accum[tile] = fma(partial, (float8)(coefficient), accum[tile]); \
        } \
    } \
    _Pragma("unroll") \
    for (int tile = 0; tile < BN / 16; ++tile) { \
        const int col = col_base + tile * 16 + lane; \
        _Pragma("unroll") \
        for (int r = 0; r < 8; ++r) if (row + r < rows && col < output_width) \
            output[(size_t)(row + r) * output_width + col] = accum[tile][r] + bias[col]; \
    } \
}
Q35PT_PACKED_FACTORED(q35l_prefill_xmx_packed_factored_m256_n32, 256, 32)
Q35PT_PACKED_FACTORED(q35l_prefill_xmx_packed_factored_m128_n128, 128, 128)
Q35PT_PACKED_FACTORED(q35l_prefill_xmx_packed_factored_m128_n32, 128, 32)
Q35PT_PACKED_FACTORED(q35l_prefill_xmx_packed_factored_m64_n128, 64, 128)
#undef Q35PT_PACKED_FACTORED

// Double row reuse and halve decoded weight columns per workgroup. The total
// output tile is still 8192 elements. Use original row-major input conversion
// and exact integer IQ grids, with FP32 coefficients applied per K16.
#define Q35PT_FACTORED_WIDE_M(NAME, BYTES, GRID, SCALE, FALLBACK) \
__attribute__((intel_reqd_sub_group_size(16))) \
__attribute__((reqd_work_group_size(16, 32, 1))) \
__kernel void NAME(__global const ushort* xhigh, __global const ushort* xlow, \
    __global const uchar* packed, __global const float* bias, __global float* output, \
    int rows, int input_width, int output_width, __global const float* input, __global const int* range_status) { \
    const int row_base = get_group_id(1) * 256, col_base = get_group_id(0) * 32; \
    if (range_status[0] != 0) { FALLBACK(input, packed, bias, output, rows, input_width, output_width, row_base, col_base, 256, 32); return; } \
    const int lane = get_local_id(0), subgroup = get_local_id(1), tid = subgroup * 16 + lane; \
    __local ushort ahigh[256][33], alow[256][33]; \
    __local uint grids[32 * 16]; \
    __local float scales[2][32]; \
    float8 accum[2] = {(float8)(0.0f), (float8)(0.0f)}; \
    for (int base = 0; base < input_width; base += 32) { \
        for (int i = tid; i < 256 * 32; i += 512) { \
            const int row = row_base + i / 32, k = base + i % 32; \
            ahigh[i / 32][i % 32] = row < rows ? xhigh[(size_t)row * input_width + k] : (ushort)0; \
            alow[i / 32][i % 32] = row < rows ? xlow[(size_t)row * input_width + k] : (ushort)0; \
        } \
        for (int i = tid; i < 32 * 4; i += 512) { \
            const int col_inner = i / 4, col = col_base + col_inner; \
            const int octet = base / 8 + i % 4; \
            float8 decoded = (float8)(0.0f); float scale = 0.0f; \
            if (col < output_width) { \
                __global const uchar* block = packed + ((size_t)col * (input_width / 256) + octet / 32) * BYTES; \
                const float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8)); \
                decoded = GRID(block, octet % 32); scale = SCALE(block, octet % 32, d); \
            } \
            if ((i & 1) == 0) scales[(i & 3) / 2][col_inner] = scale; \
            const ushort8 values = as_ushort8(convert_half8_rte(decoded)); \
            const int first_pair = (i % 4) * 4; \
            _Pragma("unroll") \
            for (int pair = 0; pair < 4; ++pair) { \
                const int p = first_pair + pair; \
                const int dst = ((col_inner / 16) * 2 + p / 8) * 128 + (p % 8) * 16 + col_inner % 16; \
                grids[dst] = (uint)values[2 * pair] | ((uint)values[2 * pair + 1] << 16); \
            } \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        _Pragma("unroll") \
        for (int part = 0; part < 2; ++part) { \
            short8 ah, al; \
            _Pragma("unroll") \
            for (int r = 0; r < 8; ++r) { \
                ah[r] = as_short(ahigh[subgroup * 8 + r][part * 16 + lane]); \
                al[r] = as_short(alow[subgroup * 8 + r][part * 16 + lane]); \
            } \
            _Pragma("unroll") \
            for (int tile = 0; tile < 2; ++tile) { \
                const int8 b = q35pt_xmx_panel_block(grids + (tile * 2 + part) * 128, lane); \
                float8 partial = intel_sub_group_f16_f16_matrix_mad_k16(ah, b, (float8)(0.0f)); \
                partial = intel_sub_group_f16_f16_matrix_mad_k16(al, b, partial); \
                accum[tile] = fma(partial, (float8)(scales[part][tile * 16 + lane]), accum[tile]); \
            } \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    _Pragma("unroll") \
    for (int tile = 0; tile < 2; ++tile) { \
        const int col = col_base + tile * 16 + lane; \
        _Pragma("unroll") \
        for (int r = 0; r < 8; ++r) { \
            const int row = row_base + subgroup * 8 + r; \
            if (row < rows && col < output_width) output[(size_t)row * output_width + col] = accum[tile][r] + bias[col]; \
        } \
    } \
}
Q35PT_FACTORED_WIDE_M(q35l_prefill_xmx_iq2_s_factored_m256_n32, 82, q35p_iq2_integer_grid, q35p_iq2_integer_scale, q35pt_packed_fallback_iq2)
Q35PT_FACTORED_WIDE_M(q35l_prefill_xmx_iq3_s_factored_m256_n32, 110, q35p_iq3_integer_grid, q35p_iq3_integer_scale, q35pt_packed_fallback_iq3)
#undef Q35PT_FACTORED_WIDE_M
#endif
