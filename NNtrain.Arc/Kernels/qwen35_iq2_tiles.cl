// Bounded-panel IQ2_S XMX products. The source GGUF bytes remain resident.
// Requires qwen35_iq.cl earlier in the OpenCL source list. No IQ quantization
// is applied to activations: they use the existing high/residual F16 pair.
// IQ2_S signed grid * odd coefficient is an integer of magnitude <= 1333,
// exactly representable in F16. Only its original block d/8 is kept separate.
// Factoring d changes the FP32 summation order; output is not bit-identical
// to the original F32 dot product. All XMX accumulators and outputs are F32.
#if defined(ARC_XMX) && ARC_SG == 16 && defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

inline short8 q35s_iq2_integer8(__global const uchar* block, int octet) {
    const int group = octet >> 2, part = octet & 3;
    const int gridIndex = block[2 + octet]
        | (((block[66 + group] >> (part * 2)) & 3) << 8);
    const short8 grid = convert_short8(as_uchar8(q35l_iq2s_grid[gridIndex]));
    const int odd = 1 + 2 * ((block[74 + group] >> ((part >> 1) * 4)) & 15);
    const short8 negative = ((short8)(block[34 + octet])
        & (short8)(1, 2, 4, 8, 16, 32, 64, 128)) != (short8)(0);
    return select(grid, -grid, negative) * (short8)(odd);
}

// Power-of-two row normalization retains relative precision for tiny
// training gradients. Never impose the absolute F16 subnormal floor on dY.
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void q35s_input_scales(__global const float* input, __global float* scales,
    __global int* rangeStatus, int M, int K) {
    const int row = get_group_id(0), tid = get_local_id(0);
    __local float maxima[128];
    float value = 0.0f;
    for (int k = tid; k < K; k += 128) {
        const float x = input[(size_t)row * K + k];
        if (!isfinite(x) || fabs(x) > 65504.0f) atomic_or(rangeStatus, 1);
        else value = fmax(value, fabs(x));
    }
    maxima[tid] = value; barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) maxima[tid] = fmax(maxima[tid], maxima[tid + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        int exponent = 0;
        if (maxima[0] > 0.0f) frexp(maxima[0], &exponent);
        if (exponent < -120) atomic_or(rangeStatus, 1);
        scales[row] = ldexp(1.0f, clamp(exponent, -120, 120));
    }
}

// Same panel layout as q35l_prefill_xmx_pack_input_f16x2, normalized per row.
// writes eight values. K is a multiple of 16. The caller clears rangeStatus
// before packing and retains it until all dependent products have finished.
// Half layout: [K/16][ceil(M/8)][8 rows][16 K values].
// Logical global size: (K/16) * ceil(M/8) * 16 (round to chosen local size).
__kernel void q35s_pack_input_f16x2(__global const float* input,
    __global ushort* high, __global ushort* low, __global int* rangeStatus,
    int M, int K, __global const float* rowScales) {
    const size_t id = get_global_id(0), rowTiles = ((size_t)M + 7) / 8;
    const size_t count = (size_t)(K / 16) * rowTiles * 16;
    if (id >= count) return;
    const int kb = (int)(id / (rowTiles * 16));
    const int row = (int)((id / 16) % rowTiles) * 8 + (int)((id % 16) / 2);
    const int k = kb * 16 + (int)(id & 1) * 8;
    const float8 value = row < M ? vload8(0, input + (size_t)row * K + k) : (float8)(0);
    if (any(!isfinite(value)) || any(fabs(value) > (float8)(65504.0f))) {
        atomic_or(rangeStatus, 1);
        vstore8((ushort8)(0), id, high);
        vstore8((ushort8)(0), id, low);
        return;
    }
    const float8 normalized = value / (float8)(row < M ? rowScales[row] : 1.0f);
    const half8 hi = convert_half8_rte(normalized);
    vstore8(as_ushort8(hi), id, high);
    vstore8(as_ushort8(convert_half8_rte(normalized - convert_float8(hi))), id, low);
}

// Experimental traversal order: [ceil(M/8)][K/16][8 rows][16 K values].
// A subgroup advances only 256 bytes for each K16 rather than crossing all
// token rows; conversion and per-row scaling are otherwise identical.
// Logical workitems and allocation sizes match the K-major pack above.
__kernel void q35s_pack_input_f16x2_rowmajor(__global const float* input,
    __global ushort* high, __global ushort* low, __global int* rangeStatus,
    int M, int K, __global const float* rowScales) {
    const size_t id = get_global_id(0), rowTiles = ((size_t)M + 7) / 8;
    const size_t kTiles = (size_t)K / 16, count = rowTiles * kTiles * 16;
    if (id >= count) return;
    const int kb = (int)((id / 16) % kTiles);
    const int row = (int)(id / (kTiles * 16)) * 8 + (int)((id % 16) / 2);
    const int k = kb * 16 + (int)(id & 1) * 8;
    const float8 value = row < M ? vload8(0, input + (size_t)row * K + k) : (float8)(0);
    if (any(!isfinite(value)) || any(fabs(value) > (float8)(65504.0f))) {
        atomic_or(rangeStatus, 1);
        vstore8((ushort8)(0), id, high);
        vstore8((ushort8)(0), id, low);
        return;
    }
    const float8 normalized = value / (float8)(row < M ? rowScales[row] : 1.0f);
    const half8 hi = convert_half8_rte(normalized);
    vstore8(as_ushort8(hi), id, high);
    vstore8(as_ushort8(convert_half8_rte(normalized - convert_float8(hi))), id, low);
}

// Decode only [columnStart, columnStart+panelColumns) of W[N,K].
// B pair layout: [K/16][ceil(panelColumns/16)][8 K pairs][16 columns].
// blockD layout: [K/256][ceil(panelColumns/16)*16 columns], storing d/8.
// Logical global size: (K/16) * ceil(panelColumns/16) * 32.
// panelColumns need not be a multiple of 16; padding is initialized to zero.
__kernel void q35s_pack_iq2_b(__global const uchar* raw,
    __global uint* panels, __global float* blockD, __global int* rangeStatus,
    int N, int K, int columnStart, int panelColumns) {
    const size_t id = get_global_id(0), colTiles = ((size_t)panelColumns + 15) / 16;
    const size_t count = (size_t)(K / 16) * colTiles * 32;
    if (id >= count) return;
    const int kb = (int)(id / (colTiles * 32));
    const int col = (int)((id / 32) % colTiles) * 16 + (int)(id % 16);
    const int octet = (int)((id % 32) / 16);
    const int k = kb * 16 + octet * 8;
    ushort8 values = (ushort8)(0);
    float d = 0.0f;
    if (col < panelColumns && columnStart + col < N) {
        __global const uchar* block = raw
            + ((size_t)(columnStart + col) * (K / 256) + k / 256) * 82;
        values = as_ushort8(convert_half8_rte(convert_float8(q35s_iq2_integer8(block, (k & 255) / 8))));
        if ((k & 255) == 0) {
            d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8)) * 0.125f;
            if (!isfinite(d)) atomic_or(rangeStatus, 1);
        }
    }
    const size_t offset = (id / 32) * 128 + octet * 64 + col % 16;
    panels[offset]      = (uint)values.s0 | ((uint)values.s1 << 16);
    panels[offset + 16] = (uint)values.s2 | ((uint)values.s3 << 16);
    panels[offset + 32] = (uint)values.s4 | ((uint)values.s5 << 16);
    panels[offset + 48] = (uint)values.s6 | ((uint)values.s7 << 16);
    if ((k & 255) == 0) blockD[(size_t)(k / 256) * (colTiles * 16) + col] = d;
}

// Rare device-only fallback. The original FP32 activation and encoded
// weight are read when the conversion range flag is set. One subgroup owns
// exactly the same output tile as its XMX variant; partial tiles are masked.
inline void q35s_forward_fallback(__global const float* input,
    __global const uchar* raw, __global const float* bias, __global float* output,
    int M, int N, int K, int columnStart, int panelColumns,
    int rowBase, int colBase, int tileRows, int tileColumns) {
    const int lane = get_sub_group_local_id();
    for (int rowOffset = 0; rowOffset < tileRows; ++rowOffset) {
        const int row = rowBase + rowOffset;
        if (row >= M) continue;
        for (int colOffset = lane; colOffset < tileColumns; colOffset += 16) {
            const int panelCol = colBase + colOffset, col = columnStart + panelCol;
            if (panelCol >= panelColumns || col >= N) continue;
            float sum = 0.0f;
            for (int b = 0; b < K / 256; ++b) {
                __global const uchar* block = raw + ((size_t)col * (K / 256) + b) * 82;
                const float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8));
                for (int k = 0; k < 256; ++k)
                    sum = fma(input[(size_t)row * K + b * 256 + k],
                        q35l_iq2_s_value(block, k, d, 0.0f), sum);
            }
            output[(size_t)row * N + col] = sum + bias[col];
        }
    }
}

// One SG16 owns (RM*8)x(CN*16). Any local size divisible by 16 is allowed.
// Global work-items: ceil(M/(RM*8))*ceil(panelColumns/(CN*16))*16,
// rounded to the selected local size. bPanel and bD describe this panel;
// output/bias use the complete N-wide row stride.
#define Q35S_FORWARD(NAME, RM, CN, UNROLL, ROW_MAJOR) \
__attribute__((intel_reqd_sub_group_size(16))) \
__kernel void NAME(__global const ushort* aHigh, __global const ushort* aLow, \
    __global const uint* bPanel, __global const float* bD, \
    __global const float* bias, __global float* output, \
    __global const float* input, __global const uchar* raw, \
    __global const int* rangeStatus, int M, int N, int K, int columnStart, int panelColumns, __global const float* rowScales) { \
    const size_t tile = get_group_id(0) * (get_local_size(0) / 16) + get_sub_group_id(); \
    const int mTiles = (M + (RM)*8 - 1) / ((RM)*8); \
    const int nTiles = (panelColumns + (CN)*16 - 1) / ((CN)*16); \
    if (tile >= (size_t)mTiles * nTiles) return; \
    const int lane = get_sub_group_local_id(); \
    const int rowBase = (int)(tile / nTiles) * ((RM)*8); \
    const int colBase = (int)(tile % nTiles) * ((CN)*16); \
    if (rangeStatus[0] != 0) { \
        q35s_forward_fallback(input, raw, bias, output, M, N, K, columnStart, panelColumns, \
            rowBase, colBase, (RM)*8, (CN)*16); return; \
    } \
    const int aTiles = (M + 7) / 8, bTiles = (panelColumns + 15) / 16; \
    float8 acc[RM][CN]; \
    _Pragma("unroll") \
    for (int rb = 0; rb < RM; ++rb) { \
        _Pragma("unroll") \
        for (int cb = 0; cb < CN; ++cb) acc[rb][cb] = (float8)(0); \
    } \
    for (int block = 0; block < K / 256; ++block) { \
        float8 blockSum[RM][CN]; \
        _Pragma("unroll") \
        for (int rb = 0; rb < RM; ++rb) { \
            _Pragma("unroll") \
            for (int cb = 0; cb < CN; ++cb) blockSum[rb][cb] = (float8)(0); \
        } \
        _Pragma(UNROLL) \
        for (int part = 0; part < 16; ++part) { \
            const int kb = block * 16 + part; \
            short8 avh[RM], avl[RM]; int8 bv[CN]; \
            _Pragma("unroll") \
            for (int rb = 0; rb < RM; ++rb) { \
                avh[rb] = (short8)(0); avl[rb] = (short8)(0); \
                if (rowBase + rb * 8 < M) { \
                    const size_t offset = (ROW_MAJOR \
                        ? ((size_t)(rowBase / 8 + rb) * (K / 16) + kb) \
                        : ((size_t)kb * aTiles + rowBase / 8 + rb)) * 128; \
                    avh[rb] = as_short8(intel_sub_group_block_read_us8(aHigh + offset)); \
                    avl[rb] = as_short8(intel_sub_group_block_read_us8(aLow + offset)); \
                } \
            } \
            _Pragma("unroll") \
            for (int cb = 0; cb < CN; ++cb) { \
                bv[cb] = (int8)(0); \
                if (colBase + cb * 16 < panelColumns) \
                    bv[cb] = as_int8(intel_sub_group_block_read8( \
                        bPanel + ((size_t)kb * bTiles + colBase / 16 + cb) * 128)); \
            } \
            _Pragma("unroll") \
            for (int rb = 0; rb < RM; ++rb) { \
                _Pragma("unroll") \
                for (int cb = 0; cb < CN; ++cb) { \
                    blockSum[rb][cb] = intel_sub_group_f16_f16_matrix_mad_k16(avh[rb], bv[cb], blockSum[rb][cb]); \
                    blockSum[rb][cb] = intel_sub_group_f16_f16_matrix_mad_k16(avl[rb], bv[cb], blockSum[rb][cb]); \
                } \
            } \
        } \
        _Pragma("unroll") \
        for (int rb = 0; rb < RM; ++rb) { \
            _Pragma("unroll") \
            for (int cb = 0; cb < CN; ++cb) { \
                const int col = colBase + cb * 16 + lane; \
                const float d = col < panelColumns ? bD[(size_t)block * (bTiles * 16) + col] : 0.0f; \
                acc[rb][cb] = fma(blockSum[rb][cb], (float8)(d), acc[rb][cb]); \
            } \
        } \
    } \
    _Pragma("unroll") \
    for (int rb = 0; rb < RM; ++rb) { \
        _Pragma("unroll") \
        for (int cb = 0; cb < CN; ++cb) { \
            const int panelCol = colBase + cb * 16 + lane, col = columnStart + panelCol; \
            if (panelCol < panelColumns && col < N) { \
                _Pragma("unroll") \
                for (int r = 0; r < 8; ++r) if (rowBase + rb * 8 + r < M) \
                    output[(size_t)(rowBase + rb * 8 + r) * N + col] = acc[rb][cb][r] * rowScales[rowBase + rb * 8 + r] + bias[col]; \
            } \
        } \
    } \
}
Q35S_FORWARD(q35s_forward_m8n16, 1, 1, "unroll 1", 0)
Q35S_FORWARD(q35s_forward_m16n16, 2, 1, "unroll 1", 0)
Q35S_FORWARD(q35s_forward_m16n32, 2, 2, "unroll 1", 0)
Q35S_FORWARD(q35s_forward_m32n32, 4, 2, "unroll 1", 0)
Q35S_FORWARD(q35s_forward_u2_m16n32, 2, 2, "unroll 2", 0)
Q35S_FORWARD(q35s_forward_u2_m32n32, 4, 2, "unroll 2", 0)
Q35S_FORWARD(q35s_forward_u2_rowmajor_m16n32, 2, 2, "unroll 2", 1)
#undef Q35S_FORWARD

// Transpose/backward path: raw W[K,N] is multiplied by dY[M,K]. Its IQ2_S
// block d varies over the reduction K, so it cannot be factored outside a
// K256 sum. Instead a bounded destination-column panel stores the full
// decoded weight as high/residual F16; all four products are accumulated.
// N is divisible by 256; the common input pack requires K divisible by 16.
// bHigh/bLow layout is [ceil(K/16)][ceil(panelColumns/16)][8 K pairs][16 columns].
// Logical workitems: ceil(K/16)*ceil(panelColumns/16)*128.
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void q35s_transpose_scales(__global const uchar* raw, __global float* scales,
    __global int* rangeStatus, int N, int K) {
    const int blockCol = get_group_id(0), tid = get_local_id(0);
    __local float maxima[128];
    float value = 0.0f;
    for (int k = tid; k < K; k += 128) {
        __global const uchar* block = raw + ((size_t)k * (N / 256) + blockCol) * 82;
        const float d = fabs(q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8)));
        if (!isfinite(d)) atomic_or(rangeStatus, 1);
        else value = fmax(value, d);
    }
    maxima[tid] = value; barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) maxima[tid] = fmax(maxima[tid], maxima[tid + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        int exponent = 0;
        if (maxima[0] > 0.0f) frexp(maxima[0] * 166.625f, &exponent);
        scales[blockCol] = ldexp(1.0f, exponent);
    }
}

__kernel void q35s_pack_iq2_bt(__global const uchar* raw,
    __global uint* bHigh, __global uint* bLow, __global int* rangeStatus,
    int N, int K, int columnStart, int panelColumns, __global const float* weightScales) {
    const size_t id = get_global_id(0), colTiles = ((size_t)panelColumns + 15) / 16;
    const size_t count = ((size_t)(K + 15) / 16) * colTiles * 128;
    if (id >= count) return;
    const int kb = (int)(id / (colTiles * 128));
    const int panelCol = (int)((id / 128) % colTiles) * 16 + (int)(id % 16);
    const int col = columnStart + panelCol;
    const int k = kb * 16 + (int)((id % 128) / 16) * 2;
    float2 value = (float2)(0);
    if (panelCol < panelColumns && col < N) {
        if (k < K) {
            __global const uchar* block = raw + ((size_t)k * (N / 256) + col / 256) * 82;
            const float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8));
            value.s0 = q35l_iq2_s_value(block, col & 255, d, 0.0f);
        }
        if (k + 1 < K) {
            __global const uchar* block = raw + ((size_t)(k + 1) * (N / 256) + col / 256) * 82;
            const float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8));
            value.s1 = q35l_iq2_s_value(block, col & 255, d, 0.0f);
        }
    }
    if (any(!isfinite(value)) || any(fabs(value) > (float2)(65504.0f))) {
        atomic_or(rangeStatus, 1); bHigh[id] = 0; bLow[id] = 0; return;
    }
    value /= (float2)(col < N ? weightScales[col / 256] : 1.0f);
    const half2 hi = convert_half2_rte(value);
    bHigh[id] = as_uint(hi);
    bLow[id] = as_uint(convert_half2_rte(value - convert_float2(hi)));
}

inline void q35s_transpose_fallback(__global const float* input,
    __global const uchar* raw, __global float* output,
    int M, int N, int K, int columnStart, int panelColumns,
    int rowBase, int colBase, int tileRows, int tileColumns, int addToOutput) {
    const int lane = get_sub_group_local_id();
    for (int rowOffset = 0; rowOffset < tileRows; ++rowOffset) {
        const int row = rowBase + rowOffset;
        if (row >= M) continue;
        for (int colOffset = lane; colOffset < tileColumns; colOffset += 16) {
            const int panelCol = colBase + colOffset, col = columnStart + panelCol;
            if (panelCol >= panelColumns || col >= N) continue;
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                __global const uchar* block = raw + ((size_t)k * (N / 256) + col / 256) * 82;
                const float d = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8));
                sum = fma(input[(size_t)row * K + k], q35l_iq2_s_value(block, col & 255, d, 0.0f), sum);
            }
            const size_t dst = (size_t)row * N + col;
            output[dst] = addToOutput ? output[dst] + sum : sum;
        }
    }
}

// Grid and workgroup requirements match the corresponding forward tile.
// Output uses the full N-wide row stride, never the panel width.
#define Q35S_TRANSPOSE(NAME, RM, CN) \
__attribute__((intel_reqd_sub_group_size(16))) \
__kernel void NAME(__global const ushort* aHigh, __global const ushort* aLow, \
    __global const uint* bHigh, __global const uint* bLow, \
    __global float* output, __global const float* input, __global const uchar* raw, \
    __global const int* rangeStatus, int M, int N, int K, int columnStart, int panelColumns, int addToOutput, \
    __global const float* rowScales, __global const float* weightScales) { \
    const size_t tile = get_group_id(0) * (get_local_size(0) / 16) + get_sub_group_id(); \
    const int mTiles = (M + (RM)*8 - 1) / ((RM)*8); \
    const int nTiles = (panelColumns + (CN)*16 - 1) / ((CN)*16); \
    if (tile >= (size_t)mTiles * nTiles) return; \
    const int lane = get_sub_group_local_id(); \
    const int rowBase = (int)(tile / nTiles) * ((RM)*8); \
    const int colBase = (int)(tile % nTiles) * ((CN)*16); \
    if (rangeStatus[0] != 0) { \
        q35s_transpose_fallback(input, raw, output, M, N, K, columnStart, panelColumns, \
            rowBase, colBase, (RM)*8, (CN)*16, addToOutput); return; \
    } \
    const int aTiles = (M + 7) / 8, bTiles = (panelColumns + 15) / 16; \
    float8 acc[RM][CN]; \
    _Pragma("unroll") \
    for (int rb = 0; rb < RM; ++rb) { \
        _Pragma("unroll") \
        for (int cb = 0; cb < CN; ++cb) acc[rb][cb] = (float8)(0); \
    } \
    for (int kb = 0; kb < (K + 15) / 16; ++kb) { \
        short8 avh[RM], avl[RM]; int8 bvh[CN], bvl[CN]; \
        _Pragma("unroll") \
        for (int rb = 0; rb < RM; ++rb) { \
            avh[rb] = (short8)(0); avl[rb] = (short8)(0); \
            if (rowBase + rb * 8 < M) { \
                const size_t offset = ((size_t)kb * aTiles + rowBase / 8 + rb) * 128; \
                avh[rb] = as_short8(intel_sub_group_block_read_us8(aHigh + offset)); \
                avl[rb] = as_short8(intel_sub_group_block_read_us8(aLow + offset)); \
            } \
        } \
        _Pragma("unroll") \
        for (int cb = 0; cb < CN; ++cb) { \
            bvh[cb] = (int8)(0); bvl[cb] = (int8)(0); \
            if (colBase + cb * 16 < panelColumns) { \
                const size_t offset = ((size_t)kb * bTiles + colBase / 16 + cb) * 128; \
                bvh[cb] = as_int8(intel_sub_group_block_read8(bHigh + offset)); \
                bvl[cb] = as_int8(intel_sub_group_block_read8(bLow + offset)); \
            } \
        } \
        _Pragma("unroll") \
        for (int rb = 0; rb < RM; ++rb) { \
            _Pragma("unroll") \
            for (int cb = 0; cb < CN; ++cb) { \
                acc[rb][cb] = intel_sub_group_f16_f16_matrix_mad_k16(avh[rb], bvh[cb], acc[rb][cb]); \
                acc[rb][cb] = intel_sub_group_f16_f16_matrix_mad_k16(avl[rb], bvh[cb], acc[rb][cb]); \
                acc[rb][cb] = intel_sub_group_f16_f16_matrix_mad_k16(avh[rb], bvl[cb], acc[rb][cb]); \
                acc[rb][cb] = intel_sub_group_f16_f16_matrix_mad_k16(avl[rb], bvl[cb], acc[rb][cb]); \
            } \
        } \
    } \
    _Pragma("unroll") \
    for (int rb = 0; rb < RM; ++rb) { \
        _Pragma("unroll") \
        for (int cb = 0; cb < CN; ++cb) { \
            const int panelCol = colBase + cb * 16 + lane, col = columnStart + panelCol; \
            if (panelCol < panelColumns && col < N) { \
                _Pragma("unroll") \
                for (int r = 0; r < 8; ++r) if (rowBase + rb * 8 + r < M) { \
                    const size_t dst = (size_t)(rowBase + rb * 8 + r) * N + col; \
                    const float result = (acc[rb][cb][r] * weightScales[col / 256]) * rowScales[rowBase + rb * 8 + r]; \
                    output[dst] = addToOutput ? output[dst] + result : result; \
                } \
            } \
        } \
    } \
}
Q35S_TRANSPOSE(q35s_transpose_m8n16, 1, 1)
Q35S_TRANSPOSE(q35s_transpose_m16n16, 2, 1)
Q35S_TRANSPOSE(q35s_transpose_m16n32, 2, 2)
Q35S_TRANSPOSE(q35s_transpose_m32n32, 4, 2)
#undef Q35S_TRANSPOSE

// Direct-GGUF BSLM candidate: one WG computes 128 rows x 64 columns.
// A stays in global high/residual-half panels. Decode one complete K256
// integer-weight tile into 32 KiB of local memory, reducing synchronization
// to two barriers per GGUF block. No global decoded weight panel is needed.
// Global=(ceil(N/64)*16,ceil(M/128)*16), local=(16,16).
// rowScale undoes the activation's per-row power-of-two normalization.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void q35s_forward_gguf_bslm(__global const ushort* aHigh,
    __global const ushort* aLow, __global const uchar* raw,
    __global const float* bias, __global float* output, int M, int K, int N,
    __global const float* input, __global const int* rangeStatus,
    __global const float* rowScale) {
    const int lane = get_local_id(0), subgroup = get_local_id(1);
    const int tid = subgroup * 16 + lane;
    const int rowBase = get_group_id(1) * 128, colBase = get_group_id(0) * 64;
    const int row = rowBase + subgroup * 8, aTiles = (M + 7) / 8;
    if (rangeStatus[0] != 0) {
        q35s_forward_fallback(input, raw, bias, output, M, N, K,
            colBase, min(64, N - colBase), row, 0, 8, 64);
        return;
    }
    // Exactly 32 KiB: storing another 256-byte d array in local memory would
    // prevent two resident WGs on a 64 KiB SLM slice. Each lane instead reads
    // its four block coefficients from the encoded, cacheable source.
    __local uint bPanel[8192];
    float8 accum[4] = {(float8)(0), (float8)(0), (float8)(0), (float8)(0)};
    for (int blockIndex = 0; blockIndex < K / 256; ++blockIndex) {
        float coefficients[4];
        int invalid = 0;
        #pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
            const int col = colBase + tile * 16 + lane;
            coefficients[tile] = 0.0f;
            if (col < N) {
                __global const uchar* block = raw + ((size_t)col * (K / 256) + blockIndex) * 82;
                coefficients[tile] = q35l_half_to_float((ushort)block[0] | ((ushort)block[1] << 8)) * 0.125f;
                invalid |= !isfinite(coefficients[tile]);
            }
        }
        // Every subgroup reads the same 64 coefficients, so this reduction
        // gives the same decision to the entire WG before either barrier.
        // Recompute the whole tile, discarding earlier finite block sums:
        // factoring Inf outside mixed-sign products changes NaN semantics.
        if (sub_group_reduce_add(invalid) != 0) {
            q35s_forward_fallback(input, raw, bias, output, M, N, K,
                colBase, min(64, N - colBase), row, 0, 8, 64);
            return;
        }
        for (int item = tid; item < 64 * 32; item += 256) {
            const int localCol = item % 64, octet = item / 64;
            const int col = colBase + localCol;
            ushort8 values = (ushort8)(0);
            if (col < N) {
                __global const uchar* block = raw + ((size_t)col * (K / 256) + blockIndex) * 82;
                values = as_ushort8(convert_half8_rte(convert_float8(q35s_iq2_integer8(block, octet))));
            }
            const int destination = ((octet / 2) * 4 + localCol / 16) * 128
                + (octet % 2) * 64 + localCol % 16;
            bPanel[destination] = (uint)values.s0 | ((uint)values.s1 << 16);
            bPanel[destination + 16] = (uint)values.s2 | ((uint)values.s3 << 16);
            bPanel[destination + 32] = (uint)values.s4 | ((uint)values.s5 << 16);
            bPanel[destination + 48] = (uint)values.s6 | ((uint)values.s7 << 16);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float8 partial[4] = {(float8)(0), (float8)(0), (float8)(0), (float8)(0)};
        #pragma unroll 2
        for (int part = 0; part < 16; ++part) {
            const size_t aOffset = ((size_t)(blockIndex * 16 + part) * aTiles + row / 8) * 128;
            short8 ah = (short8)(0), al = (short8)(0);
            if (row < M) {
                ah = as_short8(intel_sub_group_block_read_us8(aHigh + aOffset));
                al = as_short8(intel_sub_group_block_read_us8(aLow + aOffset));
            }
            #pragma unroll
            for (int tile = 0; tile < 4; ++tile) {
                const __local uint* start = bPanel + (part * 4 + tile) * 128;
                int8 b;
                #pragma unroll
                for (int pair = 0; pair < 8; ++pair) b[pair] = as_int(start[pair * 16 + lane]);
                partial[tile] = intel_sub_group_f16_f16_matrix_mad_k16(ah, b, partial[tile]);
                partial[tile] = intel_sub_group_f16_f16_matrix_mad_k16(al, b, partial[tile]);
            }
        }
        #pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
            accum[tile] = fma(partial[tile], (float8)(coefficients[tile]), accum[tile]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        const int col = colBase + tile * 16 + lane;
        if (col < N) {
            #pragma unroll
            for (int r = 0; r < 8; ++r) if (row + r < M)
                output[(size_t)(row + r) * N + col] = accum[tile][r] * rowScale[row + r] + bias[col];
        }
    }
}

#endif
