// Exact-F32 Qwen3-VL projection: a 64x64 output tile per 16x16 group.
// The GGUF F16 weights are output-major, so loading their reduction axis
// cooperatively avoids the previous output-strided global transactions.
// Four rows by four columns stay in registers per work item. Accumulation
// still follows the original increasing-K FMA order, including the epilogue.
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

inline float q35v_fast_weight(__global const ushort * weights, size_t index) {
#ifdef cl_khr_fp16
    return vload_half(index, (__global const half *)weights);
#else
    return q35v_half_to_float(weights[index]);
#endif
}

// A two-component F16 expansion retains the F32 input's low bits instead of
// directly rounding normalized activations to F16. Weights remain exactly the
// original GGUF F16 values. XMX accumulation itself is F32. This candidate is
// separate from the exact-FMA kernel so the caller can validate/select it.
#if defined(ARC_XMX) && ARC_SG == 16 && defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#ifdef cl_intel_subgroups_short
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#endif

__kernel void q35v_linear_pack_a_f16(__global const float * input,
    __global ushort * high, __global ushort * low, int rows, int input_width) {
    const int id = get_global_id(0), row_blocks = (rows + 7) / 8;
    const int k_blocks = (input_width + 15) / 16;
    if (id >= row_blocks * k_blocks * 128) return;
    const int k = (id / (row_blocks * 128)) * 16 + id % 16;
    const int row = ((id / 128) % row_blocks) * 8 + (id % 128) / 16;
    const float value = row < rows && k < input_width
        ? input[(size_t)row * input_width + k] : 0.0f;
    const half hi = convert_half_rte(value);
    high[id] = as_ushort(hi);
    low[id] = as_ushort(convert_half_rte(value - convert_float(hi)));
}

__kernel void q35v_linear_pack_b_f16(__global const ushort * input,
    __global uint * packed, int input_width, int output_width) {
    const int id = get_global_id(0), col_blocks = (output_width + 15) / 16;
    const int pairs = ((input_width + 15) / 16) * 8;
    if (id >= col_blocks * 16 * pairs) return;
    const int col = id / pairs, pair = id % pairs, k = pair * 2;
    const ushort first = col < output_width && k < input_width
        ? input[(size_t)col * input_width + k] : (ushort)0;
    const ushort second = col < output_width && k + 1 < input_width
        ? input[(size_t)col * input_width + k + 1] : (ushort)0;
    const int dst = ((pair / 8) * col_blocks + col / 16) * 128
        + (pair % 8) * 16 + col % 16;
    packed[dst] = (uint)first | ((uint)second << 16);
}

// Packing normalized activations once and reusing packed weights removes SLM
// staging/barriers from the reduction loop. The row tail remains fully padded.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void q35v_linear_f16_xmx_packed(__global const ushort * high,
    __global const ushort * low, __global const uint * weight,
    __global const float * bias, __global float * output,
    int rows, int input_width, int output_width, int gelu) {
    const int lane = get_local_id(0), subgroup = get_local_id(1);
    const int row = get_group_id(1) * 128 + subgroup * 8;
    const int col_base = get_group_id(0) * 64;
    const int row_blocks = (rows + 7) / 8, col_blocks = (output_width + 15) / 16;
    const int k_blocks = (input_width + 15) / 16;
    float8 sums[4] = {(float8)(0.0f), (float8)(0.0f),
        (float8)(0.0f), (float8)(0.0f)};
    for (int block = 0; block < k_blocks; ++block) {
        short8 hi = (short8)(0), lo = (short8)(0);
        if (row < rows) {
            const size_t offset = ((size_t)block * row_blocks + row / 8) * 128 + lane;
#ifdef cl_intel_subgroups_short
            hi = as_short8(intel_sub_group_block_read_us8(high + offset - lane));
            lo = as_short8(intel_sub_group_block_read_us8(low + offset - lane));
#else
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                hi[r] = as_short(high[offset + r * 16]);
                lo[r] = as_short(low[offset + r * 16]);
            }
#endif
        }
        #pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
            const int col_block = col_base / 16 + tile;
            int8 w = (int8)(0);
            if (col_block < col_blocks) {
                const size_t offset = ((size_t)block * col_blocks + col_block) * 128 + lane;
                w = as_int8(intel_sub_group_block_read8(weight + offset - lane));
            }
            sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(hi, w, sums[tile]);
            sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(lo, w, sums[tile]);
        }
    }
    #pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        const int col = col_base + tile * 16 + lane;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (row + r < rows && col < output_width) {
                const float value = sums[tile][r] + bias[col];
                output[(size_t)(row + r) * output_width + col] = gelu ? q35v_gelu(value) : value;
            }
        }
    }
}

inline int8 q35v_fast_panel(const __local uint * panel, int lane) {
    int8 values;
    #pragma unroll
    for (int r = 0; r < 8; ++r) values[r] = as_int(panel[r * 16 + lane]);
    return values;
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void q35v_linear_f16_xmx(__global const float * input,
    __global const ushort * weight, __global const float * bias,
    __global float * output, int rows, int input_width, int output_width,
    int gelu) {
    const int lane = get_local_id(0), subgroup = get_local_id(1);
    const int tid = subgroup * 16 + lane;
    const int row_base = get_group_id(1) * 128;
    const int col_base = get_group_id(0) * 64;
    __local ushort high[128][33], low[128][33];
    __local uint weights[64 * 16];
    float8 sums[4] = {(float8)(0.0f), (float8)(0.0f),
        (float8)(0.0f), (float8)(0.0f)};
    for (int base = 0; base < input_width; base += 32) {
        for (int i = tid; i < 128 * 32; i += 256) {
            const int r = i / 32, k = i % 32;
            const int row = row_base + r;
            float value = row < rows && base + k < input_width
                ? input[(size_t)row * input_width + base + k] : 0.0f;
            const half hi = convert_half_rte(value);
            high[r][k] = as_ushort(hi);
            low[r][k] = as_ushort(convert_half_rte(value - convert_float(hi)));
        }
        // Each work item loads four adjacent K values for one output channel.
        // Packing changes only the layout; no weight conversion is performed.
        for (int i = tid; i < 64 * 16; i += 256) {
            const int n = i / 16, pair = i % 16;
            const int col = col_base + n, k = base + pair * 2;
            ushort first = (ushort)0, second = (ushort)0;
            if (col < output_width && k < input_width)
                first = weight[(size_t)col * input_width + k];
            if (col < output_width && k + 1 < input_width)
                second = weight[(size_t)col * input_width + k + 1];
            const int dst = ((n / 16) * 2 + pair / 8) * 128
                + (pair % 8) * 16 + n % 16;
            weights[dst] = (uint)first | ((uint)second << 16);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            short8 hi, lo;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                hi[r] = as_short(high[subgroup * 8 + r][part * 16 + lane]);
                lo[r] = as_short(low[subgroup * 8 + r][part * 16 + lane]);
            }
            #pragma unroll
            for (int tile = 0; tile < 4; ++tile) {
                const int8 w = q35v_fast_panel(weights + (tile * 2 + part) * 128, lane);
                sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(hi, w, sums[tile]);
                sums[tile] = intel_sub_group_f16_f16_matrix_mad_k16(lo, w, sums[tile]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        const int col = col_base + tile * 16 + lane;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int row = row_base + subgroup * 8 + r;
            if (row < rows && col < output_width) {
                const float value = sums[tile][r] + bias[col];
                output[(size_t)row * output_width + col] = gelu ? q35v_gelu(value) : value;
            }
        }
    }
}
#endif

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void q35v_linear_f16_fast(__global const float * input,
    __global const ushort * weight, __global const float * bias,
    __global float * output, int rows, int input_width, int output_width,
    int gelu) {
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int tid = ly * 16 + lx;
    const int row_base = get_group_id(1) * 64;
    const int col_base = get_group_id(0) * 64;
    // Padding prevents output-major weight reads from conflicting in SLM.
    __local float activations[64][33];
    __local float weights[64][33];
    float4 sum0 = (float4)(0.0f), sum1 = (float4)(0.0f);
    float4 sum2 = (float4)(0.0f), sum3 = (float4)(0.0f);
    for (int base = 0; base < input_width; base += 32) {
        for (int i = tid; i < 64 * 32; i += 256) {
            const int item = i / 32, k = base + i % 32;
            const int row = row_base + item, col = col_base + item;
            activations[item][i % 32] = row < rows && k < input_width
                ? input[(size_t)row * input_width + k] : 0.0f;
            weights[item][i % 32] = col < output_width && k < input_width
                ? q35v_fast_weight(weight, (size_t)col * input_width + k) : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            const float4 w = (float4)(weights[lx][k], weights[lx + 16][k],
                weights[lx + 32][k], weights[lx + 48][k]);
            sum0 = fma((float4)(activations[ly][k]), w, sum0);
            sum1 = fma((float4)(activations[ly + 16][k]), w, sum1);
            sum2 = fma((float4)(activations[ly + 32][k]), w, sum2);
            sum3 = fma((float4)(activations[ly + 48][k]), w, sum3);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float4 sums[4] = {sum0, sum1, sum2, sum3};
    #pragma unroll
    for (int row_part = 0; row_part < 4; ++row_part) {
        const int row = row_base + ly + row_part * 16;
        #pragma unroll
        for (int col_part = 0; col_part < 4; ++col_part) {
            const int col = col_base + lx + col_part * 16;
            if (row < rows && col < output_width) {
                const float result = sums[row_part][col_part] + bias[col];
                output[(size_t)row * output_width + col] = gelu ? q35v_gelu(result) : result;
            }
        }
    }
}
