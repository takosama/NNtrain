#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>

#include "mix8_diagnostics.cuh"

#if defined(_WIN32)
#define NNTRAIN_EXPORT extern "C" __declspec(dllexport)
#else
#define NNTRAIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
constexpr std::uint32_t nntrain_abi_major = 1;
constexpr std::uint32_t nntrain_abi_minor = 31;
constexpr std::uint32_t nntrain_abi_version_value =
    (nntrain_abi_major << 16) | nntrain_abi_minor;

enum nntrain_native_operation : std::uint32_t {
    nntrain_operation_none = 0,
    nntrain_operation_device_count = 1,
    nntrain_operation_device_name = 2,
    nntrain_operation_set_device = 3,
    nntrain_operation_synchronize = 4,
    nntrain_operation_memory_info = 5,
    nntrain_operation_allocate = 6,
    nntrain_operation_free = 7,
    nntrain_operation_memset = 8,
    nntrain_operation_copy_h2d = 9,
    nntrain_operation_copy_d2h = 10,
    nntrain_operation_host_allocate = 11,
    nntrain_operation_host_free = 12,
    nntrain_operation_stream_create = 13,
    nntrain_operation_stream_destroy = 14,
    nntrain_operation_stream_synchronize = 15,
    nntrain_operation_event_create = 16,
    nntrain_operation_event_destroy = 17,
    nntrain_operation_event_record = 18,
    nntrain_operation_event_query = 19,
    nntrain_operation_event_synchronize = 20,
    nntrain_operation_copy_d2h_async = 21,
    nntrain_operation_copy_h2d_async = 22,
    nntrain_operation_copy_d2d = 23,
    nntrain_operation_peer_access = 24,
    nntrain_operation_capabilities = 25,
    nntrain_operation_bfp8_quantize = 26,
    nntrain_operation_bfp8_dequantize_f32 = 27,
    nntrain_operation_bfp8_dequantize_bf16 = 28,
    nntrain_operation_bfp8_quantize_bf16 = 29,
    nntrain_operation_bfp8_requantize_i32 = 30,
    nntrain_operation_bfp8_transpose_i8 = 31,
    nntrain_operation_memset_async = 32,
    nntrain_operation_copy_d2d_async = 33,
    nntrain_operation_graph_begin_capture = 74,
    nntrain_operation_graph_end_capture = 75,
    nntrain_operation_graph_instantiate = 76,
    nntrain_operation_graph_launch = 77,
    nntrain_operation_graph_destroy = 78,
    nntrain_operation_graph_exec_destroy = 79,
    nntrain_operation_graph_rng_step = 80,
    nntrain_operation_graph_counter_set = 81,
    nntrain_operation_graph_counter_advance = 82,
    nntrain_operation_graph_dropout_forward = 83,
    nntrain_operation_graph_add_dropout_forward = 84,
    nntrain_operation_graph_dropout_backward = 85,
    nntrain_operation_graph_add_dropout_backward = 86,
};

struct nntrain_native_error_info {
    std::uint32_t struct_size;
    std::uint32_t abi_version;
    std::uint64_t sequence;
    std::int32_t status;
    std::int32_t device;
    std::uint32_t operation;
    std::uint32_t reserved;
};

constexpr std::size_t error_ring_capacity = 64;
std::uint64_t error_sequence = 0;
std::mutex last_error_mutex;
std::array<nntrain_native_error_info, error_ring_capacity> error_ring{};
nntrain_native_error_info last_error{
    sizeof(nntrain_native_error_info),
    nntrain_abi_version_value,
    0,
    static_cast<std::int32_t>(cudaSuccess),
    -1,
    nntrain_operation_none,
    0,
};

int complete(
    nntrain_native_operation operation,
    int device,
    int status) {
    // The successful hot path performs neither a lock nor an atomic update.
    if (status == static_cast<int>(cudaSuccess))
        return status;

    std::lock_guard<std::mutex> guard(last_error_mutex);
    const std::uint64_t sequence = ++error_sequence;
    nntrain_native_error_info snapshot{
        sizeof(nntrain_native_error_info),
        nntrain_abi_version_value,
        sequence,
        static_cast<std::int32_t>(status),
        static_cast<std::int32_t>(device),
        static_cast<std::uint32_t>(operation),
        0,
    };
    last_error = snapshot;
    error_ring[(sequence - 1) % error_ring_capacity] = snapshot;
    return status;
}

thread_local int selected_device = -1;

int select_device(int device) {
    if (selected_device == device)
        return static_cast<int>(cudaSuccess);
    cudaError_t status = cudaSetDevice(device);
    if (status == cudaSuccess)
        selected_device = device;
    return static_cast<int>(status);
}

// The DLL is compiled with --use_fast_math for the compute kernels. BFP8 is
// a storage codec, however, and its public CPU reference promises identical
// round-to-nearest-even payloads. Keep scale derivation and code selection on
// explicit IEEE round-to-nearest division so approximate reciprocal lowering
// cannot move a value across an Int8 midpoint (most visible with block scales).
__device__ __forceinline__ float bfp8_scale_from_maximum(float maximum) {
    return maximum == 0.0f ? 1.0f : __fdiv_rn(maximum, 127.0f);
}

__device__ __forceinline__ int bfp8_quantized_code(
    float value,
    float scale) {
    int quantized = __float2int_rn(__fdiv_rn(value, scale));
    return max(-127, min(127, quantized));
}

__global__ void bfp8_quantize_f32_kernel(
    const float* source,
    signed char* payload,
    float* scales,
    int length,
    int quantization_block_size) {
    const long long start =
        static_cast<long long>(blockIdx.x) * quantization_block_size;
    const long long end = min(
        static_cast<long long>(length),
        start + quantization_block_size);

    float local_maximum = 0.0f;
    for (long long index = start + threadIdx.x;
         index < end;
         index += blockDim.x) {
        local_maximum = fmaxf(local_maximum, fabsf(source[index]));
    }

    __shared__ float reduction[256];
    reduction[threadIdx.x] = local_maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x],
                reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        scales[blockIdx.x] = bfp8_scale_from_maximum(reduction[0]);
    }
    __syncthreads();

    const float scale = scales[blockIdx.x];
    for (long long index = start + threadIdx.x;
         index < end;
         index += blockDim.x) {
        int quantized = bfp8_quantized_code(source[index], scale);
        payload[index] = static_cast<signed char>(quantized);
    }
}

__global__ void bfp8_quantize_f32_roundtrip_kernel(
    float* source,
    signed char* payload,
    float* scales,
    int length,
    int quantization_block_size) {
    const long long start =
        static_cast<long long>(blockIdx.x) * quantization_block_size;
    const long long end = min(
        static_cast<long long>(length),
        start + quantization_block_size);
    float local_maximum = 0.0f;
    for (long long index = start + threadIdx.x;
         index < end;
         index += blockDim.x) {
        local_maximum = fmaxf(local_maximum, fabsf(source[index]));
    }
    __shared__ float reduction[256];
    reduction[threadIdx.x] = local_maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        scales[blockIdx.x] = bfp8_scale_from_maximum(reduction[0]);
    __syncthreads();
    const float scale = scales[blockIdx.x];
    for (long long index = start + threadIdx.x;
         index < end;
         index += blockDim.x) {
        const int quantized = bfp8_quantized_code(source[index], scale);
        payload[index] = static_cast<signed char>(quantized);
        source[index] = static_cast<float>(quantized) * scale;
    }
}

// mix8_32 keeps FP32 master weights/gradients but persists Adam moments in
// block-scaled BFP8. One block owns one quantization block, so moment decode,
// EMA update, scale reduction, requantization and parameter update share the
// values in registers instead of crossing six kernel boundaries.
__global__ void adamw_block_bfp8_state_kernel(
    float* data,
    const float* gradient,
    signed char* first_payload,
    float* first_scales,
    signed char* second_payload,
    float* second_scales,
    int length,
    float beta1,
    float beta2,
    float learning_rate,
    float weight_decay,
    float update_scale,
    float scaled_epsilon,
    int apply_weight_decay,
    int* finite_status) {
    constexpr int block_size = 128;
    __shared__ float first_maximum[block_size];
    __shared__ float second_maximum[block_size];
    __shared__ int invalid;
    if (threadIdx.x == 0)
        invalid = *finite_status;
    __syncthreads();

    const int index = blockIdx.x * block_size + threadIdx.x;
    float first = 0.f;
    float second = 0.f;
    if (index < length && invalid == 0) {
        const float g = gradient[index];
        first = fmaf(
            beta1,
            static_cast<float>(first_payload[index]) * first_scales[blockIdx.x],
            (1.f - beta1) * g);
        second = fmaf(
            beta2,
            static_cast<float>(second_payload[index]) * second_scales[blockIdx.x],
            (1.f - beta2) * g * g);
        if (!isfinite(g) || !isfinite(first) || !isfinite(second)
            || second < 0.f) {
            atomicExch(&invalid, 1);
        }
    }
    first_maximum[threadIdx.x] = fabsf(first);
    second_maximum[threadIdx.x] = fabsf(second);
    __syncthreads();
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            first_maximum[threadIdx.x] = fmaxf(
                first_maximum[threadIdx.x], first_maximum[threadIdx.x + stride]);
            second_maximum[threadIdx.x] = fmaxf(
                second_maximum[threadIdx.x], second_maximum[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (invalid != 0) {
        if (threadIdx.x == 0)
            atomicExch(finite_status, 1);
        return;
    }
    if (threadIdx.x == 0) {
        first_scales[blockIdx.x] =
            bfp8_scale_from_maximum(first_maximum[0]);
        second_scales[blockIdx.x] =
            bfp8_scale_from_maximum(second_maximum[0]);
    }
    __syncthreads();
    if (index >= length)
        return;

    const float first_scale = first_scales[blockIdx.x];
    const float second_scale = second_scales[blockIdx.x];
    const int first_code = bfp8_quantized_code(first, first_scale);
    const int second_code = bfp8_quantized_code(second, second_scale);
    first_payload[index] = static_cast<signed char>(first_code);
    second_payload[index] = static_cast<signed char>(second_code);
    first = static_cast<float>(first_code) * first_scale;
    second = static_cast<float>(second_code) * second_scale;

    const float variance = fmaxf(second, 0.5f * second_scale);
    float parameter = data[index];
    if (apply_weight_decay)
        parameter *= 1.f - learning_rate * weight_decay;
    parameter -= update_scale * first /
        (sqrtf(variance) + scaled_epsilon);
    if (!isfinite(parameter)) {
        atomicExch(finite_status, 1);
        return;
    }
    data[index] = parameter;
}

// The NekoMuon statistics kernel must observe the quantized persistent
// moments. Fuse decode/update/requantize and materialize that exact state into
// the shared FP32 workspace; the existing block reducer then computes stable
// FP32 statistics in one additional launch.
__global__ void nekomuon_block_bfp8_moments_kernel(
    const float* gradient,
    signed char* fast_payload,
    float* fast_scales,
    signed char* slow_payload,
    float* slow_scales,
    float* fast_roundtrip,
    float* slow_roundtrip,
    int length,
    float beta_fast,
    float beta_slow,
    bool nesterov,
    int* finite_status) {
    constexpr int block_size = 128;
    __shared__ float fast_maximum[block_size];
    __shared__ float slow_maximum[block_size];
    __shared__ int invalid;
    if (threadIdx.x == 0)
        invalid = *finite_status;
    __syncthreads();

    const int index = blockIdx.x * block_size + threadIdx.x;
    float fast = 0.f;
    float slow = 0.f;
    if (index < length && invalid == 0) {
        const float g = gradient[index];
        fast = fmaf(
            beta_fast,
            static_cast<float>(fast_payload[index]) * fast_scales[blockIdx.x],
            (1.f - beta_fast) * g);
        slow = nesterov
            ? fmaf(beta_fast, fast, (1.f - beta_fast) * g)
            : fmaf(beta_slow,
                static_cast<float>(slow_payload[index])
                    * slow_scales[blockIdx.x],
                (1.f - beta_slow) * g);
        if (!isfinite(g) || !isfinite(fast) || !isfinite(slow))
            atomicExch(&invalid, 1);
    }
    fast_maximum[threadIdx.x] = fabsf(fast);
    slow_maximum[threadIdx.x] = fabsf(slow);
    __syncthreads();
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            fast_maximum[threadIdx.x] = fmaxf(
                fast_maximum[threadIdx.x], fast_maximum[threadIdx.x + stride]);
            slow_maximum[threadIdx.x] = fmaxf(
                slow_maximum[threadIdx.x], slow_maximum[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (invalid != 0) {
        if (threadIdx.x == 0)
            atomicExch(finite_status, 1);
        return;
    }
    if (threadIdx.x == 0) {
        fast_scales[blockIdx.x] =
            bfp8_scale_from_maximum(fast_maximum[0]);
        slow_scales[blockIdx.x] =
            bfp8_scale_from_maximum(slow_maximum[0]);
    }
    __syncthreads();
    if (index >= length)
        return;
    const float fast_scale = fast_scales[blockIdx.x];
    const float slow_scale = slow_scales[blockIdx.x];
    const int fast_code = bfp8_quantized_code(fast, fast_scale);
    const int slow_code = bfp8_quantized_code(slow, slow_scale);
    fast_payload[index] = static_cast<signed char>(fast_code);
    slow_payload[index] = static_cast<signed char>(slow_code);
    fast_roundtrip[index] = static_cast<float>(fast_code) * fast_scale;
    slow_roundtrip[index] = static_cast<float>(slow_code) * slow_scale;
}

// Tensor-wide BFP8 must inspect the complete tensor before any payload byte
// can be published. Running that reduction in the generic one-block kernel
// leaves only 256 threads active even for multi-million-element parameters.
// Use a bounded grid-stride reduction and an exact non-negative atomic max;
// IEEE754 bit ordering is identical to numeric ordering for fabsf results.
__global__ void bfp8_quantize_f32_tensor_max_kernel(
    const float* source,
    float* maximum,
    int length) {
    float local_maximum = 0.0f;
    const int stride = blockDim.x * gridDim.x;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < length;
         index += stride) {
        local_maximum = fmaxf(local_maximum, fabsf(source[index]));
    }

    __shared__ float reduction[256];
    reduction[threadIdx.x] = local_maximum;
    __syncthreads();
    for (int reduction_stride = blockDim.x / 2;
         reduction_stride > 0;
         reduction_stride >>= 1) {
        if (threadIdx.x < reduction_stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x],
                reduction[threadIdx.x + reduction_stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMax(
            reinterpret_cast<unsigned int*>(maximum),
            __float_as_uint(reduction[0]));
    }
}

__global__ void bfp8_quantize_f32_tensor_payload_kernel(
    const float* source,
    signed char* payload,
    const float* maximum,
    int length) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    const float scale = bfp8_scale_from_maximum(maximum[0]);
    payload[index] = static_cast<signed char>(
        bfp8_quantized_code(source[index], scale));
}

__global__ void bfp8_dequantize_f32_kernel(
    const signed char* payload,
    const float* scales,
    float* destination,
    int length,
    int quantization_block_size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    destination[index] =
        static_cast<float>(payload[index]) *
        scales[index / quantization_block_size];
}

__global__ void bfp8_dequantize_bf16_kernel(
    const signed char* payload,
    const float* scales,
    __nv_bfloat16* destination,
    int length,
    int quantization_block_size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    const float value =
        static_cast<float>(payload[index]) *
        scales[index / quantization_block_size];
    destination[index] = __float2bfloat16_rn(value);
}

// Assign one warp to each common mix8_32 scale block so the scale is read once
// per warp and the runtime integer division in the scalar kernels disappears.
// Eight independent scale blocks share a CTA; this is especially important for
// 32-value blocks, where the generic one-CTA-per-scale implementation otherwise
// leaves seven eighths of a 256-thread CTA idle.
constexpr int kBfp8ScaleBlocksPerCta = 8;
constexpr int kMix8DiagnosticMaximumCtas = 4096;

__global__ void bfp8_dequantize_f32_block32_kernel(
    const signed char* payload,
    const float* scales,
    float* destination,
    int length) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    // A 256-thread CTA contains eight warps, and each aligned warp owns one
    // complete 32-value scale block. The shift is the compile-time-specialized
    // form of the generic runtime division.
    destination[index] =
        static_cast<float>(payload[index]) * scales[index >> 5];
}

__global__ void bfp8_dequantize_bf16_block32_kernel(
    const signed char* payload,
    const float* scales,
    __nv_bfloat16* destination,
    int length) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    destination[index] = __float2bfloat16_rn(
        static_cast<float>(payload[index]) * scales[index >> 5]);
}

template <int QuantizationBlockSize>
__global__ void bfp8_quantize_f32_warp_blocks_kernel(
    const float* source,
    signed char* payload,
    float* scales,
    int length,
    int scale_count) {
    const int warp = threadIdx.x / warpSize;
    const int lane = threadIdx.x & (warpSize - 1);
    const int scale_index =
        blockIdx.x * kBfp8ScaleBlocksPerCta + warp;
    if (scale_index >= scale_count)
        return;
    const int start = scale_index * QuantizationBlockSize;
    const int end = min(length, start + QuantizationBlockSize);
    float maximum = 0.f;
    for (int index = start + lane; index < end; index += warpSize) {
        maximum = fmaxf(maximum, fabsf(source[index]));
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        maximum = fmaxf(
            maximum,
            __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    const float scale = bfp8_scale_from_maximum(
        __shfl_sync(0xffffffffu, maximum, 0));
    if (lane == 0)
        scales[scale_index] = scale;
    for (int index = start + lane; index < end; index += warpSize) {
        payload[index] = static_cast<signed char>(
            bfp8_quantized_code(source[index], scale));
    }
}

// Diagnostic publication is deliberately fused with quantization.  Reading
// the previous payload before replacing it gives an exact code-change count;
// comparing the FP32 authority with the newly published value gives the
// retained master residual in units of the new block quantum.  A bounded,
// grid-stride launch keeps aggregate atomic traffic independent of model size.
template <int QuantizationBlockSize>
__global__ void bfp8_quantize_f32_diagnostic_warp_blocks_kernel(
    const float* source,
    signed char* payload,
    float* scales,
    int length,
    int scale_count,
    nntrain_mix8_diagnostic_accumulator* metrics) {
    const int warp = threadIdx.x / warpSize;
    const int lane = threadIdx.x & (warpSize - 1);
    float local_residual_ratio_squared_sum = 0.f;
    unsigned long long local_changed_code_count = 0;
    unsigned long long local_element_count = 0;

    for (int scale_index =
             blockIdx.x * kBfp8ScaleBlocksPerCta + warp;
         scale_index < scale_count;
         scale_index += gridDim.x * kBfp8ScaleBlocksPerCta) {
        const int start = scale_index * QuantizationBlockSize;
        const int end = min(length, start + QuantizationBlockSize);
        float maximum = 0.f;
        for (int index = start + lane; index < end; index += warpSize)
            maximum = fmaxf(maximum, fabsf(source[index]));
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            maximum = fmaxf(
                maximum,
                __shfl_down_sync(0xffffffffu, maximum, offset));
        }
        const float scale = bfp8_scale_from_maximum(
            __shfl_sync(0xffffffffu, maximum, 0));
        if (lane == 0)
            scales[scale_index] = scale;

        for (int index = start + lane; index < end; index += warpSize) {
            const signed char previous_code = payload[index];
            const int quantized = bfp8_quantized_code(source[index], scale);
            payload[index] = static_cast<signed char>(quantized);
            local_changed_code_count +=
                previous_code != static_cast<signed char>(quantized);
            ++local_element_count;
            const float residual = source[index]
                - static_cast<float>(quantized) * scale;
            const float normalized = __fdiv_rn(residual, scale);
            local_residual_ratio_squared_sum += normalized * normalized;
        }
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_residual_ratio_squared_sum += __shfl_down_sync(
            0xffffffffu, local_residual_ratio_squared_sum, offset);
        local_changed_code_count += __shfl_down_sync(
            0xffffffffu, local_changed_code_count, offset);
        local_element_count += __shfl_down_sync(
            0xffffffffu, local_element_count, offset);
    }

    __shared__ float residual_by_warp[kBfp8ScaleBlocksPerCta];
    __shared__ unsigned long long
        changed_by_warp[kBfp8ScaleBlocksPerCta];
    __shared__ unsigned long long
        elements_by_warp[kBfp8ScaleBlocksPerCta];
    if (lane == 0) {
        residual_by_warp[warp] = local_residual_ratio_squared_sum;
        changed_by_warp[warp] = local_changed_code_count;
        elements_by_warp[warp] = local_element_count;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float residual_sum = 0.f;
        unsigned long long changed_sum = 0;
        unsigned long long element_sum = 0;
        for (int index = 0; index < kBfp8ScaleBlocksPerCta; ++index) {
            residual_sum += residual_by_warp[index];
            changed_sum += changed_by_warp[index];
            element_sum += elements_by_warp[index];
        }
        atomicAdd(
            &metrics->residual_step_ratio_squared_sum,
            static_cast<double>(residual_sum));
        atomicAdd(&metrics->changed_code_count, changed_sum);
        atomicAdd(&metrics->element_count, element_sum);
    }
}

__global__ void bfp8_quantize_f32_diagnostic_kernel(
    const float* source,
    signed char* payload,
    float* scales,
    int length,
    int quantization_block_size,
    int scale_count,
    nntrain_mix8_diagnostic_accumulator* metrics) {
    float local_residual_ratio_squared_sum = 0.f;
    unsigned long long local_changed_code_count = 0;
    unsigned long long local_element_count = 0;
    __shared__ float maximum_reduction[256];

    for (int scale_index = blockIdx.x;
         scale_index < scale_count;
         scale_index += gridDim.x) {
        const long long start =
            static_cast<long long>(scale_index) * quantization_block_size;
        const long long end = min(
            static_cast<long long>(length),
            start + quantization_block_size);
        float local_maximum = 0.f;
        for (long long index = start + threadIdx.x;
             index < end;
             index += blockDim.x) {
            local_maximum = fmaxf(local_maximum, fabsf(source[index]));
        }
        maximum_reduction[threadIdx.x] = local_maximum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                maximum_reduction[threadIdx.x] = fmaxf(
                    maximum_reduction[threadIdx.x],
                    maximum_reduction[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        const float scale =
            bfp8_scale_from_maximum(maximum_reduction[0]);
        if (threadIdx.x == 0)
            scales[scale_index] = scale;
        __syncthreads();

        for (long long index = start + threadIdx.x;
             index < end;
             index += blockDim.x) {
            const signed char previous_code = payload[index];
            const int quantized = bfp8_quantized_code(source[index], scale);
            payload[index] = static_cast<signed char>(quantized);
            local_changed_code_count +=
                previous_code != static_cast<signed char>(quantized);
            ++local_element_count;
            const float residual = source[index]
                - static_cast<float>(quantized) * scale;
            const float normalized = __fdiv_rn(residual, scale);
            local_residual_ratio_squared_sum += normalized * normalized;
        }
        __syncthreads();
    }

    __shared__ float residual_reduction[256];
    __shared__ unsigned long long changed_reduction[256];
    __shared__ unsigned long long element_reduction[256];
    residual_reduction[threadIdx.x] =
        local_residual_ratio_squared_sum;
    changed_reduction[threadIdx.x] = local_changed_code_count;
    element_reduction[threadIdx.x] = local_element_count;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            residual_reduction[threadIdx.x] +=
                residual_reduction[threadIdx.x + stride];
            changed_reduction[threadIdx.x] +=
                changed_reduction[threadIdx.x + stride];
            element_reduction[threadIdx.x] +=
                element_reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(
            &metrics->residual_step_ratio_squared_sum,
            static_cast<double>(residual_reduction[0]));
        atomicAdd(&metrics->changed_code_count, changed_reduction[0]);
        atomicAdd(&metrics->element_count, element_reduction[0]);
    }
}

template <int QuantizationBlockSize>
__global__ void bfp8_quantize_f32_roundtrip_warp_blocks_kernel(
    float* source,
    signed char* payload,
    float* scales,
    int length,
    int scale_count) {
    const int warp = threadIdx.x / warpSize;
    const int lane = threadIdx.x & (warpSize - 1);
    const int scale_index =
        blockIdx.x * kBfp8ScaleBlocksPerCta + warp;
    if (scale_index >= scale_count)
        return;
    const int start = scale_index * QuantizationBlockSize;
    const int end = min(length, start + QuantizationBlockSize);
    float maximum = 0.f;
    for (int index = start + lane; index < end; index += warpSize) {
        maximum = fmaxf(maximum, fabsf(source[index]));
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        maximum = fmaxf(
            maximum,
            __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    const float scale = bfp8_scale_from_maximum(
        __shfl_sync(0xffffffffu, maximum, 0));
    if (lane == 0)
        scales[scale_index] = scale;
    for (int index = start + lane; index < end; index += warpSize) {
        const int quantized = bfp8_quantized_code(source[index], scale);
        payload[index] = static_cast<signed char>(quantized);
        source[index] = static_cast<float>(quantized) * scale;
    }
}

template <int QuantizationBlockSize>
__global__ void bfp8_dequantize_bf16_warp_blocks_kernel(
    const signed char* payload,
    const float* scales,
    __nv_bfloat16* destination,
    int length,
    int scale_count) {
    const int warp = threadIdx.x / warpSize;
    const int lane = threadIdx.x & (warpSize - 1);
    const int scale_index =
        blockIdx.x * kBfp8ScaleBlocksPerCta + warp;
    if (scale_index >= scale_count)
        return;
    const int start = scale_index * QuantizationBlockSize;
    const int end = min(length, start + QuantizationBlockSize);
    const float scale = scales[scale_index];
    if (end - start == QuantizationBlockSize
        && QuantizationBlockSize == 128) {
        // One lane owns four adjacent payload bytes.  The 128-value scale
        // block is naturally aligned, so char4 loads and bfloat162 stores
        // turn 128 scalar memory instructions into 32 + 64 vector accesses.
        const char4 quantized =
            reinterpret_cast<const char4*>(payload + start)[lane];
        __nv_bfloat162* output =
            reinterpret_cast<__nv_bfloat162*>(destination + start);
        output[lane * 2] = __floats2bfloat162_rn(
            static_cast<float>(quantized.x) * scale,
            static_cast<float>(quantized.y) * scale);
        output[lane * 2 + 1] = __floats2bfloat162_rn(
            static_cast<float>(quantized.z) * scale,
            static_cast<float>(quantized.w) * scale);
        return;
    }
    for (int index = start + lane; index < end; index += warpSize) {
        destination[index] = __float2bfloat16_rn(
            static_cast<float>(payload[index]) * scale);
    }
}

__global__ void bfp8_dequantize_bf16_tensor_kernel(
    const signed char* payload,
    const float* scales,
    __nv_bfloat16* destination,
    int length) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) {
        destination[index] = __float2bfloat16_rn(
            static_cast<float>(payload[index]) * scales[0]);
    }
}

__global__ void bfp8_quantize_bf16_kernel(
    const __nv_bfloat16* source,
    signed char* payload,
    float* scales,
    int length,
    int quantization_block_size) {
    const long long start =
        static_cast<long long>(blockIdx.x) * quantization_block_size;
    const long long end = min(
        static_cast<long long>(length),
        start + quantization_block_size);

    float local_maximum = 0.0f;
    for (long long index = start + threadIdx.x;
         index < end;
         index += blockDim.x) {
        local_maximum = fmaxf(
            local_maximum,
            fabsf(__bfloat162float(source[index])));
    }

    __shared__ float reduction[256];
    reduction[threadIdx.x] = local_maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x],
                reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        scales[blockIdx.x] = bfp8_scale_from_maximum(reduction[0]);
    }
    __syncthreads();

    const float scale = scales[blockIdx.x];
    for (long long index = start + threadIdx.x;
         index < end;
         index += blockDim.x) {
        int quantized = bfp8_quantized_code(
            __bfloat162float(source[index]), scale);
        payload[index] = static_cast<signed char>(quantized);
    }
}

template <int QuantizationBlockSize>
__global__ void bfp8_quantize_bf16_warp_blocks_kernel(
    const __nv_bfloat16* source,
    signed char* payload,
    float* scales,
    int length,
    int scale_count) {
    const int warp = threadIdx.x / warpSize;
    const int lane = threadIdx.x & (warpSize - 1);
    const int scale_index =
        blockIdx.x * kBfp8ScaleBlocksPerCta + warp;
    if (scale_index >= scale_count)
        return;
    const int start = scale_index * QuantizationBlockSize;
    const int end = min(length, start + QuantizationBlockSize);
    float maximum = 0.f;
    if (end - start == QuantizationBlockSize
        && QuantizationBlockSize == 128) {
        const int base = start + lane * 4;
        const float x = __bfloat162float(source[base]);
        const float y = __bfloat162float(source[base + 1]);
        const float z = __bfloat162float(source[base + 2]);
        const float w = __bfloat162float(source[base + 3]);
        maximum = fmaxf(
            fmaxf(fabsf(x), fabsf(y)),
            fmaxf(fabsf(z), fabsf(w)));
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            maximum = fmaxf(
                maximum,
                __shfl_down_sync(0xffffffffu, maximum, offset));
        }
        const float scale = bfp8_scale_from_maximum(
            __shfl_sync(0xffffffffu, maximum, 0));
        if (lane == 0)
            scales[scale_index] = scale;
        char4 quantized;
        quantized.x = static_cast<signed char>(
            bfp8_quantized_code(x, scale));
        quantized.y = static_cast<signed char>(
            bfp8_quantized_code(y, scale));
        quantized.z = static_cast<signed char>(
            bfp8_quantized_code(z, scale));
        quantized.w = static_cast<signed char>(
            bfp8_quantized_code(w, scale));
        reinterpret_cast<char4*>(payload + start)[lane] = quantized;
        return;
    }
    for (int index = start + lane; index < end; index += warpSize) {
        maximum = fmaxf(
            maximum,
            fabsf(__bfloat162float(source[index])));
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        maximum = fmaxf(
            maximum,
            __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    const float scale = bfp8_scale_from_maximum(
        __shfl_sync(0xffffffffu, maximum, 0));
    if (lane == 0)
        scales[scale_index] = scale;
    for (int index = start + lane; index < end; index += warpSize) {
        payload[index] = static_cast<signed char>(bfp8_quantized_code(
            __bfloat162float(source[index]), scale));
    }
}

__global__ void bfp8_quantize_bf16_tensor_max_kernel(
    const __nv_bfloat16* source,
    float* maximum,
    int length) {
    float local_maximum = 0.0f;
    const int stride = blockDim.x * gridDim.x;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < length;
         index += stride) {
        local_maximum = fmaxf(
            local_maximum,
            fabsf(__bfloat162float(source[index])));
    }

    __shared__ float reduction[256];
    reduction[threadIdx.x] = local_maximum;
    __syncthreads();
    for (int reduction_stride = blockDim.x / 2;
         reduction_stride > 0;
         reduction_stride >>= 1) {
        if (threadIdx.x < reduction_stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x],
                reduction[threadIdx.x + reduction_stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMax(
            reinterpret_cast<unsigned int*>(maximum),
            __float_as_uint(reduction[0]));
    }
}

__global__ void bfp8_quantize_bf16_tensor_payload_kernel(
    const __nv_bfloat16* source,
    signed char* payload,
    const float* maximum,
    int length) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    const float scale = bfp8_scale_from_maximum(maximum[0]);
    payload[index] = static_cast<signed char>(bfp8_quantized_code(
        __bfloat162float(source[index]), scale));
}

__device__ __forceinline__ float bfp8_scaled_int32_value(
    const int* source,
    const float* left_scales,
    const float* right_scales,
    const signed char* bias_payload,
    const float* bias_scales,
    int index,
    int output_width,
    int bias_block_size,
    bool apply_relu) {
    float value = static_cast<float>(source[index]) *
        left_scales[0] * right_scales[0];
    if (bias_payload != nullptr) {
        const int bias_index = index % output_width;
        value += static_cast<float>(bias_payload[bias_index]) *
            bias_scales[bias_index / bias_block_size];
    }
    return apply_relu ? fmaxf(value, 0.0f) : value;
}

__global__ void bfp8_requantize_i32_kernel(
    const int* source,
    const float* left_scales,
    const float* right_scales,
    const signed char* bias_payload,
    const float* bias_scales,
    signed char* payload,
    float* scales,
    int length,
    int output_width,
    int quantization_block_size,
    int bias_block_size,
    bool apply_relu) {
    const long long start =
        static_cast<long long>(blockIdx.x) * quantization_block_size;
    const long long end = min(
        static_cast<long long>(length),
        start + quantization_block_size);

    float local_maximum = 0.0f;
    for (long long index = start + threadIdx.x;
         index < end;
         index += blockDim.x) {
        local_maximum = fmaxf(
            local_maximum,
            fabsf(bfp8_scaled_int32_value(
                source,
                left_scales,
                right_scales,
                bias_payload,
                bias_scales,
                static_cast<int>(index),
                output_width,
                bias_block_size,
                apply_relu)));
    }

    __shared__ float reduction[256];
    reduction[threadIdx.x] = local_maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x],
                reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        scales[blockIdx.x] = bfp8_scale_from_maximum(reduction[0]);
    }
    __syncthreads();

    const float output_scale = scales[blockIdx.x];
    for (long long index = start + threadIdx.x;
         index < end;
         index += blockDim.x) {
        const float value = bfp8_scaled_int32_value(
            source,
            left_scales,
            right_scales,
            bias_payload,
            bias_scales,
            static_cast<int>(index),
            output_width,
            bias_block_size,
            apply_relu);
        int quantized = bfp8_quantized_code(value, output_scale);
        payload[index] = static_cast<signed char>(quantized);
    }
}

__global__ void bfp8_requantize_i32_tensor_max_kernel(
    const int* source,
    const float* left_scales,
    const float* right_scales,
    const signed char* bias_payload,
    const float* bias_scales,
    float* maximum,
    int length,
    int output_width,
    int bias_block_size,
    bool apply_relu) {
    float local_maximum = 0.0f;
    const int stride = blockDim.x * gridDim.x;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < length;
         index += stride) {
        local_maximum = fmaxf(
            local_maximum,
            fabsf(bfp8_scaled_int32_value(
                source,
                left_scales,
                right_scales,
                bias_payload,
                bias_scales,
                index,
                output_width,
                bias_block_size,
                apply_relu)));
    }

    __shared__ float reduction[256];
    reduction[threadIdx.x] = local_maximum;
    __syncthreads();
    for (int reduction_stride = blockDim.x / 2;
         reduction_stride > 0;
         reduction_stride >>= 1) {
        if (threadIdx.x < reduction_stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x],
                reduction[threadIdx.x + reduction_stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // Every candidate is non-negative, so IEEE754 bit ordering matches
        // numeric ordering and unsigned atomicMax is exact.
        atomicMax(
            reinterpret_cast<unsigned int*>(maximum),
            __float_as_uint(reduction[0]));
    }
}

__global__ void bfp8_requantize_i32_tensor_payload_kernel(
    const int* source,
    const float* left_scales,
    const float* right_scales,
    const signed char* bias_payload,
    const float* bias_scales,
    signed char* payload,
    const float* maximum,
    int length,
    int output_width,
    int bias_block_size,
    bool apply_relu) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    const float maximum_value = maximum[0];
    const float scale = bfp8_scale_from_maximum(maximum_value);
    const float value = bfp8_scaled_int32_value(
        source,
        left_scales,
        right_scales,
        bias_payload,
        bias_scales,
        index,
        output_width,
        bias_block_size,
        apply_relu);
    int quantized = bfp8_quantized_code(value, scale);
    payload[index] = static_cast<signed char>(quantized);
}

__global__ void bfp8_finalize_tensor_scale_kernel(float* maximum) {
    const float maximum_value = maximum[0];
    maximum[0] = bfp8_scale_from_maximum(maximum_value);
}

__global__ void bfp8_transpose_i8_kernel(
    const signed char* source,
    signed char* destination,
    int rows,
    int columns) {
    __shared__ signed char tile[32][33];
    int column = blockIdx.x * 32 + threadIdx.x;
    int row = blockIdx.y * 32 + threadIdx.y;
    for (int offset = 0; offset < 32; offset += 8) {
        if (column < columns && row + offset < rows) {
            tile[threadIdx.y + offset][threadIdx.x] =
                source[(row + offset) * columns + column];
        }
    }
    __syncthreads();

    int destination_row = blockIdx.x * 32 + threadIdx.y;
    int destination_column = blockIdx.y * 32 + threadIdx.x;
    for (int offset = 0; offset < 32; offset += 8) {
        if (destination_row + offset < columns
            && destination_column < rows) {
            destination[(destination_row + offset) * rows +
                destination_column] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

__global__ void graph_rng_advance_kernel(
    unsigned long long* step_counter) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        atomicAdd(step_counter, 1ull);
}

__global__ void graph_rng_set_kernel(
    unsigned long long* step_counter,
    unsigned long long value) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *step_counter = value;
}

__device__ __forceinline__ unsigned int graph_rng_hash(
    unsigned long long step,
    unsigned long long operation_seed,
    unsigned int index) {
    unsigned long long value = step
        ^ operation_seed
        ^ (static_cast<unsigned long long>(index) *
            0x9e3779b97f4a7c15ull);
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    value ^= value >> 31;
    return static_cast<unsigned int>(value >> 32);
}

__global__ void graph_dropout_mask_kernel(
    const unsigned long long* step_counter,
    unsigned int seed,
    unsigned int threshold,
    float keep_scale,
    float* output,
    int length) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    const unsigned int random = graph_rng_hash(
        *step_counter,
        seed,
        static_cast<unsigned int>(index));
    output[index] = random >= threshold ? keep_scale : 0.0f;
}

__device__ __forceinline__ float graph_dropout_multiplier(
    const unsigned long long* step_counter,
    unsigned long long operation_seed,
    int index,
    unsigned int threshold,
    float keep_scale) {
    const unsigned int random = graph_rng_hash(
        *step_counter,
        operation_seed,
        static_cast<unsigned int>(index));
    return random >= threshold ? keep_scale : 0.0f;
}

__global__ void graph_dropout_forward_float_kernel(
    const unsigned long long* step_counter,
    unsigned long long operation_seed,
    unsigned int threshold,
    float keep_scale,
    const float* input,
    float* output,
    int length) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    output[index] = input[index] * graph_dropout_multiplier(
        step_counter,
        operation_seed,
        index,
        threshold,
        keep_scale);
}

__global__ void graph_dropout_forward_bf16_kernel(
    const unsigned long long* step_counter,
    unsigned long long operation_seed,
    unsigned int threshold,
    float keep_scale,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int length) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    const float value = __bfloat162float(input[index])
        * graph_dropout_multiplier(
            step_counter,
            operation_seed,
            index,
            threshold,
            keep_scale);
    output[index] = __float2bfloat16_rn(value);
}

__global__ void graph_add_dropout_forward_float_kernel(
    const unsigned long long* step_counter,
    unsigned long long operation_seed,
    unsigned int threshold,
    float keep_scale,
    const float* residual,
    const float* branch,
    float* output,
    int length) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    output[index] = residual[index] + branch[index]
        * graph_dropout_multiplier(
            step_counter,
            operation_seed,
            index,
            threshold,
            keep_scale);
}

__global__ void graph_add_dropout_forward_bf16_kernel(
    const unsigned long long* step_counter,
    unsigned long long operation_seed,
    unsigned int threshold,
    float keep_scale,
    const __nv_bfloat16* residual,
    const __nv_bfloat16* branch,
    __nv_bfloat16* output,
    int length) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    const float value = __bfloat162float(residual[index])
        + __bfloat162float(branch[index])
            * graph_dropout_multiplier(
                step_counter,
                operation_seed,
                index,
                threshold,
                keep_scale);
    output[index] = __float2bfloat16_rn(value);
}

__global__ void graph_dropout_backward_float_kernel(
    const unsigned long long* step_counter,
    unsigned long long operation_seed,
    unsigned int threshold,
    float keep_scale,
    const float* output_gradient,
    float* input_gradient,
    int length) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    input_gradient[index] += output_gradient[index]
        * graph_dropout_multiplier(
            step_counter,
            operation_seed,
            index,
            threshold,
            keep_scale);
}

__global__ void graph_add_dropout_backward_float_kernel(
    const unsigned long long* step_counter,
    unsigned long long operation_seed,
    unsigned int threshold,
    float keep_scale,
    const float* output_gradient,
    float* residual_gradient,
    float* branch_gradient,
    int length,
    int same_parent) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    const float gradient = output_gradient[index];
    const float multiplier = graph_dropout_multiplier(
        step_counter,
        operation_seed,
        index,
        threshold,
        keep_scale);
    if (same_parent != 0) {
        residual_gradient[index] += gradient * (1.0f + multiplier);
    } else {
        residual_gradient[index] += gradient;
        branch_gradient[index] += gradient * multiplier;
    }
}
}

NNTRAIN_EXPORT std::uint32_t nntrain_abi_version() {
    return nntrain_abi_version_value;
}

NNTRAIN_EXPORT int nntrain_last_error(
    nntrain_native_error_info* destination,
    size_t destination_size) {
    if (destination == nullptr ||
        destination_size < sizeof(nntrain_native_error_info)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    std::lock_guard<std::mutex> guard(last_error_mutex);
    *destination = last_error;
    return static_cast<int>(cudaSuccess);
}

NNTRAIN_EXPORT int nntrain_error_snapshot(
    int status,
    int device,
    std::uint32_t operation,
    nntrain_native_error_info* destination,
    size_t destination_size) {
    if (destination == nullptr ||
        destination_size < sizeof(nntrain_native_error_info)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    std::lock_guard<std::mutex> guard(last_error_mutex);
    const std::uint64_t available =
        error_sequence < error_ring_capacity
            ? error_sequence
            : error_ring_capacity;
    for (std::uint64_t offset = 0; offset < available; ++offset) {
        const std::uint64_t sequence = error_sequence - offset;
        const nntrain_native_error_info& candidate =
            error_ring[(sequence - 1) % error_ring_capacity];
        if (candidate.sequence == sequence &&
            candidate.status == status &&
            candidate.device == device &&
            candidate.operation == operation) {
            *destination = candidate;
            return static_cast<int>(cudaSuccess);
        }
    }
    return static_cast<int>(cudaErrorInvalidValue);
}

NNTRAIN_EXPORT int nntrain_capability_bitmap(
    int device,
    unsigned long long* bitmap,
    int* compute_capability_major,
    int* compute_capability_minor) {
    if (bitmap == nullptr || compute_capability_major == nullptr ||
        compute_capability_minor == nullptr) {
        return complete(
            nntrain_operation_capabilities,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }

    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_capabilities,
            device,
            static_cast<int>(status));
    }

    constexpr unsigned long long tensor_cores = 1ull << 0;
    constexpr unsigned long long bfloat16 = 1ull << 1;
    constexpr unsigned long long flash_attention = 1ull << 2;
    constexpr unsigned long long fused_layer_norm = 1ull << 3;
    constexpr unsigned long long forget_memory = 1ull << 4;
    constexpr unsigned long long block_reduced_muon = 1ull << 5;
    constexpr unsigned long long asynchronous_gradient_reduction = 1ull << 6;
    constexpr unsigned long long cuda_graphs = 1ull << 7;
    constexpr unsigned long long bfp8_quantization = 1ull << 8;
    constexpr unsigned long long int8_tensor_cores = 1ull << 9;
    constexpr unsigned long long fused_first_order_optimizers = 1ull << 10;

    unsigned long long features = 0;
    // The distributed binary contains native cubins for SM80/86/89/90 and a
    // forward-compatible PTX image. Its BF16/Tensor Core paths require SM80+.
    if (properties.major >= 8) {
        features = tensor_cores |
            bfloat16 |
            flash_attention |
            fused_layer_norm |
            forget_memory |
            block_reduced_muon |
            asynchronous_gradient_reduction |
            cuda_graphs |
            bfp8_quantization |
            int8_tensor_cores |
            fused_first_order_optimizers;
    }

    *bitmap = features;
    *compute_capability_major = properties.major;
    *compute_capability_minor = properties.minor;
    return static_cast<int>(cudaSuccess);
}

NNTRAIN_EXPORT int nntrain_cuda_bfp8_quantize_f32(
    int device,
    const float* source,
    signed char* payload,
    float* scales,
    int length,
    int quantization_block_size,
    void* stream) {
    if (source == nullptr || payload == nullptr || scales == nullptr ||
        length <= 0 || quantization_block_size <= 0) {
        return complete(
            nntrain_operation_bfp8_quantize,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_bfp8_quantize,
            device,
            status);
    }

    const long long blocks64 =
        (static_cast<long long>(length) + quantization_block_size - 1) /
        quantization_block_size;
    if (blocks64 > 2147483647ll) {
        return complete(
            nntrain_operation_bfp8_quantize,
            device,
            static_cast<int>(cudaErrorInvalidConfiguration));
    }
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (quantization_block_size >= length) {
        status = static_cast<int>(cudaMemsetAsync(
            scales,
            0,
            sizeof(float),
            cuda_stream));
        if (status != cudaSuccess) {
            return complete(
                nntrain_operation_bfp8_quantize,
                device,
                status);
        }
        constexpr int threads = 256;
        const int element_blocks = (length - 1) / threads + 1;
        const int reduction_blocks = min(element_blocks, 1024);
        bfp8_quantize_f32_tensor_max_kernel<<<
            reduction_blocks, threads, 0, cuda_stream>>>(
                source, scales, length);
        bfp8_quantize_f32_tensor_payload_kernel<<<
            element_blocks, threads, 0, cuda_stream>>>(
                source, payload, scales, length);
        bfp8_finalize_tensor_scale_kernel<<<1, 1, 0, cuda_stream>>>(scales);
    }
    else if (quantization_block_size == 32) {
        constexpr int threads = kBfp8ScaleBlocksPerCta * 32;
        const int scale_count = static_cast<int>(blocks64);
        const int scale_block_grid =
            (scale_count + kBfp8ScaleBlocksPerCta - 1) /
            kBfp8ScaleBlocksPerCta;
        bfp8_quantize_f32_warp_blocks_kernel<32><<<
            scale_block_grid, threads, 0, cuda_stream>>>(
                source, payload, scales, length, scale_count);
    }
    else {
        bfp8_quantize_f32_kernel<<<
            static_cast<int>(blocks64),
            256,
            0,
            cuda_stream>>>(
                source,
                payload,
                scales,
                length,
                quantization_block_size);
    }
    return complete(
        nntrain_operation_bfp8_quantize,
        device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_mix8_diagnostics_reset(
    int device,
    void* metrics,
    void* stream) {
    if (metrics == nullptr) {
        return complete(
            nntrain_operation_memset_async,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_memset_async,
            device,
            status);
    }
    status = static_cast<int>(cudaMemsetAsync(
        metrics,
        0,
        sizeof(nntrain_mix8_diagnostic_accumulator),
        reinterpret_cast<cudaStream_t>(stream)));
    return complete(nntrain_operation_memset_async, device, status);
}

NNTRAIN_EXPORT int nntrain_cuda_bfp8_quantize_f32_diagnostic(
    int device,
    const float* source,
    signed char* payload,
    float* scales,
    int length,
    int quantization_block_size,
    void* metrics,
    void* stream) {
    if (source == nullptr || payload == nullptr || scales == nullptr
        || metrics == nullptr || length <= 0
        || quantization_block_size <= 0) {
        return complete(
            nntrain_operation_bfp8_quantize,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_bfp8_quantize,
            device,
            status);
    }
    const long long blocks64 =
        (static_cast<long long>(length) + quantization_block_size - 1)
        / quantization_block_size;
    if (blocks64 > 2147483647ll) {
        return complete(
            nntrain_operation_bfp8_quantize,
            device,
            static_cast<int>(cudaErrorInvalidConfiguration));
    }

    const int scale_count = static_cast<int>(blocks64);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    auto* accumulator =
        reinterpret_cast<nntrain_mix8_diagnostic_accumulator*>(metrics);
    if (quantization_block_size == 32) {
        constexpr int threads = kBfp8ScaleBlocksPerCta * 32;
        const int required_ctas =
            (scale_count + kBfp8ScaleBlocksPerCta - 1)
            / kBfp8ScaleBlocksPerCta;
        const int grid = min(required_ctas, kMix8DiagnosticMaximumCtas);
        bfp8_quantize_f32_diagnostic_warp_blocks_kernel<32><<<
            grid, threads, 0, cuda_stream>>>(
                source,
                payload,
                scales,
                length,
                scale_count,
                accumulator);
    }
    else {
        const int grid = min(scale_count, kMix8DiagnosticMaximumCtas);
        bfp8_quantize_f32_diagnostic_kernel<<<
            grid, 256, 0, cuda_stream>>>(
                source,
                payload,
                scales,
                length,
                quantization_block_size,
                scale_count,
                accumulator);
    }
    return complete(
        nntrain_operation_bfp8_quantize,
        device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_bfp8_dequantize_f32(
    int device,
    const signed char* payload,
    const float* scales,
    float* destination,
    int length,
    int quantization_block_size,
    void* stream) {
    if (payload == nullptr || scales == nullptr || destination == nullptr ||
        length <= 0 || quantization_block_size <= 0) {
        return complete(
            nntrain_operation_bfp8_dequantize_f32,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_bfp8_dequantize_f32,
            device,
            status);
    }

    constexpr int threads = 256;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (quantization_block_size == 32) {
        const int blocks = (length + threads - 1) / threads;
        bfp8_dequantize_f32_block32_kernel<<<
            blocks, threads, 0, cuda_stream>>>(
                payload, scales, destination, length);
    }
    else {
        const int blocks = (length + threads - 1) / threads;
        bfp8_dequantize_f32_kernel<<<
            blocks,
            threads,
            0,
            cuda_stream>>>(
                payload,
                scales,
                destination,
                length,
                quantization_block_size);
    }
    return complete(
        nntrain_operation_bfp8_dequantize_f32,
        device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_bfp8_quantize_f32_roundtrip(
    int device,
    float* source,
    signed char* payload,
    float* scales,
    int length,
    int quantization_block_size,
    void* stream) {
    if (source == nullptr || payload == nullptr || scales == nullptr
        || length <= 0 || quantization_block_size <= 0
        || quantization_block_size >= length) {
        return complete(
            nntrain_operation_bfp8_quantize,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(nntrain_operation_bfp8_quantize, device, status);
    }
    const long long blocks64 =
        (static_cast<long long>(length) + quantization_block_size - 1)
        / quantization_block_size;
    if (blocks64 > 2147483647ll) {
        return complete(
            nntrain_operation_bfp8_quantize,
            device,
            static_cast<int>(cudaErrorInvalidConfiguration));
    }
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (quantization_block_size == 32) {
        constexpr int threads = kBfp8ScaleBlocksPerCta * 32;
        const int scale_count = static_cast<int>(blocks64);
        const int scale_block_grid =
            (scale_count + kBfp8ScaleBlocksPerCta - 1) /
            kBfp8ScaleBlocksPerCta;
        bfp8_quantize_f32_roundtrip_warp_blocks_kernel<32><<<
            scale_block_grid, threads, 0, cuda_stream>>>(
                source, payload, scales, length, scale_count);
    }
    else {
        bfp8_quantize_f32_roundtrip_kernel<<<
            static_cast<int>(blocks64),
            256,
            0,
            cuda_stream>>>(
                source,
                payload,
                scales,
                length,
                quantization_block_size);
    }
    return complete(
        nntrain_operation_bfp8_quantize,
        device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_adamw_block_bfp8_state(
    int device,
    float* data,
    const float* gradient,
    signed char* first_payload,
    float* first_scales,
    signed char* second_payload,
    float* second_scales,
    int length,
    float beta1,
    float beta2,
    float learning_rate,
    float weight_decay,
    float update_scale,
    float scaled_epsilon,
    int apply_weight_decay,
    int* finite_status,
    void* stream) {
    if (data == nullptr || gradient == nullptr || first_payload == nullptr
        || first_scales == nullptr || second_payload == nullptr
        || second_scales == nullptr || length <= 0
        || finite_status == nullptr) {
        return complete(
            nntrain_operation_bfp8_quantize,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_bfp8_quantize, device, status);
    constexpr int threads = 128;
    const int blocks = (length + threads - 1) / threads;
    adamw_block_bfp8_state_kernel<<<
        blocks,
        threads,
        0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
            data,
            gradient,
            first_payload,
            first_scales,
            second_payload,
            second_scales,
            length,
            beta1,
            beta2,
            learning_rate,
            weight_decay,
            update_scale,
            scaled_epsilon,
            apply_weight_decay,
            finite_status);
    return complete(
        nntrain_operation_bfp8_quantize,
        device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_nekomuon_block_bfp8_moments(
    int device,
    const float* gradient,
    signed char* fast_payload,
    float* fast_scales,
    signed char* slow_payload,
    float* slow_scales,
    float* fast_roundtrip,
    float* slow_roundtrip,
    int length,
    float beta_fast,
    float beta_slow,
    int* finite_status,
    void* stream) {
    if (gradient == nullptr || fast_payload == nullptr || fast_scales == nullptr
        || slow_payload == nullptr || slow_scales == nullptr
        || fast_roundtrip == nullptr || slow_roundtrip == nullptr
        || length <= 0 || finite_status == nullptr) {
        return complete(
            nntrain_operation_bfp8_quantize,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_bfp8_quantize, device, status);
    constexpr int threads = 128;
    const int blocks = (length + threads - 1) / threads;
    nekomuon_block_bfp8_moments_kernel<<<
        blocks,
        threads,
        0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
            gradient,
            fast_payload,
            fast_scales,
            slow_payload,
            slow_scales,
            fast_roundtrip,
            slow_roundtrip,
            length,
            beta_fast,
            beta_slow,
            false,
            finite_status);
    return complete(
        nntrain_operation_bfp8_quantize,
        device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_muon_block_bfp8_moments(
    int device, const float* gradient, signed char* fast_payload,
    float* fast_scales, signed char* direction_payload,
    float* direction_scales, float* fast_roundtrip,
    float* direction_roundtrip, int length, float beta,
    int* finite_status, void* stream) {
    if (gradient == nullptr || fast_payload == nullptr || fast_scales == nullptr
        || direction_payload == nullptr || direction_scales == nullptr
        || fast_roundtrip == nullptr || direction_roundtrip == nullptr
        || length <= 0 || finite_status == nullptr) {
        return complete(nntrain_operation_bfp8_quantize, device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_bfp8_quantize, device, status);
    constexpr int threads = 128;
    const int blocks = (length + threads - 1) / threads;
    nekomuon_block_bfp8_moments_kernel<<<blocks, threads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
            gradient, fast_payload, fast_scales, direction_payload,
            direction_scales, fast_roundtrip, direction_roundtrip,
            length, beta, beta, true, finite_status);
    return complete(nntrain_operation_bfp8_quantize, device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_bfp8_dequantize_bf16(
    int device,
    const signed char* payload,
    const float* scales,
    __nv_bfloat16* destination,
    int length,
    int quantization_block_size,
    void* stream) {
    if (payload == nullptr || scales == nullptr || destination == nullptr ||
        length <= 0 || quantization_block_size <= 0) {
        return complete(
            nntrain_operation_bfp8_dequantize_bf16,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_bfp8_dequantize_bf16,
            device,
            status);
    }

    constexpr int threads = 256;
    const int blocks = (length + threads - 1) / threads;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (quantization_block_size == 128) {
        const int scale_count = (length + 127) / 128;
        const int scale_block_grid =
            (scale_count + kBfp8ScaleBlocksPerCta - 1) /
            kBfp8ScaleBlocksPerCta;
        bfp8_dequantize_bf16_warp_blocks_kernel<128><<<
            scale_block_grid, threads, 0, cuda_stream>>>(
                payload, scales, destination, length, scale_count);
    }
    else if (quantization_block_size == 32) {
        bfp8_dequantize_bf16_block32_kernel<<<
            blocks, threads, 0, cuda_stream>>>(
                payload, scales, destination, length);
    }
    else if (quantization_block_size >= length) {
        bfp8_dequantize_bf16_tensor_kernel<<<
            blocks, threads, 0, cuda_stream>>>(
                payload, scales, destination, length);
    }
    else {
        bfp8_dequantize_bf16_kernel<<<
            blocks, threads, 0, cuda_stream>>>(
                payload,
                scales,
                destination,
                length,
                quantization_block_size);
    }
    return complete(
        nntrain_operation_bfp8_dequantize_bf16,
        device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_bfp8_quantize_bf16(
    int device,
    const __nv_bfloat16* source,
    signed char* payload,
    float* scales,
    int length,
    int quantization_block_size,
    void* stream) {
    if (source == nullptr || payload == nullptr || scales == nullptr ||
        length <= 0 || quantization_block_size <= 0) {
        return complete(
            nntrain_operation_bfp8_quantize_bf16,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_bfp8_quantize_bf16,
            device,
            status);
    }

    const long long blocks64 =
        (static_cast<long long>(length) + quantization_block_size - 1) /
        quantization_block_size;
    if (blocks64 > 2147483647ll) {
        return complete(
            nntrain_operation_bfp8_quantize_bf16,
            device,
            static_cast<int>(cudaErrorInvalidConfiguration));
    }
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (quantization_block_size >= length) {
        status = static_cast<int>(cudaMemsetAsync(
            scales,
            0,
            sizeof(float),
            cuda_stream));
        if (status != cudaSuccess) {
            return complete(
                nntrain_operation_bfp8_quantize_bf16,
                device,
                status);
        }
        constexpr int threads = 256;
        const int element_blocks = (length - 1) / threads + 1;
        const int reduction_blocks = min(element_blocks, 1024);
        bfp8_quantize_bf16_tensor_max_kernel<<<
            reduction_blocks, threads, 0, cuda_stream>>>(
                source, scales, length);
        bfp8_quantize_bf16_tensor_payload_kernel<<<
            element_blocks, threads, 0, cuda_stream>>>(
                source, payload, scales, length);
        bfp8_finalize_tensor_scale_kernel<<<1, 1, 0, cuda_stream>>>(scales);
    }
    else if (quantization_block_size == 128) {
        constexpr int threads =
            kBfp8ScaleBlocksPerCta * 32;
        const int scale_count = static_cast<int>(blocks64);
        const int scale_block_grid =
            (scale_count + kBfp8ScaleBlocksPerCta - 1) /
            kBfp8ScaleBlocksPerCta;
        bfp8_quantize_bf16_warp_blocks_kernel<128><<<
            scale_block_grid, threads, 0, cuda_stream>>>(
                source, payload, scales, length, scale_count);
    }
    else if (quantization_block_size == 32) {
        constexpr int threads =
            kBfp8ScaleBlocksPerCta * 32;
        const int scale_count = static_cast<int>(blocks64);
        const int scale_block_grid =
            (scale_count + kBfp8ScaleBlocksPerCta - 1) /
            kBfp8ScaleBlocksPerCta;
        bfp8_quantize_bf16_warp_blocks_kernel<32><<<
            scale_block_grid, threads, 0, cuda_stream>>>(
                source, payload, scales, length, scale_count);
    }
    else {
        bfp8_quantize_bf16_kernel<<<
            static_cast<int>(blocks64),
            256,
            0,
            cuda_stream>>>(
                source,
                payload,
                scales,
                length,
                quantization_block_size);
    }
    return complete(
        nntrain_operation_bfp8_quantize_bf16,
        device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_bfp8_requantize_i32(
    int device,
    const int* source,
    const float* left_scales,
    const float* right_scales,
    const signed char* bias_payload,
    const float* bias_scales,
    signed char* payload,
    float* scales,
    int length,
    int output_width,
    int quantization_block_size,
    int bias_block_size,
    int apply_relu,
    void* stream) {
    const bool has_bias = bias_payload != nullptr || bias_scales != nullptr;
    if (source == nullptr || left_scales == nullptr || right_scales == nullptr ||
        payload == nullptr || scales == nullptr || length <= 0 ||
        output_width <= 0 || quantization_block_size <= 0 ||
        (has_bias && (bias_payload == nullptr || bias_scales == nullptr ||
            bias_block_size <= 0))) {
        return complete(
            nntrain_operation_bfp8_requantize_i32,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_bfp8_requantize_i32,
            device,
            status);
    }

    const long long blocks64 =
        (static_cast<long long>(length) + quantization_block_size - 1) /
        quantization_block_size;
    if (blocks64 > 2147483647ll) {
        return complete(
            nntrain_operation_bfp8_requantize_i32,
            device,
            static_cast<int>(cudaErrorInvalidConfiguration));
    }
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (quantization_block_size >= length) {
        status = static_cast<int>(cudaMemsetAsync(
            scales,
            0,
            sizeof(float),
            cuda_stream));
        if (status != cudaSuccess) {
            return complete(
                nntrain_operation_bfp8_requantize_i32,
                device,
                status);
        }
        constexpr int threads = 256;
        const int element_blocks = (length + threads - 1) / threads;
        const int reduction_blocks = min(element_blocks, 1024);
        bfp8_requantize_i32_tensor_max_kernel<<<
            reduction_blocks,
            threads,
            0,
            cuda_stream>>>(
                source,
                left_scales,
                right_scales,
                bias_payload,
                bias_scales,
                scales,
                length,
                output_width,
                bias_block_size,
                apply_relu != 0);
        bfp8_requantize_i32_tensor_payload_kernel<<<
            element_blocks,
            threads,
            0,
            cuda_stream>>>(
                source,
                left_scales,
                right_scales,
                bias_payload,
                bias_scales,
                payload,
                scales,
                length,
                output_width,
                bias_block_size,
                apply_relu != 0);
        bfp8_finalize_tensor_scale_kernel<<<1, 1, 0, cuda_stream>>>(scales);
    } else {
        bfp8_requantize_i32_kernel<<<
            static_cast<int>(blocks64),
            256,
            0,
            cuda_stream>>>(
                source,
                left_scales,
                right_scales,
                bias_payload,
                bias_scales,
                payload,
                scales,
                length,
                output_width,
                quantization_block_size,
                bias_block_size,
                apply_relu != 0);
    }
    return complete(
        nntrain_operation_bfp8_requantize_i32,
        device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_bfp8_transpose_i8(
    int device,
    const signed char* source,
    signed char* destination,
    int rows,
    int columns,
    void* stream) {
    if (source == nullptr || destination == nullptr || rows <= 0 ||
        columns <= 0) {
        return complete(
            nntrain_operation_bfp8_transpose_i8,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    const long long length = static_cast<long long>(rows) * columns;
    if (length > 2147483647ll) {
        return complete(
            nntrain_operation_bfp8_transpose_i8,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_bfp8_transpose_i8,
            device,
            status);
    }
    const dim3 threads(32, 8);
    const dim3 blocks(
        (static_cast<unsigned int>(columns) + 31u) / 32u,
        (static_cast<unsigned int>(rows) + 31u) / 32u);
    bfp8_transpose_i8_kernel<<<
        blocks,
        threads,
        0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
            source,
            destination,
            rows,
            columns);
    return complete(
        nntrain_operation_bfp8_transpose_i8,
        device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_device_count(int* count) {
    if (count == nullptr)
        return complete(nntrain_operation_device_count, -1,
            static_cast<int>(cudaErrorInvalidValue));
    return complete(nntrain_operation_device_count, -1,
        static_cast<int>(cudaGetDeviceCount(count)));
}

NNTRAIN_EXPORT int nntrain_cuda_device_name(
    int device,
    char* destination,
    int capacity) {
    if (destination == nullptr || capacity <= 0)
        return complete(nntrain_operation_device_name, device,
            static_cast<int>(cudaErrorInvalidValue));
    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_device_name, device,
            static_cast<int>(status));
#if defined(_WIN32)
    strncpy_s(destination, static_cast<size_t>(capacity), properties.name,
        _TRUNCATE);
#else
    std::strncpy(destination, properties.name, static_cast<size_t>(capacity));
    destination[capacity - 1] = '\0';
#endif
    return static_cast<int>(cudaSuccess);
}

NNTRAIN_EXPORT int nntrain_cuda_set_device(int device) {
    return complete(
        nntrain_operation_set_device, device, select_device(device));
}

NNTRAIN_EXPORT int nntrain_cuda_synchronize(int device) {
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_synchronize, device, status);
    return complete(nntrain_operation_synchronize, device,
        static_cast<int>(cudaDeviceSynchronize()));
}

NNTRAIN_EXPORT int nntrain_cuda_mem_info(
    int device,
    size_t* free_bytes,
    size_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr)
        return complete(nntrain_operation_memory_info, device,
            static_cast<int>(cudaErrorInvalidValue));
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_memory_info, device, status);
    return complete(nntrain_operation_memory_info, device,
        static_cast<int>(cudaMemGetInfo(free_bytes, total_bytes)));
}

NNTRAIN_EXPORT int nntrain_cuda_malloc(
    int device,
    size_t bytes,
    void** pointer) {
    if (pointer == nullptr)
        return complete(nntrain_operation_allocate, device,
            static_cast<int>(cudaErrorInvalidValue));
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_allocate, device, status);
    cudaError_t allocation_status = cudaMalloc(pointer, bytes);
    if (allocation_status != cudaSuccess) {
        // Managed pools may release cached buffers and retry. Clear CUDA's
        // per-thread last-error slot so a successful retry is not reported by
        // the next kernel's cudaPeekAtLastError call.
        (void)cudaGetLastError();
    }
    return complete(nntrain_operation_allocate, device,
        static_cast<int>(allocation_status));
}

NNTRAIN_EXPORT int nntrain_cuda_free(int device, void* pointer) {
    if (pointer == nullptr)
        return static_cast<int>(cudaSuccess);
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_free, device, status);
    return complete(nntrain_operation_free, device,
        static_cast<int>(cudaFree(pointer)));
}

NNTRAIN_EXPORT int nntrain_cuda_memset(
    int device,
    void* destination,
    int value,
    size_t bytes) {
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_memset, device, status);
    return complete(nntrain_operation_memset, device,
        static_cast<int>(cudaMemset(destination, value, bytes)));
}

NNTRAIN_EXPORT int nntrain_cuda_memset_async(
    int device,
    void* destination,
    int value,
    size_t bytes,
    void* stream) {
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_memset_async, device, status);
    return complete(nntrain_operation_memset_async, device,
        static_cast<int>(cudaMemsetAsync(
            destination,
            value,
            bytes,
            reinterpret_cast<cudaStream_t>(stream))));
}

NNTRAIN_EXPORT int nntrain_cuda_copy_h2d(
    int device,
    void* destination,
    const void* source,
    size_t bytes) {
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_copy_h2d, device, status);
    return complete(nntrain_operation_copy_h2d, device,
        static_cast<int>(cudaMemcpy(
            destination, source, bytes, cudaMemcpyHostToDevice)));
}

NNTRAIN_EXPORT int nntrain_cuda_copy_h2d_async(
    int device,
    void* destination,
    const void* source,
    size_t bytes,
    void* stream) {
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_copy_h2d_async, device, status);
    return complete(nntrain_operation_copy_h2d_async, device,
        static_cast<int>(cudaMemcpyAsync(
            destination,
            source,
            bytes,
            cudaMemcpyHostToDevice,
            reinterpret_cast<cudaStream_t>(stream))));
}

NNTRAIN_EXPORT int nntrain_cuda_copy_d2h(
    int device,
    void* destination,
    const void* source,
    size_t bytes) {
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_copy_d2h, device, status);
    return complete(nntrain_operation_copy_d2h, device,
        static_cast<int>(cudaMemcpy(
            destination, source, bytes, cudaMemcpyDeviceToHost)));
}

NNTRAIN_EXPORT int nntrain_cuda_copy_d2h_async(
    int device,
    void* destination,
    const void* source,
    size_t bytes,
    void* stream) {
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_copy_d2h_async, device, status);
    return complete(nntrain_operation_copy_d2h_async, device,
        static_cast<int>(cudaMemcpyAsync(
            destination,
            source,
            bytes,
            cudaMemcpyDeviceToHost,
            reinterpret_cast<cudaStream_t>(stream))));
}

NNTRAIN_EXPORT int nntrain_cuda_host_alloc(size_t bytes, void** pointer) {
    if (pointer == nullptr)
        return complete(nntrain_operation_host_allocate, -1,
            static_cast<int>(cudaErrorInvalidValue));
    return complete(nntrain_operation_host_allocate, -1,
        static_cast<int>(cudaMallocHost(pointer, bytes)));
}

NNTRAIN_EXPORT int nntrain_cuda_host_free(void* pointer) {
    if (pointer == nullptr)
        return static_cast<int>(cudaSuccess);
    return complete(nntrain_operation_host_free, -1,
        static_cast<int>(cudaFreeHost(pointer)));
}

NNTRAIN_EXPORT int nntrain_cuda_stream_create(int device, void** stream) {
    if (stream == nullptr)
        return complete(nntrain_operation_stream_create, device,
            static_cast<int>(cudaErrorInvalidValue));
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_stream_create, device, status);
    return complete(nntrain_operation_stream_create, device,
        static_cast<int>(cudaStreamCreateWithFlags(
            reinterpret_cast<cudaStream_t*>(stream),
            cudaStreamNonBlocking)));
}

NNTRAIN_EXPORT int nntrain_cuda_stream_destroy(int device, void* stream) {
    if (stream == nullptr)
        return static_cast<int>(cudaSuccess);
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_stream_destroy, device, status);
    return complete(nntrain_operation_stream_destroy, device,
        static_cast<int>(cudaStreamDestroy(
            reinterpret_cast<cudaStream_t>(stream))));
}

NNTRAIN_EXPORT int nntrain_cuda_stream_synchronize(
    int device,
    void* stream) {
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_stream_synchronize, device, status);
    return complete(nntrain_operation_stream_synchronize, device,
        static_cast<int>(cudaStreamSynchronize(
            reinterpret_cast<cudaStream_t>(stream))));
}

NNTRAIN_EXPORT int nntrain_cuda_stream_begin_capture(
    int device,
    void* stream) {
    if (stream == nullptr) {
        return complete(
            nntrain_operation_graph_begin_capture,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_graph_begin_capture,
            device,
            status);
    }
    return complete(
        nntrain_operation_graph_begin_capture,
        device,
        static_cast<int>(cudaStreamBeginCapture(
            reinterpret_cast<cudaStream_t>(stream),
            cudaStreamCaptureModeThreadLocal)));
}

NNTRAIN_EXPORT int nntrain_cuda_stream_end_capture(
    int device,
    void* stream,
    void** graph) {
    if (stream == nullptr || graph == nullptr) {
        return complete(
            nntrain_operation_graph_end_capture,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    *graph = nullptr;
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_graph_end_capture,
            device,
            status);
    }
    return complete(
        nntrain_operation_graph_end_capture,
        device,
        static_cast<int>(cudaStreamEndCapture(
            reinterpret_cast<cudaStream_t>(stream),
            reinterpret_cast<cudaGraph_t*>(graph))));
}

NNTRAIN_EXPORT int nntrain_cuda_graph_instantiate(
    int device,
    void* graph,
    void** executable) {
    if (graph == nullptr || executable == nullptr) {
        return complete(
            nntrain_operation_graph_instantiate,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    *executable = nullptr;
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_graph_instantiate,
            device,
            status);
    }
    return complete(
        nntrain_operation_graph_instantiate,
        device,
        static_cast<int>(cudaGraphInstantiateWithFlags(
            reinterpret_cast<cudaGraphExec_t*>(executable),
            reinterpret_cast<cudaGraph_t>(graph),
            0)));
}

NNTRAIN_EXPORT int nntrain_cuda_graph_launch(
    int device,
    void* executable,
    void* stream) {
    if (executable == nullptr || stream == nullptr) {
        return complete(
            nntrain_operation_graph_launch,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_graph_launch, device, status);
    return complete(
        nntrain_operation_graph_launch,
        device,
        static_cast<int>(cudaGraphLaunch(
            reinterpret_cast<cudaGraphExec_t>(executable),
            reinterpret_cast<cudaStream_t>(stream))));
}

NNTRAIN_EXPORT int nntrain_cuda_graph_destroy(int device, void* graph) {
    if (graph == nullptr)
        return static_cast<int>(cudaSuccess);
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_graph_destroy, device, status);
    return complete(
        nntrain_operation_graph_destroy,
        device,
        static_cast<int>(cudaGraphDestroy(
            reinterpret_cast<cudaGraph_t>(graph))));
}

NNTRAIN_EXPORT int nntrain_cuda_graph_exec_destroy(
    int device,
    void* executable) {
    if (executable == nullptr)
        return static_cast<int>(cudaSuccess);
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_graph_exec_destroy,
            device,
            status);
    }
    return complete(
        nntrain_operation_graph_exec_destroy,
        device,
        static_cast<int>(cudaGraphExecDestroy(
            reinterpret_cast<cudaGraphExec_t>(executable))));
}

NNTRAIN_EXPORT int nntrain_cuda_graph_dropout_mask(
    int device,
    unsigned long long* step_counter,
    unsigned int seed,
    float dropout_probability,
    float* output,
    int length,
    void* stream) {
    if (step_counter == nullptr || output == nullptr || length <= 0 ||
        stream == nullptr || !(dropout_probability >= 0.0f) ||
        !(dropout_probability < 1.0f)) {
        return complete(
            nntrain_operation_graph_rng_step,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_graph_rng_step, device, status);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    graph_rng_advance_kernel<<<1, 1, 0, cuda_stream>>>(step_counter);
    cudaError_t launch_error = cudaPeekAtLastError();
    if (launch_error != cudaSuccess) {
        return complete(
            nntrain_operation_graph_rng_step,
            device,
            static_cast<int>(launch_error));
    }
    const unsigned int threshold = static_cast<unsigned int>(
        static_cast<double>(dropout_probability) * 4294967296.0);
    const float keep_scale = 1.0f / (1.0f - dropout_probability);
    constexpr int threads = 256;
    graph_dropout_mask_kernel<<<
        (length + threads - 1) / threads,
        threads,
        0,
        cuda_stream>>>(
            step_counter,
            seed,
            threshold,
            keep_scale,
            output,
            length);
    return complete(
        nntrain_operation_graph_rng_step,
        device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_graph_counter_set(
    int device,
    unsigned long long* step_counter,
    unsigned long long value,
    void* stream) {
    if (step_counter == nullptr || stream == nullptr) {
        return complete(
            nntrain_operation_graph_counter_set,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_graph_counter_set, device, status);
    graph_rng_set_kernel<<<
        1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            step_counter,
            value);
    return complete(
        nntrain_operation_graph_counter_set,
        device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_graph_counter_advance(
    int device,
    unsigned long long* step_counter,
    void* stream) {
    if (step_counter == nullptr || stream == nullptr) {
        return complete(
            nntrain_operation_graph_counter_advance,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_graph_counter_advance,
            device,
            status);
    }
    graph_rng_advance_kernel<<<
        1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(step_counter);
    return complete(
        nntrain_operation_graph_counter_advance,
        device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_graph_dropout_forward_float(
    int device,
    const unsigned long long* step_counter,
    unsigned long long operation_seed,
    float dropout_probability,
    const float* input,
    float* output,
    int length,
    void* stream) {
    if (step_counter == nullptr || input == nullptr || output == nullptr ||
        length <= 0 || stream == nullptr ||
        !(dropout_probability >= 0.0f) ||
        !(dropout_probability < 1.0f)) {
        return complete(
            nntrain_operation_graph_dropout_forward,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_graph_dropout_forward,
            device,
            status);
    }
    const unsigned int threshold = static_cast<unsigned int>(
        static_cast<double>(dropout_probability) * 4294967296.0);
    const float keep_scale = 1.0f / (1.0f - dropout_probability);
    constexpr int threads = 256;
    graph_dropout_forward_float_kernel<<<
        (length + threads - 1) / threads,
        threads,
        0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
            step_counter,
            operation_seed,
            threshold,
            keep_scale,
            input,
            output,
            length);
    return complete(
        nntrain_operation_graph_dropout_forward,
        device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_graph_dropout_forward_bf16(
    int device,
    const unsigned long long* step_counter,
    unsigned long long operation_seed,
    float dropout_probability,
    const unsigned short* input,
    unsigned short* output,
    int length,
    void* stream) {
    if (step_counter == nullptr || input == nullptr || output == nullptr ||
        length <= 0 || stream == nullptr ||
        !(dropout_probability >= 0.0f) ||
        !(dropout_probability < 1.0f)) {
        return complete(
            nntrain_operation_graph_dropout_forward,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_graph_dropout_forward,
            device,
            status);
    }
    const unsigned int threshold = static_cast<unsigned int>(
        static_cast<double>(dropout_probability) * 4294967296.0);
    const float keep_scale = 1.0f / (1.0f - dropout_probability);
    constexpr int threads = 256;
    graph_dropout_forward_bf16_kernel<<<
        (length + threads - 1) / threads,
        threads,
        0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
            step_counter,
            operation_seed,
            threshold,
            keep_scale,
            reinterpret_cast<const __nv_bfloat16*>(input),
            reinterpret_cast<__nv_bfloat16*>(output),
            length);
    return complete(
        nntrain_operation_graph_dropout_forward,
        device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_graph_add_dropout_forward_float(
    int device,
    const unsigned long long* step_counter,
    unsigned long long operation_seed,
    float dropout_probability,
    const float* residual,
    const float* branch,
    float* output,
    int length,
    void* stream) {
    if (step_counter == nullptr || residual == nullptr || branch == nullptr ||
        output == nullptr || length <= 0 || stream == nullptr ||
        !(dropout_probability >= 0.0f) ||
        !(dropout_probability < 1.0f)) {
        return complete(
            nntrain_operation_graph_add_dropout_forward,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_graph_add_dropout_forward,
            device,
            status);
    }
    const unsigned int threshold = static_cast<unsigned int>(
        static_cast<double>(dropout_probability) * 4294967296.0);
    const float keep_scale = 1.0f / (1.0f - dropout_probability);
    constexpr int threads = 256;
    graph_add_dropout_forward_float_kernel<<<
        (length + threads - 1) / threads,
        threads,
        0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
            step_counter,
            operation_seed,
            threshold,
            keep_scale,
            residual,
            branch,
            output,
            length);
    return complete(
        nntrain_operation_graph_add_dropout_forward,
        device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_graph_add_dropout_forward_bf16(
    int device,
    const unsigned long long* step_counter,
    unsigned long long operation_seed,
    float dropout_probability,
    const unsigned short* residual,
    const unsigned short* branch,
    unsigned short* output,
    int length,
    void* stream) {
    if (step_counter == nullptr || residual == nullptr || branch == nullptr ||
        output == nullptr || length <= 0 || stream == nullptr ||
        !(dropout_probability >= 0.0f) ||
        !(dropout_probability < 1.0f)) {
        return complete(
            nntrain_operation_graph_add_dropout_forward,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_graph_add_dropout_forward,
            device,
            status);
    }
    const unsigned int threshold = static_cast<unsigned int>(
        static_cast<double>(dropout_probability) * 4294967296.0);
    const float keep_scale = 1.0f / (1.0f - dropout_probability);
    constexpr int threads = 256;
    graph_add_dropout_forward_bf16_kernel<<<
        (length + threads - 1) / threads,
        threads,
        0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
            step_counter,
            operation_seed,
            threshold,
            keep_scale,
            reinterpret_cast<const __nv_bfloat16*>(residual),
            reinterpret_cast<const __nv_bfloat16*>(branch),
            reinterpret_cast<__nv_bfloat16*>(output),
            length);
    return complete(
        nntrain_operation_graph_add_dropout_forward,
        device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_graph_dropout_backward_float(
    int device,
    const unsigned long long* step_counter,
    unsigned long long operation_seed,
    float dropout_probability,
    const float* output_gradient,
    float* input_gradient,
    int length,
    void* stream) {
    if (step_counter == nullptr || output_gradient == nullptr ||
        input_gradient == nullptr || length <= 0 || stream == nullptr ||
        !(dropout_probability >= 0.0f) ||
        !(dropout_probability < 1.0f)) {
        return complete(
            nntrain_operation_graph_dropout_backward,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_graph_dropout_backward,
            device,
            status);
    }
    const unsigned int threshold = static_cast<unsigned int>(
        static_cast<double>(dropout_probability) * 4294967296.0);
    const float keep_scale = 1.0f / (1.0f - dropout_probability);
    constexpr int threads = 256;
    graph_dropout_backward_float_kernel<<<
        (length + threads - 1) / threads,
        threads,
        0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
            step_counter,
            operation_seed,
            threshold,
            keep_scale,
            output_gradient,
            input_gradient,
            length);
    return complete(
        nntrain_operation_graph_dropout_backward,
        device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_graph_add_dropout_backward_float(
    int device,
    const unsigned long long* step_counter,
    unsigned long long operation_seed,
    float dropout_probability,
    const float* output_gradient,
    float* residual_gradient,
    float* branch_gradient,
    int length,
    int same_parent,
    void* stream) {
    if (step_counter == nullptr || output_gradient == nullptr ||
        residual_gradient == nullptr || branch_gradient == nullptr ||
        length <= 0 || stream == nullptr ||
        !(dropout_probability >= 0.0f) ||
        !(dropout_probability < 1.0f)) {
        return complete(
            nntrain_operation_graph_add_dropout_backward,
            device,
            static_cast<int>(cudaErrorInvalidValue));
    }
    int status = select_device(device);
    if (status != cudaSuccess) {
        return complete(
            nntrain_operation_graph_add_dropout_backward,
            device,
            status);
    }
    const unsigned int threshold = static_cast<unsigned int>(
        static_cast<double>(dropout_probability) * 4294967296.0);
    const float keep_scale = 1.0f / (1.0f - dropout_probability);
    constexpr int threads = 256;
    graph_add_dropout_backward_float_kernel<<<
        (length + threads - 1) / threads,
        threads,
        0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
            step_counter,
            operation_seed,
            threshold,
            keep_scale,
            output_gradient,
            residual_gradient,
            branch_gradient,
            length,
            same_parent);
    return complete(
        nntrain_operation_graph_add_dropout_backward,
        device,
        static_cast<int>(cudaPeekAtLastError()));
}

NNTRAIN_EXPORT int nntrain_cuda_event_create(int device, void** event) {
    if (event == nullptr)
        return complete(nntrain_operation_event_create, device,
            static_cast<int>(cudaErrorInvalidValue));
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_event_create, device, status);
    return complete(nntrain_operation_event_create, device,
        static_cast<int>(cudaEventCreateWithFlags(
            reinterpret_cast<cudaEvent_t*>(event), cudaEventDisableTiming)));
}

NNTRAIN_EXPORT int nntrain_cuda_event_destroy(int device, void* event) {
    if (event == nullptr)
        return static_cast<int>(cudaSuccess);
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_event_destroy, device, status);
    return complete(nntrain_operation_event_destroy, device,
        static_cast<int>(cudaEventDestroy(
            reinterpret_cast<cudaEvent_t>(event))));
}

NNTRAIN_EXPORT int nntrain_cuda_event_record(
    int device,
    void* event,
    void* stream) {
    if (event == nullptr)
        return complete(nntrain_operation_event_record, device,
            static_cast<int>(cudaErrorInvalidValue));
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_event_record, device, status);
    return complete(nntrain_operation_event_record, device,
        static_cast<int>(cudaEventRecord(
            reinterpret_cast<cudaEvent_t>(event),
            reinterpret_cast<cudaStream_t>(stream))));
}

NNTRAIN_EXPORT int nntrain_cuda_event_query(int device, void* event) {
    if (event == nullptr)
        return complete(nntrain_operation_event_query, device,
            static_cast<int>(cudaErrorInvalidValue));
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_event_query, device, status);
    return complete(nntrain_operation_event_query, device,
        static_cast<int>(cudaEventQuery(
            reinterpret_cast<cudaEvent_t>(event))));
}

NNTRAIN_EXPORT int nntrain_cuda_copy_d2h_async_record(
    int device,
    void* destination,
    const void* source,
    size_t bytes,
    void* stream,
    void* event) {
    if (destination == nullptr || source == nullptr || event == nullptr)
        return complete(nntrain_operation_copy_d2h_async, device,
            static_cast<int>(cudaErrorInvalidValue));
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_copy_d2h_async, device, status);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t copy_status = cudaMemcpyAsync(
        destination, source, bytes, cudaMemcpyDeviceToHost, cuda_stream);
    if (copy_status != cudaSuccess)
        return complete(nntrain_operation_copy_d2h_async, device,
            static_cast<int>(copy_status));
    return complete(nntrain_operation_copy_d2h_async, device,
        static_cast<int>(cudaEventRecord(
            reinterpret_cast<cudaEvent_t>(event), cuda_stream)));
}

NNTRAIN_EXPORT int nntrain_cuda_copy_h2d_async_record(
    int device,
    void* destination,
    const void* source,
    size_t bytes,
    void* stream,
    void* event) {
    if (destination == nullptr || source == nullptr || event == nullptr)
        return complete(nntrain_operation_copy_h2d_async, device,
            static_cast<int>(cudaErrorInvalidValue));
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_copy_h2d_async, device, status);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t copy_status = cudaMemcpyAsync(
        destination, source, bytes, cudaMemcpyHostToDevice, cuda_stream);
    if (copy_status != cudaSuccess)
        return complete(nntrain_operation_copy_h2d_async, device,
            static_cast<int>(copy_status));
    return complete(nntrain_operation_copy_h2d_async, device,
        static_cast<int>(cudaEventRecord(
            reinterpret_cast<cudaEvent_t>(event), cuda_stream)));
}

NNTRAIN_EXPORT int nntrain_cuda_event_synchronize(int device, void* event) {
    if (event == nullptr)
        return complete(nntrain_operation_event_synchronize, device,
            static_cast<int>(cudaErrorInvalidValue));
    int status = select_device(device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_event_synchronize, device, status);
    return complete(nntrain_operation_event_synchronize, device,
        static_cast<int>(cudaEventSynchronize(
            reinterpret_cast<cudaEvent_t>(event))));
}

NNTRAIN_EXPORT int nntrain_cuda_copy_d2d(
    int destination_device,
    void* destination,
    int source_device,
    const void* source,
    size_t bytes) {
    if (destination_device == source_device) {
        int status = select_device(destination_device);
        if (status != cudaSuccess)
            return complete(nntrain_operation_copy_d2d,
                destination_device, status);
        return complete(nntrain_operation_copy_d2d, destination_device,
            static_cast<int>(cudaMemcpy(
                destination, source, bytes, cudaMemcpyDeviceToDevice)));
    }
    return complete(nntrain_operation_copy_d2d, destination_device,
        static_cast<int>(cudaMemcpyPeer(
            destination, destination_device, source, source_device, bytes)));
}

NNTRAIN_EXPORT int nntrain_cuda_copy_d2d_async(
    int destination_device,
    void* destination,
    int source_device,
    const void* source,
    size_t bytes,
    void* stream) {
    int status = select_device(destination_device);
    if (status != cudaSuccess)
        return complete(nntrain_operation_copy_d2d_async,
            destination_device, status);

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (destination_device == source_device) {
        return complete(nntrain_operation_copy_d2d_async,
            destination_device,
            static_cast<int>(cudaMemcpyAsync(
                destination,
                source,
                bytes,
                cudaMemcpyDeviceToDevice,
                cuda_stream)));
    }
    return complete(nntrain_operation_copy_d2d_async,
        destination_device,
        static_cast<int>(cudaMemcpyPeerAsync(
            destination,
            destination_device,
            source,
            source_device,
            bytes,
            cuda_stream)));
}

NNTRAIN_EXPORT int nntrain_cuda_can_access_peer(
    int device,
    int peer_device,
    int* can_access) {
    if (can_access == nullptr)
        return complete(nntrain_operation_peer_access, device,
            static_cast<int>(cudaErrorInvalidValue));
    return complete(nntrain_operation_peer_access, device,
        static_cast<int>(cudaDeviceCanAccessPeer(
            can_access, device, peer_device)));
}

NNTRAIN_EXPORT const char* nntrain_cuda_error_string(int status) {
    return cudaGetErrorString(static_cast<cudaError_t>(status));
}
