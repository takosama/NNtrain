#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>

#if defined(_WIN32)
#define NNTRAIN_EXPORT extern "C" __declspec(dllexport)
#else
#define NNTRAIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
constexpr std::uint32_t nntrain_abi_major = 1;
constexpr std::uint32_t nntrain_abi_minor = 3;
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
        scales[blockIdx.x] = reduction[0] == 0.0f
            ? 1.0f
            : reduction[0] / 127.0f;
    }
    __syncthreads();

    const float scale = scales[blockIdx.x];
    for (long long index = start + threadIdx.x;
         index < end;
         index += blockDim.x) {
        int quantized = __float2int_rn(source[index] / scale);
        quantized = max(-127, min(127, quantized));
        payload[index] = static_cast<signed char>(quantized);
    }
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
        scales[blockIdx.x] = reduction[0] == 0.0f
            ? 1.0f
            : reduction[0] / 127.0f;
    }
    __syncthreads();

    const float scale = scales[blockIdx.x];
    for (long long index = start + threadIdx.x;
         index < end;
         index += blockDim.x) {
        int quantized = __float2int_rn(
            __bfloat162float(source[index]) / scale);
        quantized = max(-127, min(127, quantized));
        payload[index] = static_cast<signed char>(quantized);
    }
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
        scales[blockIdx.x] = reduction[0] == 0.0f
            ? 1.0f
            : reduction[0] / 127.0f;
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
        int quantized = __float2int_rn(value / output_scale);
        quantized = max(-127, min(127, quantized));
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
    const float scale = maximum_value == 0.0f
        ? 1.0f
        : maximum_value / 127.0f;
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
    int quantized = __float2int_rn(value / scale);
    quantized = max(-127, min(127, quantized));
    payload[index] = static_cast<signed char>(quantized);
}

__global__ void bfp8_finalize_tensor_scale_kernel(float* maximum) {
    const float maximum_value = maximum[0];
    maximum[0] = maximum_value == 0.0f
        ? 1.0f
        : maximum_value / 127.0f;
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
    constexpr unsigned long long bfp8_quantization = 1ull << 8;
    constexpr unsigned long long int8_tensor_cores = 1ull << 9;

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
            bfp8_quantization |
            int8_tensor_cores;
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
    bfp8_quantize_f32_kernel<<<
        static_cast<int>(blocks64),
        256,
        0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
            source,
            payload,
            scales,
            length,
            quantization_block_size);
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
    const int blocks = (length + threads - 1) / threads;
    bfp8_dequantize_f32_kernel<<<
        blocks,
        threads,
        0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
            payload,
            scales,
            destination,
            length,
            quantization_block_size);
    return complete(
        nntrain_operation_bfp8_dequantize_f32,
        device,
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
    bfp8_dequantize_bf16_kernel<<<
        blocks,
        threads,
        0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
            payload,
            scales,
            destination,
            length,
            quantization_block_size);
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
    bfp8_quantize_bf16_kernel<<<
        static_cast<int>(blocks64),
        256,
        0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
            source,
            payload,
            scales,
            length,
            quantization_block_size);
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
