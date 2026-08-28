#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define NNTRAIN_EXPORT extern "C" __declspec(dllexport)
#else
#define NNTRAIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Keep cuda_runtime_bridge.cu's thread-local selected-device cache coherent.
// Every exported entry point with an explicit device must use the bridge as
// the single device-selection authority.
extern "C" int nntrain_cuda_set_device(int device);

namespace {

constexpr int threads_per_block = 256;
constexpr int tensor_reduction_values_per_block = 1024;

// BFP8 is a persistent storage codec. Keep its scale and code selection
// deterministic even though this translation unit is built with
// --use_fast_math for the surrounding training kernels.
__device__ __forceinline__ float bfp8_scale_from_maximum(float maximum) {
    return maximum == 0.0f ? 1.0f : __fdiv_rn(maximum, 127.0f);
}

__device__ __forceinline__ int bfp8_quantized_code(
    float value,
    float scale) {
    return max(-127, min(127, __float2int_rn(__fdiv_rn(value, scale))));
}

struct embedding_reader {
    const signed char* table_payload;
    const float* table_scales;
    const int* indices;
    int table_rows;
    int table_block_size;
    int width;

    __device__ __forceinline__ float operator()(int output_index) const {
        const int position = output_index / width;
        const int column = output_index - position * width;
        const int row = indices[position];
        // Managed dispatch validates every index. Keep the native guard as a
        // last line of defence against an out-of-bounds device access if this
        // ABI is called directly.
        if (static_cast<unsigned int>(row) >=
            static_cast<unsigned int>(table_rows)) {
            return 0.0f;
        }
        const int source_index = row * width + column;
        return static_cast<float>(table_payload[source_index]) *
            table_scales[source_index / table_block_size];
    }
};

struct embedding_positions_reader {
    const signed char* token_payload;
    const float* token_scales;
    const signed char* position_payload;
    const float* position_scales;
    const int* indices;
    int token_rows;
    int token_block_size;
    int position_block_size;
    int sequence_length;
    int width;

    __device__ __forceinline__ float operator()(int output_index) const {
        const int position = output_index / width;
        const int column = output_index - position * width;
        const int row = indices[position];
        if (static_cast<unsigned int>(row) >=
            static_cast<unsigned int>(token_rows)) {
            return 0.0f;
        }
        const int token_index = row * width + column;
        const int position_index =
            (position % sequence_length) * width + column;
        const float token =
            static_cast<float>(token_payload[token_index]) *
            token_scales[token_index / token_block_size];
        const float positional =
            static_cast<float>(position_payload[position_index]) *
            position_scales[position_index / position_block_size];
        return token + positional;
    }
};

__device__ __forceinline__ float reduce_maximum(float value) {
    __shared__ float reduction[threads_per_block];
    reduction[threadIdx.x] = value;
    __syncthreads();
    for (int stride = threads_per_block / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x],
                reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return reduction[0];
}

template <typename Reader>
__global__ void selected_partial_maximum_kernel(
    Reader reader,
    float* partial_maxima,
    int length) {
    const int start = blockIdx.x * tensor_reduction_values_per_block;
    const int end = start + tensor_reduction_values_per_block < length
        ? start + tensor_reduction_values_per_block
        : length;
    float maximum = 0.0f;
    for (int index = start + threadIdx.x;
         index < end;
         index += threads_per_block) {
        maximum = fmaxf(maximum, fabsf(reader(index)));
    }
    maximum = reduce_maximum(maximum);
    if (threadIdx.x == 0)
        partial_maxima[blockIdx.x] = maximum;
}

__global__ void tensor_scale_kernel(
    const float* partial_maxima,
    int partial_count,
    float* output_scale) {
    float maximum = 0.0f;
    for (int index = threadIdx.x;
         index < partial_count;
         index += threads_per_block) {
        maximum = fmaxf(maximum, partial_maxima[index]);
    }
    maximum = reduce_maximum(maximum);
    if (threadIdx.x == 0) {
        // One is the canonical positive scale for an all-zero BFP block.
        output_scale[0] = bfp8_scale_from_maximum(maximum);
    }
}

template <typename Reader>
__global__ void quantize_with_tensor_scale_kernel(
    Reader reader,
    signed char* output_payload,
    const float* output_scale,
    int length) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    const int quantized = bfp8_quantized_code(
        reader(index), output_scale[0]);
    output_payload[index] = static_cast<signed char>(quantized);
}

template <typename Reader>
__global__ void quantize_with_block_scales_kernel(
    Reader reader,
    signed char* output_payload,
    float* output_scales,
    int output_block_size,
    int length) {
    const long long start =
        static_cast<long long>(blockIdx.x) * output_block_size;
    const long long candidate_end = start + output_block_size;
    const long long end = candidate_end < length ? candidate_end : length;
    float maximum = 0.0f;
    for (long long index = start + threadIdx.x;
         index < end;
         index += threads_per_block) {
        maximum = fmaxf(
            maximum,
            fabsf(reader(static_cast<int>(index))));
    }
    maximum = reduce_maximum(maximum);

    __shared__ float block_scale;
    if (threadIdx.x == 0) {
        block_scale = bfp8_scale_from_maximum(maximum);
        output_scales[blockIdx.x] = block_scale;
    }
    __syncthreads();

    for (long long index = start + threadIdx.x;
         index < end;
         index += threads_per_block) {
        const int quantized = bfp8_quantized_code(
            reader(static_cast<int>(index)), block_scale);
        output_payload[index] = static_cast<signed char>(quantized);
    }
}

template <typename Reader>
cudaError_t launch_embedding_quantization(
    Reader reader,
    signed char* output_payload,
    float* output_scales,
    int output_block_size,
    int output_scale_count,
    float* workspace,
    int workspace_length,
    int length,
    cudaStream_t stream) {
    if (output_scale_count == 1) {
        const int partial_count =
            (length - 1) / tensor_reduction_values_per_block + 1;
        if (workspace == nullptr || workspace_length < partial_count)
            return cudaErrorInvalidValue;
        selected_partial_maximum_kernel<<<
            partial_count, threads_per_block, 0, stream>>>(
                reader, workspace, length);
        cudaError_t status = cudaPeekAtLastError();
        if (status != cudaSuccess)
            return status;
        tensor_scale_kernel<<<1, threads_per_block, 0, stream>>>(
            workspace, partial_count, output_scales);
        status = cudaPeekAtLastError();
        if (status != cudaSuccess)
            return status;
        const int blocks = (length - 1) / threads_per_block + 1;
        quantize_with_tensor_scale_kernel<<<
            blocks, threads_per_block, 0, stream>>>(
                reader, output_payload, output_scales, length);
        return cudaPeekAtLastError();
    }

    const int expected_scale_count = static_cast<int>(
        (static_cast<long long>(length) + output_block_size - 1) /
        output_block_size);
    if (expected_scale_count != output_scale_count)
        return cudaErrorInvalidValue;
    quantize_with_block_scales_kernel<<<
        output_scale_count, threads_per_block, 0, stream>>>(
            reader,
            output_payload,
            output_scales,
            output_block_size,
            length);
    return cudaPeekAtLastError();
}

bool invalid_common_arguments(
    int device,
    const signed char* table_payload,
    const float* table_scales,
    int table_length,
    int table_block_size,
    const int* indices,
    int index_count,
    int width,
    signed char* output_payload,
    float* output_scales,
    int output_block_size,
    int output_scale_count) {
    return device < 0 || table_payload == nullptr || table_scales == nullptr ||
        indices == nullptr || output_payload == nullptr ||
        output_scales == nullptr || table_length <= 0 ||
        table_block_size <= 0 || index_count <= 0 || width <= 0 ||
        table_length % width != 0 || output_block_size <= 0 ||
        output_scale_count <= 0;
}

} // namespace

NNTRAIN_EXPORT int nntrain_bfp8_embedding_forward(
    int device,
    const signed char* table_payload,
    const float* table_scales,
    int table_length,
    int table_block_size,
    const int* indices,
    int index_count,
    int width,
    signed char* output_payload,
    float* output_scales,
    int output_block_size,
    int output_scale_count,
    float* workspace,
    int workspace_length,
    void* stream) {
    if (invalid_common_arguments(
            device,
            table_payload,
            table_scales,
            table_length,
            table_block_size,
            indices,
            index_count,
            width,
            output_payload,
            output_scales,
            output_block_size,
            output_scale_count)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const long long output_length =
        static_cast<long long>(index_count) * width;
    if (output_length > 0x7fffffffLL)
        return static_cast<int>(cudaErrorInvalidValue);
    cudaError_t status = static_cast<cudaError_t>(
        nntrain_cuda_set_device(device));
    if (status != cudaSuccess)
        return static_cast<int>(status);

    embedding_reader reader{
        table_payload,
        table_scales,
        indices,
        table_length / width,
        table_block_size,
        width,
    };
    status = launch_embedding_quantization(
        reader,
        output_payload,
        output_scales,
        output_block_size,
        output_scale_count,
        workspace,
        workspace_length,
        static_cast<int>(output_length),
        static_cast<cudaStream_t>(stream));
    return static_cast<int>(status);
}

NNTRAIN_EXPORT int nntrain_bfp8_embedding_positions_forward(
    int device,
    const signed char* token_payload,
    const float* token_scales,
    int token_length,
    int token_block_size,
    const signed char* position_payload,
    const float* position_scales,
    int position_length,
    int position_block_size,
    const int* indices,
    int index_count,
    int sequence_length,
    int width,
    signed char* output_payload,
    float* output_scales,
    int output_block_size,
    int output_scale_count,
    float* workspace,
    int workspace_length,
    void* stream) {
    if (invalid_common_arguments(
            device,
            token_payload,
            token_scales,
            token_length,
            token_block_size,
            indices,
            index_count,
            width,
            output_payload,
            output_scales,
            output_block_size,
            output_scale_count) ||
        position_payload == nullptr || position_scales == nullptr ||
        position_length <= 0 || position_block_size <= 0 ||
        position_length % width != 0 || sequence_length <= 0 ||
        sequence_length > position_length / width) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const long long output_length =
        static_cast<long long>(index_count) * width;
    if (output_length > 0x7fffffffLL)
        return static_cast<int>(cudaErrorInvalidValue);
    cudaError_t status = static_cast<cudaError_t>(
        nntrain_cuda_set_device(device));
    if (status != cudaSuccess)
        return static_cast<int>(status);

    embedding_positions_reader reader{
        token_payload,
        token_scales,
        position_payload,
        position_scales,
        indices,
        token_length / width,
        token_block_size,
        position_block_size,
        sequence_length,
        width,
    };
    status = launch_embedding_quantization(
        reader,
        output_payload,
        output_scales,
        output_block_size,
        output_scale_count,
        workspace,
        workspace_length,
        static_cast<int>(output_length),
        static_cast<cudaStream_t>(stream));
    return static_cast<int>(status);
}
