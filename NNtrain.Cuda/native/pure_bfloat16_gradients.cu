#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <cmath>

#if defined(_WIN32)
#define NNTRAIN_EXPORT extern "C" __declspec(dllexport)
#else
#define NNTRAIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
constexpr int kThreads = 256;

int blocks_for(int length) {
    return (length + kThreads - 1) / kThreads;
}

int launch_status() {
    return static_cast<int>(cudaPeekAtLastError());
}

__device__ __forceinline__ float bf16_load(
    const unsigned short* values,
    int index) {
    return __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(values)[index]);
}

__device__ __forceinline__ void bf16_store(
    unsigned short* values,
    int index,
    float value) {
    reinterpret_cast<__nv_bfloat16*>(values)[index] =
        __float2bfloat16_rn(value);
}

__device__ __forceinline__ void bf16_accumulate(
    unsigned short* values,
    int index,
    float contribution) {
    bf16_store(values, index, bf16_load(values, index) + contribution);
}

__global__ void embedding_occurrence_map_kernel(
    const int* indices,
    int* hash_keys,
    int* hash_heads,
    int* occurrence_next,
    int* unique_slots,
    int* unique_count,
    int position_count,
    int hash_mask) {
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= position_count)
        return;

    const int token = indices[position];
    unsigned int slot =
        static_cast<unsigned int>(token) * 2654435761u
        & static_cast<unsigned int>(hash_mask);
    for (int probe = 0; probe <= hash_mask; ++probe) {
        const int previous = atomicCAS(hash_keys + slot, -1, token);
        if (previous == -1 || previous == token) {
            if (previous == -1) {
                const int ordinal = atomicAdd(unique_count, 1);
                unique_slots[ordinal] = static_cast<int>(slot);
            }
            occurrence_next[position] =
                atomicExch(hash_heads + slot, position);
            return;
        }
        slot = (slot + 1u) & static_cast<unsigned int>(hash_mask);
    }
}

// One block owns one unique token row. Every destination element therefore
// has exactly one writer and needs no atomic BF16 update. Occurrences reduce
// in FP32 and the completed contribution is rounded once when accumulated
// into the authoritative BF16 gradient.
__global__ void embedding_token_owner_bf16_gradient_kernel(
    const unsigned short* output_gradient,
    unsigned short* token_gradient,
    const int* hash_keys,
    const int* hash_heads,
    const int* occurrence_next,
    const int* unique_slots,
    const int* unique_count,
    int width) {
    const int ordinal = static_cast<int>(blockIdx.x);
    if (ordinal >= *unique_count)
        return;
    const int slot = unique_slots[ordinal];
    const int token = hash_keys[slot];
    const int first = hash_heads[slot];

    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        float sum = 0.0f;
        for (int position = first;
             position >= 0;
             position = occurrence_next[position]) {
            sum += bf16_load(
                output_gradient,
                position * width + column);
        }
        bf16_accumulate(token_gradient, token * width + column, sum);
    }
}

__global__ void embedding_position_owner_bf16_gradient_kernel(
    const unsigned short* output_gradient,
    unsigned short* position_gradient,
    int position_count,
    int sequence,
    int width,
    int column_blocks) {
    const int position = static_cast<int>(blockIdx.x) / column_blocks;
    const int column_block = static_cast<int>(blockIdx.x)
        - position * column_blocks;
    const int column = column_block * blockDim.x + threadIdx.x;
    if (position >= sequence || column >= width)
        return;

    float sum = 0.0f;
    for (int source_position = position;
         source_position < position_count;
         source_position += sequence) {
        sum += bf16_load(
            output_gradient,
            source_position * width + column);
    }
    bf16_accumulate(
        position_gradient,
        position * width + column,
        sum);
}

int embedding_hash_capacity(int position_count) {
    if (position_count <= 0 || position_count > (1 << 29))
        return 0;
    int capacity = 2;
    const int requested = position_count * 2;
    while (capacity < requested)
        capacity <<= 1;
    return capacity;
}

long long embedding_workspace_ints(int position_count) {
    const int hash_capacity = embedding_hash_capacity(position_count);
    if (hash_capacity == 0)
        return 0;
    return 2LL * hash_capacity + 2LL * position_count + 1LL;
}

int launch_embedding_bf16_gradient(
    const int* indices,
    const unsigned short* output_gradient,
    unsigned short* token_gradient,
    unsigned short* position_gradient,
    int* workspace,
    int workspace_ints,
    int length,
    int sequence,
    int width,
    cudaStream_t stream) {
    if (indices == nullptr || output_gradient == nullptr
        || token_gradient == nullptr || workspace == nullptr
        || length <= 0 || width <= 0 || length % width != 0
        || sequence < 0 || (sequence != 0 && position_gradient == nullptr)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int position_count = length / width;
    if (sequence != 0 && position_count % sequence != 0)
        return static_cast<int>(cudaErrorInvalidValue);
    const int hash_capacity = embedding_hash_capacity(position_count);
    const long long required = embedding_workspace_ints(position_count);
    if (hash_capacity == 0 || required > workspace_ints)
        return static_cast<int>(cudaErrorInvalidValue);

    int* hash_keys = workspace;
    int* hash_heads = hash_keys + hash_capacity;
    int* occurrence_next = hash_heads + hash_capacity;
    int* unique_slots = occurrence_next + position_count;
    int* unique_count = unique_slots + position_count;

    cudaError_t status = cudaMemsetAsync(
        hash_keys,
        0xff,
        static_cast<size_t>(2LL * hash_capacity) * sizeof(int),
        stream);
    if (status != cudaSuccess)
        return static_cast<int>(status);
    status = cudaMemsetAsync(unique_count, 0, sizeof(int), stream);
    if (status != cudaSuccess)
        return static_cast<int>(status);

    embedding_occurrence_map_kernel<<<
        blocks_for(position_count), kThreads, 0, stream>>>(
            indices,
            hash_keys,
            hash_heads,
            occurrence_next,
            unique_slots,
            unique_count,
            position_count,
            hash_capacity - 1);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return static_cast<int>(status);

    embedding_token_owner_bf16_gradient_kernel<<<
        position_count, kThreads, 0, stream>>>(
            output_gradient,
            token_gradient,
            hash_keys,
            hash_heads,
            occurrence_next,
            unique_slots,
            unique_count,
            width);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return static_cast<int>(status);

    if (sequence != 0) {
        const int column_blocks = blocks_for(width);
        embedding_position_owner_bf16_gradient_kernel<<<
            sequence * column_blocks, kThreads, 0, stream>>>(
                output_gradient,
                position_gradient,
                position_count,
                sequence,
                width,
                column_blocks);
        status = cudaPeekAtLastError();
    }
    return static_cast<int>(status);
}

__device__ __forceinline__ float dropout_multiplier(
    unsigned int seed,
    int index,
    unsigned int threshold,
    float scale) {
    unsigned int counter = static_cast<unsigned int>(index + 1);
    unsigned int bits = seed + 0x9E3779B9u * counter;
    bits ^= bits >> 16;
    bits *= 0x7FEB352Du;
    bits ^= bits >> 15;
    bits *= 0x846CA68Bu;
    bits ^= bits >> 16;
    return bits < threshold ? 0.0f : scale;
}

__global__ void dropout_backward_bf16_gradient_kernel(
    const unsigned short* output_gradient,
    unsigned short* input_gradient,
    int length,
    unsigned int seed,
    unsigned int threshold,
    float scale) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) {
        bf16_accumulate(
            input_gradient,
            index,
            bf16_load(output_gradient, index)
                * dropout_multiplier(seed, index, threshold, scale));
    }
}

__global__ void add_dropout_backward_bf16_gradient_kernel(
    const unsigned short* output_gradient,
    unsigned short* residual_gradient,
    unsigned short* branch_gradient,
    int length,
    int same_parent,
    unsigned int seed,
    unsigned int threshold,
    float scale) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    const float gradient = bf16_load(output_gradient, index);
    const float multiplier = dropout_multiplier(
        seed, index, threshold, scale);
    if (same_parent != 0) {
        bf16_accumulate(
            residual_gradient,
            index,
            gradient * (1.0f + multiplier));
    } else {
        bf16_accumulate(residual_gradient, index, gradient);
        bf16_accumulate(
            branch_gradient,
            index,
            gradient * multiplier);
    }
}

__global__ void linear_bias_backward_bf16_gradient_kernel(
    const unsigned short* output_gradient,
    unsigned short* bias_gradient,
    int rows,
    int width) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= width)
        return;
    float sum = 0.0f;
    for (int row = 0; row < rows; ++row)
        sum += bf16_load(output_gradient, row * width + column);
    bf16_accumulate(bias_gradient, column, sum);
}

__global__ void bf16_gradient_squared_sum_kernel(
    const unsigned short* values,
    int length,
    double* result) {
    __shared__ double sums[kThreads];
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    double value = 0.0;
    if (index < length) {
        const double gradient = bf16_load(values, index);
        value = gradient * gradient;
    }
    sums[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(result, sums[0]);
}

__global__ void bf16_gradient_scale_kernel(
    unsigned short* values,
    int length,
    float multiplier) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) {
        bf16_store(
            values,
            index,
            bf16_load(values, index) * multiplier);
    }
}

__device__ __forceinline__ unsigned int graph_rng_hash(
    unsigned long long step,
    unsigned long long operation_seed,
    unsigned int index) {
    unsigned long long value = step
        ^ operation_seed
        ^ (static_cast<unsigned long long>(index)
            * 0x9e3779b97f4a7c15ull);
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    value ^= value >> 31;
    return static_cast<unsigned int>(value >> 32);
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

__global__ void graph_dropout_backward_bf16_gradient_kernel(
    const unsigned long long* step_counter,
    unsigned long long operation_seed,
    unsigned int threshold,
    float keep_scale,
    const unsigned short* output_gradient,
    unsigned short* input_gradient,
    int length) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    bf16_accumulate(
        input_gradient,
        index,
        bf16_load(output_gradient, index)
            * graph_dropout_multiplier(
                step_counter,
                operation_seed,
                index,
                threshold,
                keep_scale));
}

__global__ void graph_add_dropout_backward_bf16_gradient_kernel(
    const unsigned long long* step_counter,
    unsigned long long operation_seed,
    unsigned int threshold,
    float keep_scale,
    const unsigned short* output_gradient,
    unsigned short* residual_gradient,
    unsigned short* branch_gradient,
    int length,
    int same_parent) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    const float gradient = bf16_load(output_gradient, index);
    const float multiplier = graph_dropout_multiplier(
        step_counter,
        operation_seed,
        index,
        threshold,
        keep_scale);
    if (same_parent != 0) {
        bf16_accumulate(
            residual_gradient,
            index,
            gradient * (1.0f + multiplier));
    } else {
        bf16_accumulate(residual_gradient, index, gradient);
        bf16_accumulate(
            branch_gradient,
            index,
            gradient * multiplier);
    }
}
}

NNTRAIN_EXPORT int nntrain_tensor_embedding_backward_reduced_bf16_gradient(
    const int* indices,
    const unsigned short* output_gradient,
    unsigned short* table_gradient,
    int* workspace,
    int workspace_ints,
    int length,
    int width,
    cudaStream_t stream) {
    return launch_embedding_bf16_gradient(
        indices,
        output_gradient,
        table_gradient,
        nullptr,
        workspace,
        workspace_ints,
        length,
        0,
        width,
        stream);
}

NNTRAIN_EXPORT int
nntrain_tensor_embedding_positions_backward_reduced_bf16_gradient(
    const int* indices,
    const unsigned short* output_gradient,
    unsigned short* token_gradient,
    unsigned short* position_gradient,
    int* workspace,
    int workspace_ints,
    int length,
    int sequence,
    int width,
    cudaStream_t stream) {
    return launch_embedding_bf16_gradient(
        indices,
        output_gradient,
        token_gradient,
        position_gradient,
        workspace,
        workspace_ints,
        length,
        sequence,
        width,
        stream);
}

NNTRAIN_EXPORT int nntrain_tensor_dropout_backward_bf16_gradient(
    const unsigned short* output_gradient,
    unsigned short* input_gradient,
    int length,
    unsigned int seed,
    unsigned int threshold,
    float scale,
    cudaStream_t stream) {
    if (!output_gradient || !input_gradient || length <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    dropout_backward_bf16_gradient_kernel<<<
        blocks_for(length), kThreads, 0, stream>>>(
            output_gradient,
            input_gradient,
            length,
            seed,
            threshold,
            scale);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_tensor_add_dropout_backward_bf16_gradient(
    const unsigned short* output_gradient,
    unsigned short* residual_gradient,
    unsigned short* branch_gradient,
    int length,
    int same_parent,
    unsigned int seed,
    unsigned int threshold,
    float scale,
    cudaStream_t stream) {
    if (!output_gradient || !residual_gradient || !branch_gradient
        || length <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    add_dropout_backward_bf16_gradient_kernel<<<
        blocks_for(length), kThreads, 0, stream>>>(
            output_gradient,
            residual_gradient,
            branch_gradient,
            length,
            same_parent,
            seed,
            threshold,
            scale);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_tensor_linear_bias_backward_bf16_gradient(
    const unsigned short* output_gradient,
    unsigned short* bias_gradient,
    int rows,
    int width,
    cudaStream_t stream) {
    if (!output_gradient || !bias_gradient || rows <= 0 || width <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    linear_bias_backward_bf16_gradient_kernel<<<
        blocks_for(width), kThreads, 0, stream>>>(
            output_gradient, bias_gradient, rows, width);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_tensor_bf16_gradient_squared_sum(
    const unsigned short* values,
    int length,
    double* result,
    cudaStream_t stream) {
    if (!values || length <= 0 || !result)
        return static_cast<int>(cudaErrorInvalidValue);
    bf16_gradient_squared_sum_kernel<<<
        blocks_for(length), kThreads, 0, stream>>>(
            values, length, result);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_tensor_bf16_gradient_scale(
    unsigned short* values,
    int length,
    float multiplier,
    cudaStream_t stream) {
    if (!values || length <= 0 || !std::isfinite(multiplier))
        return static_cast<int>(cudaErrorInvalidValue);
    bf16_gradient_scale_kernel<<<
        blocks_for(length), kThreads, 0, stream>>>(
            values, length, multiplier);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_cuda_graph_dropout_backward_bf16_gradient(
    const unsigned long long* step_counter,
    unsigned long long operation_seed,
    float dropout_probability,
    const unsigned short* output_gradient,
    unsigned short* input_gradient,
    int length,
    cudaStream_t stream) {
    if (!step_counter || !output_gradient || !input_gradient || length <= 0
        || !stream || !(dropout_probability >= 0.0f)
        || !(dropout_probability < 1.0f)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const unsigned int threshold = static_cast<unsigned int>(
        static_cast<double>(dropout_probability) * 4294967296.0);
    const float keep_scale = 1.0f / (1.0f - dropout_probability);
    graph_dropout_backward_bf16_gradient_kernel<<<
        blocks_for(length), kThreads, 0, stream>>>(
            step_counter,
            operation_seed,
            threshold,
            keep_scale,
            output_gradient,
            input_gradient,
            length);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_cuda_graph_add_dropout_backward_bf16_gradient(
    const unsigned long long* step_counter,
    unsigned long long operation_seed,
    float dropout_probability,
    const unsigned short* output_gradient,
    unsigned short* residual_gradient,
    unsigned short* branch_gradient,
    int length,
    int same_parent,
    cudaStream_t stream) {
    if (!step_counter || !output_gradient || !residual_gradient
        || !branch_gradient || length <= 0 || !stream
        || !(dropout_probability >= 0.0f)
        || !(dropout_probability < 1.0f)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const unsigned int threshold = static_cast<unsigned int>(
        static_cast<double>(dropout_probability) * 4294967296.0);
    const float keep_scale = 1.0f / (1.0f - dropout_probability);
    graph_add_dropout_backward_bf16_gradient_kernel<<<
        blocks_for(length), kThreads, 0, stream>>>(
            step_counter,
            operation_seed,
            threshold,
            keep_scale,
            output_gradient,
            residual_gradient,
            branch_gradient,
            length,
            same_parent);
    return launch_status();
}
