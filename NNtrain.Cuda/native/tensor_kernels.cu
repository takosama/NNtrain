#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cub/block/block_radix_sort.cuh>
#include <cfloat>
#include <cmath>

#if defined(_WIN32)
#define NNTRAIN_EXPORT extern "C" __declspec(dllexport)
#else
#define NNTRAIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Keep cuda_runtime_bridge.cu's thread-local selected-device cache coherent.
// The BFP8 collective entry points switch devices on one managed thread, so
// direct cudaSetDevice calls would make a later cached bridge selection skip
// a required physical device switch.
extern "C" int nntrain_cuda_set_device(int device);

namespace {
constexpr int kThreads = 256;
thread_local cudaStream_t g_stream = nullptr;

int blocks_for(int length) {
    return (length + kThreads - 1) / kThreads;
}

int launch_status() {
    return static_cast<int>(cudaGetLastError());
}

__device__ __forceinline__ float bf16_load(const unsigned short* values,
    int index) {
    return __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(values)[index]);
}

__device__ __forceinline__ void bf16_store(unsigned short* values,
    int index, float value) {
    reinterpret_cast<__nv_bfloat16*>(values)[index] =
        __float2bfloat16_rn(value);
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
    return bits < threshold ? 0.f : scale;
}

template <typename T>
__device__ __forceinline__ float load(const T* values, int index) {
    return values[index];
}

template <>
__device__ __forceinline__ float load<unsigned short>(
    const unsigned short* values, int index) {
    return bf16_load(values, index);
}

template <typename T>
__device__ __forceinline__ void store(T* values, int index, float value) {
    values[index] = value;
}

template <>
__device__ __forceinline__ void store<unsigned short>(
    unsigned short* values, int index, float value) {
    bf16_store(values, index, value);
}

template <typename T>
__global__ void add_forward_kernel(const T* left, const T* right, T* output,
    int length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length)
        store(output, index, load(left, index) + load(right, index));
}

__global__ void add_backward_kernel(const float* output_gradient,
    float* left_gradient, float* right_gradient, int length, int same_parent) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    float gradient = output_gradient[index];
    if (same_parent)
        left_gradient[index] += 2.f * gradient;
    else {
        left_gradient[index] += gradient;
        right_gradient[index] += gradient;
    }
}

// Publishes an autograd root seed without staging a host buffer.  Kernel
// arguments carry the scalar value as part of the launch command; no H2D
// memcpy is submitted.  Assignment is used when the previous authoritative
// gradient is on the host, while accumulation preserves resident leaf-gradient
// semantics across repeated backward traversals.
__global__ void scalar_gradient_seed_kernel(
    float* destination,
    float value,
    int accumulate) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        destination[0] = accumulate != 0
            ? destination[0] + value
            : value;
    }
}

// Stable value/index pair used by the two-stage vocabulary reduction.  The
// layout is intentionally identical to the managed CudaTopKCandidate record
// so the final K pairs need only one D2H copy.
struct TensorTopKCandidate {
    int index;
    float value;
};
static_assert(sizeof(TensorTopKCandidate) == 8,
    "CUDA top-K candidate ABI must remain eight bytes");

__device__ __forceinline__ bool tensor_topk_better(
    float candidate_value,
    int candidate_index,
    float current_value,
    int current_index) {
    if (current_index < 0)
        return true;
    const bool candidate_nan = isnan(candidate_value);
    const bool current_nan = isnan(current_value);
    if (candidate_nan != current_nan)
        return !candidate_nan;
    if (candidate_value > current_value)
        return true;
    if (candidate_value < current_value)
        return false;
    return candidate_index < current_index;
}

__device__ __forceinline__ float tensor_topk_load(
    const float* values,
    int index) {
    return values[index];
}

__device__ __forceinline__ float tensor_topk_load(
    const unsigned short* values,
    int index) {
    return __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(values)[index]);
}

__device__ __forceinline__ void tensor_topk_insert(
    TensorTopKCandidate* candidates,
    int k,
    int index,
    float value) {
    int insertion = k;
    for (int slot = 0; slot < k; ++slot) {
        if (tensor_topk_better(
            value,
            index,
            candidates[slot].value,
            candidates[slot].index)) {
            insertion = slot;
            break;
        }
    }
    if (insertion == k)
        return;
    for (int slot = k - 1; slot > insertion; --slot)
        candidates[slot] = candidates[slot - 1];
    candidates[insertion] = { index, value };
}

__device__ __forceinline__ float tensor_topk_negative_infinity() {
    return __int_as_float(static_cast<int>(0xff800000u));
}

__device__ __forceinline__ bool tensor_topk_candidate_better(
    const TensorTopKCandidate& candidate,
    const TensorTopKCandidate& current) {
    if (candidate.index < 0)
        return false;
    if (current.index < 0)
        return true;
    return tensor_topk_better(
        candidate.value,
        candidate.index,
        current.value,
        current.index);
}

__device__ __forceinline__ unsigned long long tensor_topk_sort_key(
    float value,
    int index) {
    unsigned int value_rank = 0;
    if (!isnan(value)) {
        unsigned int bits = __float_as_uint(value);
        // The comparison contract treats positive and negative zero as equal.
        if ((bits & 0x7fffffffu) == 0)
            bits = 0;
        value_rank = (bits & 0x80000000u) != 0
            ? ~bits
            : bits ^ 0x80000000u;
    }
    // Descending radix order then means higher numeric value first and, for
    // exact collisions (including NaNs), smaller logical token index first.
    return (static_cast<unsigned long long>(value_rank) << 32)
        | (0xffffffffu - static_cast<unsigned int>(index));
}

template <typename T>
__global__ void tensor_topk_stage_one_kernel(
    const T* values,
    int offset,
    int count,
    int k,
    TensorTopKCandidate* workspace,
    int reduction_blocks) {
    __shared__ TensorTopKCandidate reduction[256];
    TensorTopKCandidate* local = workspace + blockIdx.x * k;
    const int begin = static_cast<int>(
        static_cast<long long>(count) * blockIdx.x / reduction_blocks);
    const int end = static_cast<int>(
        static_cast<long long>(count) * (blockIdx.x + 1)
        / reduction_blocks);

    // Vocabulary-sized inputs are partitioned into at most one candidate per
    // thread.  Each rank is then one block reduction; the winning thread
    // retires its candidate before the next rank.  This replaces the old
    // serial O(partition*K) insertion loop while retaining stable ties.
    if (end - begin <= blockDim.x) {
        const int local_index = begin + threadIdx.x;
        TensorTopKCandidate candidate = local_index < end
            ? TensorTopKCandidate{
                local_index,
                tensor_topk_load(values, offset + local_index) }
            : TensorTopKCandidate{
                -1, tensor_topk_negative_infinity() };
        // Block radix sort handles K>4 in one collective and orders a
        // composite numeric-value/token-index key. Keep tiny K on the
        // lower-latency repeated reduction.
        if (k > 4) {
            using BlockSort = cub::BlockRadixSort<
                unsigned long long, 256, 1, int>;
            __shared__ typename BlockSort::TempStorage sort_storage;
            unsigned long long keys[1] = {
                candidate.index >= 0
                    ? tensor_topk_sort_key(candidate.value, candidate.index)
                    : 0ull };
            int sorted_indices[1] = { candidate.index };
            BlockSort(sort_storage).SortDescending(keys, sorted_indices);
            const int sorted_index = sorted_indices[0];
            if (threadIdx.x < k) {
                local[threadIdx.x] = sorted_index >= 0
                    ? TensorTopKCandidate{
                        sorted_index,
                        tensor_topk_load(values, offset + sorted_index) }
                    : TensorTopKCandidate{
                        -1, tensor_topk_negative_infinity() };
            }
            return;
        }
        for (int rank = 0; rank < k; ++rank) {
            reduction[threadIdx.x] = candidate;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    const TensorTopKCandidate other =
                        reduction[threadIdx.x + stride];
                    TensorTopKCandidate& current = reduction[threadIdx.x];
                    if (other.index >= 0 && tensor_topk_better(
                        other.value,
                        other.index,
                        current.value,
                        current.index)) {
                        current = other;
                    }
                }
                __syncthreads();
            }
            const TensorTopKCandidate winner = reduction[0];
            if (threadIdx.x == 0)
                local[rank] = winner;
            if (candidate.index == winner.index)
                candidate = { -1, tensor_topk_negative_infinity() };
            __syncthreads();
        }
        return;
    }

    // Very large rows can exceed the 1024-partition cap.  Preserve correctness
    // by rescanning each thread's strided slice while excluding prior winners;
    // generation vocabularies use the fast one-candidate branch above.
    for (int rank = 0; rank < k; ++rank) {
        TensorTopKCandidate candidate = {
            -1, tensor_topk_negative_infinity() };
        for (int local_index = begin + threadIdx.x;
             local_index < end;
             local_index += blockDim.x) {
            bool selected = false;
            for (int previous = 0; previous < rank; ++previous) {
                if (local[previous].index == local_index) {
                    selected = true;
                    break;
                }
            }
            if (selected)
                continue;
            const float value = tensor_topk_load(
                values, offset + local_index);
            if (tensor_topk_better(
                value,
                local_index,
                candidate.value,
                candidate.index)) {
                candidate = { local_index, value };
            }
        }
        reduction[threadIdx.x] = candidate;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const TensorTopKCandidate other =
                    reduction[threadIdx.x + stride];
                TensorTopKCandidate& current = reduction[threadIdx.x];
                if (other.index >= 0 && tensor_topk_better(
                    other.value,
                    other.index,
                    current.value,
                    current.index)) {
                    current = other;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
            local[rank] = reduction[0];
        __syncthreads();
    }
}

__global__ void tensor_topk_stage_two_kernel(
    const TensorTopKCandidate* workspace,
    int k,
    int reduction_blocks,
    TensorTopKCandidate* output) {
    __shared__ int positions[1024];
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    for (int block = 0; block < reduction_blocks; ++block)
        positions[block] = 0;
    // Each stage-one list is already sorted.  A K-way merge examines only the
    // current head of each list rather than reinserting blocks*K candidates.
    for (int rank = 0; rank < k; ++rank) {
        TensorTopKCandidate winner = {
            -1, tensor_topk_negative_infinity() };
        int winner_block = -1;
        for (int block = 0; block < reduction_blocks; ++block) {
            const int position = positions[block];
            if (position >= k)
                continue;
            const TensorTopKCandidate candidate =
                workspace[block * k + position];
            if (candidate.index >= 0 && tensor_topk_better(
                candidate.value,
                candidate.index,
                winner.value,
                winner.index)) {
                winner = candidate;
                winner_block = block;
            }
        }
        output[rank] = winner;
        if (winner_block >= 0)
            ++positions[winner_block];
    }
}

template <typename T>
int launch_tensor_topk(
    const T* values,
    int offset,
    int count,
    int k,
    TensorTopKCandidate* workspace,
    int reduction_blocks,
    TensorTopKCandidate* output) {
    if (!values || offset < 0 || count <= 0 || k <= 0 || k > 64
        || k > count || !workspace || reduction_blocks <= 0
        || reduction_blocks > 1024 || !output) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    tensor_topk_stage_one_kernel<<<
        reduction_blocks, 256, 0, g_stream>>>(
            values,
            offset,
            count,
            k,
            workspace,
            reduction_blocks);
    cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return static_cast<int>(status);
    tensor_topk_stage_two_kernel<<<1, 1, 0, g_stream>>>(
        workspace, k, reduction_blocks, output);
    return launch_status();
}

template <typename T>
__global__ void embedding_forward_kernel(const T* table, const int* indices,
    T* output, int length, int width) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= length)
        return;
    int position = linear / width;
    int column = linear - position * width;
    store(output, linear, load(table, indices[position] * width + column));
}

__global__ void embedding_backward_kernel(const int* indices,
    const float* output_gradient, float* table_gradient, int length,
    int width) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= length)
        return;
    int position = linear / width;
    int column = linear - position * width;
    atomicAdd(table_gradient + indices[position] * width + column,
        output_gradient[linear]);
}

template <typename T>
__global__ void embedding_positions_forward_kernel(const T* tokens,
    const T* positions, const int* indices, T* output, int length,
    int sequence, int width) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= length)
        return;
    int token_position = linear / width;
    int column = linear - token_position * width;
    int token = indices[token_position];
    int position = token_position % sequence;
    store(output, linear, load(tokens, token * width + column)
        + load(positions, position * width + column));
}

__global__ void embedding_positions_backward_kernel(const int* indices,
    const float* output_gradient, float* token_gradient,
    float* position_gradient, int length, int sequence, int width) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= length)
        return;
    int token_position = linear / width;
    int column = linear - token_position * width;
    int token = indices[token_position];
    int position = token_position % sequence;
    float gradient = output_gradient[linear];
    atomicAdd(token_gradient + token * width + column, gradient);
    atomicAdd(position_gradient + position * width + column, gradient);
}

// The transformer embedding gradient used to issue one global atomicAdd for
// every (token-position, column) pair. A production shard with 36 x 512
// tokens and width 1024 therefore submitted 18,874,368 contending atomics for
// the token table, plus the same number for the position table. The reduced
// path below builds a compact token -> occurrence-list map on the device and
// assigns each token row to exactly one CUDA block. Table accumulation is
// consequently an ordinary owner write, including when a token occurs many
// times. The hash-map atomics are O(token positions), rather than
// O(token positions * embedding width).
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

__global__ void embedding_token_owner_backward_kernel(
    const float* output_gradient,
    float* token_gradient,
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

    // Float4 is valid only when every row begins on a 16-byte boundary. The
    // scalar tail path retains arbitrary-width semantics (for example the
    // width=257 stress shape) without a CPU fallback.
    if ((width & 3) == 0) {
        const int vector_width = width >> 2;
        const float4* source =
            reinterpret_cast<const float4*>(output_gradient);
        float4* destination = reinterpret_cast<float4*>(token_gradient);
        for (int column = threadIdx.x;
             column < vector_width;
             column += blockDim.x) {
            float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int position = first;
                 position >= 0;
                 position = occurrence_next[position]) {
                const float4 value = source[position * vector_width + column];
                sum.x += value.x;
                sum.y += value.y;
                sum.z += value.z;
                sum.w += value.w;
            }
            float4 current = destination[token * vector_width + column];
            current.x += sum.x;
            current.y += sum.y;
            current.z += sum.z;
            current.w += sum.w;
            destination[token * vector_width + column] = current;
        }
        return;
    }

    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        float sum = 0.f;
        for (int position = first;
             position >= 0;
             position = occurrence_next[position]) {
            sum += output_gradient[position * width + column];
        }
        token_gradient[token * width + column] += sum;
    }
}

__global__ void embedding_position_owner_backward_kernel(
    const float* output_gradient,
    float* position_gradient,
    int position_count,
    int sequence,
    int width,
    int column_blocks) {
    const int position = static_cast<int>(blockIdx.x) / column_blocks;
    const int column_block = static_cast<int>(blockIdx.x)
        - position * column_blocks;
    if (position >= sequence)
        return;

    if ((width & 3) == 0) {
        const int vector_width = width >> 2;
        const int column = column_block * blockDim.x + threadIdx.x;
        if (column >= vector_width)
            return;
        const float4* source =
            reinterpret_cast<const float4*>(output_gradient);
        float4* destination = reinterpret_cast<float4*>(position_gradient);
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int source_position = position;
             source_position < position_count;
             source_position += sequence) {
            const float4 value =
                source[source_position * vector_width + column];
            sum.x += value.x;
            sum.y += value.y;
            sum.z += value.z;
            sum.w += value.w;
        }
        float4 current = destination[position * vector_width + column];
        current.x += sum.x;
        current.y += sum.y;
        current.z += sum.z;
        current.w += sum.w;
        destination[position * vector_width + column] = current;
        return;
    }

    const int column = column_block * blockDim.x + threadIdx.x;
    if (column >= width)
        return;
    float sum = 0.f;
    for (int source_position = position;
         source_position < position_count;
         source_position += sequence) {
        sum += output_gradient[source_position * width + column];
    }
    position_gradient[position * width + column] += sum;
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

long long embedding_backward_workspace_ints(int position_count) {
    const int hash_capacity = embedding_hash_capacity(position_count);
    if (hash_capacity == 0)
        return 0;
    return 2LL * hash_capacity + 2LL * position_count + 1LL;
}

int launch_embedding_reduced_backward(
    const int* indices,
    const float* output_gradient,
    float* token_gradient,
    float* position_gradient,
    int* workspace,
    int workspace_ints,
    int length,
    int sequence,
    int width) {
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
    const long long required =
        embedding_backward_workspace_ints(position_count);
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
        g_stream);
    if (status != cudaSuccess)
        return static_cast<int>(status);
    status = cudaMemsetAsync(unique_count, 0, sizeof(int), g_stream);
    if (status != cudaSuccess)
        return static_cast<int>(status);

    embedding_occurrence_map_kernel<<<
        blocks_for(position_count), kThreads, 0, g_stream>>>(
            indices,
            hash_keys,
            hash_heads,
            occurrence_next,
            unique_slots,
            unique_count,
            position_count,
            hash_capacity - 1);
    status = cudaGetLastError();
    if (status != cudaSuccess)
        return static_cast<int>(status);

    // The grid is bounded by position_count; blocks beyond unique_count exit
    // immediately, avoiding a device-to-host read of the dynamic count.
    embedding_token_owner_backward_kernel<<<
        position_count, kThreads, 0, g_stream>>>(
            output_gradient,
            token_gradient,
            hash_keys,
            hash_heads,
            occurrence_next,
            unique_slots,
            unique_count,
            width);
    status = cudaGetLastError();
    if (status != cudaSuccess)
        return static_cast<int>(status);

    if (sequence != 0) {
        const int reduced_width = (width & 3) == 0 ? width / 4 : width;
        const int column_blocks = blocks_for(reduced_width);
        embedding_position_owner_backward_kernel<<<
            sequence * column_blocks, kThreads, 0, g_stream>>>(
                output_gradient,
                position_gradient,
                position_count,
                sequence,
                width,
                column_blocks);
        status = cudaGetLastError();
    }
    return static_cast<int>(status);
}

template <typename T>
__global__ void dropout_forward_kernel(const T* input, T* output, int length,
    unsigned int seed, unsigned int threshold, float scale) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length)
        store(output, index, load(input, index)
            * dropout_multiplier(seed, index, threshold, scale));
}

__global__ void dropout_backward_kernel(const float* output_gradient,
    float* input_gradient, int length, unsigned int seed,
    unsigned int threshold, float scale) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length)
        input_gradient[index] += output_gradient[index]
            * dropout_multiplier(seed, index, threshold, scale);
}

template <typename T>
__global__ void add_dropout_forward_kernel(const T* residual,
    const T* branch, T* output, int length, unsigned int seed,
    unsigned int threshold, float scale) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length)
        store(output, index, load(residual, index) + load(branch, index)
            * dropout_multiplier(seed, index, threshold, scale));
}

__global__ void add_dropout_backward_kernel(const float* output_gradient,
    float* residual_gradient, float* branch_gradient, int length,
    int same_parent, unsigned int seed, unsigned int threshold, float scale) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    float gradient = output_gradient[index];
    float multiplier = dropout_multiplier(seed, index, threshold, scale);
    if (same_parent)
        residual_gradient[index] += gradient * (1.f + multiplier);
    else {
        residual_gradient[index] += gradient;
        branch_gradient[index] += gradient * multiplier;
    }
}

template <typename T>
__global__ void linear_bias_kernel(T* output, const T* bias, int length,
    int width, int relu) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    float value = load(output, index) + load(bias, index % width);
    if (relu && value <= 0.f)
        value = 0.f;
    store(output, index, value);
}

__global__ void linear_mask_kernel(const float* output,
    float* output_gradient, int length, int relu) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length && relu && output[index] <= 0.f)
        output_gradient[index] = 0.f;
}

__global__ void linear_encode_bf16_kernel(const float* output_gradient,
    const unsigned short* output, unsigned short* encoded, int length,
    int relu) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    float gradient = output_gradient[index];
    if (relu && bf16_load(output, index) <= 0.f)
        gradient = 0.f;
    bf16_store(encoded, index, gradient);
}

__global__ void linear_encode_bfp8_relu_kernel(
    const float* output_gradient,
    const signed char* output_payload,
    unsigned short* encoded,
    int length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    // The forward ReLU is quantized before it is stored. Its dequantized sign
    // is therefore exactly the payload sign because every BFP8 scale is
    // positive. Reading the code directly avoids a full BFP8->BF16 decode.
    const float gradient = output_payload[index] > 0
        ? output_gradient[index]
        : 0.f;
    bf16_store(encoded, index, gradient);
}

__global__ void linear_mask_bf16_gradient_kernel(
    const unsigned short* output_gradient,
    const unsigned short* output,
    unsigned short* masked,
    int length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    masked[index] = bf16_load(output, index) > 0.f
        ? output_gradient[index]
        : 0;
}

template <typename T>
__global__ void linear_bias_backward_kernel(const T* output_gradient,
    float* bias_gradient, int rows, int width) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= width)
        return;
    float sum = 0.f;
    for (int row = 0; row < rows; ++row)
        sum += load(output_gradient, row * width + column);
    bias_gradient[column] += sum;
}

__global__ void scale_kernel(float* values, int length, float scale) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length)
        values[index] *= scale;
}

__global__ void accumulate_kernel(const float* source, float* destination,
    int length, int source_offset, int destination_offset) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length)
        destination[destination_offset + index] +=
            source[source_offset + index];
}

__global__ void copy_kernel(const float* source, float* destination,
    int length, int source_offset, int destination_offset) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length)
        destination[destination_offset + index] = source[source_offset + index];
}

__global__ void encode_bf16_kernel(const float* source,
    unsigned short* destination, int length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length)
        bf16_store(destination, index, source[index]);
}

__global__ void softmax_probabilities_kernel(const float* logits,
    const float* maxima, const float* inverse_sums, float* probabilities,
    int length, int columns) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear < length) {
        int row = linear / columns;
        probabilities[linear] = expf(logits[linear] - maxima[row])
            * inverse_sums[row];
    }
}

__global__ void cross_entropy_probabilities_backward_kernel(
    const float* probabilities, const int* labels, float* gradient,
    int length, int columns, int ignore_index, int valid_rows,
    float smoothing, float upstream) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= length)
        return;
    int row = linear / columns;
    int column = linear - row * columns;
    int label = labels[row];
    if (label == ignore_index)
        return;
    float target = smoothing / columns;
    if (column == label)
        target += 1.f - smoothing;
    gradient[linear] += upstream / valid_rows
        * (probabilities[linear] - target);
}

__global__ void squared_sum_kernel(const float* values, int length,
    double* result) {
    __shared__ double sums[kThreads];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double value = 0.0;
    if (index < length) {
        double x = values[index];
        value = x * x;
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

__global__ void bfp8_gradient_max_finite_kernel(const float* source,
    int length, unsigned int* maximum_bits, int* finite_status) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    float value = source[index];
    if (!isfinite(value)) {
        atomicExch(finite_status, 1);
        return;
    }
    atomicMax(maximum_bits, __float_as_uint(fabsf(value)));
}

__global__ void optimizer_accumulate_finite_status_kernel(
    const float* values, int length, int* finite_status) {
    const int stride = blockDim.x * gridDim.x;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < length; index += stride) {
        if (!isfinite(values[index]))
            atomicExch(finite_status, 1);
    }
}

__global__ void bfp8_gradient_finalize_scale_kernel(float* scale,
    const int* finite_status) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    if (*finite_status != 0) {
        *scale = 1.f;
        return;
    }
    float maximum = *scale;
    *scale = maximum == 0.f ? 1.f : __fdiv_rn(maximum, 127.f);
}

__global__ void bfp8_gradient_quantize_kernel(float* source,
    signed char* payload, const float* scale, int length,
    const int* finite_status) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length || *finite_status != 0)
        return;
    int quantized = __float2int_rn(__fdiv_rn(source[index], scale[0]));
    quantized = max(-127, min(127, quantized));
    payload[index] = static_cast<signed char>(quantized);
}

__global__ void bfp8_gradient_decode_kernel(const signed char* payload,
    const float* scale, float* destination, int length,
    const int* finite_status) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length || *finite_status != 0)
        return;
    float value = static_cast<float>(payload[index]) * scale[0];
    destination[index] = value;
}

// Accumulates the norm of the authoritative quantized representation.  One
// atomicAdd per block keeps the cost independent of any Float32 decode cache
// and avoids the per-element double atomics used by the first BFP8 reducer.
__global__ void bfp8_gradient_squared_sum_kernel(
    const signed char* payload, const float* scale, int length,
    double* squared_sum, int* finite_status) {
    __shared__ double sums[kThreads];
    const float tensor_scale = scale[0];
    const bool valid_scale = isfinite(tensor_scale) && tensor_scale > 0.f;
    if (!valid_scale && threadIdx.x == 0)
        atomicExch(finite_status, 1);

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    double value = 0.0;
    if (valid_scale && index < length) {
        const float decoded_float =
            static_cast<float>(payload[index]) * tensor_scale;
        const double decoded = static_cast<double>(decoded_float);
        value = decoded * decoded;
    }
    sums[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0 && valid_scale)
        atomicAdd(squared_sum, sums[0]);
}

__global__ void bfp8_gradient_scale_kernel(float* scale, float multiplier) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    // A positive sidecar is part of the BFP8 storage invariant.  The clamp is
    // only observable for an extreme underflowing clip coefficient; payload
    // bytes remain untouched in every case.
    scale[0] = fmaxf(scale[0] * multiplier, FLT_MIN);
}

__global__ void bfp8_gradient_sum_kernel(const signed char* local_payload,
    const float* local_scale, const signed char* remote_payload,
    const float* remote_scale, float* destination, int length,
    float reduction_scale, int* finite_status) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    float left_scale = local_scale[0];
    float right_scale = remote_scale[0];
    if (!isfinite(left_scale) || left_scale <= 0.f
        || !isfinite(right_scale) || right_scale <= 0.f) {
        atomicExch(finite_status, 1);
        return;
    }
    float value = reduction_scale * (
        static_cast<float>(local_payload[index]) * left_scale
        + static_cast<float>(remote_payload[index]) * right_scale);
    if (!isfinite(value)) {
        atomicExch(finite_status, 1);
        return;
    }
    destination[index] = value;
}

__global__ void bfp8_gradient_merge_status_kernel(int* destination,
    const int* remote) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && *remote != 0)
        *destination = 1;
}

int launch_bfp8_gradient_quantize(float* source, signed char* payload,
    float* scale, int length, int* finite_status, cudaStream_t stream,
    double* squared_sum) {
    cudaError_t status = cudaMemsetAsync(scale, 0, sizeof(float), stream);
    if (status != cudaSuccess)
        return static_cast<int>(status);
    bfp8_gradient_max_finite_kernel<<<blocks_for(length), kThreads, 0,
        stream>>>(source, length, reinterpret_cast<unsigned int*>(scale),
            finite_status);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return static_cast<int>(status);
    bfp8_gradient_finalize_scale_kernel<<<1, 1, 0, stream>>>(
        scale, finite_status);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return static_cast<int>(status);
    bfp8_gradient_quantize_kernel<<<blocks_for(length), kThreads, 0,
        stream>>>(source, payload, scale, length, finite_status);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return static_cast<int>(status);
    bfp8_gradient_decode_kernel<<<blocks_for(length), kThreads, 0,
        stream>>>(payload, scale, source, length, finite_status);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return static_cast<int>(status);
    if (squared_sum != nullptr) {
        bfp8_gradient_squared_sum_kernel<<<blocks_for(length), kThreads, 0,
            stream>>>(payload, scale, length, squared_sum, finite_status);
    }
    return launch_status();
}

__global__ void adamw_kernel(float* data, const float* gradient,
    float* first_moment, float* second_moment, int length, float beta1,
    float beta2, float learning_rate, float weight_decay, float update_scale,
    float scaled_epsilon, int apply_weight_decay, void* compute,
    int physical_bf16) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    float g = gradient[index];
    float first = beta1 * first_moment[index] + (1.f - beta1) * g;
    float second = beta2 * second_moment[index] + (1.f - beta2) * g * g;
    first_moment[index] = first;
    second_moment[index] = second;
    float parameter = data[index];
    if (apply_weight_decay)
        parameter *= 1.f - learning_rate * weight_decay;
    parameter = parameter - update_scale * first /
        (sqrtf(second) + scaled_epsilon);
    data[index] = parameter;
    if (compute != nullptr) {
        const __nv_bfloat16 value = __float2bfloat16_rn(parameter);
        if (physical_bf16)
            reinterpret_cast<__nv_bfloat16*>(compute)[index] = value;
        else
            reinterpret_cast<float*>(compute)[index] = __bfloat162float(value);
    }
}

// Pure BFP8 keeps the signed-int8 moment payloads as the sole persistent
// optimizer authority.  Updating the parameter from the transient FP32
// moments before requantizing them makes AdamW observe precision that will no
// longer exist at the next step.  More importantly, a tensor-wide BFP8 second
// moment can round a small positive value to zero while its first moment stays
// non-zero.  The ordinary Adam epsilon is far below one BFP8 quantum, so that
// combination produces an unbounded update on the following sparse-gradient
// step.  Split the operation into moment update and parameter application so
// managed code can requantize/dequantize the moments between these kernels.
__global__ void adamw_bfp8_moments_kernel(const float* gradient,
    float* first_moment, float* second_moment, int length, float beta1,
    float beta2, int* finite_status) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length || *finite_status != 0)
        return;
    const float g = gradient[index];
    const float first = fmaf(beta1, first_moment[index],
        (1.f - beta1) * g);
    const float second = fmaf(beta2, second_moment[index],
        (1.f - beta2) * g * g);
    if (!isfinite(first) || !isfinite(second) || second < 0.f) {
        atomicExch(finite_status, 1);
        return;
    }
    first_moment[index] = first;
    second_moment[index] = second;
}

__global__ void adamw_bfp8_apply_kernel(float* data,
    const float* first_moment, const float* second_moment,
    const float* second_scale, int second_scale_block_size,
    int length, float learning_rate,
    float weight_decay, float update_scale, float scaled_epsilon,
    int apply_weight_decay, int* finite_status) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length || *finite_status != 0)
        return;

    // A zero code denotes the interval [-scale/2, scale/2].  Use the
    // half-quantum uncertainty as a conservative variance floor instead of
    // pretending that the missing second moment is exactly zero.  This is
    // scale-aware (and therefore invariant to loss/gradient rescaling), while
    // scaled_epsilon still provides AdamW's configured numerical epsilon.
    const float quantum =
        second_scale[index / second_scale_block_size];
    if (!isfinite(quantum) || !(quantum > 0.f)) {
        atomicExch(finite_status, 1);
        return;
    }
    const float variance_floor = 0.5f * quantum;
    const float variance = fmaxf(second_moment[index], variance_floor);
    const float denominator = sqrtf(variance) + scaled_epsilon;
    float parameter = data[index];
    if (apply_weight_decay)
        parameter *= 1.f - learning_rate * weight_decay;
    parameter -= update_scale * first_moment[index] / denominator;
    if (!isfinite(parameter)) {
        atomicExch(finite_status, 1);
        return;
    }
    data[index] = parameter;
}

__global__ void adamw_bf16_state_kernel(float* data, const float* gradient,
    unsigned short* first_moment, unsigned short* second_moment, int length,
    float beta1, float beta2, float learning_rate, float weight_decay,
    float update_scale, float scaled_epsilon, int apply_weight_decay,
    void* compute, int physical_bf16) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    float g = gradient[index];
    float previous_first = bf16_load(first_moment, index);
    float previous_second = bf16_load(second_moment, index);
    float first = beta1 * previous_first + (1.f - beta1) * g;
    float second = beta2 * previous_second + (1.f - beta2) * g * g;
    bf16_store(first_moment, index, first);
    bf16_store(second_moment, index, second);
    float parameter = data[index];
    if (apply_weight_decay)
        parameter *= 1.f - learning_rate * weight_decay;
    parameter = parameter - update_scale * first /
        (sqrtf(second) + scaled_epsilon);
    data[index] = parameter;
    if (compute != nullptr) {
        const __nv_bfloat16 value = __float2bfloat16_rn(parameter);
        if (physical_bf16)
            reinterpret_cast<__nv_bfloat16*>(compute)[index] = value;
        else
            reinterpret_cast<float*>(compute)[index] = __bfloat162float(value);
    }
}

// Pure BF16 has no persistent FP32 master weight. Parameter, gradient and
// both moments are read and published as BF16; FP32 exists only in registers
// for the element-local update and accumulation arithmetic.
__global__ void adamw_pure_bf16_kernel(__nv_bfloat16* data,
    const __nv_bfloat16* gradient, __nv_bfloat16* first_moment,
    __nv_bfloat16* second_moment, int length, float beta1, float beta2,
    float learning_rate, float weight_decay, float update_scale,
    float scaled_epsilon, int apply_weight_decay) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    const float g = __bfloat162float(gradient[index]);
    float first = fmaf(
        beta1, __bfloat162float(first_moment[index]), (1.f - beta1) * g);
    float second = fmaf(
        beta2, __bfloat162float(second_moment[index]),
        (1.f - beta2) * g * g);
    first_moment[index] = __float2bfloat16_rn(first);
    second_moment[index] = __float2bfloat16_rn(second);
    first = __bfloat162float(first_moment[index]);
    second = __bfloat162float(second_moment[index]);
    float parameter = __bfloat162float(data[index]);
    if (apply_weight_decay)
        parameter *= 1.f - learning_rate * weight_decay;
    parameter -= update_scale * first / (sqrtf(second) + scaled_epsilon);
    data[index] = __float2bfloat16_rn(parameter);
}

struct AdamWChunkDescriptor {
    float* data;
    const float* gradient;
    void* first_moment;
    void* second_moment;
    void* compute;
    int offset;
    int length;
    int apply_weight_decay;
    int physical_bf16;
    int bfloat16_state;
    int pure_bfloat16;
};
static_assert(sizeof(AdamWChunkDescriptor) == 64,
    "AdamW descriptor ABI must match managed layout");

__global__ void adamw_multi_tensor_kernel(
    const AdamWChunkDescriptor* chunks,
    int chunk_count,
    float beta1,
    float beta2,
    float learning_rate,
    float weight_decay,
    float update_scale,
    float scaled_epsilon) {
    const int chunk_index = blockIdx.x;
    if (chunk_index >= chunk_count)
        return;
    const AdamWChunkDescriptor chunk = chunks[chunk_index];
    for (int local = threadIdx.x; local < chunk.length;
         local += blockDim.x) {
        const int index = chunk.offset + local;
        const float g = chunk.pure_bfloat16
            ? __bfloat162float(
                reinterpret_cast<const __nv_bfloat16*>(
                    chunk.gradient)[index])
            : chunk.gradient[index];
        float first;
        float second;
        if (chunk.bfloat16_state) {
            auto* first_bf16 =
                reinterpret_cast<unsigned short*>(chunk.first_moment);
            auto* second_bf16 =
                reinterpret_cast<unsigned short*>(chunk.second_moment);
            first = fmaf(beta1, bf16_load(first_bf16, index),
                (1.f - beta1) * g);
            second = fmaf(beta2, bf16_load(second_bf16, index),
                (1.f - beta2) * g * g);
            bf16_store(first_bf16, index, first);
            bf16_store(second_bf16, index, second);
            if (chunk.pure_bfloat16) {
                first = bf16_load(first_bf16, index);
                second = bf16_load(second_bf16, index);
            }
        }
        else {
            auto* first_float = reinterpret_cast<float*>(chunk.first_moment);
            auto* second_float = reinterpret_cast<float*>(chunk.second_moment);
            first = fmaf(beta1, first_float[index], (1.f - beta1) * g);
            second = fmaf(beta2, second_float[index],
                (1.f - beta2) * g * g);
            first_float[index] = first;
            second_float[index] = second;
        }
        float parameter = chunk.pure_bfloat16
            ? __bfloat162float(
                reinterpret_cast<const __nv_bfloat16*>(chunk.data)[index])
            : chunk.data[index];
        if (chunk.apply_weight_decay)
            parameter *= 1.f - learning_rate * weight_decay;
        parameter -= update_scale * first /
            (sqrtf(second) + scaled_epsilon);
        if (chunk.pure_bfloat16) {
            reinterpret_cast<__nv_bfloat16*>(chunk.data)[index] =
                __float2bfloat16_rn(parameter);
        }
        else {
            chunk.data[index] = parameter;
        }
        if (chunk.compute != nullptr) {
            const __nv_bfloat16 value = __float2bfloat16_rn(parameter);
            if (chunk.physical_bf16) {
                reinterpret_cast<__nv_bfloat16*>(chunk.compute)[index] = value;
            }
            else {
                reinterpret_cast<float*>(chunk.compute)[index] =
                    __bfloat162float(value);
            }
        }
    }
}

__global__ void publish_bf16_kernel(const float* master, void* compute,
    int length, int physical) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    __nv_bfloat16 value = __float2bfloat16_rn(master[index]);
    if (physical)
        reinterpret_cast<__nv_bfloat16*>(compute)[index] = value;
    else
        reinterpret_cast<float*>(compute)[index] = __bfloat162float(value);
}

__global__ void gather_optimizer_stats_kernel(
    const float* const* sources,
    float* destination,
    int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count * 4)
        return;
    destination[index] = sources[index >> 2][index & 3];
}

__global__ void neko_moments_kernel(const float* gradient, float* fast,
    float* slow, float* fast_hat, float* slow_hat, int length,
    float beta_fast, float beta_slow, float fast_correction,
    float slow_correction) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    float next_fast = beta_fast * fast[index]
        + (1.f - beta_fast) * gradient[index];
    float next_slow = beta_slow * slow[index]
        + (1.f - beta_slow) * gradient[index];
    fast[index] = next_fast;
    slow[index] = next_slow;
    fast_hat[index] = next_fast / fast_correction;
    slow_hat[index] = next_slow / slow_correction;
}

__global__ void neko_initialize_kernel(const float* source,
    float* destination, int length, int original_rows, int original_columns,
    int transpose, float inverse_norm) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= length)
        return;
    if (!transpose) {
        destination[linear] = source[linear] * inverse_norm;
        return;
    }
    int row = linear / original_columns;
    int column = linear - row * original_columns;
    destination[column * original_rows + row] = source[linear] * inverse_norm;
}

__global__ void neko_initialize_corrected_kernel(const float* source,
    float* destination, int length, int original_rows, int original_columns,
    int transpose, float inverse_fast_correction, float inverse_norm) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= length)
        return;
    float value = (source[linear] * inverse_fast_correction) * inverse_norm;
    if (!transpose) {
        destination[linear] = value;
        return;
    }
    int row = linear / original_columns;
    int column = linear - row * original_columns;
    destination[column * original_rows + row] = value;
}

__global__ void neko_initialize_bf16_corrected_kernel(
    const __nv_bfloat16* source, float* destination, int length,
    int original_rows, int original_columns, int transpose,
    float inverse_fast_correction, float inverse_norm) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= length)
        return;
    const float value = (__bfloat162float(source[linear])
        * inverse_fast_correction) * inverse_norm;
    if (!transpose) {
        destination[linear] = value;
        return;
    }
    const int row = linear / original_columns;
    const int column = linear - row * original_columns;
    destination[column * original_rows + row] = value;
}

__global__ void neko_update_device_control_kernel(
    const float* stats,
    float* confidence,
    int* finite_status,
    float epsilon,
    float rho) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    const double fast_norm = static_cast<double>(stats[1]);
    const double slow_norm = static_cast<double>(stats[2]);
    const double residual_norm = static_cast<double>(stats[3]);
    const double denominator = sqrt(fast_norm) * sqrt(slow_norm)
        + static_cast<double>(epsilon);
    const double alignment = fmax(
        0.0, static_cast<double>(stats[0]) / denominator);
    const double persistence = slow_norm /
        (slow_norm + residual_norm + static_cast<double>(epsilon));
    const float raw = static_cast<float>(fmin(
        1.0, fmax(0.0, alignment * persistence)));
    const float next = fminf(
        1.f, fmaxf(0.f, fmaf(rho, confidence[0], (1.f - rho) * raw)));
    if (!isfinite(next) || !isfinite(stats[0]) || !isfinite(stats[1])
        || !isfinite(stats[2]) || !isfinite(stats[3])) {
        atomicExch(finite_status, 1);
        return;
    }
    confidence[0] = next;
}

__global__ void neko_initialize_device_stats_kernel(
    const float* source,
    float* destination,
    int length,
    int original_rows,
    int original_columns,
    int transpose,
    float inverse_fast_correction,
    const float* stats,
    float epsilon,
    int* finite_status) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= length)
        return;
    const float denominator = sqrtf(stats[1]) + epsilon;
    const float inverse_norm = 1.f / denominator;
    const float value = source[linear]
        * inverse_fast_correction * inverse_norm;
    if (!isfinite(value)) {
        atomicExch(finite_status, 1);
        return;
    }
    if (!transpose) {
        destination[linear] = value;
        return;
    }
    const int row = linear / original_columns;
    const int column = linear - row * original_columns;
    destination[column * original_rows + row] = value;
}

__global__ void neko_initialize_bf16_device_stats_kernel(
    const __nv_bfloat16* source, float* destination, int length,
    int original_rows, int original_columns, int transpose,
    float inverse_fast_correction, const float* stats, float epsilon,
    int* finite_status) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= length)
        return;
    const float inverse_norm = 1.f / (sqrtf(stats[1]) + epsilon);
    const float value = __bfloat162float(source[linear])
        * inverse_fast_correction * inverse_norm;
    if (!isfinite(value)) {
        atomicExch(finite_status, 1);
        return;
    }
    if (!transpose) {
        destination[linear] = value;
        return;
    }
    const int row = linear / original_columns;
    const int column = linear - row * original_columns;
    destination[column * original_rows + row] = value;
}

__global__ void neko_interpolate_kernel(float* current, const float* next,
    int length, float fraction) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length)
        current[index] += fraction * (next[index] - current[index]);
}

__global__ void neko_transpose_back_kernel(const float* source,
    float* destination, int length, int original_rows, int original_columns) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= length)
        return;
    int row = linear / original_columns;
    int column = linear - row * original_columns;
    destination[linear] = source[column * original_rows + row];
}

__global__ void neko_apply_kernel(float* data, const float* update,
    int length, float learning_rate, float final_scale, float weight_decay,
    int apply_weight_decay) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    float parameter = data[index];
    if (apply_weight_decay)
        parameter -= learning_rate * weight_decay * parameter;
    data[index] = parameter - learning_rate * final_scale * update[index];
}

__global__ void neko_apply_bf16_kernel(__nv_bfloat16* data,
    const float* update, int length, float learning_rate, float final_scale,
    float weight_decay, int apply_weight_decay) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    float parameter = __bfloat162float(data[index]);
    if (apply_weight_decay)
        parameter -= learning_rate * weight_decay * parameter;
    parameter -= learning_rate * final_scale * update[index];
    data[index] = __float2bfloat16_rn(parameter);
}

__global__ void neko_combine_kernel(const float* gram, float* gram_squared,
    int length, int rows, float a, float b, float c) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= length)
        return;
    int row = linear / rows;
    int column = linear - row * rows;
    gram_squared[linear] = b * gram[linear] + c * gram_squared[linear]
        + (row == column ? a : 0.f);
}

__global__ void neko_combine_batched_kernel(
    const float* gram,
    float* gram_squared,
    int matrix_length,
    int total_length,
    int rows,
    float a,
    float b,
    float c) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total_length)
        return;
    int matrix_linear = linear % matrix_length;
    int row = matrix_linear / rows;
    int column = matrix_linear - row * rows;
    gram_squared[linear] = b * gram[linear] + c * gram_squared[linear]
        + (row == column ? a : 0.f);
}

__device__ __forceinline__ float round_bf16_operand(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

template <bool BFloat16Operands>
__global__ void symmetric_gram_kernel(const float* source,
    float* destination, int rows, int columns) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int length = rows * rows;
    if (linear >= length)
        return;
    int row = linear / rows;
    int other = linear - row * rows;
    float sum = 0.f;
    for (int column = 0; column < columns; ++column) {
        float left = source[row * columns + column];
        float right = source[other * columns + column];
        if constexpr (BFloat16Operands) {
            left = round_bf16_operand(left);
            right = round_bf16_operand(right);
        }
        sum = fmaf(left, right, sum);
    }
    destination[linear] = sum;
}

template <bool BFloat16Operands>
__global__ void newton_schulz_kernel(const float* source,
    const float* gram, const float* gram_squared, float* destination,
    int rows, int columns, float a, float b, float c) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= rows * columns)
        return;
    int row = linear / columns;
    int column = linear - row * columns;
    if constexpr (BFloat16Operands) {
        float result = 0.f;
        int coefficient_offset = row * rows;
        for (int inner = 0; inner < rows; ++inner) {
            float coefficient = fmaf(
                c,
                gram_squared[coefficient_offset + inner],
                b * gram[coefficient_offset + inner]);
            if (row == inner)
                coefficient += a;
            coefficient = round_bf16_operand(coefficient);
            const float source_value = round_bf16_operand(
                source[inner * columns + column]);
            result = fmaf(coefficient, source_value, result);
        }
        destination[linear] = result;
        return;
    }
    float result = a * source[linear];
    int coefficient_offset = row * rows;
    for (int inner = 0; inner < rows; ++inner) {
        float coefficient = b * gram[coefficient_offset + inner]
            + c * gram_squared[coefficient_offset + inner];
        result += coefficient * source[inner * columns + column];
    }
    destination[linear] = result;
}

__device__ __forceinline__ float stable_sigmoid(float value) {
    if (value >= 0.f)
        return 1.f / (1.f + expf(-value));
    float exponential = expf(value);
    return exponential / (1.f + exponential);
}

__device__ __forceinline__ float forget_load(const float* float_values,
    const unsigned short* bf16_values, int index, int bfloat16) {
    return bfloat16 ? bf16_load(bf16_values, index) : float_values[index];
}

__device__ __forceinline__ void forget_store(float* float_values,
    unsigned short* bf16_values, int index, float value, int bfloat16) {
    if (bfloat16)
        bf16_store(bf16_values, index, value);
    else
        float_values[index] = value;
}

__global__ void forget_forward_kernel(const float* projected,
    const unsigned short* projected_bf16, float* output,
    unsigned short* output_bf16, float* states, float* state, int workers,
    int sequence, int projection_width, int key_width, int value_width,
    float retention_floor, int memory_variant, int bfloat16) {
    int worker = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker >= workers)
        return;
    int use_v3 = memory_variant == 1;
    int use_drn = memory_variant == 2;
    int batch = worker / value_width;
    int value_index = worker - batch * value_width;
    int matrix_size = key_width * value_width;
    int projected_batch = batch * sequence * projection_width;
    int output_batch = batch * sequence * value_width;
    int state_batch = batch * matrix_size;
    int states_batch = batch * sequence * matrix_size;
    float inverse_sqrt_key = rsqrtf(static_cast<float>(key_width));
    for (int time = 0; time < sequence; ++time) {
        int projected_offset = projected_batch + time * projection_width;
        int key_offset = projected_offset + key_width;
        int value_offset = key_offset + key_width;
        int gate_offset = value_offset + value_width;
        int beta_offset = gate_offset + value_width;
        int row = state_batch + value_index * key_width;
        float gate = stable_sigmoid(forget_load(projected, projected_bf16,
            gate_offset + value_index, bfloat16));
        float retention = use_drn ? gate
            : retention_floor + (1.f - retention_floor) * gate;
        float beta = stable_sigmoid(forget_load(projected, projected_bf16,
            beta_offset + value_index, bfloat16));
        float write = (use_v3 || use_drn) ? beta
            : (1.f - retention) * beta;
        float value = tanhf(forget_load(projected, projected_bf16,
            value_offset + value_index, bfloat16));
        float key_squared_norm = use_drn ? 1e-8f : 1e-6f;
        float query_squared_norm = 1e-8f;
        if (use_v3 || use_drn) {
            for (int key = 0; key < key_width; ++key) {
                float key_tanh = tanhf(forget_load(projected,
                    projected_bf16, key_offset + key, bfloat16));
                key_squared_norm += key_tanh * key_tanh;
                if (use_drn) {
                    float query_tanh = tanhf(forget_load(projected,
                        projected_bf16, projected_offset + key, bfloat16));
                    query_squared_norm += query_tanh * query_tanh;
                }
            }
        }
        float key_scale = (use_v3 || use_drn)
            ? rsqrtf(key_squared_norm) : inverse_sqrt_key;
        float query_scale = use_drn
            ? rsqrtf(query_squared_norm) : inverse_sqrt_key;
        if (use_drn) {
            float recalled = 0.f;
            for (int key = 0; key < key_width; ++key) {
                float query = tanhf(forget_load(projected, projected_bf16,
                    projected_offset + key, bfloat16)) * query_scale;
                recalled += state[row + key] * query;
            }
            forget_store(output, output_bf16,
                output_batch + time * value_width + value_index, recalled,
                bfloat16);
        }
        float predicted = 0.f;
        for (int key = 0; key < key_width; ++key) {
            float normalized_key = tanhf(forget_load(projected,
                projected_bf16, key_offset + key, bfloat16)) * key_scale;
            predicted += state[row + key] * normalized_key;
        }
        if (use_v3)
            predicted *= retention;
        float delta = write * (value - predicted);
        for (int key = 0; key < key_width; ++key) {
            float normalized_key = tanhf(forget_load(projected,
                projected_bf16, key_offset + key, bfloat16)) * key_scale;
            state[row + key] = retention * state[row + key]
                + delta * normalized_key;
        }
        if (!use_drn) {
            float recalled = 0.f;
            for (int key = 0; key < key_width; ++key) {
                float query = tanhf(forget_load(projected, projected_bf16,
                    projected_offset + key, bfloat16)) * query_scale;
                recalled += state[row + key] * query;
            }
            forget_store(output, output_bf16,
                output_batch + time * value_width + value_index, recalled,
                bfloat16);
        }
        int state_time = states_batch + time * matrix_size;
        int row_offset = value_index * key_width;
        for (int key = 0; key < key_width; ++key)
            states[state_time + row_offset + key] =
                state[state_batch + row_offset + key];
    }
}

__global__ void forget_backward_kernel(const float* projected,
    const unsigned short* projected_bf16, float* projected_gradient,
    const float* output_gradient, const float* states, float* state_gradient,
    float* previous_gradient, int workers, int sequence,
    int projection_width, int key_width, int value_width,
    float retention_floor, int memory_variant, int bfloat16) {
    int worker = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker >= workers)
        return;
    int use_v3 = memory_variant == 1;
    int use_drn = memory_variant == 2;
    int batch = worker / value_width;
    int value_index = worker - batch * value_width;
    int matrix_size = key_width * value_width;
    int projected_batch = batch * sequence * projection_width;
    int output_batch = batch * sequence * value_width;
    int states_batch = batch * sequence * matrix_size;
    int gradient_batch = batch * matrix_size;
    float inverse_sqrt_key = rsqrtf(static_cast<float>(key_width));
    for (int time = sequence - 1; time >= 0; --time) {
        int projected_offset = projected_batch + time * projection_width;
        int query_offset = projected_offset;
        int key_offset = query_offset + key_width;
        int value_offset = key_offset + key_width;
        int gate_offset = value_offset + value_width;
        int beta_offset = gate_offset + value_width;
        int current_state = states_batch + time * matrix_size;
        int previous_state = states_batch + (time - 1) * matrix_size;
        int row = value_index * key_width;
        int gradient_row = gradient_batch + row;
        for (int key = 0; key < key_width; ++key)
            previous_gradient[gradient_row + key] = 0.f;
        float recalled_gradient = output_gradient[
            output_batch + time * value_width + value_index];
        if (use_drn) {
            float query_squared_norm = 1e-8f;
            for (int key = 0; key < key_width; ++key) {
                float q = tanhf(forget_load(projected, projected_bf16,
                    query_offset + key, bfloat16));
                query_squared_norm += q * q;
            }
            float query_scale = rsqrtf(query_squared_norm);
            float query_dot_gradient = 0.f;
            for (int key = 0; key < key_width; ++key) {
                float previous = time == 0 ? 0.f
                    : states[previous_state + row + key];
                float query_gradient = previous * recalled_gradient;
                query_dot_gradient += tanhf(forget_load(projected,
                    projected_bf16, query_offset + key, bfloat16))
                    * query_gradient;
            }
            for (int key = 0; key < key_width; ++key) {
                float previous = time == 0 ? 0.f
                    : states[previous_state + row + key];
                float q = tanhf(forget_load(projected, projected_bf16,
                    query_offset + key, bfloat16));
                float query_gradient = previous * recalled_gradient;
                float tanh_gradient = query_gradient * query_scale
                    - q * query_dot_gradient * query_scale * query_scale
                        * query_scale;
                atomicAdd(projected_gradient + query_offset + key,
                    tanh_gradient * (1.f - q * q));
                previous_gradient[gradient_row + key] +=
                    q * query_scale * recalled_gradient;
            }
        }
        else {
            for (int key = 0; key < key_width; ++key) {
                float q = tanhf(forget_load(projected, projected_bf16,
                    query_offset + key, bfloat16));
                float normalized_query = q * inverse_sqrt_key;
                float derivative = (1.f - q * q) * inverse_sqrt_key;
                atomicAdd(projected_gradient + query_offset + key,
                    states[current_state + row + key] * recalled_gradient
                        * derivative);
                state_gradient[gradient_row + key] +=
                    normalized_query * recalled_gradient;
            }
        }
        float gate = stable_sigmoid(forget_load(projected, projected_bf16,
            gate_offset + value_index, bfloat16));
        float retention = use_drn ? gate
            : retention_floor + (1.f - retention_floor) * gate;
        float beta = stable_sigmoid(forget_load(projected, projected_bf16,
            beta_offset + value_index, bfloat16));
        float write = (use_v3 || use_drn) ? beta
            : (1.f - retention) * beta;
        float value = tanhf(forget_load(projected, projected_bf16,
            value_offset + value_index, bfloat16));
        float key_squared_norm = use_drn ? 1e-8f : 1e-6f;
        if (use_v3 || use_drn) {
            for (int key = 0; key < key_width; ++key) {
                float k = tanhf(forget_load(projected, projected_bf16,
                    key_offset + key, bfloat16));
                key_squared_norm += k * k;
            }
        }
        float key_norm = (use_v3 || use_drn)
            ? sqrtf(key_squared_norm) : sqrtf(static_cast<float>(key_width));
        float key_scale = 1.f / key_norm;
        float predicted = 0.f;
        float state_gradient_dot_key = 0.f;
        float retention_gradient = 0.f;
        for (int key = 0; key < key_width; ++key) {
            float previous = time == 0 ? 0.f
                : states[previous_state + row + key];
            float gradient = state_gradient[gradient_row + key];
            float key_value = tanhf(forget_load(projected, projected_bf16,
                key_offset + key, bfloat16)) * key_scale;
            predicted += previous * key_value;
            state_gradient_dot_key += gradient * key_value;
            retention_gradient += gradient * previous;
        }
        float retained_prediction = use_v3 ? retention * predicted : predicted;
        float error = value - retained_prediction;
        float write_gradient = error * state_gradient_dot_key;
        float error_gradient = write * state_gradient_dot_key;
        if (use_v3)
            retention_gradient -= error_gradient * predicted;
        else if (!use_drn)
            retention_gradient -= write_gradient * beta;
        projected_gradient[value_offset + value_index] +=
            error_gradient * (1.f - value * value);
        projected_gradient[gate_offset + value_index] += retention_gradient
            * (use_drn ? 1.f : 1.f - retention_floor)
            * gate * (1.f - gate);
        projected_gradient[beta_offset + value_index] += write_gradient
            * ((use_v3 || use_drn) ? 1.f : 1.f - retention)
            * beta * (1.f - beta);
        float key_dot_gradient = 0.f;
        if (use_v3 || use_drn) {
            for (int key = 0; key < key_width; ++key) {
                float previous = time == 0 ? 0.f
                    : states[previous_state + row + key];
                float gradient = state_gradient[gradient_row + key];
                float key_gradient = gradient * write * error
                    - previous * error_gradient * (use_v3 ? retention : 1.f);
                key_dot_gradient += tanhf(forget_load(projected,
                    projected_bf16, key_offset + key, bfloat16)) * key_gradient;
            }
        }
        for (int key = 0; key < key_width; ++key) {
            float previous = time == 0 ? 0.f
                : states[previous_state + row + key];
            float gradient = state_gradient[gradient_row + key];
            float key_tanh = tanhf(forget_load(projected, projected_bf16,
                key_offset + key, bfloat16));
            float key_value = key_tanh * key_scale;
            float key_gradient = gradient * write * error
                - previous * error_gradient * (use_v3 ? retention : 1.f);
            float tanh_gradient = (use_v3 || use_drn)
                ? key_gradient * key_scale
                    - key_tanh * key_dot_gradient * key_scale * key_scale
                        * key_scale
                : key_gradient * key_scale;
            atomicAdd(projected_gradient + key_offset + key,
                tanh_gradient * (1.f - key_tanh * key_tanh));
            float recurrent = use_v3
                ? retention * (gradient - key_value * error_gradient)
                : gradient * retention - key_value * error_gradient;
            if (use_drn)
                previous_gradient[gradient_row + key] += recurrent;
            else
                previous_gradient[gradient_row + key] = recurrent;
        }
        for (int key = 0; key < key_width; ++key)
            state_gradient[gradient_row + key] =
                previous_gradient[gradient_row + key];
    }
}

template <typename T>
__global__ void cross_entropy_stats_kernel(const T* logits,
    const int* labels, float* maxima, float* inverse_sums, float* row_losses,
    int rows, int columns, int ignore_index, int valid_rows,
    float smoothing) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows)
        return;

    // One block owns a complete vocabulary row. Reducing first inside each
    // warp cuts the full-block barriers from eighteen to three at the
    // production 256-thread launch shape. Accumulation stays FP32.
    constexpr int kWarpSize = 32;
    constexpr int kWarps = kThreads / kWarpSize;
    __shared__ float warp_maxima[kWarps];
    __shared__ float warp_sums[kWarps];
    __shared__ float block_maximum;
    __shared__ float block_logit_sum;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    const int offset = row * columns;
    float maximum = -FLT_MAX;
    float logit_sum = 0.f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = load(logits, offset + column);
        maximum = fmaxf(maximum, value);
        logit_sum += value;
    }

    for (int delta = kWarpSize / 2; delta > 0; delta >>= 1) {
        maximum = fmaxf(
            maximum,
            __shfl_down_sync(0xffffffffu, maximum, delta));
        logit_sum += __shfl_down_sync(0xffffffffu, logit_sum, delta);
    }
    if (lane == 0) {
        warp_maxima[warp] = maximum;
        warp_sums[warp] = logit_sum;
    }
    __syncthreads();

    if (warp == 0) {
        maximum = lane < kWarps ? warp_maxima[lane] : -FLT_MAX;
        logit_sum = lane < kWarps ? warp_sums[lane] : 0.f;
        for (int delta = kWarpSize / 2; delta > 0; delta >>= 1) {
            maximum = fmaxf(
                maximum,
                __shfl_down_sync(0xffffffffu, maximum, delta));
            logit_sum += __shfl_down_sync(
                0xffffffffu, logit_sum, delta);
        }
        if (lane == 0) {
            block_maximum = maximum;
            block_logit_sum = logit_sum;
        }
    }
    __syncthreads();

    maximum = block_maximum;
    float exponential_sum = 0.f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        // CUDA's approximate exponential is accurate well beyond the BF16
        // operand quantum while providing the SFU throughput required by a
        // full-vocabulary softmax. Forward and backward deliberately use the
        // same intrinsic so the normalized gradient is self-consistent.
        exponential_sum +=
            __expf(load(logits, offset + column) - maximum);
    }
    for (int delta = kWarpSize / 2; delta > 0; delta >>= 1) {
        exponential_sum += __shfl_down_sync(
            0xffffffffu, exponential_sum, delta);
    }
    if (lane == 0)
        warp_sums[warp] = exponential_sum;
    __syncthreads();

    if (warp == 0) {
        exponential_sum = lane < kWarps ? warp_sums[lane] : 0.f;
        for (int delta = kWarpSize / 2; delta > 0; delta >>= 1) {
            exponential_sum += __shfl_down_sync(
                0xffffffffu, exponential_sum, delta);
        }
        if (lane == 0) {
            maxima[row] = maximum;
            inverse_sums[row] = 1.f / exponential_sum;
            const int label = labels[row];
            if (label == ignore_index) {
                row_losses[row] = 0.f;
                return;
            }
            const float normalizer = maximum + logf(exponential_sum);
            const float nll = normalizer - load(logits, offset + label);
            const float uniform = normalizer - block_logit_sum / columns;
            row_losses[row] =
                ((1.f - smoothing) * nll + smoothing * uniform)
                / valid_rows;
        }
    }
}

__global__ void reduce_loss_kernel(const float* row_losses, float* loss,
    int rows) {
    __shared__ float values[kThreads];
    float sum = 0.f;
    for (int row = threadIdx.x; row < rows; row += blockDim.x)
        sum += row_losses[row];
    values[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            values[threadIdx.x] += values[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        loss[0] = values[0];
}

template <typename T>
__global__ void cross_entropy_backward_rows_kernel(const T* logits,
    const float* maxima, const float* inverse_sums, const int* labels,
    float* gradient, const float* upstream, int rows, int columns,
    int ignore_index, int valid_rows, float smoothing) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows)
        return;
    const int label = labels[row];
    if (label == ignore_index)
        return;
    const int offset = row * columns;
    const float maximum = maxima[row];
    const float inverse_sum = inverse_sums[row];
    const float uniform_target = smoothing / columns;
    const float true_target = 1.f - smoothing;
    const float scale = upstream[0] / valid_rows;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int linear = offset + column;
        const float probability =
            __expf(load(logits, linear) - maximum) * inverse_sum;
        const float target = uniform_target
            + (column == label ? true_target : 0.f);
        gradient[linear] += scale * (probability - target);
    }
}

template <typename T>
__global__ void cross_entropy_backward_bf16_output_rows_kernel(
    const T* logits,
    const float* maxima, const float* inverse_sums, const int* labels,
    unsigned short* gradient, const float* upstream, int rows, int columns,
    int ignore_index, int valid_rows, float smoothing) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows)
        return;
    const int label = labels[row];
    const int offset = row * columns;
    if (label == ignore_index) {
        for (int column = threadIdx.x;
             column < columns;
             column += blockDim.x) {
            bf16_store(gradient, offset + column, 0.f);
        }
        return;
    }
    const float maximum = maxima[row];
    const float inverse_sum = inverse_sums[row];
    const float uniform_target = smoothing / columns;
    const float true_target = 1.f - smoothing;
    const float scale = upstream[0] / valid_rows;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int linear = offset + column;
        const float probability =
            __expf(load(logits, linear) - maximum) * inverse_sum;
        const float target = uniform_target
            + (column == label ? true_target : 0.f);
        bf16_store(gradient, linear, scale * (probability - target));
    }
}

__global__ void decode_bf16_kernel(const unsigned short* source,
    float* destination, int length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length)
        destination[index] = bf16_load(source, index);
}
}

#define NNTRAIN_LAUNCH_1D(kernel, length, ...) \
    kernel<<<blocks_for(length), kThreads, 0, g_stream>>>(__VA_ARGS__); \
    return launch_status()

NNTRAIN_EXPORT int nntrain_cuda_use_external_stream(void* stream) {
    g_stream = reinterpret_cast<cudaStream_t>(stream);
    return static_cast<int>(cudaSuccess);
}

NNTRAIN_EXPORT int nntrain_tensor_add_float(const float* left,
    const float* right, float* output, int length) {
    NNTRAIN_LAUNCH_1D(add_forward_kernel<float>, length,
        left, right, output, length);
}

NNTRAIN_EXPORT int nntrain_tensor_add_bf16(const unsigned short* left,
    const unsigned short* right, unsigned short* output, int length) {
    NNTRAIN_LAUNCH_1D(add_forward_kernel<unsigned short>, length,
        left, right, output, length);
}

NNTRAIN_EXPORT int nntrain_tensor_add_backward(const float* output_gradient,
    float* left_gradient, float* right_gradient, int length,
    int same_parent) {
    NNTRAIN_LAUNCH_1D(add_backward_kernel, length, output_gradient,
        left_gradient, right_gradient, length, same_parent);
}

NNTRAIN_EXPORT int nntrain_tensor_topk_float(
    const float* values,
    int offset,
    int count,
    int k,
    TensorTopKCandidate* workspace,
    int reduction_blocks,
    TensorTopKCandidate* output) {
    return launch_tensor_topk(
        values,
        offset,
        count,
        k,
        workspace,
        reduction_blocks,
        output);
}

NNTRAIN_EXPORT int nntrain_tensor_topk_bf16(
    const unsigned short* values,
    int offset,
    int count,
    int k,
    TensorTopKCandidate* workspace,
    int reduction_blocks,
    TensorTopKCandidate* output) {
    return launch_tensor_topk(
        values,
        offset,
        count,
        k,
        workspace,
        reduction_blocks,
        output);
}

NNTRAIN_EXPORT int nntrain_tensor_embedding_float(const float* table,
    const int* indices, float* output, int length, int width) {
    NNTRAIN_LAUNCH_1D(embedding_forward_kernel<float>, length,
        table, indices, output, length, width);
}

NNTRAIN_EXPORT int nntrain_tensor_embedding_bf16(
    const unsigned short* table, const int* indices, unsigned short* output,
    int length, int width) {
    NNTRAIN_LAUNCH_1D(embedding_forward_kernel<unsigned short>, length,
        table, indices, output, length, width);
}

NNTRAIN_EXPORT int nntrain_tensor_embedding_backward(const int* indices,
    const float* output_gradient, float* table_gradient, int length,
    int width) {
    NNTRAIN_LAUNCH_1D(embedding_backward_kernel, length, indices,
        output_gradient, table_gradient, length, width);
}

NNTRAIN_EXPORT int nntrain_tensor_embedding_backward_reduced(
    const int* indices, const float* output_gradient, float* table_gradient,
    int* workspace, int workspace_ints, int length, int width) {
    return launch_embedding_reduced_backward(
        indices,
        output_gradient,
        table_gradient,
        nullptr,
        workspace,
        workspace_ints,
        length,
        0,
        width);
}

NNTRAIN_EXPORT int nntrain_tensor_embedding_positions_float(
    const float* tokens, const float* positions, const int* indices,
    float* output, int length, int sequence, int width) {
    NNTRAIN_LAUNCH_1D(embedding_positions_forward_kernel<float>, length,
        tokens, positions, indices, output, length, sequence, width);
}

NNTRAIN_EXPORT int nntrain_tensor_embedding_positions_bf16(
    const unsigned short* tokens, const unsigned short* positions,
    const int* indices, unsigned short* output, int length, int sequence,
    int width) {
    NNTRAIN_LAUNCH_1D(embedding_positions_forward_kernel<unsigned short>,
        length, tokens, positions, indices, output, length, sequence, width);
}

NNTRAIN_EXPORT int nntrain_tensor_embedding_positions_backward(
    const int* indices, const float* output_gradient, float* token_gradient,
    float* position_gradient, int length, int sequence, int width) {
    NNTRAIN_LAUNCH_1D(embedding_positions_backward_kernel, length, indices,
        output_gradient, token_gradient, position_gradient, length, sequence,
        width);
}

NNTRAIN_EXPORT int nntrain_tensor_embedding_positions_backward_reduced(
    const int* indices, const float* output_gradient, float* token_gradient,
    float* position_gradient, int* workspace, int workspace_ints, int length,
    int sequence, int width) {
    return launch_embedding_reduced_backward(
        indices,
        output_gradient,
        token_gradient,
        position_gradient,
        workspace,
        workspace_ints,
        length,
        sequence,
        width);
}

NNTRAIN_EXPORT int nntrain_tensor_dropout_float(const float* input,
    float* output, int length, unsigned int seed, unsigned int threshold,
    float scale) {
    NNTRAIN_LAUNCH_1D(dropout_forward_kernel<float>, length, input, output,
        length, seed, threshold, scale);
}

NNTRAIN_EXPORT int nntrain_tensor_dropout_bf16(const unsigned short* input,
    unsigned short* output, int length, unsigned int seed,
    unsigned int threshold, float scale) {
    NNTRAIN_LAUNCH_1D(dropout_forward_kernel<unsigned short>, length, input,
        output, length, seed, threshold, scale);
}

NNTRAIN_EXPORT int nntrain_tensor_dropout_backward(
    const float* output_gradient, float* input_gradient, int length,
    unsigned int seed, unsigned int threshold, float scale) {
    NNTRAIN_LAUNCH_1D(dropout_backward_kernel, length, output_gradient,
        input_gradient, length, seed, threshold, scale);
}

NNTRAIN_EXPORT int nntrain_tensor_add_dropout_float(const float* residual,
    const float* branch, float* output, int length, unsigned int seed,
    unsigned int threshold, float scale) {
    NNTRAIN_LAUNCH_1D(add_dropout_forward_kernel<float>, length, residual,
        branch, output, length, seed, threshold, scale);
}

NNTRAIN_EXPORT int nntrain_tensor_add_dropout_bf16(
    const unsigned short* residual, const unsigned short* branch,
    unsigned short* output, int length, unsigned int seed,
    unsigned int threshold, float scale) {
    NNTRAIN_LAUNCH_1D(add_dropout_forward_kernel<unsigned short>, length,
        residual, branch, output, length, seed, threshold, scale);
}

NNTRAIN_EXPORT int nntrain_tensor_add_dropout_backward(
    const float* output_gradient, float* residual_gradient,
    float* branch_gradient, int length, int same_parent, unsigned int seed,
    unsigned int threshold, float scale) {
    NNTRAIN_LAUNCH_1D(add_dropout_backward_kernel, length, output_gradient,
        residual_gradient, branch_gradient, length, same_parent, seed,
        threshold, scale);
}

NNTRAIN_EXPORT int nntrain_tensor_linear_bias_float(float* output,
    const float* bias, int length, int width, int relu) {
    NNTRAIN_LAUNCH_1D(linear_bias_kernel<float>, length, output, bias, length,
        width, relu);
}

NNTRAIN_EXPORT int nntrain_tensor_linear_bias_bf16(unsigned short* output,
    const unsigned short* bias, int length, int width, int relu) {
    NNTRAIN_LAUNCH_1D(linear_bias_kernel<unsigned short>, length, output, bias,
        length, width, relu);
}

NNTRAIN_EXPORT int nntrain_tensor_linear_mask_float(const float* output,
    float* output_gradient, int length, int relu) {
    NNTRAIN_LAUNCH_1D(linear_mask_kernel, length, output, output_gradient,
        length, relu);
}

NNTRAIN_EXPORT int nntrain_tensor_linear_encode_bf16(
    const float* output_gradient, const unsigned short* output,
    unsigned short* encoded, int length, int relu) {
    NNTRAIN_LAUNCH_1D(linear_encode_bf16_kernel, length, output_gradient,
        output, encoded, length, relu);
}

NNTRAIN_EXPORT int nntrain_tensor_linear_encode_bfp8_relu(
    const float* output_gradient,
    const signed char* output_payload,
    unsigned short* encoded,
    int length) {
    NNTRAIN_LAUNCH_1D(
        linear_encode_bfp8_relu_kernel,
        length,
        output_gradient,
        output_payload,
        encoded,
        length);
}

NNTRAIN_EXPORT int nntrain_tensor_linear_mask_bf16_gradient(
    const unsigned short* output_gradient,
    const unsigned short* output,
    unsigned short* masked,
    int length) {
    NNTRAIN_LAUNCH_1D(
        linear_mask_bf16_gradient_kernel,
        length,
        output_gradient,
        output,
        masked,
        length);
}

NNTRAIN_EXPORT int nntrain_tensor_linear_bias_backward_float(
    const float* output_gradient, float* bias_gradient, int rows, int width) {
    NNTRAIN_LAUNCH_1D(linear_bias_backward_kernel<float>, width,
        output_gradient, bias_gradient, rows, width);
}

NNTRAIN_EXPORT int nntrain_tensor_linear_bias_backward_bf16(
    const unsigned short* output_gradient, float* bias_gradient, int rows,
    int width) {
    NNTRAIN_LAUNCH_1D(linear_bias_backward_kernel<unsigned short>, width,
        output_gradient, bias_gradient, rows, width);
}

NNTRAIN_EXPORT int nntrain_tensor_scale(float* values, int length,
    float scale) {
    NNTRAIN_LAUNCH_1D(scale_kernel, length, values, length, scale);
}

NNTRAIN_EXPORT int nntrain_tensor_accumulate_scalar(
    float* destination,
    float value,
    int accumulate) {
    scalar_gradient_seed_kernel<<<1, 1, 0, g_stream>>>(
        destination,
        value,
        accumulate);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_tensor_accumulate(const float* source,
    float* destination, int length, int source_offset,
    int destination_offset) {
    NNTRAIN_LAUNCH_1D(accumulate_kernel, length, source, destination, length,
        source_offset, destination_offset);
}

NNTRAIN_EXPORT int nntrain_tensor_copy(const float* source,
    float* destination, int length, int source_offset,
    int destination_offset) {
    NNTRAIN_LAUNCH_1D(copy_kernel, length, source, destination, length,
        source_offset, destination_offset);
}

NNTRAIN_EXPORT int nntrain_tensor_encode_bf16(const float* source,
    unsigned short* destination, int length) {
    NNTRAIN_LAUNCH_1D(encode_bf16_kernel, length, source, destination, length);
}

NNTRAIN_EXPORT int nntrain_tensor_softmax_probabilities(
    const float* logits, const float* maxima, const float* inverse_sums,
    float* probabilities, int length, int columns) {
    NNTRAIN_LAUNCH_1D(softmax_probabilities_kernel, length, logits, maxima,
        inverse_sums, probabilities, length, columns);
}

NNTRAIN_EXPORT int nntrain_tensor_cross_entropy_probabilities_backward(
    const float* probabilities, const int* labels, float* gradient,
    int length, int columns, int ignore_index, int valid_rows,
    float smoothing, float upstream) {
    NNTRAIN_LAUNCH_1D(cross_entropy_probabilities_backward_kernel, length,
        probabilities, labels, gradient, length, columns, ignore_index,
        valid_rows, smoothing, upstream);
}

NNTRAIN_EXPORT int nntrain_tensor_squared_sum(const float* values,
    int length, double* result) {
    NNTRAIN_LAUNCH_1D(squared_sum_kernel, length, values, length, result);
}

NNTRAIN_EXPORT int nntrain_optimizer_adamw(float* data,
    const float* gradient, float* first_moment, float* second_moment,
    int length, float beta1, float beta2, float learning_rate,
    float weight_decay, float update_scale, float scaled_epsilon,
    int apply_weight_decay) {
    NNTRAIN_LAUNCH_1D(adamw_kernel, length, data, gradient, first_moment,
        second_moment, length, beta1, beta2, learning_rate, weight_decay,
        update_scale, scaled_epsilon, apply_weight_decay, nullptr, 0);
}

NNTRAIN_EXPORT int nntrain_optimizer_adamw_bfp8_moments(
    const float* gradient,
    float* first_moment,
    float* second_moment,
    int length,
    float beta1,
    float beta2,
    int* finite_status) {
    if (!gradient || !first_moment || !second_moment || length <= 0
        || !finite_status) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    NNTRAIN_LAUNCH_1D(adamw_bfp8_moments_kernel, length, gradient,
        first_moment, second_moment, length, beta1, beta2, finite_status);
}

NNTRAIN_EXPORT int nntrain_optimizer_adamw_bfp8_apply(
    float* data,
    const float* first_moment,
    const float* second_moment,
    const float* second_scale,
    int second_scale_block_size,
    int length,
    float learning_rate,
    float weight_decay,
    float update_scale,
    float scaled_epsilon,
    int apply_weight_decay,
    int* finite_status) {
    if (!data || !first_moment || !second_moment || !second_scale
        || second_scale_block_size <= 0 || length <= 0 || !finite_status) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    NNTRAIN_LAUNCH_1D(adamw_bfp8_apply_kernel, length, data, first_moment,
        second_moment, second_scale, second_scale_block_size, length,
        learning_rate, weight_decay,
        update_scale, scaled_epsilon, apply_weight_decay, finite_status);
}

NNTRAIN_EXPORT int nntrain_optimizer_adamw_publish(float* data,
    const float* gradient, float* first_moment, float* second_moment,
    void* compute, int physical_bf16, int length, float beta1, float beta2,
    float learning_rate, float weight_decay, float update_scale,
    float scaled_epsilon, int apply_weight_decay) {
    NNTRAIN_LAUNCH_1D(adamw_kernel, length, data, gradient, first_moment,
        second_moment, length, beta1, beta2, learning_rate, weight_decay,
        update_scale, scaled_epsilon, apply_weight_decay, compute,
        physical_bf16);
}

NNTRAIN_EXPORT int nntrain_optimizer_adamw_bf16_state(float* data,
    const float* gradient, unsigned short* first_moment,
    unsigned short* second_moment, int length, float beta1, float beta2,
    float learning_rate, float weight_decay, float update_scale,
    float scaled_epsilon, int apply_weight_decay) {
    NNTRAIN_LAUNCH_1D(adamw_bf16_state_kernel, length, data, gradient,
        first_moment, second_moment, length, beta1, beta2, learning_rate,
        weight_decay, update_scale, scaled_epsilon, apply_weight_decay,
        nullptr, 0);
}

NNTRAIN_EXPORT int nntrain_optimizer_adamw_bf16_state_publish(float* data,
    const float* gradient, unsigned short* first_moment,
    unsigned short* second_moment, void* compute, int physical_bf16,
    int length, float beta1, float beta2, float learning_rate,
    float weight_decay, float update_scale, float scaled_epsilon,
    int apply_weight_decay) {
    NNTRAIN_LAUNCH_1D(adamw_bf16_state_kernel, length, data, gradient,
        first_moment, second_moment, length, beta1, beta2, learning_rate,
        weight_decay, update_scale, scaled_epsilon, apply_weight_decay,
        compute, physical_bf16);
}

NNTRAIN_EXPORT int nntrain_optimizer_adamw_pure_bf16(
    unsigned short* data, const unsigned short* gradient,
    unsigned short* first_moment, unsigned short* second_moment, int length,
    float beta1, float beta2, float learning_rate, float weight_decay,
    float update_scale, float scaled_epsilon, int apply_weight_decay) {
    if (!data || !gradient || !first_moment || !second_moment
        || length <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    NNTRAIN_LAUNCH_1D(adamw_pure_bf16_kernel, length,
        reinterpret_cast<__nv_bfloat16*>(data),
        reinterpret_cast<const __nv_bfloat16*>(gradient),
        reinterpret_cast<__nv_bfloat16*>(first_moment),
        reinterpret_cast<__nv_bfloat16*>(second_moment), length, beta1,
        beta2, learning_rate, weight_decay, update_scale, scaled_epsilon,
        apply_weight_decay);
}

NNTRAIN_EXPORT int nntrain_optimizer_adamw_multi_tensor(
    const AdamWChunkDescriptor* chunks,
    int chunk_count,
    float beta1,
    float beta2,
    float learning_rate,
    float weight_decay,
    float update_scale,
    float scaled_epsilon) {
    if (!chunks || chunk_count <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    adamw_multi_tensor_kernel<<<chunk_count, kThreads, 0, g_stream>>>(
        chunks, chunk_count, beta1, beta2, learning_rate, weight_decay,
        update_scale, scaled_epsilon);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_optimizer_accumulate_finite_status(
    const float* values, int length, int* finite_status) {
    if (!values || length <= 0 || !finite_status)
        return static_cast<int>(cudaErrorInvalidValue);
    optimizer_accumulate_finite_status_kernel<<<
        blocks_for(length), kThreads, 0, g_stream>>>(
            values, length, finite_status);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_optimizer_publish_bf16(const float* master,
    void* compute, int length, int physical) {
    NNTRAIN_LAUNCH_1D(publish_bf16_kernel, length, master, compute, length,
        physical);
}

NNTRAIN_EXPORT int nntrain_optimizer_gather_stats(
    const float* const* sources,
    float* destination,
    int count) {
    NNTRAIN_LAUNCH_1D(gather_optimizer_stats_kernel, count * 4,
        sources, destination, count);
}

NNTRAIN_EXPORT int nntrain_optimizer_neko_moments(const float* gradient,
    float* fast, float* slow, float* fast_hat, float* slow_hat, int length,
    float beta_fast, float beta_slow, float fast_correction,
    float slow_correction) {
    NNTRAIN_LAUNCH_1D(neko_moments_kernel, length, gradient, fast, slow,
        fast_hat, slow_hat, length, beta_fast, beta_slow, fast_correction,
        slow_correction);
}

NNTRAIN_EXPORT int nntrain_optimizer_neko_initialize(const float* source,
    float* destination, int length, int original_rows, int original_columns,
    int transpose, float inverse_norm) {
    NNTRAIN_LAUNCH_1D(neko_initialize_kernel, length, source, destination,
        length, original_rows, original_columns, transpose, inverse_norm);
}

NNTRAIN_EXPORT int nntrain_optimizer_neko_initialize_corrected(
    const float* source, float* destination, int length,
    int original_rows, int original_columns, int transpose,
    float inverse_fast_correction, float inverse_norm) {
    if (!(inverse_fast_correction > 0.f))
        return static_cast<int>(cudaErrorInvalidValue);
    NNTRAIN_LAUNCH_1D(neko_initialize_corrected_kernel, length,
        source, destination, length, original_rows, original_columns,
        transpose, inverse_fast_correction, inverse_norm);
}

NNTRAIN_EXPORT int nntrain_optimizer_neko_initialize_bf16_corrected(
    const unsigned short* source, float* destination, int length,
    int original_rows, int original_columns, int transpose,
    float inverse_fast_correction, float inverse_norm) {
    if (!source || !destination || length <= 0 || original_rows <= 0
        || original_columns <= 0 || !(inverse_fast_correction > 0.f)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    NNTRAIN_LAUNCH_1D(neko_initialize_bf16_corrected_kernel, length,
        reinterpret_cast<const __nv_bfloat16*>(source), destination, length,
        original_rows, original_columns, transpose,
        inverse_fast_correction, inverse_norm);
}

NNTRAIN_EXPORT int nntrain_optimizer_neko_update_device_control(
    const float* stats, float* confidence, int* finite_status,
    float epsilon, float rho) {
    if (!stats || !confidence || !finite_status || !(epsilon > 0.f)
        || rho < 0.f || rho > 1.f) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    neko_update_device_control_kernel<<<1, 1, 0, g_stream>>>(
        stats, confidence, finite_status, epsilon, rho);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_optimizer_neko_initialize_device_stats(
    const float* source, float* destination, int length,
    int original_rows, int original_columns, int transpose,
    float inverse_fast_correction, const float* stats, float epsilon,
    int* finite_status) {
    if (!source || !destination || !stats || !finite_status || length <= 0
        || original_rows <= 0 || original_columns <= 0
        || !(inverse_fast_correction > 0.f) || !(epsilon > 0.f)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    NNTRAIN_LAUNCH_1D(neko_initialize_device_stats_kernel, length,
        source, destination, length, original_rows, original_columns,
        transpose, inverse_fast_correction, stats, epsilon, finite_status);
}

NNTRAIN_EXPORT int nntrain_optimizer_neko_initialize_bf16_device_stats(
    const unsigned short* source, float* destination, int length,
    int original_rows, int original_columns, int transpose,
    float inverse_fast_correction, const float* stats, float epsilon,
    int* finite_status) {
    if (!source || !destination || !stats || !finite_status || length <= 0
        || original_rows <= 0 || original_columns <= 0
        || !(inverse_fast_correction > 0.f) || !(epsilon > 0.f)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    NNTRAIN_LAUNCH_1D(neko_initialize_bf16_device_stats_kernel, length,
        reinterpret_cast<const __nv_bfloat16*>(source), destination, length,
        original_rows, original_columns, transpose,
        inverse_fast_correction, stats, epsilon, finite_status);
}

NNTRAIN_EXPORT int nntrain_optimizer_neko_interpolate(float* current,
    const float* next, int length, float fraction) {
    NNTRAIN_LAUNCH_1D(neko_interpolate_kernel, length, current, next, length,
        fraction);
}

NNTRAIN_EXPORT int nntrain_optimizer_neko_transpose_back(
    const float* source, float* destination, int length, int original_rows,
    int original_columns) {
    NNTRAIN_LAUNCH_1D(neko_transpose_back_kernel, length, source, destination,
        length, original_rows, original_columns);
}

NNTRAIN_EXPORT int nntrain_optimizer_neko_apply(float* data,
    const float* update, int length, float learning_rate, float final_scale,
    float weight_decay, int apply_weight_decay) {
    NNTRAIN_LAUNCH_1D(neko_apply_kernel, length, data, update, length,
        learning_rate, final_scale, weight_decay, apply_weight_decay);
}

NNTRAIN_EXPORT int nntrain_optimizer_neko_apply_bf16(unsigned short* data,
    const float* update, int length, float learning_rate, float final_scale,
    float weight_decay, int apply_weight_decay) {
    if (!data || !update || length <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    NNTRAIN_LAUNCH_1D(neko_apply_bf16_kernel, length,
        reinterpret_cast<__nv_bfloat16*>(data), update, length,
        learning_rate, final_scale, weight_decay, apply_weight_decay);
}

NNTRAIN_EXPORT int nntrain_optimizer_neko_combine(const float* gram,
    float* gram_squared, int length, int rows, float a, float b, float c) {
    NNTRAIN_LAUNCH_1D(neko_combine_kernel, length, gram, gram_squared, length,
        rows, a, b, c);
}

NNTRAIN_EXPORT int nntrain_optimizer_neko_combine_batched(
    const float* gram, float* gram_squared, int matrix_length,
    int batch, int rows, float a, float b, float c) {
    if (!gram || !gram_squared || matrix_length <= 0 || batch <= 0
        || rows <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    int total_length = matrix_length * batch;
    NNTRAIN_LAUNCH_1D(
        neko_combine_batched_kernel,
        total_length,
        gram,
        gram_squared,
        matrix_length,
        total_length,
        rows,
        a,
        b,
        c);
}

NNTRAIN_EXPORT int nntrain_optimizer_symmetric_gram(const float* source,
    float* destination, int rows, int columns) {
    int length = rows * rows;
    NNTRAIN_LAUNCH_1D(symmetric_gram_kernel<false>, length, source, destination,
        rows, columns);
}

NNTRAIN_EXPORT int nntrain_optimizer_symmetric_gram_bf16_operands(
    const float* source, float* destination, int rows, int columns) {
    int length = rows * rows;
    NNTRAIN_LAUNCH_1D(symmetric_gram_kernel<true>, length, source, destination,
        rows, columns);
}

NNTRAIN_EXPORT int nntrain_optimizer_newton_schulz(const float* source,
    const float* gram, const float* gram_squared, float* destination,
    int rows, int columns, float a, float b, float c) {
    int length = rows * columns;
    NNTRAIN_LAUNCH_1D(newton_schulz_kernel<false>, length, source, gram,
        gram_squared, destination, rows, columns, a, b, c);
}

NNTRAIN_EXPORT int nntrain_optimizer_newton_schulz_bf16_operands(
    const float* source, const float* gram, const float* gram_squared,
    float* destination, int rows, int columns, float a, float b, float c) {
    int length = rows * columns;
    NNTRAIN_LAUNCH_1D(newton_schulz_kernel<true>, length, source, gram,
        gram_squared, destination, rows, columns, a, b, c);
}

NNTRAIN_EXPORT int nntrain_forget_forward(const float* projected,
    const unsigned short* projected_bf16, float* output,
    unsigned short* output_bf16, float* states, float* state, int batch,
    int sequence, int projection_width, int key_width, int value_width,
    float retention_floor, int memory_variant, int bfloat16) {
    int workers = batch * value_width;
    NNTRAIN_LAUNCH_1D(forget_forward_kernel, workers, projected,
        projected_bf16, output, output_bf16, states, state, workers, sequence,
        projection_width, key_width, value_width, retention_floor,
        memory_variant, bfloat16);
}

NNTRAIN_EXPORT int nntrain_forget_backward(const float* projected,
    const unsigned short* projected_bf16, float* projected_gradient,
    const float* output_gradient, const float* states, float* state_gradient,
    float* previous_gradient, int batch, int sequence, int projection_width,
    int key_width, int value_width, float retention_floor,
    int memory_variant, int bfloat16) {
    int workers = batch * value_width;
    NNTRAIN_LAUNCH_1D(forget_backward_kernel, workers, projected,
        projected_bf16, projected_gradient, output_gradient, states,
        state_gradient, previous_gradient, workers, sequence,
        projection_width, key_width, value_width, retention_floor,
        memory_variant, bfloat16);
}

NNTRAIN_EXPORT int nntrain_tensor_cross_entropy_float(const float* logits,
    const int* labels, float* maxima, float* inverse_sums,
    float* row_losses, float* loss,
    int rows, int columns, int ignore_index, int valid_rows,
    float smoothing) {
    cross_entropy_stats_kernel<float><<<rows, kThreads, 0, g_stream>>>(logits, labels,
        maxima, inverse_sums, row_losses, rows, columns, ignore_index, valid_rows,
        smoothing);
    reduce_loss_kernel<<<1, kThreads, 0, g_stream>>>(row_losses, loss, rows);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_tensor_cross_entropy_bf16(
    const unsigned short* logits, const int* labels, float* maxima,
    float* inverse_sums, float* row_losses, float* loss,
    int rows, int columns,
    int ignore_index, int valid_rows, float smoothing) {
    cross_entropy_stats_kernel<unsigned short><<<rows, kThreads, 0, g_stream>>>(logits,
        labels, maxima, inverse_sums, row_losses, rows, columns, ignore_index,
        valid_rows, smoothing);
    reduce_loss_kernel<<<1, kThreads, 0, g_stream>>>(row_losses, loss, rows);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_tensor_cross_entropy_backward_float(
    const float* logits, const float* maxima, const float* inverse_sums,
    const int* labels, float* gradient, const float* upstream, int length,
    int columns, int ignore_index, int valid_rows, float smoothing) {
    if (columns <= 0 || length <= 0 || length % columns != 0)
        return static_cast<int>(cudaErrorInvalidValue);
    const int rows = length / columns;
    cross_entropy_backward_rows_kernel<float><<<
        rows, kThreads, 0, g_stream>>>(logits, maxima, inverse_sums, labels,
            gradient, upstream, rows, columns, ignore_index, valid_rows,
            smoothing);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_tensor_cross_entropy_backward_bf16(
    const unsigned short* logits, const float* maxima,
    const float* inverse_sums, const int* labels, float* gradient,
    const float* upstream, int length, int columns, int ignore_index,
    int valid_rows, float smoothing) {
    if (columns <= 0 || length <= 0 || length % columns != 0)
        return static_cast<int>(cudaErrorInvalidValue);
    const int rows = length / columns;
    cross_entropy_backward_rows_kernel<unsigned short><<<
        rows, kThreads, 0, g_stream>>>(logits, maxima, inverse_sums, labels,
            gradient, upstream, rows, columns, ignore_index, valid_rows,
            smoothing);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_tensor_cross_entropy_backward_bf16_output(
    const unsigned short* logits, const float* maxima,
    const float* inverse_sums, const int* labels, unsigned short* gradient,
    const float* upstream, int length, int columns, int ignore_index,
    int valid_rows, float smoothing) {
    if (columns <= 0 || length <= 0 || length % columns != 0)
        return static_cast<int>(cudaErrorInvalidValue);
    const int rows = length / columns;
    cross_entropy_backward_bf16_output_rows_kernel<unsigned short><<<
        rows, kThreads, 0, g_stream>>>(logits, maxima, inverse_sums, labels,
            gradient, upstream, rows, columns, ignore_index, valid_rows,
            smoothing);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_tensor_decode_bf16(
    const unsigned short* source, float* destination, int length) {
    NNTRAIN_LAUNCH_1D(
        decode_bf16_kernel, length, source, destination, length);
}

NNTRAIN_EXPORT int nntrain_bfp8_gradient_quantize(
    int device, float* source, signed char* payload, float* scale,
    int length, int* finite_status, cudaStream_t stream) {
    if (device < 0 || !source || !payload || !scale || length <= 0
        || !finite_status) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = static_cast<cudaError_t>(
        nntrain_cuda_set_device(device));
    if (status != cudaSuccess)
        return static_cast<int>(status);
    return launch_bfp8_gradient_quantize(
        source, payload, scale, length, finite_status, stream, nullptr);
}

NNTRAIN_EXPORT int nntrain_bfp8_gradient_quantize_accumulate(
    int device, float* source, signed char* payload, float* scale,
    int length, int* finite_status, double* squared_sum,
    cudaStream_t stream) {
    if (device < 0 || !source || !payload || !scale || length <= 0
        || !finite_status || !squared_sum) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = static_cast<cudaError_t>(
        nntrain_cuda_set_device(device));
    if (status != cudaSuccess)
        return static_cast<int>(status);
    return launch_bfp8_gradient_quantize(
        source, payload, scale, length, finite_status, stream, squared_sum);
}

NNTRAIN_EXPORT int nntrain_bfp8_gradient_squared_sum(
    int device, const signed char* payload, const float* scale, int length,
    double* squared_sum, int* finite_status, cudaStream_t stream) {
    if (device < 0 || !payload || !scale || length <= 0 || !squared_sum
        || !finite_status) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = static_cast<cudaError_t>(
        nntrain_cuda_set_device(device));
    if (status != cudaSuccess)
        return static_cast<int>(status);
    bfp8_gradient_squared_sum_kernel<<<blocks_for(length), kThreads, 0,
        stream>>>(payload, scale, length, squared_sum, finite_status);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_bfp8_gradient_scale(
    int device, float* scale, float multiplier, cudaStream_t stream) {
    if (device < 0 || !scale || !isfinite(multiplier) || !(multiplier > 0.f))
        return static_cast<int>(cudaErrorInvalidValue);
    cudaError_t status = static_cast<cudaError_t>(
        nntrain_cuda_set_device(device));
    if (status != cudaSuccess)
        return static_cast<int>(status);
    bfp8_gradient_scale_kernel<<<1, 1, 0, stream>>>(scale, multiplier);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_bfp8_gradient_reduce(
    int primary_device, int secondary_device,
    const signed char* local_payload, const float* local_scale,
    const signed char* remote_payload, const float* remote_scale,
    signed char* remote_payload_staging, float* remote_scale_staging,
    float* reduced, signed char* output_payload, float* output_scale,
    int length, float reduction_scale, int* finite_status,
    const int* remote_finite_status, int* remote_status_staging,
    double* squared_sum, cudaStream_t communication_stream,
    cudaEvent_t local_ready, cudaEvent_t remote_ready,
    cudaEvent_t reduced_ready) {
    if (primary_device < 0 || secondary_device < 0
        || primary_device == secondary_device || !local_payload
        || !local_scale || !remote_payload || !remote_scale
        || !remote_payload_staging || !remote_scale_staging || !reduced
        || !output_payload || !output_scale || length <= 0
        || !isfinite(reduction_scale) || !finite_status
        || !remote_finite_status || !remote_status_staging || !squared_sum
        || !communication_stream || !local_ready || !remote_ready
        || !reduced_ready) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = static_cast<cudaError_t>(
        nntrain_cuda_set_device(primary_device));
    if (status != cudaSuccess)
        return static_cast<int>(status);
    status = cudaStreamWaitEvent(communication_stream, local_ready, 0);
    if (status != cudaSuccess)
        return static_cast<int>(status);
    status = cudaStreamWaitEvent(communication_stream, remote_ready, 0);
    if (status != cudaSuccess)
        return static_cast<int>(status);
    status = cudaMemcpyPeerAsync(remote_payload_staging, primary_device,
        remote_payload, secondary_device,
        static_cast<size_t>(length) * sizeof(signed char),
        communication_stream);
    if (status != cudaSuccess)
        return static_cast<int>(status);
    status = cudaMemcpyPeerAsync(remote_scale_staging, primary_device,
        remote_scale, secondary_device, sizeof(float),
        communication_stream);
    if (status != cudaSuccess)
        return static_cast<int>(status);
    status = cudaMemcpyPeerAsync(remote_status_staging, primary_device,
        remote_finite_status, secondary_device, sizeof(int),
        communication_stream);
    if (status != cudaSuccess)
        return static_cast<int>(status);
    bfp8_gradient_merge_status_kernel<<<1, 1, 0,
        communication_stream>>>(finite_status, remote_status_staging);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return static_cast<int>(status);
    bfp8_gradient_sum_kernel<<<blocks_for(length), kThreads, 0,
        communication_stream>>>(local_payload, local_scale,
            remote_payload_staging, remote_scale_staging, reduced, length,
            reduction_scale, finite_status);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return static_cast<int>(status);
    int launch = launch_bfp8_gradient_quantize(reduced, output_payload,
        output_scale, length, finite_status, communication_stream,
        squared_sum);
    if (launch != static_cast<int>(cudaSuccess))
        return launch;
    return static_cast<int>(cudaEventRecord(
        reduced_ready, communication_stream));
}

NNTRAIN_EXPORT int nntrain_bfp8_gradient_broadcast(
    int destination_device, int source_device,
    const signed char* source_payload, const float* source_scale,
    signed char* destination_payload, float* destination_scale,
    float* destination_float, int length, int* destination_finite_status,
    cudaStream_t destination_stream, cudaEvent_t source_ready) {
    if (destination_device < 0 || source_device < 0
        || destination_device == source_device || !source_payload
        || !source_scale || !destination_payload || !destination_scale
        || !destination_float || length <= 0 || !destination_finite_status
        || !destination_stream || !source_ready) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = static_cast<cudaError_t>(
        nntrain_cuda_set_device(destination_device));
    if (status != cudaSuccess)
        return static_cast<int>(status);
    status = cudaStreamWaitEvent(destination_stream, source_ready, 0);
    if (status != cudaSuccess)
        return static_cast<int>(status);
    status = cudaMemcpyPeerAsync(destination_payload, destination_device,
        source_payload, source_device,
        static_cast<size_t>(length) * sizeof(signed char),
        destination_stream);
    if (status != cudaSuccess)
        return static_cast<int>(status);
    status = cudaMemcpyPeerAsync(destination_scale, destination_device,
        source_scale, source_device, sizeof(float), destination_stream);
    if (status != cudaSuccess)
        return static_cast<int>(status);
    bfp8_gradient_decode_kernel<<<blocks_for(length), kThreads, 0,
        destination_stream>>>(destination_payload, destination_scale,
            destination_float, length, destination_finite_status);
    return launch_status();
}
