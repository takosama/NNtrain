#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>
#include <new>

#if defined(_WIN32)
#define NNTRAIN_EXPORT extern "C" __declspec(dllexport)
#else
#define NNTRAIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Keep cuda_runtime_bridge.cu's thread-local selected-device cache coherent.
// The host-staged gradient pipeline switches devices on managed thread-pool
// threads; calling cudaSetDevice directly here left that cache stale and a
// later managed launch could consequently use another GPU's pointers.
extern "C" int nntrain_cuda_set_device(int device);

namespace {
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kQueriesPerWarp = 4;
constexpr int kColumnsPerLane = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int kLayerNormThreads = 256;
constexpr int kLayerNormValuesPerThread = 4;
constexpr int kLayerNormCachedColumns =
    kLayerNormThreads * kLayerNormValuesPerThread;
constexpr int kLayerNormParameterColumns = 32;
constexpr int kLayerNormParameterRows = 8;
constexpr int kLayerNormRowsPerTile = 1024;
constexpr int kNekoMuonThreads = 256;
constexpr int kNekoMuonMaxBlocks = 128;
constexpr int kGradientBucketThreads = 256;
constexpr int kAttentionTensorCoreTile = 16;
constexpr int kAttentionTensorCoreMaxHead = 128;
constexpr int kAttentionTensorCoreThreads = 256;

template <bool async_load>
__device__ __forceinline__ void copy_bfloat16_pair(
    __nv_bfloat16* destination,
    const __nv_bfloat16* source,
    int valid_bytes) {
#if __CUDA_ARCH__ >= 800
    if constexpr (async_load) {
        const unsigned int shared_address =
            static_cast<unsigned int>(__cvta_generic_to_shared(destination));
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
            :: "r"(shared_address), "l"(source), "r"(valid_bytes));
    }
    else {
        destination[0] = valid_bytes >= 2
            ? source[0]
            : __float2bfloat16_rn(0.f);
        destination[1] = valid_bytes >= 4
            ? source[1]
            : __float2bfloat16_rn(0.f);
    }
#else
    destination[0] = valid_bytes >= 2
        ? source[0]
        : __float2bfloat16_rn(0.f);
    destination[1] = valid_bytes >= 4
        ? source[1]
        : __float2bfloat16_rn(0.f);
#endif
}

template <bool async_load>
__device__ __forceinline__ void commit_async_copy_group() {
#if __CUDA_ARCH__ >= 800
    if constexpr (async_load)
        asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <bool async_load>
__device__ __forceinline__ void wait_for_async_copy_group() {
#if __CUDA_ARCH__ >= 800
    if constexpr (async_load)
        asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return __shfl_sync(0xffffffffu, value, 0);
}

template <typename T>
__device__ __forceinline__ float load_value(const T* values, int index);

template <>
__device__ __forceinline__ float load_value<float>(
    const float* values, int index) {
    return values[index];
}

template <>
__device__ __forceinline__ float load_value<__nv_bfloat16>(
    const __nv_bfloat16* values, int index) {
    return __bfloat162float(values[index]);
}

template <typename T>
__device__ __forceinline__ void store_value(
    T* values, int index, float value);

template <typename T>
__device__ __forceinline__ float round_to_storage(float value);

template <>
__device__ __forceinline__ void store_value<float>(
    float* values, int index, float value) {
    values[index] = value;
}

template <>
__device__ __forceinline__ void store_value<__nv_bfloat16>(
    __nv_bfloat16* values, int index, float value) {
    values[index] = __float2bfloat16_rn(value);
}

template <typename T>
__device__ __forceinline__ void accumulate_layer_norm_gradient(
    T* values, int index, float value);

template <>
__device__ __forceinline__ void accumulate_layer_norm_gradient<float>(
    float* values, int index, float value) {
    values[index] += value;
}

template <>
__device__ __forceinline__ void
accumulate_layer_norm_gradient<__nv_bfloat16>(
    __nv_bfloat16* values, int index, float value) {
    // Used only when the managed autograd graph proves that this branch has
    // no earlier gradient. It is therefore an initializing store, not an
    // accumulation into BF16 state.
    values[index] = __float2bfloat16_rn(value);
}

template <typename T>
__device__ __forceinline__ void accumulate_attention_gradient(
    T* values, int index, float value);

template <>
__device__ __forceinline__ void accumulate_attention_gradient<float>(
    float* values, int index, float value) {
    values[index] += value;
}

template <>
__device__ __forceinline__ void
accumulate_attention_gradient<__nv_bfloat16>(
    __nv_bfloat16* values, int index, float value) {
    // This specialization is used only for a fresh, single-consumer
    // attention projection gradient, so no prior value must be accumulated.
    values[index] = __float2bfloat16_rn(value);
}

template <>
__device__ __forceinline__ float round_to_storage<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ float round_to_storage<__nv_bfloat16>(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

template <typename T>
__global__ void attention_forward_tiled(
    const T* __restrict__ qkv,
    T* __restrict__ output,
    float* __restrict__ log_sum_exp,
    int sequence,
    int model_width,
    int heads,
    int causal) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int query = blockIdx.x * kWarpsPerBlock + warp;
    const int batch_head = blockIdx.y;
    if (query >= sequence)
        return;

    const int head = batch_head % heads;
    const int batch_index = batch_head / heads;
    const int head_width = model_width / heads;
    const int projected_width = 3 * model_width;
    const int batch_base = batch_index * sequence * projected_width;
    const int head_base = head * head_width;
    const int query_base =
        batch_base + query * projected_width + head_base;
    const int output_base =
        (batch_index * sequence + query) * model_width + head_base;
    const int last_key = causal ? query : sequence - 1;
    const float scale = rsqrtf((float)head_width);

    float accumulator[kColumnsPerLane] = {0.f, 0.f, 0.f, 0.f};
    float maximum = -3.402823466e+38F;
    float denominator = 0.f;
    for (int key = 0; key <= last_key; ++key) {
        const int key_base =
            batch_base + key * projected_width + model_width + head_base;
        float score_partial = 0.f;
        #pragma unroll
        for (int item = 0; item < kColumnsPerLane; ++item) {
            const int column = lane + item * kWarpSize;
            if (column < head_width) {
                score_partial = fmaf(
                    load_value(qkv, query_base + column),
                    load_value(qkv, key_base + column),
                    score_partial);
            }
        }
        const float score = warp_sum(score_partial) * scale;
        const float next_maximum = fmaxf(maximum, score);
        const float old_scale = expf(maximum - next_maximum);
        const float new_scale = expf(score - next_maximum);
        denominator = denominator * old_scale + new_scale;
        const int value_base = key_base + model_width;
        #pragma unroll
        for (int item = 0; item < kColumnsPerLane; ++item) {
            const int column = lane + item * kWarpSize;
            if (column < head_width) {
                accumulator[item] = accumulator[item] * old_scale
                    + new_scale * load_value(qkv, value_base + column);
            }
        }
        maximum = next_maximum;
    }

    const float inverse = 1.f / denominator;
    #pragma unroll
    for (int item = 0; item < kColumnsPerLane; ++item) {
        const int column = lane + item * kWarpSize;
        if (column < head_width) {
            store_value(
                output,
                output_base + column,
                accumulator[item] * inverse);
        }
    }
    if (lane == 0) {
        log_sum_exp[batch_head * sequence + query] =
            maximum + logf(denominator);
    }
}

template <typename T>
__global__ void attention_backward_register_tiled(
    const T* __restrict__ qkv,
    const T* __restrict__ output,
    const float* __restrict__ output_gradient,
    const float* __restrict__ log_sum_exp,
    float* __restrict__ qkv_gradient,
    int sequence,
    int model_width,
    int heads,
    int causal) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int warp_tile =
        blockIdx.x * kWarpsPerBlock + warp;
    const int first_query = warp_tile * kQueriesPerWarp;
    const int batch_head = blockIdx.y;
    if (first_query >= sequence)
        return;

    const int head = batch_head % heads;
    const int batch_index = batch_head / heads;
    const int head_width = model_width / heads;
    const int projected_width = 3 * model_width;
    const int batch_base = batch_index * sequence * projected_width;
    const int head_base = head * head_width;
    const int last_query =
        min(sequence - 1, first_query + kQueriesPerWarp - 1);
    const int last_key = causal ? last_query : sequence - 1;
    const float scale = rsqrtf((float)head_width);

    float row_delta[kQueriesPerWarp] = {};
    float dq[kQueriesPerWarp][kColumnsPerLane] = {};
    #pragma unroll
    for (int query_item = 0;
         query_item < kQueriesPerWarp;
         ++query_item) {
        const int query = first_query + query_item;
        if (query < sequence) {
            const int output_base =
                (batch_index * sequence + query) * model_width + head_base;
            float partial = 0.f;
            #pragma unroll
            for (int item = 0; item < kColumnsPerLane; ++item) {
                const int column = lane + item * kWarpSize;
                if (column < head_width) {
                    partial = fmaf(
                        output_gradient[output_base + column],
                        load_value(output, output_base + column),
                        partial);
                }
            }
            row_delta[query_item] = warp_sum(partial);
        }
    }

    for (int key = 0; key <= last_key; ++key) {
        const int key_base =
            batch_base + key * projected_width + model_width + head_base;
        const int value_base = key_base + model_width;
        float key_gradient[kColumnsPerLane] = {0.f, 0.f, 0.f, 0.f};
        float value_gradient[kColumnsPerLane] = {0.f, 0.f, 0.f, 0.f};

        #pragma unroll
        for (int query_item = 0;
             query_item < kQueriesPerWarp;
             ++query_item) {
            const int query = first_query + query_item;
            if (query >= sequence || (causal && key > query))
                continue;
            const int query_base =
                batch_base + query * projected_width + head_base;
            const int output_base =
                (batch_index * sequence + query) * model_width + head_base;
            float score_partial = 0.f;
            float probability_gradient_partial = 0.f;
            #pragma unroll
            for (int item = 0; item < kColumnsPerLane; ++item) {
                const int column = lane + item * kWarpSize;
                if (column < head_width) {
                    score_partial = fmaf(
                        load_value(qkv, query_base + column),
                        load_value(qkv, key_base + column),
                        score_partial);
                    probability_gradient_partial = fmaf(
                        output_gradient[output_base + column],
                        load_value(qkv, value_base + column),
                        probability_gradient_partial);
                }
            }
            const float probability = expf(
                warp_sum(score_partial) * scale
                - log_sum_exp[batch_head * sequence + query]);
            const float score_gradient = probability
                * (warp_sum(probability_gradient_partial)
                    - row_delta[query_item])
                * scale;
            #pragma unroll
            for (int item = 0; item < kColumnsPerLane; ++item) {
                const int column = lane + item * kWarpSize;
                if (column < head_width) {
                    dq[query_item][item] = fmaf(
                        score_gradient,
                        load_value(qkv, key_base + column),
                        dq[query_item][item]);
                    key_gradient[item] = fmaf(
                        score_gradient,
                        load_value(qkv, query_base + column),
                        key_gradient[item]);
                    value_gradient[item] = fmaf(
                        probability,
                        output_gradient[output_base + column],
                        value_gradient[item]);
                }
            }
        }

        #pragma unroll
        for (int item = 0; item < kColumnsPerLane; ++item) {
            const int column = lane + item * kWarpSize;
            if (column < head_width) {
                atomicAdd(
                    qkv_gradient + key_base + column,
                    key_gradient[item]);
                atomicAdd(
                    qkv_gradient + value_base + column,
                    value_gradient[item]);
            }
        }
    }

    #pragma unroll
    for (int query_item = 0;
         query_item < kQueriesPerWarp;
         ++query_item) {
        const int query = first_query + query_item;
        if (query < sequence) {
            const int query_base =
                batch_base + query * projected_width + head_base;
            #pragma unroll
            for (int item = 0; item < kColumnsPerLane; ++item) {
                const int column = lane + item * kWarpSize;
                if (column < head_width) {
                    qkv_gradient[query_base + column] +=
                        dq[query_item][item];
                }
            }
        }
    }
}

template <typename T>
int launch_forward(
    const T* qkv,
    T* output,
    float* log_sum_exp,
    int batch,
    int sequence,
    int model_width,
    int heads,
    int causal,
    cudaStream_t stream) {
    if (!qkv || !output || !log_sum_exp || batch <= 0 || sequence <= 0
        || heads <= 0 || model_width % heads
        || model_width / heads > kWarpSize * kColumnsPerLane) {
        return (int)cudaErrorInvalidValue;
    }
    dim3 grid(
        (sequence + kWarpsPerBlock - 1) / kWarpsPerBlock,
        batch * heads);
    attention_forward_tiled<<<grid, kThreadsPerBlock, 0, stream>>>(
        qkv, output, log_sum_exp, sequence, model_width, heads, causal);
    return (int)cudaPeekAtLastError();
}

template <typename T>
int launch_backward(
    const T* qkv,
    const T* output,
    const float* output_gradient,
    const float* log_sum_exp,
    float* qkv_gradient,
    int batch,
    int sequence,
    int model_width,
    int heads,
    int causal,
    cudaStream_t stream) {
    if (!qkv || !output || !output_gradient || !log_sum_exp
        || !qkv_gradient || batch <= 0 || sequence <= 0 || heads <= 0
        || model_width % heads
        || model_width / heads > kWarpSize * kColumnsPerLane) {
        return (int)cudaErrorInvalidValue;
    }
    const int queries_per_block =
        kWarpsPerBlock * kQueriesPerWarp;
    dim3 grid(
        (sequence + queries_per_block - 1) / queries_per_block,
        batch * heads);
    attention_backward_register_tiled<<<
        grid, kThreadsPerBlock, 0, stream>>>(
        qkv, output, output_gradient, log_sum_exp, qkv_gradient,
        sequence, model_width, heads, causal);
    return (int)cudaPeekAtLastError();
}

__device__ __forceinline__ float block_sum(float value);

// Move one strided 16-row attention tile into the compact shared-memory
// layout consumed by WMMA. Loads are issued as aligned BF16 pairs so Ampere
// and newer devices can overlap cp.async traffic with the previous tile's
// matrix products. Invalid rows/columns are zero-filled by cp.async.
template <int max_head, bool async_load>
__device__ __forceinline__ void load_projected_bfloat16_tile(
    __nv_bfloat16* destination,
    const __nv_bfloat16* source,
    int batch_base,
    int row_start,
    int sequence,
    int projected_width,
    int projection_offset,
    int head_base,
    int head_width) {
    constexpr int tile = kAttentionTensorCoreTile;
    for (int pair = threadIdx.x; pair < tile * max_head / 2;
         pair += blockDim.x) {
        const int index = pair * 2;
        const int row = index / max_head;
        const int column = index % max_head;
        const int source_row = row_start + row;
        const int valid_bytes = source_row < sequence && column < head_width
            ? (column + 1 < head_width ? 4 : 2)
            : 0;
        const int source_index = valid_bytes
            ? batch_base + source_row * projected_width
                + projection_offset + head_base + column
            : batch_base;
        copy_bfloat16_pair<async_load>(
            destination + index, source + source_index, valid_bytes);
    }
}

template <int max_head, bool async_load>
__device__ __forceinline__ void load_output_gradient_tile(
    __nv_bfloat16* destination,
    const __nv_bfloat16* source,
    int batch_index,
    int row_start,
    int sequence,
    int model_width,
    int head_base,
    int head_width) {
    constexpr int tile = kAttentionTensorCoreTile;
    for (int pair = threadIdx.x; pair < tile * max_head / 2;
         pair += blockDim.x) {
        const int index = pair * 2;
        const int row = index / max_head;
        const int column = index % max_head;
        const int source_row = row_start + row;
        const int valid_bytes = source_row < sequence && column < head_width
            ? (column + 1 < head_width ? 4 : 2)
            : 0;
        const int source_index = valid_bytes
            ? (batch_index * sequence + source_row) * model_width
                + head_base + column
            : 0;
        copy_bfloat16_pair<async_load>(
            destination + index, source + source_index, valid_bytes);
    }
}

template <int max_head, bool async_load>
__device__ __forceinline__ void load_output_gradient_tile(
    __nv_bfloat16* destination,
    const float* source,
    int batch_index,
    int row_start,
    int sequence,
    int model_width,
    int head_base,
    int head_width) {
    constexpr int tile = kAttentionTensorCoreTile;
    for (int index = threadIdx.x; index < tile * max_head;
         index += blockDim.x) {
        const int row = index / max_head;
        const int column = index % max_head;
        const int source_row = row_start + row;
        destination[index] = source_row < sequence && column < head_width
            ? __float2bfloat16_rn(source[
                (batch_index * sequence + source_row) * model_width
                    + head_base + column])
            : __float2bfloat16_rn(0.f);
    }
}

// BF16 Tensor Core attention uses one 16-query or 16-key tile per CTA. The
// head dimension is padded to a multiple of 16 up to 128, so this path is not
// specialized for any single model width. Matrix products use BF16 operands
// and Float32 accumulation; softmax statistics and published gradients remain
// Float32.
template <int max_head, bool async_load>
__global__ void attention_forward_tensor_core_bf16(
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ log_sum_exp,
    int sequence,
    int model_width,
    int heads,
    int causal) {
    using namespace nvcuda;
    constexpr int tile = kAttentionTensorCoreTile;
    __shared__ __nv_bfloat16 query_tile[tile * max_head];
    // Ping-pong K/V tiles let global-memory loading of tile N+1 overlap the
    // Tensor Core and online-softmax work for tile N.
    __shared__ __nv_bfloat16 key_tile[2][tile * max_head];
    __shared__ __nv_bfloat16 value_tile[2][tile * max_head];
    __shared__ __nv_bfloat16 probabilities[tile * tile];
    __shared__ float scores[tile * tile];
    __shared__ float product[tile * max_head];
    __shared__ float accumulated[tile * max_head];
    __shared__ float row_maximum[tile];
    __shared__ float row_denominator[tile];
    __shared__ float old_output_scale[tile];

    const int warp = threadIdx.x / kWarpSize;
    const int query_start = blockIdx.x * tile;
    const int batch_head = blockIdx.y;
    const int head = batch_head % heads;
    const int batch_index = batch_head / heads;
    const int head_width = model_width / heads;
    const int head_tiles = (head_width + tile - 1) / tile;
    const int projected_width = 3 * model_width;
    const int batch_base = batch_index * sequence * projected_width;
    const int head_base = head * head_width;
    const float scale = rsqrtf((float)head_width);

    for (int index = threadIdx.x; index < tile * max_head;
         index += blockDim.x) {
        const int row = index / max_head;
        const int column = index % max_head;
        const int query = query_start + row;
        query_tile[index] = query < sequence && column < head_width
            ? qkv[batch_base + query * projected_width + head_base + column]
            : __float2bfloat16_rn(0.f);
        accumulated[index] = 0.f;
        product[index] = 0.f;
    }
    if (threadIdx.x < tile) {
        row_maximum[threadIdx.x] = -3.402823466e+38F;
        row_denominator[threadIdx.x] = 0.f;
        old_output_scale[threadIdx.x] = 0.f;
    }
    __syncthreads();

    const int key_limit = causal
        ? min(sequence, query_start + tile)
        : sequence;

    // Each thread moves aligned BF16 pairs.  cp.async zero-fills the invalid
    // tail, so no separate clearing pass is required for partial tiles.
    for (int pair = threadIdx.x; pair < tile * max_head / 2;
         pair += blockDim.x) {
        const int index = pair * 2;
        const int row = index / max_head;
        const int column = index % max_head;
        const int key = row;
        const int valid_bytes = key < sequence && column < head_width
            ? (column + 1 < head_width ? 4 : 2)
            : 0;
        const int key_base = valid_bytes
            ? batch_base + key * projected_width + model_width
                + head_base + column
            : batch_base;
        copy_bfloat16_pair<async_load>(
            key_tile[0] + index, qkv + key_base, valid_bytes);
        copy_bfloat16_pair<async_load>(
            value_tile[0] + index,
            qkv + key_base + (valid_bytes ? model_width : 0),
            valid_bytes);
    }
    commit_async_copy_group<async_load>();

    for (int key_start = 0; key_start < key_limit; key_start += tile) {
        const int buffer_index = (key_start / tile) & 1;
        wait_for_async_copy_group<async_load>();
        __syncthreads();
        __nv_bfloat16* current_key = key_tile[buffer_index];
        __nv_bfloat16* current_value = value_tile[buffer_index];

        const int next_key_start = key_start + tile;
        if (next_key_start < key_limit) {
            const int next_buffer = buffer_index ^ 1;
            for (int pair = threadIdx.x; pair < tile * max_head / 2;
                 pair += blockDim.x) {
                const int index = pair * 2;
                const int row = index / max_head;
                const int column = index % max_head;
                const int key = next_key_start + row;
                const int valid_bytes = key < sequence && column < head_width
                    ? (column + 1 < head_width ? 4 : 2)
                    : 0;
                const int key_base = valid_bytes
                    ? batch_base + key * projected_width + model_width
                        + head_base + column
                    : batch_base;
                copy_bfloat16_pair<async_load>(
                    key_tile[next_buffer] + index,
                    qkv + key_base,
                    valid_bytes);
                copy_bfloat16_pair<async_load>(
                    value_tile[next_buffer] + index,
                    qkv + key_base + (valid_bytes ? model_width : 0),
                    valid_bytes);
            }
            commit_async_copy_group<async_load>();
        }

        if (warp == 0) {
            wmma::fragment<wmma::accumulator, tile, tile, tile, float>
                score_fragment;
            wmma::fill_fragment(score_fragment, 0.f);
            for (int dimension_tile = 0;
                 dimension_tile < head_tiles;
                 ++dimension_tile) {
                wmma::fragment<wmma::matrix_a, tile, tile, tile,
                    __nv_bfloat16, wmma::row_major>
                    query_fragment;
                wmma::fragment<wmma::matrix_b, tile, tile, tile,
                    __nv_bfloat16, wmma::col_major>
                    key_fragment;
                const int dimension = dimension_tile * tile;
                wmma::load_matrix_sync(
                    query_fragment, query_tile + dimension, max_head);
                wmma::load_matrix_sync(
                    key_fragment, current_key + dimension, max_head);
                wmma::mma_sync(
                    score_fragment,
                    query_fragment,
                    key_fragment,
                    score_fragment);
            }
            wmma::store_matrix_sync(
                scores, score_fragment, tile, wmma::mem_row_major);
        }
        __syncthreads();

        if (threadIdx.x < tile) {
            const int row = threadIdx.x;
            const int query = query_start + row;
            float tile_maximum = -3.402823466e+38F;
            if (query < sequence) {
                for (int key_item = 0; key_item < tile; ++key_item) {
                    const int key = key_start + key_item;
                    if (key < sequence && (!causal || key <= query)) {
                        tile_maximum = fmaxf(
                            tile_maximum,
                            scores[row * tile + key_item] * scale);
                    }
                }
            }
            const float previous_maximum = row_maximum[row];
            const float next_maximum = fmaxf(previous_maximum, tile_maximum);
            const float previous_scale = isinf(previous_maximum)
                ? 0.f
                : expf(previous_maximum - next_maximum);
            float tile_sum = 0.f;
            for (int key_item = 0; key_item < tile; ++key_item) {
                const int key = key_start + key_item;
                float probability = 0.f;
                if (query < sequence && key < sequence
                    && (!causal || key <= query)) {
                    probability = expf(
                        scores[row * tile + key_item] * scale
                        - next_maximum);
                }
                probabilities[row * tile + key_item] =
                    __float2bfloat16_rn(probability);
                tile_sum += probability;
            }
            old_output_scale[row] = previous_scale;
            row_denominator[row] =
                row_denominator[row] * previous_scale + tile_sum;
            row_maximum[row] = next_maximum;
        }
        __syncthreads();

        if (warp < head_tiles) {
            wmma::fragment<wmma::matrix_a, tile, tile, tile,
                __nv_bfloat16, wmma::row_major>
                probability_fragment;
            wmma::fragment<wmma::matrix_b, tile, tile, tile,
                __nv_bfloat16, wmma::row_major>
                value_fragment;
            wmma::fragment<wmma::accumulator, tile, tile, tile, float>
                output_fragment;
            wmma::fill_fragment(output_fragment, 0.f);
            wmma::load_matrix_sync(
                probability_fragment, probabilities, tile);
            wmma::load_matrix_sync(
                value_fragment, current_value + warp * tile, max_head);
            wmma::mma_sync(
                output_fragment,
                probability_fragment,
                value_fragment,
                output_fragment);
            wmma::store_matrix_sync(
                product + warp * tile,
                output_fragment,
                max_head,
                wmma::mem_row_major);
        }
        __syncthreads();

        for (int index = threadIdx.x; index < tile * max_head;
             index += blockDim.x) {
            const int row = index / max_head;
            const int column = index % max_head;
            if (query_start + row < sequence && column < head_width) {
                accumulated[index] = accumulated[index]
                    * old_output_scale[row] + product[index];
            }
        }
        __syncthreads();
    }

    for (int index = threadIdx.x; index < tile * max_head;
         index += blockDim.x) {
        const int row = index / max_head;
        const int column = index % max_head;
        const int query = query_start + row;
        if (query < sequence && column < head_width) {
            const int output_index =
                (batch_index * sequence + query) * model_width
                + head_base + column;
            output[output_index] = __float2bfloat16_rn(
                accumulated[index] / row_denominator[row]);
        }
    }
    if (threadIdx.x < tile) {
        const int query = query_start + threadIdx.x;
        if (query < sequence) {
            log_sum_exp[batch_head * sequence + query] =
                row_maximum[threadIdx.x]
                + logf(row_denominator[threadIdx.x]);
        }
    }
}

template <typename output_gradient_t>
__global__ void attention_row_delta_bf16(
    const __nv_bfloat16* __restrict__ output,
    const output_gradient_t* __restrict__ output_gradient,
    float* __restrict__ row_delta,
    int sequence,
    int model_width,
    int heads) {
    const int batch_head_query = blockIdx.x;
    const int query = batch_head_query % sequence;
    const int batch_head = batch_head_query / sequence;
    const int head = batch_head % heads;
    const int batch_index = batch_head / heads;
    const int head_width = model_width / heads;
    const int offset = (batch_index * sequence + query) * model_width
        + head * head_width;
    float partial = 0.f;
    for (int column = threadIdx.x; column < head_width;
         column += blockDim.x) {
        partial = fmaf(
            load_value(output_gradient, offset + column),
            __bfloat162float(output[offset + column]),
            partial);
    }
    // The training shape uses head_width=32. A single warp consumes the row
    // without the shared-memory round trip and two block barriers required by
    // the general reduction. Wider heads retain the block reduction path.
    if (blockDim.x == kWarpSize) {
        partial = warp_sum(partial);
        if (threadIdx.x == 0)
            row_delta[batch_head_query] = partial;
    }
    else {
        partial = block_sum(partial);
        if (threadIdx.x == 0)
            row_delta[batch_head_query] = partial;
    }
}

__global__ void attention_incremental_bf16(
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    __nv_bfloat16* __restrict__ output,
    int position,
    int cache_capacity,
    int model_width,
    int heads) {
    extern __shared__ float scores[];
    const int head = blockIdx.x;
    const int head_width = model_width / heads;
    const int head_base = head * head_width;
    const int key_offset = model_width + head_base;
    const int value_offset = 2 * model_width + head_base;
    const int cache_row = position * model_width + head_base;
    for (int column = threadIdx.x; column < head_width;
         column += blockDim.x) {
        key_cache[cache_row + column] = qkv[key_offset + column];
        value_cache[cache_row + column] = qkv[value_offset + column];
    }
    __syncthreads();

    const float scale = rsqrtf((float)head_width);
    for (int key = 0; key <= position; ++key) {
        float partial = 0.f;
        const int key_base = key * model_width + head_base;
        for (int column = threadIdx.x; column < head_width;
             column += blockDim.x) {
            partial = fmaf(
                __bfloat162float(qkv[head_base + column]),
                __bfloat162float(key_cache[key_base + column]),
                partial);
        }
        partial = block_sum(partial);
        if (threadIdx.x == 0)
            scores[key] = partial * scale;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float maximum = -3.402823466e+38F;
        for (int key = 0; key <= position; ++key)
            maximum = fmaxf(maximum, scores[key]);
        float denominator = 0.f;
        for (int key = 0; key <= position; ++key) {
            const float probability = expf(scores[key] - maximum);
            scores[key] = probability;
            denominator += probability;
        }
        scores[cache_capacity] = 1.f / denominator;
    }
    __syncthreads();

    const float inverse_denominator = scores[cache_capacity];
    for (int column = threadIdx.x; column < head_width;
         column += blockDim.x) {
        float value = 0.f;
        for (int key = 0; key <= position; ++key) {
            value = fmaf(
                scores[key],
                __bfloat162float(value_cache[
                    key * model_width + head_base + column]),
                value);
        }
        output[head_base + column] =
            __float2bfloat16_rn(value * inverse_denominator);
    }
}

__global__ void attention_prefill_cache_bf16(
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    int length,
    int model_width) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    const int row = index / model_width;
    const int column = index - row * model_width;
    const int projected = row * 3 * model_width + column;
    key_cache[index] = qkv[projected + model_width];
    value_cache[index] = qkv[projected + 2 * model_width];
}

template <int max_head, bool async_load,
    typename qkv_gradient_t, typename output_gradient_t>
__global__ void attention_backward_query_owner_bf16(
    const __nv_bfloat16* __restrict__ qkv,
    const output_gradient_t* __restrict__ output_gradient,
    const float* __restrict__ log_sum_exp,
    const float* __restrict__ row_delta,
    qkv_gradient_t* __restrict__ qkv_gradient,
    int sequence,
    int model_width,
    int heads,
    int causal) {
    using namespace nvcuda;
    constexpr int tile = kAttentionTensorCoreTile;
    __shared__ __nv_bfloat16 query_tile[tile * max_head];
    __shared__ __nv_bfloat16 key_tile[2][tile * max_head];
    __shared__ __nv_bfloat16 value_tile[2][tile * max_head];
    __shared__ __nv_bfloat16 output_gradient_tile[tile * max_head];
    __shared__ __nv_bfloat16 score_gradients[tile * tile];
    __shared__ float scores[tile * tile];
    __shared__ float probability_gradients[tile * tile];
    __shared__ float product[tile * max_head];
    __shared__ float accumulated[tile * max_head];

    const int warp = threadIdx.x / kWarpSize;
    const int query_start = blockIdx.x * tile;
    const int batch_head = blockIdx.y;
    const int head = batch_head % heads;
    const int batch_index = batch_head / heads;
    const int head_width = model_width / heads;
    const int head_tiles = (head_width + tile - 1) / tile;
    const int projected_width = 3 * model_width;
    const int batch_base = batch_index * sequence * projected_width;
    const int head_base = head * head_width;
    const float scale = rsqrtf((float)head_width);

    for (int index = threadIdx.x; index < tile * max_head;
         index += blockDim.x) {
        accumulated[index] = 0.f;
        product[index] = 0.f;
    }
    load_projected_bfloat16_tile<max_head, async_load>(
        query_tile, qkv, batch_base, query_start, sequence,
        projected_width, 0, head_base, head_width);
    load_output_gradient_tile<max_head, async_load>(
        output_gradient_tile, output_gradient, batch_index, query_start,
        sequence, model_width, head_base, head_width);
    commit_async_copy_group<async_load>();
    wait_for_async_copy_group<async_load>();
    __syncthreads();

    const int key_limit = causal
        ? min(sequence, query_start + tile)
        : sequence;
    load_projected_bfloat16_tile<max_head, async_load>(
        key_tile[0], qkv, batch_base, 0, sequence,
        projected_width, model_width, head_base, head_width);
    load_projected_bfloat16_tile<max_head, async_load>(
        value_tile[0], qkv, batch_base, 0, sequence,
        projected_width, 2 * model_width, head_base, head_width);
    commit_async_copy_group<async_load>();
    for (int key_start = 0; key_start < key_limit; key_start += tile) {
        const int buffer_index = (key_start / tile) & 1;
        wait_for_async_copy_group<async_load>();
        __syncthreads();
        __nv_bfloat16* current_key = key_tile[buffer_index];
        __nv_bfloat16* current_value = value_tile[buffer_index];

        const int next_key_start = key_start + tile;
        if (next_key_start < key_limit) {
            const int next_buffer = buffer_index ^ 1;
            load_projected_bfloat16_tile<max_head, async_load>(
                key_tile[next_buffer], qkv, batch_base, next_key_start,
                sequence, projected_width, model_width, head_base,
                head_width);
            load_projected_bfloat16_tile<max_head, async_load>(
                value_tile[next_buffer], qkv, batch_base, next_key_start,
                sequence, projected_width, 2 * model_width, head_base,
                head_width);
            commit_async_copy_group<async_load>();
        }

        if (warp == 0 || warp == 1) {
            wmma::fragment<wmma::accumulator, tile, tile, tile, float>
                accumulator_fragment;
            wmma::fill_fragment(accumulator_fragment, 0.f);
            for (int dimension_tile = 0;
                 dimension_tile < head_tiles;
                 ++dimension_tile) {
                wmma::fragment<wmma::matrix_a, tile, tile, tile,
                    __nv_bfloat16, wmma::row_major>
                    left_fragment;
                wmma::fragment<wmma::matrix_b, tile, tile, tile,
                    __nv_bfloat16, wmma::col_major>
                    right_fragment;
                const int dimension = dimension_tile * tile;
                wmma::load_matrix_sync(
                    left_fragment,
                    (warp == 0 ? query_tile : output_gradient_tile)
                        + dimension,
                    max_head);
                wmma::load_matrix_sync(
                    right_fragment,
                    (warp == 0 ? current_key : current_value) + dimension,
                    max_head);
                wmma::mma_sync(
                    accumulator_fragment,
                    left_fragment,
                    right_fragment,
                    accumulator_fragment);
            }
            wmma::store_matrix_sync(
                warp == 0 ? scores : probability_gradients,
                accumulator_fragment,
                tile,
                wmma::mem_row_major);
        }
        __syncthreads();

        for (int index = threadIdx.x; index < tile * tile;
             index += blockDim.x) {
            const int row = index / tile;
            const int key_item = index % tile;
            const int query = query_start + row;
            const int key = key_start + key_item;
            float probability = 0.f;
            float score_gradient = 0.f;
            if (query < sequence && key < sequence
                && (!causal || key <= query)) {
                probability = expf(
                    scores[index] * scale
                    - log_sum_exp[batch_head * sequence + query]);
                score_gradient = probability
                    * (probability_gradients[index]
                        - row_delta[batch_head * sequence + query]);
            }
            score_gradients[index] = __float2bfloat16_rn(score_gradient);
        }
        __syncthreads();

        if (warp < head_tiles) {
            wmma::fragment<wmma::matrix_a, tile, tile, tile,
                __nv_bfloat16, wmma::row_major>
                score_gradient_fragment;
            wmma::fragment<wmma::matrix_b, tile, tile, tile,
                __nv_bfloat16, wmma::row_major>
                key_fragment;
            wmma::fragment<wmma::accumulator, tile, tile, tile, float>
                gradient_fragment;
            wmma::fill_fragment(gradient_fragment, 0.f);
            wmma::load_matrix_sync(
                score_gradient_fragment, score_gradients, tile);
            wmma::load_matrix_sync(
                key_fragment, current_key + warp * tile, max_head);
            wmma::mma_sync(
                gradient_fragment,
                score_gradient_fragment,
                key_fragment,
                gradient_fragment);
            wmma::store_matrix_sync(
                product + warp * tile,
                gradient_fragment,
                max_head,
                wmma::mem_row_major);
        }
        __syncthreads();
        for (int index = threadIdx.x; index < tile * max_head;
             index += blockDim.x) {
            const int row = index / max_head;
            const int column = index % max_head;
            if (query_start + row < sequence && column < head_width)
                accumulated[index] += product[index];
        }
        __syncthreads();
    }

    for (int index = threadIdx.x; index < tile * max_head;
         index += blockDim.x) {
        const int row = index / max_head;
        const int column = index % max_head;
        const int query = query_start + row;
        if (query < sequence && column < head_width) {
            accumulate_attention_gradient(
                qkv_gradient,
                batch_base + query * projected_width
                    + head_base + column,
                accumulated[index] * scale);
        }
    }
}

template <int max_head, bool parallel_dkv, bool async_load,
    typename qkv_gradient_t, typename output_gradient_t>
__global__ void attention_backward_key_owner_bf16(
    const __nv_bfloat16* __restrict__ qkv,
    const output_gradient_t* __restrict__ output_gradient,
    const float* __restrict__ log_sum_exp,
    const float* __restrict__ row_delta,
    qkv_gradient_t* __restrict__ qkv_gradient,
    int sequence,
    int model_width,
    int heads,
    int causal) {
    using namespace nvcuda;
    constexpr int tile = kAttentionTensorCoreTile;
    // The <=64 paths fit two Q/dO stages below the portable 48 KiB shared
    // memory limit. The 128-wide fallback keeps one stage so sm80/86/89/90
    // all remain launchable without an opt-in dynamic shared-memory limit.
    constexpr bool pipelined_load = async_load && max_head <= 64;
    __shared__ __nv_bfloat16 query_tile[
        pipelined_load ? 2 : 1][tile * max_head];
    __shared__ __nv_bfloat16 key_tile[tile * max_head];
    __shared__ __nv_bfloat16 value_tile[tile * max_head];
    __shared__ __nv_bfloat16 output_gradient_tile[
        pipelined_load ? 2 : 1][tile * max_head];
    __shared__ __nv_bfloat16 probabilities[tile * tile];
    __shared__ __nv_bfloat16 score_gradients[tile * tile];
    __shared__ float scores[tile * tile];
    __shared__ float probability_gradients[tile * tile];
    __shared__ float product[tile * max_head];
    __shared__ float value_product[
        parallel_dkv ? tile * max_head : 1];
    __shared__ float accumulated_key[tile * max_head];
    __shared__ float accumulated_value[tile * max_head];

    const int warp = threadIdx.x / kWarpSize;
    const int key_start = blockIdx.x * tile;
    const int batch_head = blockIdx.y;
    const int head = batch_head % heads;
    const int batch_index = batch_head / heads;
    const int head_width = model_width / heads;
    const int head_tiles = (head_width + tile - 1) / tile;
    const int projected_width = 3 * model_width;
    const int batch_base = batch_index * sequence * projected_width;
    const int head_base = head * head_width;
    const float scale = rsqrtf((float)head_width);

    for (int index = threadIdx.x; index < tile * max_head;
         index += blockDim.x) {
        accumulated_key[index] = 0.f;
        accumulated_value[index] = 0.f;
        product[index] = 0.f;
    }
    load_projected_bfloat16_tile<max_head, async_load>(
        key_tile, qkv, batch_base, key_start, sequence,
        projected_width, model_width, head_base, head_width);
    load_projected_bfloat16_tile<max_head, async_load>(
        value_tile, qkv, batch_base, key_start, sequence,
        projected_width, 2 * model_width, head_base, head_width);
    commit_async_copy_group<async_load>();
    wait_for_async_copy_group<async_load>();
    __syncthreads();

    const int first_query_tile = causal ? key_start : 0;
    if constexpr (pipelined_load) {
        load_projected_bfloat16_tile<max_head, async_load>(
            query_tile[0], qkv, batch_base, first_query_tile, sequence,
            projected_width, 0, head_base, head_width);
        load_output_gradient_tile<max_head, async_load>(
            output_gradient_tile[0], output_gradient, batch_index,
            first_query_tile, sequence, model_width, head_base, head_width);
        commit_async_copy_group<async_load>();
    }
    for (int query_start = first_query_tile;
         query_start < sequence;
         query_start += tile) {
        if constexpr (!pipelined_load) {
            load_projected_bfloat16_tile<max_head, async_load>(
                query_tile[0], qkv, batch_base, query_start, sequence,
                projected_width, 0, head_base, head_width);
            load_output_gradient_tile<max_head, async_load>(
                output_gradient_tile[0], output_gradient, batch_index,
                query_start, sequence, model_width, head_base, head_width);
            commit_async_copy_group<async_load>();
        }
        const int buffer_index = pipelined_load
            ? ((query_start - first_query_tile) / tile) & 1
            : 0;
        wait_for_async_copy_group<async_load>();
        __syncthreads();
        __nv_bfloat16* current_query = query_tile[buffer_index];
        __nv_bfloat16* current_output_gradient =
            output_gradient_tile[buffer_index];

        if constexpr (pipelined_load) {
          const int next_query_start = query_start + tile;
          if (next_query_start < sequence) {
            const int next_buffer = buffer_index ^ 1;
            load_projected_bfloat16_tile<max_head, async_load>(
                query_tile[next_buffer], qkv, batch_base, next_query_start,
                sequence, projected_width, 0, head_base, head_width);
            load_output_gradient_tile<max_head, async_load>(
                output_gradient_tile[next_buffer], output_gradient,
                batch_index, next_query_start, sequence, model_width,
                head_base, head_width);
            commit_async_copy_group<async_load>();
          }
        }

        if (warp == 0 || warp == 1) {
            wmma::fragment<wmma::accumulator, tile, tile, tile, float>
                accumulator_fragment;
            wmma::fill_fragment(accumulator_fragment, 0.f);
            for (int dimension_tile = 0;
                 dimension_tile < head_tiles;
                 ++dimension_tile) {
                wmma::fragment<wmma::matrix_a, tile, tile, tile,
                    __nv_bfloat16, wmma::row_major>
                    left_fragment;
                wmma::fragment<wmma::matrix_b, tile, tile, tile,
                    __nv_bfloat16, wmma::col_major>
                    right_fragment;
                const int dimension = dimension_tile * tile;
                wmma::load_matrix_sync(
                    left_fragment,
                    (warp == 0 ? current_query : current_output_gradient)
                        + dimension,
                    max_head);
                wmma::load_matrix_sync(
                    right_fragment,
                    (warp == 0 ? key_tile : value_tile) + dimension,
                    max_head);
                wmma::mma_sync(
                    accumulator_fragment,
                    left_fragment,
                    right_fragment,
                    accumulator_fragment);
            }
            wmma::store_matrix_sync(
                warp == 0 ? scores : probability_gradients,
                accumulator_fragment,
                tile,
                wmma::mem_row_major);
        }
        __syncthreads();

        for (int index = threadIdx.x; index < tile * tile;
             index += blockDim.x) {
            const int query_item = index / tile;
            const int key_item = index % tile;
            const int query = query_start + query_item;
            const int key = key_start + key_item;
            float probability = 0.f;
            float score_gradient = 0.f;
            if (query < sequence && key < sequence
                && (!causal || key <= query)) {
                probability = expf(
                    scores[index] * scale
                    - log_sum_exp[batch_head * sequence + query]);
                score_gradient = probability
                    * (probability_gradients[index]
                        - row_delta[batch_head * sequence + query]);
            }
            probabilities[index] = __float2bfloat16_rn(probability);
            score_gradients[index] = __float2bfloat16_rn(score_gradient);
        }
        __syncthreads();

        if (parallel_dkv && warp < 2 * head_tiles) {
            const bool value_gradient = warp >= head_tiles;
            const int dimension_warp = warp % head_tiles;
            wmma::fragment<wmma::matrix_a, tile, tile, tile,
                __nv_bfloat16, wmma::col_major>
                left_fragment;
            wmma::fragment<wmma::matrix_b, tile, tile, tile,
                __nv_bfloat16, wmma::row_major>
                right_fragment;
            wmma::fragment<wmma::accumulator, tile, tile, tile, float>
                gradient_fragment;
            wmma::fill_fragment(gradient_fragment, 0.f);
            wmma::load_matrix_sync(
                left_fragment,
                value_gradient ? probabilities : score_gradients,
                tile);
            wmma::load_matrix_sync(
                right_fragment,
                (value_gradient ? current_output_gradient : current_query)
                    + dimension_warp * tile,
                max_head);
            wmma::mma_sync(
                gradient_fragment,
                left_fragment,
                right_fragment,
                gradient_fragment);
            wmma::store_matrix_sync(
                (value_gradient ? value_product : product)
                    + dimension_warp * tile,
                gradient_fragment,
                max_head,
                wmma::mem_row_major);
        }
        else if (!parallel_dkv && warp < head_tiles) {
            wmma::fragment<wmma::matrix_a, tile, tile, tile,
                __nv_bfloat16, wmma::col_major>
                score_gradient_fragment;
            wmma::fragment<wmma::matrix_b, tile, tile, tile,
                __nv_bfloat16, wmma::row_major>
                query_fragment;
            wmma::fragment<wmma::accumulator, tile, tile, tile, float>
                gradient_fragment;
            wmma::fill_fragment(gradient_fragment, 0.f);
            wmma::load_matrix_sync(
                score_gradient_fragment, score_gradients, tile);
            wmma::load_matrix_sync(
                query_fragment, current_query + warp * tile, max_head);
            wmma::mma_sync(
                gradient_fragment,
                score_gradient_fragment,
                query_fragment,
                gradient_fragment);
            wmma::store_matrix_sync(
                product + warp * tile,
                gradient_fragment,
                max_head,
                wmma::mem_row_major);
        }
        __syncthreads();
        for (int index = threadIdx.x; index < tile * max_head;
             index += blockDim.x) {
            const int row = index / max_head;
            const int column = index % max_head;
            if (key_start + row < sequence && column < head_width) {
                accumulated_key[index] += product[index];
                if (parallel_dkv)
                    accumulated_value[index] += value_product[index];
            }
        }
        __syncthreads();

        if (!parallel_dkv && warp < head_tiles) {
            wmma::fragment<wmma::matrix_a, tile, tile, tile,
                __nv_bfloat16, wmma::col_major>
                probability_fragment;
            wmma::fragment<wmma::matrix_b, tile, tile, tile,
                __nv_bfloat16, wmma::row_major>
                output_gradient_fragment;
            wmma::fragment<wmma::accumulator, tile, tile, tile, float>
                gradient_fragment;
            wmma::fill_fragment(gradient_fragment, 0.f);
            wmma::load_matrix_sync(
                probability_fragment, probabilities, tile);
            wmma::load_matrix_sync(
                output_gradient_fragment,
                current_output_gradient + warp * tile,
                max_head);
            wmma::mma_sync(
                gradient_fragment,
                probability_fragment,
                output_gradient_fragment,
                gradient_fragment);
            wmma::store_matrix_sync(
                product + warp * tile,
                gradient_fragment,
                max_head,
                wmma::mem_row_major);
        }
        __syncthreads();
        if (!parallel_dkv) {
            for (int index = threadIdx.x; index < tile * max_head;
                 index += blockDim.x) {
                const int row = index / max_head;
                const int column = index % max_head;
                if (key_start + row < sequence && column < head_width)
                    accumulated_value[index] += product[index];
            }
        }
        __syncthreads();
    }

    for (int index = threadIdx.x; index < tile * max_head;
         index += blockDim.x) {
        const int row = index / max_head;
        const int column = index % max_head;
        const int key = key_start + row;
        if (key < sequence && column < head_width) {
            const int key_index = batch_base
                + key * projected_width + model_width
                + head_base + column;
            accumulate_attention_gradient(
                qkv_gradient,
                key_index,
                accumulated_key[index] * scale);
            accumulate_attention_gradient(
                qkv_gradient,
                key_index + model_width,
                accumulated_value[index]);
        }
    }
}

template <int max_head>
int launch_forward_tensor_core_bf16_specialized(
    const __nv_bfloat16* qkv,
    __nv_bfloat16* output,
    float* log_sum_exp,
    int batch,
    int sequence,
    int model_width,
    int heads,
    int causal,
    bool async_load,
    cudaStream_t stream) {
    constexpr int threads = max_head * 2;
    const dim3 grid(
        (sequence + kAttentionTensorCoreTile - 1)
            / kAttentionTensorCoreTile,
        batch * heads);
    if (async_load) {
        attention_forward_tensor_core_bf16<max_head, true><<<
            grid, threads, 0, stream>>>(
                qkv, output, log_sum_exp,
                sequence, model_width, heads, causal);
    }
    else {
        attention_forward_tensor_core_bf16<max_head, false><<<
            grid, threads, 0, stream>>>(
                qkv, output, log_sum_exp,
                sequence, model_width, heads, causal);
    }
    return (int)cudaPeekAtLastError();
}

int launch_forward_tensor_core_bf16(
    const __nv_bfloat16* qkv,
    __nv_bfloat16* output,
    float* log_sum_exp,
    int batch,
    int sequence,
    int model_width,
    int heads,
    int causal,
    bool async_load,
    cudaStream_t stream) {
    const int head_width = heads > 0 ? model_width / heads : 0;
    if (!qkv || !output || !log_sum_exp || batch <= 0 || sequence <= 0
        || heads <= 0 || model_width % heads || head_width <= 0
        || head_width > kAttentionTensorCoreMaxHead) {
        return (int)cudaErrorInvalidValue;
    }
    if (head_width <= 32)
        return launch_forward_tensor_core_bf16_specialized<32>(
            qkv, output, log_sum_exp, batch, sequence, model_width, heads,
            causal, async_load, stream);
    if (head_width <= 64)
        return launch_forward_tensor_core_bf16_specialized<64>(
            qkv, output, log_sum_exp, batch, sequence, model_width, heads,
            causal, async_load, stream);
    return launch_forward_tensor_core_bf16_specialized<128>(
        qkv, output, log_sum_exp, batch, sequence, model_width, heads,
        causal, async_load, stream);
}

template <int max_head, bool parallel_dkv, bool async_load,
    typename qkv_gradient_t, typename output_gradient_t>
int launch_backward_tensor_core_bf16_specialized(
    const __nv_bfloat16* qkv,
    const output_gradient_t* output_gradient,
    const float* log_sum_exp,
    const float* row_delta,
    qkv_gradient_t* qkv_gradient,
    int batch,
    int sequence,
    int model_width,
    int heads,
    int causal,
    cudaStream_t stream) {
    constexpr int threads = max_head * 2;
    constexpr int key_owner_threads = parallel_dkv
        ? max_head * 4
        : threads;
    const dim3 grid(
        (sequence + kAttentionTensorCoreTile - 1)
            / kAttentionTensorCoreTile,
        batch * heads);
    attention_backward_query_owner_bf16<
        max_head, async_load, qkv_gradient_t, output_gradient_t><<<
        grid, threads, 0, stream>>>(
            qkv, output_gradient, log_sum_exp, row_delta, qkv_gradient,
            sequence, model_width, heads, causal);
    cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return (int)status;
    attention_backward_key_owner_bf16<
        max_head, parallel_dkv, async_load,
        qkv_gradient_t, output_gradient_t><<<
        grid, key_owner_threads, 0, stream>>>(
            qkv, output_gradient, log_sum_exp, row_delta, qkv_gradient,
            sequence, model_width, heads, causal);
    return (int)cudaPeekAtLastError();
}

template <typename qkv_gradient_t, typename output_gradient_t>
int launch_backward_tensor_core_bf16(
    const __nv_bfloat16* qkv,
    const __nv_bfloat16* output,
    const output_gradient_t* output_gradient,
    const float* log_sum_exp,
    float* row_delta,
    qkv_gradient_t* qkv_gradient,
    int batch,
    int sequence,
    int model_width,
    int heads,
    int causal,
    bool parallel_dkv,
    bool async_load,
    cudaStream_t stream) {
    const int head_width = heads > 0 ? model_width / heads : 0;
    if (!qkv || !output || !output_gradient || !log_sum_exp || !row_delta
        || !qkv_gradient || batch <= 0 || sequence <= 0 || heads <= 0
        || model_width % heads || head_width <= 0
        || head_width > kAttentionTensorCoreMaxHead) {
        return (int)cudaErrorInvalidValue;
    }
    // Four-byte cp.async transactions require every row/head origin to remain
    // pair-aligned. Odd generic head widths retain the synchronous loader.
    async_load = async_load
        && (model_width & 1) == 0
        && (head_width & 1) == 0;
    const int row_delta_threads = head_width <= kWarpSize
        ? kWarpSize
        : 128;
    attention_row_delta_bf16<output_gradient_t><<<
        batch * heads * sequence, row_delta_threads, 0, stream>>>(
            output, output_gradient, row_delta,
            sequence, model_width, heads);
    cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return (int)status;
    if (head_width <= 32) {
        if (async_load) {
            return parallel_dkv
                ? launch_backward_tensor_core_bf16_specialized<
                    32, true, true, qkv_gradient_t, output_gradient_t>(
                    qkv, output_gradient, log_sum_exp, row_delta,
                    qkv_gradient, batch, sequence, model_width, heads,
                    causal, stream)
                : launch_backward_tensor_core_bf16_specialized<
                    32, false, true, qkv_gradient_t, output_gradient_t>(
                    qkv, output_gradient, log_sum_exp, row_delta,
                    qkv_gradient, batch, sequence, model_width, heads,
                    causal, stream);
        }
        return parallel_dkv
            ? launch_backward_tensor_core_bf16_specialized<
                32, true, false, qkv_gradient_t, output_gradient_t>(
                qkv, output_gradient, log_sum_exp, row_delta, qkv_gradient,
                batch, sequence, model_width, heads, causal, stream)
            : launch_backward_tensor_core_bf16_specialized<
                32, false, false, qkv_gradient_t, output_gradient_t>(
                qkv, output_gradient, log_sum_exp, row_delta, qkv_gradient,
                batch, sequence, model_width, heads, causal, stream);
    }
    if (head_width <= 64) {
        if (async_load) {
            return parallel_dkv
                ? launch_backward_tensor_core_bf16_specialized<
                    64, true, true, qkv_gradient_t, output_gradient_t>(
                    qkv, output_gradient, log_sum_exp, row_delta,
                    qkv_gradient, batch, sequence, model_width, heads,
                    causal, stream)
                : launch_backward_tensor_core_bf16_specialized<
                    64, false, true, qkv_gradient_t, output_gradient_t>(
                    qkv, output_gradient, log_sum_exp, row_delta,
                    qkv_gradient, batch, sequence, model_width, heads,
                    causal, stream);
        }
        return parallel_dkv
            ? launch_backward_tensor_core_bf16_specialized<
                64, true, false, qkv_gradient_t, output_gradient_t>(
                qkv, output_gradient, log_sum_exp, row_delta, qkv_gradient,
                batch, sequence, model_width, heads, causal, stream)
            : launch_backward_tensor_core_bf16_specialized<
                64, false, false, qkv_gradient_t, output_gradient_t>(
                qkv, output_gradient, log_sum_exp, row_delta, qkv_gradient,
                batch, sequence, model_width, heads, causal, stream);
    }
    return async_load
        ? launch_backward_tensor_core_bf16_specialized<
            128, false, true, qkv_gradient_t, output_gradient_t>(
            qkv, output_gradient, log_sum_exp, row_delta, qkv_gradient,
            batch, sequence, model_width, heads, causal, stream)
        : launch_backward_tensor_core_bf16_specialized<
            128, false, false, qkv_gradient_t, output_gradient_t>(
            qkv, output_gradient, log_sum_exp, row_delta, qkv_gradient,
            batch, sequence, model_width, heads, causal, stream);
}

__device__ __forceinline__ float block_sum(float value) {
    __shared__ float warp_sums[kLayerNormThreads / kWarpSize];
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    if (lane == 0)
        warp_sums[warp] = value;
    __syncthreads();
    value = threadIdx.x < blockDim.x / kWarpSize
        ? warp_sums[lane]
        : 0.f;
    if (warp == 0) {
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffu, value, offset);
        if (lane == 0)
            warp_sums[0] = value;
    }
    __syncthreads();
    return warp_sums[0];
}

__device__ __forceinline__ void block_sum_pair(float& first, float& second) {
    __shared__ float first_warps[kLayerNormThreads / kWarpSize];
    __shared__ float second_warps[kLayerNormThreads / kWarpSize];
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        first += __shfl_down_sync(0xffffffffu, first, offset);
        second += __shfl_down_sync(0xffffffffu, second, offset);
    }
    if (lane == 0) {
        first_warps[warp] = first;
        second_warps[warp] = second;
    }
    __syncthreads();
    first = threadIdx.x < blockDim.x / kWarpSize
        ? first_warps[lane]
        : 0.f;
    second = threadIdx.x < blockDim.x / kWarpSize
        ? second_warps[lane]
        : 0.f;
    if (warp == 0) {
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            first += __shfl_down_sync(0xffffffffu, first, offset);
            second += __shfl_down_sync(0xffffffffu, second, offset);
        }
        if (lane == 0) {
            first_warps[0] = first;
            second_warps[0] = second;
        }
    }
    __syncthreads();
    first = first_warps[0];
    second = second_warps[0];
}

__device__ __forceinline__ float dropout_multiplier(
    unsigned int seed,
    int index,
    unsigned int drop_threshold,
    float scale) {
    const unsigned int counter = (unsigned int)(index + 1);
    unsigned int bits = seed + 0x9E3779B9u * counter;
    bits ^= bits >> 16;
    bits *= 0x7FEB352Du;
    bits ^= bits >> 15;
    bits *= 0x846CA68Bu;
    bits ^= bits >> 16;
    return bits < drop_threshold ? 0.f : scale;
}

template <typename T, bool FuseResidualDropout>
__device__ __forceinline__ float layer_norm_input(
    const T* input,
    const T* branch,
    int index,
    unsigned int seed,
    unsigned int drop_threshold,
    float dropout_scale) {
    float value = load_value(input, index);
    if constexpr (FuseResidualDropout) {
        value += load_value(branch, index) * dropout_multiplier(
            seed, index, drop_threshold, dropout_scale);
        value = round_to_storage<T>(value);
    }
    return value;
}

template <typename T, bool FuseResidualDropout>
__global__ void layer_norm_forward_block(
    const T* __restrict__ input,
    const T* __restrict__ branch,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T* __restrict__ output,
    float* __restrict__ means,
    float* __restrict__ inverses,
    int columns,
    float epsilon,
    unsigned int seed,
    unsigned int drop_threshold,
    float dropout_scale) {
    const int row = blockIdx.x;
    const int offset = row * columns;
    if (columns <= kLayerNormCachedColumns) {
        float cached[kLayerNormValuesPerThread] = {};
        float sum = 0.f;
        #pragma unroll
        for (int item = 0; item < kLayerNormValuesPerThread; ++item) {
            const int column = threadIdx.x + item * blockDim.x;
            if (column < columns) {
                cached[item] = layer_norm_input<T, FuseResidualDropout>(
                    input, branch, offset + column, seed, drop_threshold,
                    dropout_scale);
                sum += cached[item];
            }
        }
        const float mean = block_sum(sum) / columns;
        float variance_sum = 0.f;
        #pragma unroll
        for (int item = 0; item < kLayerNormValuesPerThread; ++item) {
            const int column = threadIdx.x + item * blockDim.x;
            if (column < columns) {
                const float difference = cached[item] - mean;
                variance_sum = fmaf(
                    difference, difference, variance_sum);
            }
        }
        const float inverse = rsqrtf(
            block_sum(variance_sum) / columns + epsilon);
        if (threadIdx.x == 0) {
            means[row] = mean;
            inverses[row] = inverse;
        }
        #pragma unroll
        for (int item = 0; item < kLayerNormValuesPerThread; ++item) {
            const int column = threadIdx.x + item * blockDim.x;
            if (column < columns) {
                const int index = offset + column;
                const float xhat = (cached[item] - mean) * inverse;
                store_value(output, index, fmaf(
                    xhat,
                    load_value(gamma, column),
                    load_value(beta, column)));
            }
        }
        return;
    }
    float sum = 0.f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        sum += layer_norm_input<T, FuseResidualDropout>(
            input, branch, offset + column, seed, drop_threshold,
            dropout_scale);
    }
    const float mean = block_sum(sum) / columns;
    float variance_sum = 0.f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float difference = layer_norm_input<T, FuseResidualDropout>(
            input, branch, offset + column, seed, drop_threshold,
            dropout_scale) - mean;
        variance_sum = fmaf(difference, difference, variance_sum);
    }
    const float inverse = rsqrtf(block_sum(variance_sum) / columns + epsilon);
    if (threadIdx.x == 0) {
        means[row] = mean;
        inverses[row] = inverse;
    }
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = offset + column;
        const float xhat = (layer_norm_input<T, FuseResidualDropout>(
            input, branch, index, seed, drop_threshold, dropout_scale) - mean)
            * inverse;
        store_value(output, index,
            fmaf(xhat, load_value(gamma, column), load_value(beta, column)));
    }
}

template <typename T, bool FuseResidualDropout,
    typename output_gradient_t, typename branch_gradient_t>
__global__ void layer_norm_backward_input_block(
    const T* __restrict__ input,
    const T* __restrict__ branch,
    const T* __restrict__ gamma,
    const float* __restrict__ means,
    const float* __restrict__ inverses,
    const output_gradient_t* __restrict__ output_gradient,
    float* __restrict__ input_gradient,
    branch_gradient_t* __restrict__ branch_gradient,
    int columns,
    int same_parent,
    unsigned int seed,
    unsigned int drop_threshold,
    float dropout_scale) {
    const int row = blockIdx.x;
    const int offset = row * columns;
    const float mean = means[row];
    const float inverse = inverses[row];
    if (columns <= kLayerNormCachedColumns) {
        float cached_dxhat[kLayerNormValuesPerThread] = {};
        float cached_xhat[kLayerNormValuesPerThread] = {};
        float dxhat_sum = 0.f;
        float dxhat_xhat_sum = 0.f;
        #pragma unroll
        for (int item = 0; item < kLayerNormValuesPerThread; ++item) {
            const int column = threadIdx.x + item * blockDim.x;
            if (column < columns) {
                const int index = offset + column;
                const float dxhat = load_value(output_gradient, index)
                    * load_value(gamma, column);
                const float xhat =
                    (layer_norm_input<T, FuseResidualDropout>(
                        input, branch, index, seed, drop_threshold,
                        dropout_scale) - mean) * inverse;
                cached_dxhat[item] = dxhat;
                cached_xhat[item] = xhat;
                dxhat_sum += dxhat;
                dxhat_xhat_sum = fmaf(
                    dxhat, xhat, dxhat_xhat_sum);
            }
        }
        block_sum_pair(dxhat_sum, dxhat_xhat_sum);
        const float inverse_over_columns = inverse / columns;
        #pragma unroll
        for (int item = 0; item < kLayerNormValuesPerThread; ++item) {
            const int column = threadIdx.x + item * blockDim.x;
            if (column < columns) {
                const int index = offset + column;
                const float gradient = inverse_over_columns
                    * (columns * cached_dxhat[item] - dxhat_sum
                        - cached_xhat[item] * dxhat_xhat_sum);
                if constexpr (FuseResidualDropout) {
                    const float multiplier = dropout_multiplier(
                        seed, index, drop_threshold, dropout_scale);
                    if (same_parent) {
                        input_gradient[index] +=
                            gradient * (1.f + multiplier);
                    }
                    else {
                        input_gradient[index] += gradient;
                        accumulate_layer_norm_gradient(
                            branch_gradient, index, gradient * multiplier);
                    }
                }
                else {
                    input_gradient[index] += gradient;
                }
            }
        }
        return;
    }
    float dxhat_sum = 0.f;
    float dxhat_xhat_sum = 0.f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = offset + column;
        const float dxhat = load_value(output_gradient, index)
            * load_value(gamma, column);
        dxhat_sum += dxhat;
        const float xhat = (layer_norm_input<T, FuseResidualDropout>(
            input, branch, index, seed, drop_threshold, dropout_scale) - mean)
            * inverse;
        dxhat_xhat_sum = fmaf(dxhat, xhat, dxhat_xhat_sum);
    }
    block_sum_pair(dxhat_sum, dxhat_xhat_sum);
    const float inverse_over_columns = inverses[row] / columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = offset + column;
        const float dxhat = load_value(output_gradient, index)
            * load_value(gamma, column);
        const float xhat = (layer_norm_input<T, FuseResidualDropout>(
            input, branch, index, seed, drop_threshold, dropout_scale) - mean)
            * inverse;
        const float gradient = inverse_over_columns
            * (columns * dxhat - dxhat_sum
                - xhat * dxhat_xhat_sum);
        if constexpr (FuseResidualDropout) {
            const float multiplier = dropout_multiplier(
                seed, index, drop_threshold, dropout_scale);
            if (same_parent)
                input_gradient[index] += gradient * (1.f + multiplier);
            else {
                input_gradient[index] += gradient;
                accumulate_layer_norm_gradient(
                    branch_gradient, index, gradient * multiplier);
            }
        }
        else {
            input_gradient[index] += gradient;
        }
    }
}

template <typename T, bool FuseResidualDropout, typename output_gradient_t>
__global__ void layer_norm_backward_parameters_tiled(
    const T* __restrict__ input,
    const T* __restrict__ branch,
    const float* __restrict__ means,
    const float* __restrict__ inverses,
    const output_gradient_t* __restrict__ output_gradient,
    float* __restrict__ parameter_partials,
    int rows,
    int columns,
    int row_tiles,
    unsigned int seed,
    unsigned int drop_threshold,
    float dropout_scale) {
    __shared__ float gamma_partials
        [kLayerNormParameterRows][kLayerNormParameterColumns];
    __shared__ float beta_partials
        [kLayerNormParameterRows][kLayerNormParameterColumns];
    const int column = blockIdx.x * kLayerNormParameterColumns + threadIdx.x;
    const int row_start = blockIdx.y * kLayerNormRowsPerTile;
    const int row_end = min(rows, row_start + kLayerNormRowsPerTile);
    float gamma_sum = 0.f;
    float beta_sum = 0.f;
    if (column < columns) {
        for (int row = row_start + threadIdx.y;
             row < row_end;
             row += kLayerNormParameterRows) {
            const int index = row * columns + column;
            const float gradient = load_value(output_gradient, index);
            const float xhat = (layer_norm_input<T, FuseResidualDropout>(
                input, branch, index, seed, drop_threshold, dropout_scale)
                - means[row]) * inverses[row];
            beta_sum += gradient;
            gamma_sum = fmaf(gradient, xhat, gamma_sum);
        }
    }
    gamma_partials[threadIdx.y][threadIdx.x] = gamma_sum;
    beta_partials[threadIdx.y][threadIdx.x] = beta_sum;
    __syncthreads();
    if (threadIdx.y == 0 && column < columns) {
        gamma_sum = 0.f;
        beta_sum = 0.f;
        #pragma unroll
        for (int row_lane = 0; row_lane < kLayerNormParameterRows; ++row_lane) {
            gamma_sum += gamma_partials[row_lane][threadIdx.x];
            beta_sum += beta_partials[row_lane][threadIdx.x];
        }
        const int partial = blockIdx.y * columns + column;
        parameter_partials[partial] = gamma_sum;
        parameter_partials[row_tiles * columns + partial] = beta_sum;
    }
}

__global__ void layer_norm_backward_parameters_finalize(
    const float* __restrict__ parameter_partials,
    float* __restrict__ gamma_gradient,
    float* __restrict__ beta_gradient,
    int row_tiles,
    int columns) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns)
        return;
    float gamma_sum = 0.f;
    float beta_sum = 0.f;
    for (int tile = 0; tile < row_tiles; ++tile) {
        const int index = tile * columns + column;
        gamma_sum += parameter_partials[index];
        beta_sum += parameter_partials[row_tiles * columns + index];
    }
    gamma_gradient[column] += gamma_sum;
    beta_gradient[column] += beta_sum;
}

template <typename T, bool FuseResidualDropout>
int launch_layer_norm_forward(
    const T* input,
    const T* branch,
    const T* gamma,
    const T* beta,
    T* output,
    float* means,
    float* inverses,
    int rows,
    int columns,
    float epsilon,
    unsigned int seed,
    unsigned int drop_threshold,
    float dropout_scale,
    cudaStream_t stream) {
    if (!input || !gamma || !beta || !output || !means || !inverses
        || (FuseResidualDropout && !branch) || rows <= 0 || columns <= 0
        || epsilon <= 0.f) {
        return (int)cudaErrorInvalidValue;
    }
    layer_norm_forward_block<T, FuseResidualDropout><<<
        rows, kLayerNormThreads, 0, stream>>>(
        input, branch, gamma, beta, output, means, inverses, columns,
        epsilon, seed, drop_threshold, dropout_scale);
    return (int)cudaPeekAtLastError();
}

template <typename T, bool FuseResidualDropout,
    typename output_gradient_t, typename branch_gradient_t>
int launch_layer_norm_backward(
    const T* input,
    const T* branch,
    const T* gamma,
    const float* means,
    const float* inverses,
    const output_gradient_t* output_gradient,
    float* input_gradient,
    branch_gradient_t* branch_gradient,
    float* gamma_gradient,
    float* beta_gradient,
    float* parameter_partials,
    int rows,
    int columns,
    int same_parent,
    unsigned int seed,
    unsigned int drop_threshold,
    float dropout_scale,
    cudaStream_t stream) {
    if (!input || !gamma || !means || !inverses || !output_gradient
        || !input_gradient || !gamma_gradient || !beta_gradient
        || !parameter_partials
        || (FuseResidualDropout && (!branch || !branch_gradient))
        || rows <= 0 || columns <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    layer_norm_backward_input_block<T, FuseResidualDropout,
        output_gradient_t, branch_gradient_t><<<
        rows, kLayerNormThreads, 0, stream>>>(
        input, branch, gamma, means, inverses, output_gradient, input_gradient,
        branch_gradient, columns, same_parent, seed, drop_threshold,
        dropout_scale);
    cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return (int)status;
    const dim3 threads(
        kLayerNormParameterColumns, kLayerNormParameterRows);
    const dim3 grid(
        (columns + kLayerNormParameterColumns - 1)
            / kLayerNormParameterColumns,
        (rows + kLayerNormRowsPerTile - 1) / kLayerNormRowsPerTile);
    layer_norm_backward_parameters_tiled<T, FuseResidualDropout,
        output_gradient_t><<<
        grid, threads, 0, stream>>>(
        input, branch, means, inverses, output_gradient,
        parameter_partials, rows, columns, grid.y,
        seed, drop_threshold, dropout_scale);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return (int)status;
    layer_norm_backward_parameters_finalize<<<
        (columns + kLayerNormThreads - 1) / kLayerNormThreads,
        kLayerNormThreads, 0, stream>>>(
        parameter_partials, gamma_gradient, beta_gradient,
        grid.y, columns);
    return (int)cudaPeekAtLastError();
}

template <bool store_corrected_moments>
__global__ void nekomuon_moments_stats_block(
    const float* __restrict__ gradient,
    float* __restrict__ fast,
    float* __restrict__ slow,
    float* __restrict__ fast_hat,
    float* __restrict__ slow_hat,
    float* __restrict__ stats,
    int length,
    float beta_fast,
    float beta_slow,
    float fast_correction,
    float slow_correction) {
    __shared__ float warp_stats[4][kNekoMuonThreads / kWarpSize];
    float dot = 0.f;
    float fast_norm = 0.f;
    float slow_norm = 0.f;
    float residual_norm = 0.f;
    const int stride = blockDim.x * gridDim.x;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < length;
         index += stride) {
        const float next_fast = fmaf(
            beta_fast, fast[index], (1.f - beta_fast) * gradient[index]);
        const float next_slow = fmaf(
            beta_slow, slow[index], (1.f - beta_slow) * gradient[index]);
        const float corrected_fast = next_fast / fast_correction;
        const float corrected_slow = next_slow / slow_correction;
        const float residual = corrected_fast - corrected_slow;
        fast[index] = next_fast;
        slow[index] = next_slow;
        if constexpr (store_corrected_moments) {
            fast_hat[index] = corrected_fast;
            slow_hat[index] = corrected_slow;
        }
        dot = fmaf(corrected_fast, corrected_slow, dot);
        fast_norm = fmaf(corrected_fast, corrected_fast, fast_norm);
        slow_norm = fmaf(corrected_slow, corrected_slow, slow_norm);
        residual_norm = fmaf(residual, residual, residual_norm);
    }

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffffu, dot, offset);
        fast_norm += __shfl_down_sync(0xffffffffu, fast_norm, offset);
        slow_norm += __shfl_down_sync(0xffffffffu, slow_norm, offset);
        residual_norm += __shfl_down_sync(
            0xffffffffu, residual_norm, offset);
    }
    if (lane == 0) {
        warp_stats[0][warp] = dot;
        warp_stats[1][warp] = fast_norm;
        warp_stats[2][warp] = slow_norm;
        warp_stats[3][warp] = residual_norm;
    }
    __syncthreads();

    if (warp == 0) {
        const int warp_count = blockDim.x / kWarpSize;
        dot = lane < warp_count ? warp_stats[0][lane] : 0.f;
        fast_norm = lane < warp_count ? warp_stats[1][lane] : 0.f;
        slow_norm = lane < warp_count ? warp_stats[2][lane] : 0.f;
        residual_norm = lane < warp_count ? warp_stats[3][lane] : 0.f;
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
            fast_norm += __shfl_down_sync(
                0xffffffffu, fast_norm, offset);
            slow_norm += __shfl_down_sync(
                0xffffffffu, slow_norm, offset);
            residual_norm += __shfl_down_sync(
                0xffffffffu, residual_norm, offset);
        }
        if (lane == 0) {
            atomicAdd(stats + 0, dot);
            atomicAdd(stats + 1, fast_norm);
            atomicAdd(stats + 2, slow_norm);
            atomicAdd(stats + 3, residual_norm);
        }
    }
}

template <bool store_corrected_moments>
int launch_nekomuon_moments_stats(
    const float* gradient,
    float* fast,
    float* slow,
    float* fast_hat,
    float* slow_hat,
    float* stats,
    int length,
    float beta_fast,
    float beta_slow,
    float fast_correction,
    float slow_correction,
    cudaStream_t stream) {
    if (!gradient || !fast || !slow || !stats
        || (store_corrected_moments && (!fast_hat || !slow_hat))
        || length <= 0 || fast_correction <= 0.f
        || slow_correction <= 0.f) {
        return (int)cudaErrorInvalidValue;
    }
    const int blocks = min(
        (length + kNekoMuonThreads - 1) / kNekoMuonThreads,
        kNekoMuonMaxBlocks);
    nekomuon_moments_stats_block<store_corrected_moments><<<
        blocks, kNekoMuonThreads, 0, stream>>>(
        gradient, fast, slow, fast_hat, slow_hat, stats, length,
        beta_fast, beta_slow, fast_correction, slow_correction);
    return (int)cudaPeekAtLastError();
}

__global__ void pack_gradient_bf16(
    const float* __restrict__ source,
    __nv_bfloat16* __restrict__ destination,
    int destination_offset,
    int length) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < length;
         index += blockDim.x * gridDim.x) {
        destination[destination_offset + index] =
            __float2bfloat16_rn(source[index]);
    }
}

__global__ void sum_gradient_bf16(
    const __nv_bfloat16* __restrict__ local,
    const __nv_bfloat16* __restrict__ remote,
    float* __restrict__ reduced,
    int length,
    double* __restrict__ squared_sum) {
    __shared__ double partials[kGradientBucketThreads];
    double partial = 0.0;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < length;
         index += blockDim.x * gridDim.x) {
        float value = __bfloat162float(local[index])
            + __bfloat162float(remote[index]);
        reduced[index] = value;
        if (squared_sum)
            partial += (double)value * value;
    }
    if (!squared_sum)
        return;
    partials[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(squared_sum, partials[0]);
}

__global__ void unpack_gradient_float(
    const float* __restrict__ source,
    int source_offset,
    float* __restrict__ destination,
    int length) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < length;
         index += blockDim.x * gridDim.x) {
        destination[index] = source[source_offset + index];
    }
}

constexpr int kForgetMemoryTile = 16;
constexpr int kForgetMemoryMaxWidth = 128;

__device__ __forceinline__ void forget_memory_tensor_core_matvec(
    const __nv_bfloat16* state_tile,
    const __nv_bfloat16* vector_matrix,
    float* product,
    int key_tiles) {
    using namespace nvcuda;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.f);
    for (int key_tile = 0; key_tile < key_tiles; ++key_tile) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16,
            __nv_bfloat16, wmma::row_major> state_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16,
            __nv_bfloat16, wmma::row_major> vector_fragment;
        wmma::load_matrix_sync(
            state_fragment,
            state_tile + key_tile * kForgetMemoryTile,
            kForgetMemoryMaxWidth);
        wmma::load_matrix_sync(
            vector_fragment,
            vector_matrix + key_tile * kForgetMemoryTile * kForgetMemoryTile,
            kForgetMemoryTile);
        wmma::mma_sync(
            accumulator, state_fragment, vector_fragment, accumulator);
    }
    wmma::store_matrix_sync(
        product, accumulator, kForgetMemoryTile, wmma::mem_row_major);
}

// A CTA owns one recurrent memory.  Q/K are normalized once per token instead
// of once per value row.  16-row state tiles are read by BF16 Tensor Cores;
// the recurrence and saved state remain Float32 so long sequences do not
// accumulate a second storage-rounding error at every update.
__global__ void forget_memory_forward_tensor_core_bf16(
    const __nv_bfloat16* __restrict__ projected,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ states,
    float* __restrict__ state,
    int sequence,
    int projection_width,
    int key_width,
    int value_width,
    float retention_floor,
    int memory_variant) {
    constexpr int tile = kForgetMemoryTile;
    constexpr int max_width = kForgetMemoryMaxWidth;
    __shared__ __nv_bfloat16 state_tile[tile * max_width];
    __shared__ __nv_bfloat16 key_matrix[max_width * tile];
    __shared__ __nv_bfloat16 query_matrix[max_width * tile];
    __shared__ float product[tile * tile];
    __shared__ float normalized_key[max_width];
    __shared__ float normalized_query[max_width];
    __shared__ float predicted[max_width];
    __shared__ float recalled[max_width];
    __shared__ float retention[max_width];
    __shared__ float write_strength[max_width];
    __shared__ float values[max_width];

    const int batch = blockIdx.x;
    const int matrix_size = key_width * value_width;
    const int projected_batch = batch * sequence * projection_width;
    const int output_batch = batch * sequence * value_width;
    const int state_batch = batch * matrix_size;
    const int states_batch = batch * sequence * matrix_size;
    const int key_tiles = key_width / tile;
    const int value_tiles = value_width / tile;
    const bool use_v3 = memory_variant == 1;
    const bool use_drn = memory_variant == 2;

    for (int time = 0; time < sequence; ++time) {
        const int projected_offset = projected_batch + time * projection_width;
        const int key_offset = projected_offset + key_width;
        const int value_offset = key_offset + key_width;
        const int gate_offset = value_offset + value_width;
        const int beta_offset = gate_offset + value_width;

        if (threadIdx.x == 0) {
            float key_norm_squared = use_drn
                ? 1e-8f
                : use_v3 ? 1e-6f : (float)key_width;
            float query_norm_squared = use_drn ? 1e-8f : (float)key_width;
            for (int key = 0; key < key_width; ++key) {
                const float key_tanh = tanhf(__bfloat162float(
                    projected[key_offset + key]));
                const float query_tanh = tanhf(__bfloat162float(
                    projected[projected_offset + key]));
                normalized_key[key] = key_tanh;
                normalized_query[key] = query_tanh;
                if (use_v3 || use_drn)
                    key_norm_squared += key_tanh * key_tanh;
                if (use_drn)
                    query_norm_squared += query_tanh * query_tanh;
            }
            const float key_scale = rsqrtf(key_norm_squared);
            const float query_scale = rsqrtf(query_norm_squared);
            for (int key = 0; key < key_width; ++key) {
                normalized_key[key] *= key_scale;
                normalized_query[key] *= query_scale;
            }
        }
        for (int value_index = threadIdx.x; value_index < value_width;
             value_index += blockDim.x) {
            const float gate = 1.f / (1.f + expf(-__bfloat162float(
                projected[gate_offset + value_index])));
            const float row_retention = use_drn
                ? gate
                : retention_floor + (1.f - retention_floor) * gate;
            const float beta = 1.f / (1.f + expf(-__bfloat162float(
                projected[beta_offset + value_index])));
            retention[value_index] = row_retention;
            write_strength[value_index] = use_v3 || use_drn
                ? beta
                : (1.f - row_retention) * beta;
            values[value_index] = tanhf(__bfloat162float(
                projected[value_offset + value_index]));
        }
        __syncthreads();

        for (int index = threadIdx.x; index < key_width * tile;
             index += blockDim.x) {
            const int key = index / tile;
            const int column = index % tile;
            key_matrix[index] = __float2bfloat16_rn(
                column == 0 ? normalized_key[key] : 0.f);
            query_matrix[index] = __float2bfloat16_rn(
                column == 0 ? normalized_query[key] : 0.f);
        }
        __syncthreads();

        for (int value_tile = 0; value_tile < value_tiles; ++value_tile) {
            for (int index = threadIdx.x; index < tile * max_width;
                 index += blockDim.x) {
                const int row = index / max_width;
                const int key = index % max_width;
                const int value_index = value_tile * tile + row;
                state_tile[index] = key < key_width
                    ? __float2bfloat16_rn(
                        state[state_batch + value_index * key_width + key])
                    : __float2bfloat16_rn(0.f);
            }
            __syncthreads();
            if (threadIdx.x < kWarpSize) {
                forget_memory_tensor_core_matvec(
                    state_tile, key_matrix, product, key_tiles);
            }
            __syncthreads();
            if (threadIdx.x < tile) {
                const int value_index = value_tile * tile + threadIdx.x;
                predicted[value_index] = product[threadIdx.x * tile];
            }
            __syncthreads();
            if (use_drn) {
                if (threadIdx.x < kWarpSize) {
                    forget_memory_tensor_core_matvec(
                        state_tile, query_matrix, product, key_tiles);
                }
                __syncthreads();
                if (threadIdx.x < tile) {
                    const int value_index = value_tile * tile + threadIdx.x;
                    recalled[value_index] = product[threadIdx.x * tile];
                    output[output_batch + time * value_width + value_index] =
                        __float2bfloat16_rn(recalled[value_index]);
                }
                __syncthreads();
            }
        }

        for (int index = threadIdx.x; index < matrix_size;
             index += blockDim.x) {
            const int value_index = index / key_width;
            const int key = index % key_width;
            const float retained_prediction = use_v3
                ? retention[value_index] * predicted[value_index]
                : predicted[value_index];
            const float delta = write_strength[value_index]
                * (values[value_index] - retained_prediction);
            const float next = retention[value_index]
                    * state[state_batch + index]
                + delta * normalized_key[key];
            state[state_batch + index] = next;
            states[states_batch + time * matrix_size + index] = next;
        }
        __syncthreads();

        if (!use_drn) {
            for (int value_tile = 0; value_tile < value_tiles; ++value_tile) {
                for (int index = threadIdx.x; index < tile * max_width;
                     index += blockDim.x) {
                    const int row = index / max_width;
                    const int key = index % max_width;
                    const int value_index = value_tile * tile + row;
                    state_tile[index] = key < key_width
                        ? __float2bfloat16_rn(state[
                            state_batch + value_index * key_width + key])
                        : __float2bfloat16_rn(0.f);
                }
                __syncthreads();
                if (threadIdx.x < kWarpSize) {
                    forget_memory_tensor_core_matvec(
                        state_tile, query_matrix, product, key_tiles);
                }
                __syncthreads();
                if (threadIdx.x < tile) {
                    const int value_index = value_tile * tile + threadIdx.x;
                    output[output_batch + time * value_width + value_index] =
                        __float2bfloat16_rn(product[threadIdx.x * tile]);
                }
                __syncthreads();
            }
        }
    }
}

int gradient_blocks(int length) {
    return min(
        (length + kGradientBucketThreads - 1) / kGradientBucketThreads,
        512);
}

cudaError_t set_gradient_pipeline_device(int device) {
    return static_cast<cudaError_t>(nntrain_cuda_set_device(device));
}

struct gradient_host_pipeline {
    int source_device = -1;
    int destination_device = -1;
    int capacity = 0;
    void* host[2] = {nullptr, nullptr};
    __nv_bfloat16* device[2] = {nullptr, nullptr};
    cudaStream_t download_stream = nullptr;
    cudaStream_t upload_stream = nullptr;
    cudaEvent_t download_done[2] = {nullptr, nullptr};
    cudaEvent_t upload_done[2] = {nullptr, nullptr};
};

cudaError_t destroy_gradient_host_pipeline(
    gradient_host_pipeline* pipeline) {
    if (!pipeline)
        return cudaSuccess;
    cudaError_t first = cudaSuccess;
    auto retain = [&first](cudaError_t status) {
        if (first == cudaSuccess && status != cudaSuccess)
            first = status;
    };
    if (pipeline->source_device >= 0) {
        retain(set_gradient_pipeline_device(pipeline->source_device));
        if (pipeline->download_stream)
            retain(cudaStreamSynchronize(pipeline->download_stream));
        for (int slot = 0; slot < 2; ++slot) {
            if (pipeline->download_done[slot])
                retain(cudaEventDestroy(pipeline->download_done[slot]));
        }
        if (pipeline->download_stream)
            retain(cudaStreamDestroy(pipeline->download_stream));
    }
    if (pipeline->destination_device >= 0) {
        retain(set_gradient_pipeline_device(pipeline->destination_device));
        if (pipeline->upload_stream)
            retain(cudaStreamSynchronize(pipeline->upload_stream));
        for (int slot = 0; slot < 2; ++slot) {
            if (pipeline->upload_done[slot])
                retain(cudaEventDestroy(pipeline->upload_done[slot]));
            if (pipeline->device[slot])
                retain(cudaFree(pipeline->device[slot]));
        }
        if (pipeline->upload_stream)
            retain(cudaStreamDestroy(pipeline->upload_stream));
    }
    for (int slot = 0; slot < 2; ++slot) {
        if (pipeline->host[slot])
            retain(cudaFreeHost(pipeline->host[slot]));
    }
    delete pipeline;
    return first;
}
}

NNTRAIN_EXPORT int nntrain_forget_memory_forward_bf16_tensor_core(
    const __nv_bfloat16* projected,
    __nv_bfloat16* output,
    float* states,
    float* state,
    int batch,
    int sequence,
    int projection_width,
    int key_width,
    int value_width,
    float retention_floor,
    int memory_variant,
    cudaStream_t stream) {
    if (!projected || !output || !states || !state
        || batch <= 0 || sequence <= 0
        || key_width <= 0 || key_width > kForgetMemoryMaxWidth
        || value_width <= 0 || value_width > kForgetMemoryMaxWidth
        || key_width % kForgetMemoryTile
        || value_width % kForgetMemoryTile
        || projection_width != 2 * key_width + 3 * value_width) {
        return (int)cudaErrorInvalidValue;
    }
    forget_memory_forward_tensor_core_bf16<<<batch, 256, 0, stream>>>(
        projected, output, states, state,
        sequence, projection_width, key_width, value_width,
        retention_floor, memory_variant);
    return (int)cudaPeekAtLastError();
}

NNTRAIN_EXPORT int nntrain_flash_attention_forward(
    const float* qkv, float* output, float* log_sum_exp, int batch,
    int sequence, int model_width, int heads, int causal,
    cudaStream_t stream) {
    return launch_forward(
        qkv, output, log_sum_exp, batch, sequence, model_width, heads,
        causal, stream);
}

NNTRAIN_EXPORT int nntrain_flash_attention_backward(
    const float* qkv, const float* output, const float* output_gradient,
    const float* log_sum_exp, float* qkv_gradient, int batch,
    int sequence, int model_width, int heads, int causal,
    cudaStream_t stream) {
    return launch_backward(
        qkv, output, output_gradient, log_sum_exp, qkv_gradient,
        batch, sequence, model_width, heads, causal, stream);
}

NNTRAIN_EXPORT int nntrain_flash_attention_forward_bf16(
    const __nv_bfloat16* qkv, __nv_bfloat16* output,
    float* log_sum_exp, int batch, int sequence, int model_width,
    int heads, int causal, cudaStream_t stream) {
    return launch_forward(
        qkv, output, log_sum_exp, batch, sequence, model_width, heads,
        causal, stream);
}

NNTRAIN_EXPORT int nntrain_flash_attention_backward_bf16(
    const __nv_bfloat16* qkv, const __nv_bfloat16* output,
    const float* output_gradient, const float* log_sum_exp,
    float* qkv_gradient, int batch, int sequence, int model_width,
    int heads, int causal, cudaStream_t stream) {
    return launch_backward(
        qkv, output, output_gradient, log_sum_exp, qkv_gradient,
        batch, sequence, model_width, heads, causal, stream);
}

NNTRAIN_EXPORT int nntrain_flash_attention_forward_bf16_tensor_core(
    const __nv_bfloat16* qkv, __nv_bfloat16* output,
    float* log_sum_exp, int batch, int sequence, int model_width,
    int heads, int causal, cudaStream_t stream) {
    return launch_forward_tensor_core_bf16(
        qkv, output, log_sum_exp, batch, sequence, model_width, heads,
        causal, true, stream);
}

NNTRAIN_EXPORT int nntrain_flash_attention_forward_bf16_tensor_core_sync(
    const __nv_bfloat16* qkv, __nv_bfloat16* output,
    float* log_sum_exp, int batch, int sequence, int model_width,
    int heads, int causal, cudaStream_t stream) {
    return launch_forward_tensor_core_bf16(
        qkv, output, log_sum_exp, batch, sequence, model_width, heads,
        causal, false, stream);
}

NNTRAIN_EXPORT int nntrain_flash_attention_backward_bf16_tensor_core(
    const __nv_bfloat16* qkv, const __nv_bfloat16* output,
    const float* output_gradient, const float* log_sum_exp,
    float* row_delta, float* qkv_gradient, int batch, int sequence,
    int model_width, int heads, int causal, cudaStream_t stream) {
    return launch_backward_tensor_core_bf16(
        qkv, output, output_gradient, log_sum_exp, row_delta,
        qkv_gradient, batch, sequence, model_width, heads, causal,
        false, true, stream);
}

NNTRAIN_EXPORT int
nntrain_flash_attention_backward_bf16_tensor_core_parallel_dkv(
    const __nv_bfloat16* qkv, const __nv_bfloat16* output,
    const float* output_gradient, const float* log_sum_exp,
    float* row_delta, float* qkv_gradient, int batch, int sequence,
    int model_width, int heads, int causal, cudaStream_t stream) {
    return launch_backward_tensor_core_bf16(
        qkv, output, output_gradient, log_sum_exp, row_delta,
        qkv_gradient, batch, sequence, model_width, heads, causal,
        true, true, stream);
}

NNTRAIN_EXPORT int
nntrain_flash_attention_backward_bf16_tensor_core_bf16_gradient(
    const __nv_bfloat16* qkv, const __nv_bfloat16* output,
    const float* output_gradient, const float* log_sum_exp,
    float* row_delta, __nv_bfloat16* qkv_gradient, int batch,
    int sequence, int model_width, int heads, int causal,
    cudaStream_t stream) {
    return launch_backward_tensor_core_bf16(
        qkv, output, output_gradient, log_sum_exp, row_delta,
        qkv_gradient, batch, sequence, model_width, heads, causal,
        true, true, stream);
}

NNTRAIN_EXPORT int
nntrain_flash_attention_backward_bf16_tensor_core_bf16_io_gradient(
    const __nv_bfloat16* qkv, const __nv_bfloat16* output,
    const __nv_bfloat16* output_gradient, const float* log_sum_exp,
    float* row_delta, __nv_bfloat16* qkv_gradient, int batch,
    int sequence, int model_width, int heads, int causal,
    cudaStream_t stream) {
    return launch_backward_tensor_core_bf16(
        qkv, output, output_gradient, log_sum_exp, row_delta,
        qkv_gradient, batch, sequence, model_width, heads, causal,
        true, true, stream);
}

NNTRAIN_EXPORT int
nntrain_flash_attention_backward_bf16_tensor_core_bf16_io_gradient_sync(
    const __nv_bfloat16* qkv, const __nv_bfloat16* output,
    const __nv_bfloat16* output_gradient, const float* log_sum_exp,
    float* row_delta, __nv_bfloat16* qkv_gradient, int batch,
    int sequence, int model_width, int heads, int causal,
    cudaStream_t stream) {
    return launch_backward_tensor_core_bf16(
        qkv, output, output_gradient, log_sum_exp, row_delta,
        qkv_gradient, batch, sequence, model_width, heads, causal,
        true, false, stream);
}

NNTRAIN_EXPORT int nntrain_flash_attention_incremental_bf16(
    const __nv_bfloat16* qkv,
    __nv_bfloat16* key_cache,
    __nv_bfloat16* value_cache,
    __nv_bfloat16* output,
    int position,
    int cache_capacity,
    int model_width,
    int heads,
    cudaStream_t stream) {
    if (!qkv || !key_cache || !value_cache || !output || position < 0
        || position >= cache_capacity || cache_capacity <= 0
        || model_width <= 0 || heads <= 0 || model_width % heads != 0) {
        return (int)cudaErrorInvalidValue;
    }
    attention_incremental_bf16<<<
        heads,
        128,
        (size_t)(cache_capacity + 1) * sizeof(float),
        stream>>>(
            qkv, key_cache, value_cache, output, position, cache_capacity,
            model_width, heads);
    return (int)cudaPeekAtLastError();
}

NNTRAIN_EXPORT int nntrain_flash_attention_prefill_cache_bf16(
    const __nv_bfloat16* qkv,
    __nv_bfloat16* key_cache,
    __nv_bfloat16* value_cache,
    int sequence,
    int cache_capacity,
    int model_width,
    cudaStream_t stream) {
    if (!qkv || !key_cache || !value_cache || sequence <= 0
        || sequence > cache_capacity || model_width <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    const int length = sequence * model_width;
    attention_prefill_cache_bf16<<<
        (length + 255) / 256, 256, 0, stream>>>(
            qkv, key_cache, value_cache, length, model_width);
    return (int)cudaPeekAtLastError();
}

NNTRAIN_EXPORT int nntrain_layer_norm_forward(
    const float* input, const float* gamma, const float* beta,
    float* output, float* means, float* inverses,
    int rows, int columns, float epsilon, cudaStream_t stream) {
    return launch_layer_norm_forward<float, false>(
        input, nullptr, gamma, beta, output, means, inverses,
        rows, columns, epsilon, 0, 0, 1.f, stream);
}

NNTRAIN_EXPORT int nntrain_layer_norm_forward_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, __nv_bfloat16* output,
    float* means, float* inverses,
    int rows, int columns, float epsilon, cudaStream_t stream) {
    return launch_layer_norm_forward<__nv_bfloat16, false>(
        input, nullptr, gamma, beta, output, means, inverses,
        rows, columns, epsilon, 0, 0, 1.f, stream);
}

NNTRAIN_EXPORT int nntrain_layer_norm_backward(
    const float* input, const float* gamma, const float* means,
    const float* inverses,
    const float* output_gradient, float* input_gradient,
    float* gamma_gradient, float* beta_gradient, float* parameter_partials,
    int rows, int columns, cudaStream_t stream) {
    return launch_layer_norm_backward<float, false, float, float>(
        input, nullptr, gamma, means, inverses, output_gradient, input_gradient,
        nullptr, gamma_gradient, beta_gradient, parameter_partials,
        rows, columns, 0,
        0, 0, 1.f, stream);
}

NNTRAIN_EXPORT int nntrain_layer_norm_backward_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* gamma,
    const float* means, const float* inverses, const float* output_gradient,
    float* input_gradient, float* gamma_gradient, float* beta_gradient,
    float* parameter_partials,
    int rows, int columns, cudaStream_t stream) {
    return launch_layer_norm_backward<
        __nv_bfloat16, false, float, float>(
        input, nullptr, gamma, means, inverses, output_gradient, input_gradient,
        nullptr, gamma_gradient, beta_gradient, parameter_partials,
        rows, columns, 0,
        0, 0, 1.f, stream);
}

NNTRAIN_EXPORT int nntrain_residual_dropout_layer_norm_forward(
    const float* residual, const float* branch, const float* gamma,
    const float* beta, float* output, float* means, float* inverses,
    int rows, int columns, unsigned int seed, unsigned int drop_threshold,
    float dropout_scale, float epsilon, cudaStream_t stream) {
    return launch_layer_norm_forward<float, true>(
        residual, branch, gamma, beta, output, means, inverses,
        rows, columns, epsilon, seed, drop_threshold, dropout_scale, stream);
}

NNTRAIN_EXPORT int nntrain_residual_dropout_layer_norm_forward_bf16(
    const __nv_bfloat16* residual, const __nv_bfloat16* branch,
    const __nv_bfloat16* gamma, const __nv_bfloat16* beta,
    __nv_bfloat16* output, float* means, float* inverses,
    int rows, int columns, unsigned int seed, unsigned int drop_threshold,
    float dropout_scale, float epsilon, cudaStream_t stream) {
    return launch_layer_norm_forward<__nv_bfloat16, true>(
        residual, branch, gamma, beta, output, means, inverses,
        rows, columns, epsilon, seed, drop_threshold, dropout_scale, stream);
}

NNTRAIN_EXPORT int nntrain_residual_dropout_layer_norm_backward(
    const float* residual, const float* branch, const float* gamma,
    const float* means, const float* inverses,
    const float* output_gradient, float* residual_gradient,
    float* branch_gradient, float* gamma_gradient, float* beta_gradient,
    float* parameter_partials,
    int rows, int columns, int same_parent, unsigned int seed,
    unsigned int drop_threshold, float dropout_scale, cudaStream_t stream) {
    return launch_layer_norm_backward<float, true, float, float>(
        residual, branch, gamma, means, inverses, output_gradient,
        residual_gradient,
        branch_gradient, gamma_gradient, beta_gradient, parameter_partials,
        rows, columns,
        same_parent, seed, drop_threshold, dropout_scale, stream);
}

NNTRAIN_EXPORT int nntrain_residual_dropout_layer_norm_backward_bf16(
    const __nv_bfloat16* residual, const __nv_bfloat16* branch,
    const __nv_bfloat16* gamma, const float* means,
    const float* inverses, const float* output_gradient,
    float* residual_gradient, float* branch_gradient,
    float* gamma_gradient, float* beta_gradient,
    float* parameter_partials,
    int rows, int columns, int same_parent, unsigned int seed,
    unsigned int drop_threshold, float dropout_scale, cudaStream_t stream) {
    return launch_layer_norm_backward<
        __nv_bfloat16, true, float, float>(
        residual, branch, gamma, means, inverses, output_gradient,
        residual_gradient,
        branch_gradient, gamma_gradient, beta_gradient, parameter_partials,
        rows, columns,
        same_parent, seed, drop_threshold, dropout_scale, stream);
}

NNTRAIN_EXPORT int
nntrain_residual_dropout_layer_norm_backward_bf16_branch_gradient(
    const __nv_bfloat16* residual, const __nv_bfloat16* branch,
    const __nv_bfloat16* gamma, const float* means,
    const float* inverses, const float* output_gradient,
    float* residual_gradient, __nv_bfloat16* branch_gradient,
    float* gamma_gradient, float* beta_gradient,
    float* parameter_partials,
    int rows, int columns, unsigned int seed,
    unsigned int drop_threshold, float dropout_scale, cudaStream_t stream) {
    return launch_layer_norm_backward<
        __nv_bfloat16, true, float, __nv_bfloat16>(
        residual, branch, gamma, means, inverses, output_gradient,
        residual_gradient, branch_gradient, gamma_gradient, beta_gradient,
        parameter_partials, rows, columns, 0, seed, drop_threshold,
        dropout_scale, stream);
}

NNTRAIN_EXPORT int
nntrain_residual_dropout_layer_norm_backward_bf16_io_gradient(
    const __nv_bfloat16* residual, const __nv_bfloat16* branch,
    const __nv_bfloat16* gamma, const float* means,
    const float* inverses, const __nv_bfloat16* output_gradient,
    float* residual_gradient, __nv_bfloat16* branch_gradient,
    float* gamma_gradient, float* beta_gradient,
    float* parameter_partials,
    int rows, int columns, unsigned int seed,
    unsigned int drop_threshold, float dropout_scale, cudaStream_t stream) {
    return launch_layer_norm_backward<
        __nv_bfloat16, true, __nv_bfloat16, __nv_bfloat16>(
        residual, branch, gamma, means, inverses, output_gradient,
        residual_gradient, branch_gradient, gamma_gradient, beta_gradient,
        parameter_partials, rows, columns, 0, seed, drop_threshold,
        dropout_scale, stream);
}

NNTRAIN_EXPORT int nntrain_nekomuon_moments_stats(
    const float* gradient, float* fast, float* slow,
    float* fast_hat, float* slow_hat, float* stats, int length,
    float beta_fast, float beta_slow, float fast_correction,
    float slow_correction, cudaStream_t stream) {
    return launch_nekomuon_moments_stats<true>(
        gradient, fast, slow, fast_hat, slow_hat, stats, length,
        beta_fast, beta_slow, fast_correction, slow_correction, stream);
}

NNTRAIN_EXPORT int nntrain_nekomuon_moments_stats_compact(
    const float* gradient, float* fast, float* slow, float* stats,
    int length, float beta_fast, float beta_slow, float fast_correction,
    float slow_correction, cudaStream_t stream) {
    return launch_nekomuon_moments_stats<false>(
        gradient, fast, slow, nullptr, nullptr, stats, length,
        beta_fast, beta_slow, fast_correction, slow_correction, stream);
}

NNTRAIN_EXPORT int nntrain_gradient_comm_create(
    int device, cudaStream_t* stream) {
    if (!stream)
        return (int)cudaErrorInvalidValue;
    (void)device;
    return (int)cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking);
}

NNTRAIN_EXPORT int nntrain_gradient_event_create(
    int device, cudaEvent_t* event) {
    if (!event)
        return (int)cudaErrorInvalidValue;
    (void)device;
    return (int)cudaEventCreateWithFlags(event, cudaEventDisableTiming);
}

NNTRAIN_EXPORT int nntrain_gradient_pack_bf16(
    int device, const float* source, __nv_bfloat16* destination,
    int destination_offset, int length, cudaStream_t compute_stream) {
    if (!source || !destination || destination_offset < 0 || length <= 0)
        return (int)cudaErrorInvalidValue;
    (void)device;
    pack_gradient_bf16<<<
        gradient_blocks(length), kGradientBucketThreads, 0,
        compute_stream>>>(source, destination, destination_offset, length);
    return (int)cudaPeekAtLastError();
}

NNTRAIN_EXPORT int nntrain_gradient_record_ready(
    int device, cudaEvent_t event, cudaStream_t compute_stream) {
    (void)device;
    return (int)cudaEventRecord(event, compute_stream);
}

NNTRAIN_EXPORT int nntrain_gradient_exchange_bf16(
    int destination_device,
    int source_device,
    const __nv_bfloat16* local,
    const __nv_bfloat16* remote_source,
    __nv_bfloat16* remote_staging,
    float* reduced,
    int length,
    double* squared_sum,
    cudaStream_t communication_stream,
    cudaEvent_t local_ready,
    cudaEvent_t remote_ready) {
    if (!local || !remote_source || !remote_staging || !reduced
        || length <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    cudaError_t status = cudaStreamWaitEvent(
        communication_stream, local_ready, 0);
    if (status != cudaSuccess)
        return (int)status;
    status = cudaStreamWaitEvent(communication_stream, remote_ready, 0);
    if (status != cudaSuccess)
        return (int)status;
    status = cudaMemcpyPeerAsync(
        remote_staging,
        destination_device,
        remote_source,
        source_device,
        (size_t)length * sizeof(__nv_bfloat16),
        communication_stream);
    if (status != cudaSuccess)
        return (int)status;
    sum_gradient_bf16<<<
        gradient_blocks(length), kGradientBucketThreads, 0,
        communication_stream>>>(
            local, remote_staging, reduced, length, squared_sum);
    return (int)cudaPeekAtLastError();
}

NNTRAIN_EXPORT int nntrain_gradient_host_pipeline_create(
    int source_device,
    int destination_device,
    int chunk_elements,
    void** pipeline_pointer) {
    if (!pipeline_pointer || source_device == destination_device
        || chunk_elements <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    *pipeline_pointer = nullptr;
    gradient_host_pipeline* pipeline = new (std::nothrow)
        gradient_host_pipeline();
    if (!pipeline)
        return (int)cudaErrorMemoryAllocation;
    pipeline->source_device = source_device;
    pipeline->destination_device = destination_device;
    pipeline->capacity = chunk_elements;
    const size_t bytes =
        (size_t)chunk_elements * sizeof(__nv_bfloat16);
    cudaError_t status = set_gradient_pipeline_device(source_device);
    if (status == cudaSuccess) {
        status = cudaStreamCreateWithFlags(
            &pipeline->download_stream, cudaStreamNonBlocking);
    }
    for (int slot = 0; status == cudaSuccess && slot < 2; ++slot) {
        status = cudaEventCreateWithFlags(
            &pipeline->download_done[slot], cudaEventDisableTiming);
    }
    if (status == cudaSuccess)
        status = set_gradient_pipeline_device(destination_device);
    if (status == cudaSuccess) {
        status = cudaStreamCreateWithFlags(
            &pipeline->upload_stream, cudaStreamNonBlocking);
    }
    for (int slot = 0; status == cudaSuccess && slot < 2; ++slot) {
        status = cudaEventCreateWithFlags(
            &pipeline->upload_done[slot], cudaEventDisableTiming);
        if (status == cudaSuccess)
            status = cudaMalloc((void**)&pipeline->device[slot], bytes);
        if (status == cudaSuccess)
            status = cudaMallocHost(&pipeline->host[slot], bytes);
    }
    if (status != cudaSuccess) {
        (void)destroy_gradient_host_pipeline(pipeline);
        return (int)status;
    }
    *pipeline_pointer = pipeline;
    return (int)cudaSuccess;
}

NNTRAIN_EXPORT int nntrain_gradient_host_pipeline_exchange_bf16(
    void* pipeline_pointer,
    const __nv_bfloat16* local,
    const __nv_bfloat16* remote_source,
    float* reduced,
    int length,
    double* squared_sum,
    cudaEvent_t local_ready,
    cudaEvent_t remote_ready) {
    gradient_host_pipeline* pipeline =
        reinterpret_cast<gradient_host_pipeline*>(pipeline_pointer);
    if (!pipeline || !local || !remote_source || !reduced || length <= 0
        || !local_ready || !remote_ready) {
        return (int)cudaErrorInvalidValue;
    }

    cudaError_t status = set_gradient_pipeline_device(
        pipeline->source_device);
    if (status != cudaSuccess)
        return (int)status;
    status = cudaEventSynchronize(remote_ready);
    if (status != cudaSuccess)
        return (int)status;
    status = set_gradient_pipeline_device(pipeline->destination_device);
    if (status != cudaSuccess)
        return (int)status;
    status = cudaEventSynchronize(local_ready);
    if (status != cudaSuccess)
        return (int)status;

    const int chunks = (length + pipeline->capacity - 1)
        / pipeline->capacity;
    auto queue_download = [&](int chunk) -> cudaError_t {
        const int slot = chunk & 1;
        const int offset = chunk * pipeline->capacity;
        const int count = min(pipeline->capacity, length - offset);
        cudaError_t queue_status = set_gradient_pipeline_device(
            pipeline->source_device);
        if (queue_status != cudaSuccess)
            return queue_status;
        queue_status = cudaMemcpyAsync(
            pipeline->host[slot], remote_source + offset,
            (size_t)count * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToHost, pipeline->download_stream);
        if (queue_status != cudaSuccess)
            return queue_status;
        return cudaEventRecord(
            pipeline->download_done[slot], pipeline->download_stream);
    };

    status = queue_download(0);
    if (status == cudaSuccess && chunks > 1)
        status = queue_download(1);
    for (int chunk = 0; status == cudaSuccess && chunk < chunks; ++chunk) {
        const int slot = chunk & 1;
        const int offset = chunk * pipeline->capacity;
        const int count = min(pipeline->capacity, length - offset);
        status = set_gradient_pipeline_device(pipeline->source_device);
        if (status == cudaSuccess) {
            status = cudaEventSynchronize(
                pipeline->download_done[slot]);
        }
        if (status != cudaSuccess)
            break;
        status = set_gradient_pipeline_device(
            pipeline->destination_device);
        if (status == cudaSuccess) {
            status = cudaMemcpyAsync(
                pipeline->device[slot], pipeline->host[slot],
                (size_t)count * sizeof(__nv_bfloat16),
                cudaMemcpyHostToDevice, pipeline->upload_stream);
        }
        if (status == cudaSuccess) {
            sum_gradient_bf16<<<
                gradient_blocks(count), kGradientBucketThreads, 0,
                pipeline->upload_stream>>>(
                    local + offset, pipeline->device[slot],
                    reduced + offset, count, squared_sum);
            status = cudaPeekAtLastError();
        }
        if (status == cudaSuccess) {
            status = cudaEventRecord(
                pipeline->upload_done[slot], pipeline->upload_stream);
        }
        const int next = chunk + 2;
        if (status == cudaSuccess && next < chunks) {
            status = cudaEventSynchronize(pipeline->upload_done[slot]);
            if (status == cudaSuccess)
                status = queue_download(next);
        }
    }
    if (status == cudaSuccess) {
        status = set_gradient_pipeline_device(
            pipeline->destination_device);
    }
    if (status == cudaSuccess)
        status = cudaStreamSynchronize(pipeline->upload_stream);
    return (int)status;
}

NNTRAIN_EXPORT int nntrain_gradient_host_pipeline_destroy(
    void* pipeline_pointer) {
    return (int)destroy_gradient_host_pipeline(
        reinterpret_cast<gradient_host_pipeline*>(pipeline_pointer));
}

NNTRAIN_EXPORT int nntrain_gradient_unpack_float(
    int device, const float* source, int source_offset,
    float* destination, int length, cudaStream_t communication_stream) {
    if (!source || !destination || source_offset < 0 || length <= 0)
        return (int)cudaErrorInvalidValue;
    (void)device;
    unpack_gradient_float<<<
        gradient_blocks(length), kGradientBucketThreads, 0,
        communication_stream>>>(source, source_offset, destination, length);
    return (int)cudaPeekAtLastError();
}

NNTRAIN_EXPORT int nntrain_gradient_comm_synchronize(
    int device, cudaStream_t communication_stream) {
    (void)device;
    return (int)cudaStreamSynchronize(communication_stream);
}

NNTRAIN_EXPORT int nntrain_gradient_event_destroy(
    int device, cudaEvent_t event) {
    (void)device;
    return (int)cudaEventDestroy(event);
}

NNTRAIN_EXPORT int nntrain_gradient_comm_destroy(
    int device, cudaStream_t communication_stream) {
    (void)device;
    return (int)cudaStreamDestroy(communication_stream);
}
