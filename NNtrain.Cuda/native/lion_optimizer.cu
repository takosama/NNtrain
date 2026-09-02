#include "cuda_internal.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kThreads = 256;

struct LionFloatChunk {
    float* data;
    const float* gradient;
    float* momentum;
    unsigned short* physical_bf16;
    int offset;
    int length;
    int apply_weight_decay;
    int publish_bf16;
    int rank_one;
};

struct LionBFloat16Chunk {
    unsigned short* data;
    const unsigned short* gradient;
    unsigned short* momentum;
    int offset;
    int length;
    int apply_weight_decay;
    int rank_one;
};

struct LionMix8Block {
    float* master;
    const float* gradient;
    float* momentum;
    signed char* payload;
    float* scales;
    int offset;
    int length;
    int scale_index;
    int apply_weight_decay;
    int rank_one;
};

struct LionBfp8Tensor {
    signed char* data_payload;
    float* data_scale;
    const signed char* gradient_payload;
    const float* gradient_scale;
    signed char* momentum_payload;
    float* momentum_scale;
    int length;
    int apply_weight_decay;
    int rank_one;
};

__device__ __forceinline__ float load_bf16(
    const unsigned short* values,
    int index) {
    return __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(values)[index]);
}

__device__ __forceinline__ void store_bf16(
    unsigned short* values,
    int index,
    float value) {
    reinterpret_cast<__nv_bfloat16*>(values)[index] =
        __float2bfloat16_rn(value);
}

__device__ __forceinline__ float lion_sign(float value) {
    return value > 0.0f ? 1.0f : value < 0.0f ? -1.0f : value;
}

__device__ __forceinline__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    return value;
}

__device__ float block_max(float value) {
    __shared__ float warp_values[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_max(value);
    if (lane == 0)
        warp_values[warp] = value;
    __syncthreads();
    float result = threadIdx.x < (blockDim.x + 31) / 32
        ? warp_values[lane]
        : 0.0f;
    if (warp == 0)
        result = warp_max(result);
    __syncthreads();
    if (threadIdx.x == 0)
        warp_values[0] = result;
    __syncthreads();
    return warp_values[0];
}

__device__ __forceinline__ signed char quantize_bfp8(
    float value,
    float scale) {
    int quantized = __float2int_rn(value / scale);
    quantized = max(-127, min(127, quantized));
    return static_cast<signed char>(quantized);
}

__global__ void lion_float_kernel(
    const LionFloatChunk* chunks,
    int chunk_count,
    float beta1,
    float beta2,
    float learning_rate,
    float weight_decay,
    int decay_1d) {
    const int chunk_index = static_cast<int>(blockIdx.x);
    if (chunk_index >= chunk_count)
        return;
    const LionFloatChunk chunk = chunks[chunk_index];
    for (int local = threadIdx.x; local < chunk.length;
         local += blockDim.x) {
        const int index = chunk.offset + local;
        const float gradient = chunk.gradient[index];
        const float previous_momentum = chunk.momentum[index];
        const float direction = fmaf(
            beta1,
            previous_momentum,
            (1.0f - beta1) * gradient);
        float parameter = chunk.data[index];
        if (chunk.apply_weight_decay != 0
            || (decay_1d != 0 && chunk.rank_one != 0))
            parameter -= learning_rate * weight_decay * parameter;
        parameter -= learning_rate * lion_sign(direction);
        chunk.data[index] = parameter;
        chunk.momentum[index] = fmaf(
            beta2,
            previous_momentum,
            (1.0f - beta2) * gradient);
        if (chunk.publish_bf16 != 0)
            store_bf16(chunk.physical_bf16, index, parameter);
    }
}

__global__ void lion_bfloat16_kernel(
    const LionBFloat16Chunk* chunks,
    int chunk_count,
    float beta1,
    float beta2,
    float learning_rate,
    float weight_decay,
    int decay_1d) {
    const int chunk_index = static_cast<int>(blockIdx.x);
    if (chunk_index >= chunk_count)
        return;
    const LionBFloat16Chunk chunk = chunks[chunk_index];
    for (int local = threadIdx.x; local < chunk.length;
         local += blockDim.x) {
        const int index = chunk.offset + local;
        const float gradient = load_bf16(chunk.gradient, index);
        const float previous_momentum = load_bf16(chunk.momentum, index);
        const float direction = fmaf(
            beta1,
            previous_momentum,
            (1.0f - beta1) * gradient);
        float parameter = load_bf16(chunk.data, index);
        if (chunk.apply_weight_decay != 0
            || (decay_1d != 0 && chunk.rank_one != 0))
            parameter -= learning_rate * weight_decay * parameter;
        parameter -= learning_rate * lion_sign(direction);
        store_bf16(chunk.data, index, parameter);
        store_bf16(
            chunk.momentum,
            index,
            fmaf(beta2, previous_momentum, (1.0f - beta2) * gradient));
    }
}

__global__ void lion_mix8_kernel(
    const LionMix8Block* blocks,
    int block_count,
    float beta1,
    float beta2,
    float learning_rate,
    float weight_decay,
    int decay_1d,
    int* finite_status) {
    const int block_index = static_cast<int>(blockIdx.x);
    if (block_index >= block_count)
        return;
    const LionMix8Block block = blocks[block_index];
    float maximum = 0.0f;
    for (int local = threadIdx.x; local < block.length;
         local += blockDim.x) {
        const int index = block.offset + local;
        const float gradient = block.gradient[index];
        const float previous_momentum = block.momentum[index];
        const float direction = fmaf(
            beta1,
            previous_momentum,
            (1.0f - beta1) * gradient);
        float parameter = block.master[index];
        if (block.apply_weight_decay != 0
            || (decay_1d != 0 && block.rank_one != 0))
            parameter -= learning_rate * weight_decay * parameter;
        parameter -= learning_rate * lion_sign(direction);
        block.master[index] = parameter;
        block.momentum[index] = fmaf(
            beta2,
            previous_momentum,
            (1.0f - beta2) * gradient);
        if (!isfinite(parameter) || !isfinite(block.momentum[index])
            || !isfinite(direction))
            atomicExch(finite_status, 1);
        maximum = fmaxf(maximum, fabsf(parameter));
    }
    maximum = block_max(maximum);
    const float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
    if (threadIdx.x == 0)
        block.scales[block.scale_index] = scale;
    __syncthreads();
    for (int local = threadIdx.x; local < block.length;
         local += blockDim.x) {
        const int index = block.offset + local;
        block.payload[index] = quantize_bfp8(block.master[index], scale);
    }
}

// Tensor-wide BFP8 needs a global scale. A bounded two-scalar reduction per
// tensor lets all SMs scan large embeddings, then a second grid requantizes
// after the scale finalizer. The only FP32 scratch is four scalars per tensor:
// data/momentum maxima and the two old scales needed by the update pass.
__global__ void lion_bfp8_reduce_kernel(
    const LionBfp8Tensor* tensors,
    int tensor_count,
    float beta1,
    float beta2,
    float learning_rate,
    float weight_decay,
    int decay_1d,
    float* reduction,
    int* finite_status) {
    const int tensor_index = static_cast<int>(blockIdx.y);
    if (tensor_index >= tensor_count)
        return;
    const LionBfp8Tensor tensor = tensors[tensor_index];
    const int start = static_cast<int>(blockIdx.x) * 4096;
    if (start >= tensor.length)
        return;
    const int end = min(tensor.length, start + 4096);
    const float old_data_scale = *tensor.data_scale;
    const float gradient_scale = *tensor.gradient_scale;
    const float old_momentum_scale = *tensor.momentum_scale;
    if (!isfinite(old_data_scale) || old_data_scale <= 0.0f
        || !isfinite(gradient_scale) || gradient_scale <= 0.0f
        || !isfinite(old_momentum_scale) || old_momentum_scale <= 0.0f) {
        if (threadIdx.x == 0)
            atomicExch(finite_status, 1);
        return;
    }

    float momentum_maximum = 0.0f;
    float parameter_maximum = 0.0f;
    for (int index = start + threadIdx.x; index < end;
         index += blockDim.x) {
        const float gradient =
            static_cast<float>(tensor.gradient_payload[index])
            * gradient_scale;
        const float previous_momentum =
            static_cast<float>(tensor.momentum_payload[index])
            * old_momentum_scale;
        const float next_momentum = fmaf(
            beta2,
            previous_momentum,
            (1.0f - beta2) * gradient);
        if (!isfinite(next_momentum))
            atomicExch(finite_status, 1);
        momentum_maximum = fmaxf(momentum_maximum, fabsf(next_momentum));
        const float direction = fmaf(
            beta1,
            previous_momentum,
            (1.0f - beta1) * gradient);
        float parameter = static_cast<float>(tensor.data_payload[index])
            * old_data_scale;
        if (tensor.apply_weight_decay != 0
            || (decay_1d != 0 && tensor.rank_one != 0))
            parameter -= learning_rate * weight_decay * parameter;
        parameter -= learning_rate * lion_sign(direction);
        if (!isfinite(parameter) || !isfinite(direction))
            atomicExch(finite_status, 1);
        parameter_maximum = fmaxf(parameter_maximum, fabsf(parameter));
    }
    momentum_maximum = block_max(momentum_maximum);
    parameter_maximum = block_max(parameter_maximum);
    if (threadIdx.x == 0)
    {
        atomicMax(
            reinterpret_cast<int*>(reduction + tensor_index * 4),
            __float_as_int(parameter_maximum));
        atomicMax(
            reinterpret_cast<int*>(reduction + tensor_index * 4 + 1),
            __float_as_int(momentum_maximum));
    }
}

__global__ void lion_bfp8_finalize_scale_kernel(
    const LionBfp8Tensor* tensors,
    int tensor_count,
    float* reduction,
    int* finite_status) {
    const int tensor_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (tensor_index >= tensor_count)
        return;
    const LionBfp8Tensor tensor = tensors[tensor_index];
    const float old_data_scale = *tensor.data_scale;
    const float old_momentum_scale = *tensor.momentum_scale;
    if (!isfinite(old_data_scale) || old_data_scale <= 0.0f
        || !isfinite(old_momentum_scale) || old_momentum_scale <= 0.0f) {
        atomicExch(finite_status, 1);
    }
    reduction[tensor_index * 4 + 2] = old_data_scale;
    reduction[tensor_index * 4 + 3] = old_momentum_scale;
    const float data_maximum = reduction[tensor_index * 4];
    const float momentum_maximum = reduction[tensor_index * 4 + 1];
    *tensor.data_scale = data_maximum == 0.0f
        ? 1.0f
        : data_maximum / 127.0f;
    *tensor.momentum_scale = momentum_maximum == 0.0f
        ? 1.0f
        : momentum_maximum / 127.0f;
}

__global__ void lion_bfp8_update_kernel(
    const LionBfp8Tensor* tensors,
    int tensor_count,
    float beta1,
    float beta2,
    float learning_rate,
    float weight_decay,
    int decay_1d,
    const float* reduction,
    int* finite_status) {
    const int tensor_index = static_cast<int>(blockIdx.y);
    if (tensor_index >= tensor_count)
        return;
    const LionBfp8Tensor tensor = tensors[tensor_index];
    const int start = static_cast<int>(blockIdx.x) * 4096;
    if (start >= tensor.length)
        return;
    const int end = min(tensor.length, start + 4096);
    const float old_data_scale = reduction[tensor_index * 4 + 2];
    const float old_momentum_scale = reduction[tensor_index * 4 + 3];
    const float next_data_scale = *tensor.data_scale;
    const float next_momentum_scale = *tensor.momentum_scale;
    const float gradient_scale = *tensor.gradient_scale;
    const bool valid_scales = isfinite(old_data_scale)
        && old_data_scale > 0.0f
        && isfinite(old_momentum_scale)
        && old_momentum_scale > 0.0f
        && isfinite(gradient_scale)
        && gradient_scale > 0.0f;

    for (int index = start + threadIdx.x; index < end;
         index += blockDim.x) {
        if (!valid_scales) {
            tensor.momentum_payload[index] = 0;
            tensor.data_payload[index] = 0;
            continue;
        }
        const float gradient =
            static_cast<float>(tensor.gradient_payload[index])
            * gradient_scale;
        const float previous_momentum =
            static_cast<float>(tensor.momentum_payload[index])
            * old_momentum_scale;
        const float next_momentum = fmaf(
            beta2,
            previous_momentum,
            (1.0f - beta2) * gradient);
        const float direction = fmaf(
            beta1,
            previous_momentum,
            (1.0f - beta1) * gradient);
        float parameter = static_cast<float>(tensor.data_payload[index])
            * old_data_scale;
        if (tensor.apply_weight_decay != 0
            || (decay_1d != 0 && tensor.rank_one != 0))
            parameter -= learning_rate * weight_decay * parameter;
        parameter -= learning_rate * lion_sign(direction);
        if (!isfinite(parameter) || !isfinite(next_momentum)
            || !isfinite(direction)) {
            atomicExch(finite_status, 1);
            tensor.momentum_payload[index] = 0;
            tensor.data_payload[index] = 0;
            continue;
        }
        tensor.momentum_payload[index] =
            quantize_bfp8(next_momentum, next_momentum_scale);
        tensor.data_payload[index] =
            quantize_bfp8(parameter, next_data_scale);
    }
}

int validate_launch(
    int device,
    const void* descriptors,
    int descriptor_count,
    cudaStream_t stream) {
    // CUDA's legacy/default stream is represented by a null handle and is a
    // valid lane stream. Only descriptors and launch dimensions are required.
    if (device < 0 || descriptors == nullptr || descriptor_count <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    return nntrain::cuda::internal::select_device(device);
}

}  // namespace

NNTRAIN_EXPORT int nntrain_lion_multi_tensor_f32(
    int device,
    const LionFloatChunk* chunks,
    int chunk_count,
    float beta1,
    float beta2,
    float learning_rate,
    float weight_decay,
    int decay_1d,
    cudaStream_t stream) {
    int status = validate_launch(device, chunks, chunk_count, stream);
    if (status != static_cast<int>(cudaSuccess))
        return status;
    lion_float_kernel<<<chunk_count, kThreads, 0, stream>>>(
        chunks, chunk_count, beta1, beta2, learning_rate, weight_decay,
        decay_1d);
    return static_cast<int>(cudaPeekAtLastError());
}

NNTRAIN_EXPORT int nntrain_lion_multi_tensor_bf16(
    int device,
    const LionBFloat16Chunk* chunks,
    int chunk_count,
    float beta1,
    float beta2,
    float learning_rate,
    float weight_decay,
    int decay_1d,
    cudaStream_t stream) {
    int status = validate_launch(device, chunks, chunk_count, stream);
    if (status != static_cast<int>(cudaSuccess))
        return status;
    lion_bfloat16_kernel<<<chunk_count, kThreads, 0, stream>>>(
        chunks, chunk_count, beta1, beta2, learning_rate, weight_decay,
        decay_1d);
    return static_cast<int>(cudaPeekAtLastError());
}

NNTRAIN_EXPORT int nntrain_lion_multi_tensor_mix8(
    int device,
    const LionMix8Block* blocks,
    int block_count,
    float beta1,
    float beta2,
    float learning_rate,
    float weight_decay,
    int decay_1d,
    int* finite_status,
    cudaStream_t stream) {
    if (finite_status == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    int status = validate_launch(device, blocks, block_count, stream);
    if (status != static_cast<int>(cudaSuccess))
        return status;
    lion_mix8_kernel<<<block_count, kThreads, 0, stream>>>(
        blocks, block_count, beta1, beta2, learning_rate, weight_decay,
        decay_1d, finite_status);
    return static_cast<int>(cudaPeekAtLastError());
}

NNTRAIN_EXPORT int nntrain_lion_multi_tensor_bfp8(
    int device,
    const LionBfp8Tensor* tensors,
    int tensor_count,
    float beta1,
    float beta2,
    float learning_rate,
    float weight_decay,
    int decay_1d,
    float* reduction,
    int maximum_chunks,
    int* finite_status,
    cudaStream_t stream) {
    if (finite_status == nullptr || reduction == nullptr
        || maximum_chunks <= 0 || tensor_count > 65535)
        return static_cast<int>(cudaErrorInvalidValue);
    int status = validate_launch(device, tensors, tensor_count, stream);
    if (status != static_cast<int>(cudaSuccess))
        return status;
    cudaError_t cuda_status = cudaMemsetAsync(
        reduction,
        0,
        static_cast<size_t>(tensor_count) * 4 * sizeof(float),
        stream);
    if (cuda_status != cudaSuccess)
        return static_cast<int>(cuda_status);
    const dim3 grid(
        static_cast<unsigned int>(maximum_chunks),
        static_cast<unsigned int>(tensor_count));
    lion_bfp8_reduce_kernel<<<grid, kThreads, 0, stream>>>(
        tensors, tensor_count, beta1, beta2, learning_rate, weight_decay,
        decay_1d, reduction, finite_status);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess)
        return static_cast<int>(cuda_status);
    lion_bfp8_finalize_scale_kernel<<<
        (tensor_count + kThreads - 1) / kThreads,
        kThreads,
        0,
        stream>>>(tensors, tensor_count, reduction, finite_status);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess)
        return static_cast<int>(cuda_status);
    lion_bfp8_update_kernel<<<grid, kThreads, 0, stream>>>(
        tensors, tensor_count, beta1, beta2, learning_rate, weight_decay,
        decay_1d, reduction, finite_status);
    return static_cast<int>(cudaPeekAtLastError());
}
