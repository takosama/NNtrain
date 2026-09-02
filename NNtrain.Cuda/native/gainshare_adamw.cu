#include "cuda_internal.cuh"

#include <cuda_bf16.h>
#include <cmath>

namespace {

constexpr int kBlockSize = 256;

struct GainShareBfp8Tensor {
    signed char* data_payload;
    float* data_scale;
    const signed char* gradient_payload;
    const float* gradient_scale;
    signed char* first_payload;
    float* first_scale;
    signed char* second_payload;
    float* second_scale;
    float* direction;
    int length;
    int group_index;
    int apply_weight_decay;
    int rank_one;
};

__device__ __forceinline__ void record_non_finite(
    float value, int* finite_status) {
    if (finite_status != nullptr && !isfinite(value)) {
        atomicExch(finite_status, 1);
    }
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_sum(float value) {
    __shared__ float warp_totals[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) {
        warp_totals[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < (blockDim.x + 31) / 32
        ? warp_totals[lane]
        : 0.0f;
    if (warp == 0) {
        value = warp_sum(value);
    }
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(
            value,
            __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

__device__ __forceinline__ float block_max(float value) {
    __shared__ float warp_totals[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_max(value);
    if (lane == 0) {
        warp_totals[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < (blockDim.x + 31) / 32
        ? warp_totals[lane]
        : 0.0f;
    if (warp == 0) {
        value = warp_max(value);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        warp_totals[0] = value;
    }
    __syncthreads();
    return warp_totals[0];
}

__device__ __forceinline__ float bfp8_scale_from_maximum(float maximum) {
    return maximum == 0.0f ? 1.0f : __fdiv_rn(maximum, 127.0f);
}

__device__ __forceinline__ signed char quantize_bfp8(
    float value,
    float scale) {
    int code = __float2int_rn(__fdiv_rn(value, scale));
    code = max(-127, min(127, code));
    return static_cast<signed char>(code);
}

__global__ void prepare_fp32_kernel(
    const float* gradient,
    float* first,
    float* second,
    float* direction,
    float* group_stats,
    int group_index,
    int length,
    float beta1,
    float beta2,
    float inverse_bias_correction1,
    float inverse_bias_correction2,
    float epsilon,
    int* finite_status) {
    float alignment = 0.0f;
    float energy = 0.0f;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < length;
         index += blockDim.x * gridDim.x) {
        const float g = gradient[index];
        const float m = beta1 * first[index] + (1.0f - beta1) * g;
        const float v = beta2 * second[index] + (1.0f - beta2) * g * g;
        first[index] = m;
        second[index] = v;
        const float d = (m * inverse_bias_correction1)
            / (sqrtf(v * inverse_bias_correction2) + epsilon);
        direction[index] = d;
        alignment += g * d;
        energy += d * d;
        record_non_finite(g, finite_status);
        record_non_finite(m, finite_status);
        record_non_finite(v, finite_status);
        record_non_finite(d, finite_status);
    }
    alignment = block_sum(alignment);
    energy = block_sum(energy);
    if (threadIdx.x == 0) {
        atomicAdd(group_stats + group_index * 2, alignment);
        atomicAdd(group_stats + group_index * 2 + 1, energy);
    }
}

__global__ void prepare_bf16_kernel(
    const __nv_bfloat16* gradient,
    __nv_bfloat16* first,
    __nv_bfloat16* second,
    float* direction,
    float* group_stats,
    int group_index,
    int length,
    float beta1,
    float beta2,
    float inverse_bias_correction1,
    float inverse_bias_correction2,
    float epsilon,
    int* finite_status) {
    float alignment = 0.0f;
    float energy = 0.0f;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < length;
         index += blockDim.x * gridDim.x) {
        const float g = __bfloat162float(gradient[index]);
        const float m_value = beta1 * __bfloat162float(first[index])
            + (1.0f - beta1) * g;
        const float v_value = beta2 * __bfloat162float(second[index])
            + (1.0f - beta2) * g * g;
        const __nv_bfloat16 m_encoded = __float2bfloat16_rn(m_value);
        const __nv_bfloat16 v_encoded = __float2bfloat16_rn(v_value);
        first[index] = m_encoded;
        second[index] = v_encoded;
        // Pure BF16 state is authoritative at the kernel boundary.  Reload
        // the rounded values before producing the direction.
        const float m = __bfloat162float(m_encoded);
        const float v = __bfloat162float(v_encoded);
        const float d = (m * inverse_bias_correction1)
            / (sqrtf(v * inverse_bias_correction2) + epsilon);
        direction[index] = d;
        alignment += g * d;
        energy += d * d;
        record_non_finite(g, finite_status);
        record_non_finite(m, finite_status);
        record_non_finite(v, finite_status);
        record_non_finite(d, finite_status);
    }
    alignment = block_sum(alignment);
    energy = block_sum(energy);
    if (threadIdx.x == 0) {
        atomicAdd(group_stats + group_index * 2, alignment);
        atomicAdd(group_stats + group_index * 2 + 1, energy);
    }
}

__global__ void moments_fp32_kernel(
    const float* gradient,
    float* first,
    float* second,
    int length,
    float beta1,
    float beta2,
    int* finite_status) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < length;
         index += blockDim.x * gridDim.x) {
        const float g = gradient[index];
        const float m = beta1 * first[index] + (1.0f - beta1) * g;
        const float v = beta2 * second[index] + (1.0f - beta2) * g * g;
        first[index] = m;
        second[index] = v;
        record_non_finite(g, finite_status);
        record_non_finite(m, finite_status);
        record_non_finite(v, finite_status);
    }
}

__global__ void direction_fp32_kernel(
    const float* gradient,
    const float* first,
    const float* second,
    float* direction,
    float* group_stats,
    int group_index,
    int length,
    float inverse_bias_correction1,
    float inverse_bias_correction2,
    float epsilon,
    int* finite_status) {
    float alignment = 0.0f;
    float energy = 0.0f;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < length;
         index += blockDim.x * gridDim.x) {
        const float g = gradient[index];
        const float m = first[index];
        const float v = second[index];
        const float d = (m * inverse_bias_correction1)
            / (sqrtf(v * inverse_bias_correction2) + epsilon);
        direction[index] = d;
        alignment += g * d;
        energy += d * d;
        record_non_finite(d, finite_status);
    }
    alignment = block_sum(alignment);
    energy = block_sum(energy);
    if (threadIdx.x == 0) {
        atomicAdd(group_stats + group_index * 2, alignment);
        atomicAdd(group_stats + group_index * 2 + 1, energy);
    }
}

// Pure BFP8 keeps tensor-wide scales for data and both optimizer moments.
// The first pass recomputes the moment update directly from encoded storage
// and reduces the two new maxima without materializing FP32 tensors.
__global__ void bfp8_moment_max_kernel(
    const GainShareBfp8Tensor* tensors,
    int tensor_count,
    float beta1,
    float beta2,
    float* reduction,
    int* finite_status) {
    const int tensor_index = static_cast<int>(blockIdx.y);
    if (tensor_index >= tensor_count) {
        return;
    }
    const GainShareBfp8Tensor tensor = tensors[tensor_index];
    const int start = static_cast<int>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (static_cast<int>(blockIdx.x) * blockDim.x >= tensor.length) {
        return;
    }
    const int stride = blockDim.x * gridDim.x;
    const float gradient_scale = *tensor.gradient_scale;
    const float first_scale = *tensor.first_scale;
    const float second_scale = *tensor.second_scale;
    const bool valid_scales = isfinite(gradient_scale)
        && gradient_scale > 0.0f
        && isfinite(first_scale)
        && first_scale > 0.0f
        && isfinite(second_scale)
        && second_scale > 0.0f;
    if (!valid_scales && threadIdx.x == 0) {
        atomicExch(finite_status, 1);
    }

    float first_maximum = 0.0f;
    float second_maximum = 0.0f;
    if (valid_scales) {
        for (int index = start; index < tensor.length; index += stride) {
            const float gradient =
                static_cast<float>(tensor.gradient_payload[index])
                * gradient_scale;
            const float first = beta1
                * (static_cast<float>(tensor.first_payload[index])
                    * first_scale)
                + (1.0f - beta1) * gradient;
            const float second = beta2
                * (static_cast<float>(tensor.second_payload[index])
                    * second_scale)
                + (1.0f - beta2) * gradient * gradient;
            record_non_finite(gradient, finite_status);
            record_non_finite(first, finite_status);
            record_non_finite(second, finite_status);
            first_maximum = fmaxf(first_maximum, fabsf(first));
            second_maximum = fmaxf(second_maximum, fabsf(second));
        }
    }
    first_maximum = block_max(first_maximum);
    second_maximum = block_max(second_maximum);
    if (threadIdx.x == 0) {
        float* tensor_reduction = reduction + tensor_index * 6;
        atomicMax(
            reinterpret_cast<unsigned int*>(tensor_reduction),
            __float_as_uint(first_maximum));
        atomicMax(
            reinterpret_cast<unsigned int*>(tensor_reduction + 1),
            __float_as_uint(second_maximum));
    }
}

__global__ void bfp8_finalize_moment_scales_kernel(
    const GainShareBfp8Tensor* tensors,
    int tensor_count,
    float* reduction,
    int* finite_status) {
    const int tensor_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (tensor_index >= tensor_count) {
        return;
    }
    const GainShareBfp8Tensor tensor = tensors[tensor_index];
    float* tensor_reduction = reduction + tensor_index * 6;
    const float old_first_scale = *tensor.first_scale;
    const float old_second_scale = *tensor.second_scale;
    if (!isfinite(old_first_scale) || old_first_scale <= 0.0f
        || !isfinite(old_second_scale) || old_second_scale <= 0.0f) {
        atomicExch(finite_status, 1);
    }
    tensor_reduction[2] = old_first_scale;
    tensor_reduction[3] = old_second_scale;
    tensor_reduction[4] = 0.0f;
    *tensor.first_scale = bfp8_scale_from_maximum(tensor_reduction[0]);
    *tensor.second_scale = bfp8_scale_from_maximum(tensor_reduction[1]);
}

// Recompute the same moment expressions, publish the rounded BFP8 state, and
// immediately decode those bytes for the direction. This preserves the
// quantize -> re-decode optimizer-state boundary of the original path.
__global__ void bfp8_publish_moments_direction_kernel(
    const GainShareBfp8Tensor* tensors,
    int tensor_count,
    float inverse_bias_correction1,
    float inverse_bias_correction2,
    float beta1,
    float beta2,
    float epsilon,
    float* group_stats,
    const float* reduction,
    int* finite_status) {
    const int tensor_index = static_cast<int>(blockIdx.y);
    if (tensor_index >= tensor_count) {
        return;
    }
    const GainShareBfp8Tensor tensor = tensors[tensor_index];
    const int start = static_cast<int>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (static_cast<int>(blockIdx.x) * blockDim.x >= tensor.length) {
        return;
    }
    const int stride = blockDim.x * gridDim.x;
    const float* tensor_reduction = reduction + tensor_index * 6;
    const float gradient_scale = *tensor.gradient_scale;
    const float old_first_scale = tensor_reduction[2];
    const float old_second_scale = tensor_reduction[3];
    const float new_first_scale = *tensor.first_scale;
    const float new_second_scale = *tensor.second_scale;
    const bool valid_scales = isfinite(gradient_scale)
        && gradient_scale > 0.0f
        && isfinite(old_first_scale)
        && old_first_scale > 0.0f
        && isfinite(old_second_scale)
        && old_second_scale > 0.0f
        && isfinite(new_first_scale)
        && new_first_scale > 0.0f
        && isfinite(new_second_scale)
        && new_second_scale > 0.0f;

    float alignment = 0.0f;
    float energy = 0.0f;
    for (int index = start; index < tensor.length; index += stride) {
        if (!valid_scales) {
            tensor.first_payload[index] = 0;
            tensor.second_payload[index] = 0;
            tensor.direction[index] = 0.0f;
            continue;
        }
        const float gradient =
            static_cast<float>(tensor.gradient_payload[index])
            * gradient_scale;
        const float first = beta1
            * (static_cast<float>(tensor.first_payload[index])
                * old_first_scale)
            + (1.0f - beta1) * gradient;
        const float second = beta2
            * (static_cast<float>(tensor.second_payload[index])
                * old_second_scale)
            + (1.0f - beta2) * gradient * gradient;
        const signed char first_code = quantize_bfp8(
            first, new_first_scale);
        const signed char second_code = quantize_bfp8(
            second, new_second_scale);
        tensor.first_payload[index] = first_code;
        tensor.second_payload[index] = second_code;
        const float rounded_first =
            static_cast<float>(first_code) * new_first_scale;
        const float rounded_second =
            static_cast<float>(second_code) * new_second_scale;
        const float direction =
            (rounded_first * inverse_bias_correction1)
            / (sqrtf(rounded_second * inverse_bias_correction2) + epsilon);
        tensor.direction[index] = direction;
        alignment += gradient * direction;
        energy += direction * direction;
        record_non_finite(direction, finite_status);
    }
    alignment = block_sum(alignment);
    energy = block_sum(energy);
    if (threadIdx.x == 0) {
        atomicAdd(group_stats + tensor.group_index * 2, alignment);
        atomicAdd(group_stats + tensor.group_index * 2 + 1, energy);
    }
}

__global__ void bfp8_parameter_max_kernel(
    const GainShareBfp8Tensor* tensors,
    int tensor_count,
    const float* scales,
    float learning_rate,
    float weight_decay,
    int decay_1d,
    float* reduction,
    int* finite_status) {
    const int tensor_index = static_cast<int>(blockIdx.y);
    if (tensor_index >= tensor_count) {
        return;
    }
    const GainShareBfp8Tensor tensor = tensors[tensor_index];
    const int start = static_cast<int>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (static_cast<int>(blockIdx.x) * blockDim.x >= tensor.length) {
        return;
    }
    const int stride = blockDim.x * gridDim.x;
    const float old_data_scale = *tensor.data_scale;
    const float update_scale = learning_rate * scales[tensor.group_index];
    const float decay = 1.0f - learning_rate * weight_decay;
    const bool apply_weight_decay = tensor.apply_weight_decay != 0
        || (decay_1d != 0 && tensor.rank_one != 0);
    const bool valid_scale = isfinite(old_data_scale)
        && old_data_scale > 0.0f;
    if (!valid_scale && threadIdx.x == 0) {
        atomicExch(finite_status, 1);
    }

    float maximum = 0.0f;
    if (valid_scale) {
        for (int index = start; index < tensor.length; index += stride) {
            float value = static_cast<float>(tensor.data_payload[index])
                * old_data_scale;
            if (apply_weight_decay) {
                value *= decay;
            }
            value -= update_scale * tensor.direction[index];
            record_non_finite(value, finite_status);
            maximum = fmaxf(maximum, fabsf(value));
        }
    }
    maximum = block_max(maximum);
    if (threadIdx.x == 0) {
        atomicMax(
            reinterpret_cast<unsigned int*>(
                reduction + tensor_index * 6 + 4),
            __float_as_uint(maximum));
    }
}

__global__ void bfp8_finalize_data_scales_kernel(
    const GainShareBfp8Tensor* tensors,
    int tensor_count,
    float* reduction,
    int* finite_status) {
    const int tensor_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (tensor_index >= tensor_count) {
        return;
    }
    const GainShareBfp8Tensor tensor = tensors[tensor_index];
    float* tensor_reduction = reduction + tensor_index * 6;
    const float old_data_scale = *tensor.data_scale;
    if (!isfinite(old_data_scale) || old_data_scale <= 0.0f) {
        atomicExch(finite_status, 1);
    }
    tensor_reduction[5] = old_data_scale;
    *tensor.data_scale = bfp8_scale_from_maximum(tensor_reduction[4]);
}

__global__ void bfp8_publish_data_kernel(
    const GainShareBfp8Tensor* tensors,
    int tensor_count,
    const float* scales,
    float learning_rate,
    float weight_decay,
    int decay_1d,
    const float* reduction,
    int* finite_status) {
    const int tensor_index = static_cast<int>(blockIdx.y);
    if (tensor_index >= tensor_count) {
        return;
    }
    const GainShareBfp8Tensor tensor = tensors[tensor_index];
    const int start = static_cast<int>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (static_cast<int>(blockIdx.x) * blockDim.x >= tensor.length) {
        return;
    }
    const int stride = blockDim.x * gridDim.x;
    const float old_data_scale = reduction[tensor_index * 6 + 5];
    const float new_data_scale = *tensor.data_scale;
    const float update_scale = learning_rate * scales[tensor.group_index];
    const float decay = 1.0f - learning_rate * weight_decay;
    const bool apply_weight_decay = tensor.apply_weight_decay != 0
        || (decay_1d != 0 && tensor.rank_one != 0);
    const bool valid_scales = isfinite(old_data_scale)
        && old_data_scale > 0.0f
        && isfinite(new_data_scale)
        && new_data_scale > 0.0f;
    for (int index = start; index < tensor.length; index += stride) {
        if (!valid_scales) {
            tensor.data_payload[index] = 0;
            continue;
        }
        float value = static_cast<float>(tensor.data_payload[index])
            * old_data_scale;
        if (apply_weight_decay) {
            value *= decay;
        }
        value -= update_scale * tensor.direction[index];
        record_non_finite(value, finite_status);
        tensor.data_payload[index] = quantize_bfp8(
            value, new_data_scale);
    }
}

__global__ void compute_scales_kernel(
    const float* group_stats,
    float* alignment_ema,
    float* scales,
    int group_count,
    float rho,
    float gamma,
    float min_scale,
    float max_scale,
    float epsilon,
    int* finite_status) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    float total_energy = 0.0f;
    float weighted_alignment = 0.0f;
    for (int group = 0; group < group_count; ++group) {
        const float alignment = group_stats[group * 2];
        const float energy = group_stats[group * 2 + 1];
        const float ratio = fmaxf(alignment, 0.0f) / (energy + epsilon);
        const float previous = alignment_ema[group];
        const float smoothed = isnan(previous)
            ? ratio
            : rho * previous + (1.0f - rho) * ratio;
        alignment_ema[group] = smoothed;
        total_energy += energy;
        weighted_alignment += energy * smoothed;
        record_non_finite(smoothed, finite_status);
    }

    const float target = weighted_alignment / (total_energy + epsilon);
    float scaled_energy = 0.0f;
    const bool unit_scales = !isfinite(target) || target <= epsilon;
    for (int group = 0; group < group_count; ++group) {
        const float energy = group_stats[group * 2 + 1];
        float raw = 1.0f;
        if (!unit_scales) {
            const float relative = fmaxf(alignment_ema[group], 0.0f)
                / target;
            raw = fminf(max_scale, fmaxf(min_scale, powf(relative, gamma)));
        }
        scales[group] = raw;
        scaled_energy += raw * raw * energy;
    }
    const float normalization = total_energy > 0.0f
        ? sqrtf(total_energy / (scaled_energy + epsilon))
        : 1.0f;
    for (int group = 0; group < group_count; ++group) {
        scales[group] *= normalization;
        record_non_finite(scales[group], finite_status);
    }
}

__global__ void apply_fp32_kernel(
    float* data,
    const float* direction,
    const float* scales,
    __nv_bfloat16* bfloat16_output,
    int group_index,
    int length,
    float learning_rate,
    float weight_decay,
    int apply_weight_decay,
    int* finite_status) {
    const float update_scale = learning_rate * scales[group_index];
    const float decay = 1.0f - learning_rate * weight_decay;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < length;
         index += blockDim.x * gridDim.x) {
        float value = data[index];
        if (apply_weight_decay != 0) {
            value *= decay;
        }
        value -= update_scale * direction[index];
        data[index] = value;
        if (bfloat16_output != nullptr) {
            bfloat16_output[index] = __float2bfloat16_rn(value);
        }
        record_non_finite(value, finite_status);
    }
}

__global__ void apply_bf16_kernel(
    __nv_bfloat16* data,
    const float* direction,
    const float* scales,
    int group_index,
    int length,
    float learning_rate,
    float weight_decay,
    int apply_weight_decay,
    int* finite_status) {
    const float update_scale = learning_rate * scales[group_index];
    const float decay = 1.0f - learning_rate * weight_decay;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < length;
         index += blockDim.x * gridDim.x) {
        float value = __bfloat162float(data[index]);
        if (apply_weight_decay != 0) {
            value *= decay;
        }
        value -= update_scale * direction[index];
        data[index] = __float2bfloat16_rn(value);
        record_non_finite(value, finite_status);
    }
}

inline int block_count(int length) {
    const int required = (length + kBlockSize - 1) / kBlockSize;
    return required < 1 ? 1 : (required > 1024 ? 1024 : required);
}

}  // namespace

NNTRAIN_EXPORT int nntrain_gainshare_prepare_fp32(
    const float* gradient, float* first, float* second, float* direction,
    float* group_stats, int group_index, int length, float beta1,
    float beta2, float inverse_bias_correction1,
    float inverse_bias_correction2, float epsilon, int* finite_status,
    cudaStream_t stream) {
    prepare_fp32_kernel<<<block_count(length), kBlockSize, 0, stream>>>(
        gradient, first, second, direction, group_stats, group_index, length,
        beta1, beta2, inverse_bias_correction1,
        inverse_bias_correction2, epsilon, finite_status);
    return static_cast<int>(cudaPeekAtLastError());
}

NNTRAIN_EXPORT int nntrain_gainshare_prepare_bf16(
    const __nv_bfloat16* gradient, __nv_bfloat16* first,
    __nv_bfloat16* second, float* direction, float* group_stats,
    int group_index, int length, float beta1, float beta2,
    float inverse_bias_correction1, float inverse_bias_correction2,
    float epsilon, int* finite_status, cudaStream_t stream) {
    prepare_bf16_kernel<<<block_count(length), kBlockSize, 0, stream>>>(
        gradient, first, second, direction, group_stats, group_index, length,
        beta1, beta2, inverse_bias_correction1,
        inverse_bias_correction2, epsilon, finite_status);
    return static_cast<int>(cudaPeekAtLastError());
}

NNTRAIN_EXPORT int nntrain_gainshare_moments_fp32(
    const float* gradient, float* first, float* second, int length,
    float beta1, float beta2, int* finite_status, cudaStream_t stream) {
    moments_fp32_kernel<<<block_count(length), kBlockSize, 0, stream>>>(
        gradient, first, second, length, beta1, beta2, finite_status);
    return static_cast<int>(cudaPeekAtLastError());
}

NNTRAIN_EXPORT int nntrain_gainshare_direction_fp32(
    const float* gradient, const float* first, const float* second,
    float* direction, float* group_stats, int group_index, int length,
    float inverse_bias_correction1, float inverse_bias_correction2,
    float epsilon, int* finite_status, cudaStream_t stream) {
    direction_fp32_kernel<<<block_count(length), kBlockSize, 0, stream>>>(
        gradient, first, second, direction, group_stats, group_index, length,
        inverse_bias_correction1, inverse_bias_correction2, epsilon,
        finite_status);
    return static_cast<int>(cudaPeekAtLastError());
}

NNTRAIN_EXPORT int nntrain_gainshare_prepare_bfp8_multi_tensor(
    const GainShareBfp8Tensor* tensors,
    int tensor_count,
    int maximum_chunks,
    float* reduction,
    float* group_stats,
    float beta1,
    float beta2,
    float inverse_bias_correction1,
    float inverse_bias_correction2,
    float epsilon,
    int* finite_status,
    cudaStream_t stream) {
    if (tensors == nullptr || tensor_count <= 0 || tensor_count > 65535
        || maximum_chunks <= 0 || maximum_chunks > 1024
        || reduction == nullptr || group_stats == nullptr
        || finite_status == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaMemsetAsync(
        reduction,
        0,
        static_cast<size_t>(tensor_count) * 6 * sizeof(float),
        stream);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    const dim3 grid(
        static_cast<unsigned int>(maximum_chunks),
        static_cast<unsigned int>(tensor_count));
    bfp8_moment_max_kernel<<<grid, kBlockSize, 0, stream>>>(
        tensors, tensor_count, beta1, beta2, reduction, finite_status);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    bfp8_finalize_moment_scales_kernel<<<
        (tensor_count + kBlockSize - 1) / kBlockSize,
        kBlockSize,
        0,
        stream>>>(tensors, tensor_count, reduction, finite_status);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    bfp8_publish_moments_direction_kernel<<<
        grid, kBlockSize, 0, stream>>>(
            tensors, tensor_count, inverse_bias_correction1,
            inverse_bias_correction2, beta1, beta2, epsilon, group_stats,
            reduction, finite_status);
    return static_cast<int>(cudaPeekAtLastError());
}

NNTRAIN_EXPORT int nntrain_gainshare_apply_bfp8_multi_tensor(
    const GainShareBfp8Tensor* tensors,
    int tensor_count,
    int maximum_chunks,
    float* reduction,
    const float* scales,
    float learning_rate,
    float weight_decay,
    int decay_1d,
    int* finite_status,
    cudaStream_t stream) {
    if (tensors == nullptr || tensor_count <= 0 || tensor_count > 65535
        || maximum_chunks <= 0 || maximum_chunks > 1024
        || reduction == nullptr || scales == nullptr
        || finite_status == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const dim3 grid(
        static_cast<unsigned int>(maximum_chunks),
        static_cast<unsigned int>(tensor_count));
    bfp8_parameter_max_kernel<<<grid, kBlockSize, 0, stream>>>(
        tensors, tensor_count, scales, learning_rate, weight_decay, decay_1d,
        reduction, finite_status);
    cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    bfp8_finalize_data_scales_kernel<<<
        (tensor_count + kBlockSize - 1) / kBlockSize,
        kBlockSize,
        0,
        stream>>>(tensors, tensor_count, reduction, finite_status);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    bfp8_publish_data_kernel<<<grid, kBlockSize, 0, stream>>>(
        tensors, tensor_count, scales, learning_rate, weight_decay, decay_1d,
        reduction, finite_status);
    return static_cast<int>(cudaPeekAtLastError());
}

NNTRAIN_EXPORT int nntrain_gainshare_compute_scales(
    const float* group_stats, float* alignment_ema, float* scales,
    int group_count, float rho, float gamma, float min_scale,
    float max_scale, float epsilon, int* finite_status,
    cudaStream_t stream) {
    compute_scales_kernel<<<1, 1, 0, stream>>>(
        group_stats, alignment_ema, scales, group_count, rho, gamma,
        min_scale, max_scale, epsilon, finite_status);
    return static_cast<int>(cudaPeekAtLastError());
}

NNTRAIN_EXPORT int nntrain_gainshare_apply_fp32(
    float* data, const float* direction, const float* scales,
    __nv_bfloat16* bfloat16_output, int group_index, int length,
    float learning_rate, float weight_decay, int apply_weight_decay,
    int* finite_status, cudaStream_t stream) {
    apply_fp32_kernel<<<block_count(length), kBlockSize, 0, stream>>>(
        data, direction, scales, bfloat16_output, group_index, length,
        learning_rate, weight_decay, apply_weight_decay, finite_status);
    return static_cast<int>(cudaPeekAtLastError());
}

NNTRAIN_EXPORT int nntrain_gainshare_apply_bf16(
    __nv_bfloat16* data, const float* direction, const float* scales,
    int group_index, int length, float learning_rate, float weight_decay,
    int apply_weight_decay, int* finite_status, cudaStream_t stream) {
    apply_bf16_kernel<<<block_count(length), kBlockSize, 0, stream>>>(
        data, direction, scales, group_index, length, learning_rate,
        weight_decay, apply_weight_decay, finite_status);
    return static_cast<int>(cudaPeekAtLastError());
}
