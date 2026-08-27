#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cmath>

#if defined(_WIN32)
#define NNTRAIN_EXPORT extern "C" __declspec(dllexport)
#else
#define NNTRAIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

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
        const float g = chunk.gradient[index];
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
        float parameter = chunk.data[index];
        if (chunk.apply_weight_decay)
            parameter *= 1.f - learning_rate * weight_decay;
        parameter -= update_scale * first /
            (sqrtf(second) + scaled_epsilon);
        chunk.data[index] = parameter;
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

__global__ void symmetric_gram_kernel(const float* source,
    float* destination, int rows, int columns) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int length = rows * rows;
    if (linear >= length)
        return;
    int row = linear / rows;
    int other = linear - row * rows;
    float sum = 0.f;
    for (int column = 0; column < columns; ++column)
        sum += source[row * columns + column]
            * source[other * columns + column];
    destination[linear] = sum;
}

__global__ void newton_schulz_kernel(const float* source,
    const float* gram, const float* gram_squared, float* destination,
    int rows, int columns, float a, float b, float c) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= rows * columns)
        return;
    int row = linear / columns;
    int column = linear - row * columns;
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
    int row = blockIdx.x;
    if (row >= rows)
        return;
    __shared__ float maximum_values[kThreads];
    __shared__ float sum_values[kThreads];
    __shared__ float logit_values[kThreads];
    int offset = row * columns;
    float maximum = -FLT_MAX;
    float logit_sum = 0.f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        float value = load(logits, offset + column);
        maximum = fmaxf(maximum, value);
        logit_sum += value;
    }
    maximum_values[threadIdx.x] = maximum;
    logit_values[threadIdx.x] = logit_sum;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            maximum_values[threadIdx.x] = fmaxf(maximum_values[threadIdx.x],
                maximum_values[threadIdx.x + stride]);
            logit_values[threadIdx.x] += logit_values[threadIdx.x + stride];
        }
        __syncthreads();
    }
    maximum = maximum_values[0];
    float exponential_sum = 0.f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x)
        exponential_sum += expf(load(logits, offset + column) - maximum);
    sum_values[threadIdx.x] = exponential_sum;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sum_values[threadIdx.x] += sum_values[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x != 0)
        return;
    exponential_sum = sum_values[0];
    maxima[row] = maximum;
    inverse_sums[row] = 1.f / exponential_sum;
    int label = labels[row];
    if (label == ignore_index) {
        row_losses[row] = 0.f;
        return;
    }
    float normalizer = maximum + logf(exponential_sum);
    float nll = normalizer - load(logits, offset + label);
    float uniform = normalizer - logit_values[0] / columns;
    row_losses[row] = ((1.f - smoothing) * nll + smoothing * uniform)
        / valid_rows;
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
__global__ void cross_entropy_backward_kernel(const T* logits,
    const float* maxima, const float* inverse_sums, const int* labels,
    float* gradient, const float* upstream, int length, int columns,
    int ignore_index, int valid_rows, float smoothing) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= length)
        return;
    int row = linear / columns;
    int column = linear - row * columns;
    int label = labels[row];
    if (label == ignore_index)
        return;
    float probability = expf(load(logits, linear) - maxima[row])
        * inverse_sums[row];
    float target = smoothing / columns;
    if (column == label)
        target += 1.f - smoothing;
    gradient[linear] += upstream[0] / valid_rows * (probability - target);
}

template <typename T>
__global__ void cross_entropy_backward_bf16_output_kernel(const T* logits,
    const float* maxima, const float* inverse_sums, const int* labels,
    unsigned short* gradient, const float* upstream, int length, int columns,
    int ignore_index, int valid_rows, float smoothing) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= length)
        return;
    int row = linear / columns;
    int column = linear - row * columns;
    int label = labels[row];
    if (label == ignore_index) {
        bf16_store(gradient, linear, 0.f);
        return;
    }
    float probability = expf(load(logits, linear) - maxima[row])
        * inverse_sums[row];
    float target = smoothing / columns;
    if (column == label)
        target += 1.f - smoothing;
    bf16_store(gradient, linear,
        upstream[0] / valid_rows * (probability - target));
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
    NNTRAIN_LAUNCH_1D(symmetric_gram_kernel, length, source, destination,
        rows, columns);
}

NNTRAIN_EXPORT int nntrain_optimizer_newton_schulz(const float* source,
    const float* gram, const float* gram_squared, float* destination,
    int rows, int columns, float a, float b, float c) {
    int length = rows * columns;
    NNTRAIN_LAUNCH_1D(newton_schulz_kernel, length, source, gram,
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
    NNTRAIN_LAUNCH_1D(cross_entropy_backward_kernel<float>, length, logits,
        maxima, inverse_sums, labels, gradient, upstream, length, columns,
        ignore_index, valid_rows, smoothing);
}

NNTRAIN_EXPORT int nntrain_tensor_cross_entropy_backward_bf16(
    const unsigned short* logits, const float* maxima,
    const float* inverse_sums, const int* labels, float* gradient,
    const float* upstream, int length, int columns, int ignore_index,
    int valid_rows, float smoothing) {
    NNTRAIN_LAUNCH_1D(cross_entropy_backward_kernel<unsigned short>, length,
        logits, maxima, inverse_sums, labels, gradient, upstream, length,
        columns, ignore_index, valid_rows, smoothing);
}

NNTRAIN_EXPORT int nntrain_tensor_cross_entropy_backward_bf16_output(
    const unsigned short* logits, const float* maxima,
    const float* inverse_sums, const int* labels, unsigned short* gradient,
    const float* upstream, int length, int columns, int ignore_index,
    int valid_rows, float smoothing) {
    NNTRAIN_LAUNCH_1D(
        cross_entropy_backward_bf16_output_kernel<unsigned short>, length,
        logits, maxima, inverse_sums, labels, gradient, upstream, length,
        columns, ignore_index, valid_rows, smoothing);
}

NNTRAIN_EXPORT int nntrain_tensor_decode_bf16(
    const unsigned short* source, float* destination, int length) {
    NNTRAIN_LAUNCH_1D(
        decode_bf16_kernel, length, source, destination, length);
}
