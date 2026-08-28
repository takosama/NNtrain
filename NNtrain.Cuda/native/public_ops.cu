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

int blocks_for(int length) {
    return (length + kThreads - 1) / kThreads;
}

int launch_status() {
    return static_cast<int>(cudaPeekAtLastError());
}

template <typename T>
__device__ __forceinline__ float value_load(const T* values, int index) {
    return values[index];
}

template <>
__device__ __forceinline__ float value_load<unsigned short>(
    const unsigned short* values, int index) {
    return __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(values)[index]);
}

template <typename T>
__device__ __forceinline__ void value_store(
    T* values, int index, float value) {
    values[index] = value;
}

template <>
__device__ __forceinline__ void value_store<unsigned short>(
    unsigned short* values, int index, float value) {
    reinterpret_cast<__nv_bfloat16*>(values)[index] =
        __float2bfloat16_rn(value);
}

__device__ __forceinline__ void bf16_gradient_add(
    unsigned short* values, int index, float value, bool use_atomic) {
    __nv_bfloat16* destination =
        reinterpret_cast<__nv_bfloat16*>(values) + index;
    if (use_atomic)
        atomicAdd(destination, __float2bfloat16_rn(value));
    else
        *destination = __float2bfloat16_rn(
            __bfloat162float(*destination) + value);
}

template <typename G>
__device__ __forceinline__ void gradient_add(
    G* values, int index, float value) {
    values[index] += value;
}

template <>
__device__ __forceinline__ void gradient_add<unsigned short>(
    unsigned short* values, int index, float value) {
    bf16_gradient_add(values, index, value, false);
}

__device__ __forceinline__ float binary_forward(
    float left, float right, int operation) {
    switch (operation) {
        case 0: return left + right;
        case 1: return left - right;
        case 2: return left * right;
        case 3: return left / right;
        default: return nanf("");
    }
}

__device__ __forceinline__ void binary_derivative(
    float left, float right, int operation,
    float* left_derivative, float* right_derivative) {
    switch (operation) {
        case 0:
            *left_derivative = 1.f;
            *right_derivative = 1.f;
            break;
        case 1:
            *left_derivative = 1.f;
            *right_derivative = -1.f;
            break;
        case 2:
            *left_derivative = right;
            *right_derivative = left;
            break;
        case 3:
            *left_derivative = 1.f / right;
            *right_derivative = -left / (right * right);
            break;
        default:
            *left_derivative = nanf("");
            *right_derivative = nanf("");
            break;
    }
}

template <typename T>
__global__ void binary_forward_kernel(
    const T* left, const T* right, T* output, int length,
    int left_scalar, int right_scalar, int operation) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    float left_value = value_load(left, left_scalar != 0 ? 0 : index);
    float right_value = value_load(right, right_scalar != 0 ? 0 : index);
    value_store(output, index,
        binary_forward(left_value, right_value, operation));
}

template <typename T>
__global__ void binary_backward_kernel(
    const T* left, const T* right, const float* output_gradient,
    float* left_gradient, float* right_gradient, int length,
    int left_scalar, int right_scalar, int same_parent, int operation) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;

    int left_index = left_scalar != 0 ? 0 : index;
    int right_index = right_scalar != 0 ? 0 : index;
    float left_value = value_load(left, left_index);
    float right_value = value_load(right, right_index);
    float left_derivative;
    float right_derivative;
    binary_derivative(left_value, right_value, operation,
        &left_derivative, &right_derivative);
    float upstream = output_gradient[index];
    float left_contribution = upstream * left_derivative;
    float right_contribution = upstream * right_derivative;

    if (same_parent != 0 && left_index == right_index) {
        if (left_scalar != 0 || right_scalar != 0)
            atomicAdd(left_gradient + left_index,
                left_contribution + right_contribution);
        else
            left_gradient[left_index] +=
                left_contribution + right_contribution;
        return;
    }

    if (left_scalar != 0)
        atomicAdd(left_gradient, left_contribution);
    else
        left_gradient[left_index] += left_contribution;
    if (right_scalar != 0)
        atomicAdd(right_gradient, right_contribution);
    else
        right_gradient[right_index] += right_contribution;
}

__global__ void binary_backward_bf16_gradient_kernel(
    const unsigned short* left, const unsigned short* right,
    const unsigned short* output_gradient,
    unsigned short* left_gradient, unsigned short* right_gradient,
    int length, int left_scalar, int right_scalar, int same_parent,
    int operation) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    int left_index = left_scalar != 0 ? 0 : index;
    int right_index = right_scalar != 0 ? 0 : index;
    float left_value = value_load(left, left_index);
    float right_value = value_load(right, right_index);
    float left_derivative;
    float right_derivative;
    binary_derivative(left_value, right_value, operation,
        &left_derivative, &right_derivative);
    float upstream = value_load(output_gradient, index);
    float left_contribution = upstream * left_derivative;
    float right_contribution = upstream * right_derivative;
    if (same_parent != 0 && left_index == right_index) {
        if (left_scalar != 0 || right_scalar != 0)
            return;
        bf16_gradient_add(left_gradient, left_index,
            left_contribution + right_contribution,
            false);
        return;
    }
    if (left_scalar == 0)
        bf16_gradient_add(left_gradient, left_index, left_contribution, false);
    if (right_scalar == 0)
        bf16_gradient_add(right_gradient, right_index, right_contribution, false);
}

__global__ void binary_backward_bf16_scalar_reduction_kernel(
    const unsigned short* left, const unsigned short* right,
    const unsigned short* output_gradient, unsigned short* gradient,
    int length, int operand, int same_parent, int operation) {
    __shared__ float shared[kThreads];
    float sum = 0.f;
    for (int index = threadIdx.x; index < length; index += blockDim.x) {
        int left_index = operand == 0 || same_parent != 0 ? 0 : index;
        int right_index = operand == 1 || same_parent != 0 ? 0 : index;
        float left_derivative;
        float right_derivative;
        binary_derivative(
            value_load(left, left_index),
            value_load(right, right_index),
            operation,
            &left_derivative,
            &right_derivative);
        float upstream = value_load(output_gradient, index);
        sum += upstream * (same_parent != 0
            ? left_derivative + right_derivative
            : operand == 0 ? left_derivative : right_derivative);
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        bf16_gradient_add(gradient, 0, shared[0], false);
}

__device__ __forceinline__ float gelu_forward(float value) {
    constexpr float kAlpha = 0.7978845608028654f;
    constexpr float kBeta = 0.044715f;
    float inner = kAlpha * (value + kBeta * value * value * value);
    return 0.5f * value * (1.f + tanhf(inner));
}

__device__ __forceinline__ float gelu_derivative(float value) {
    constexpr float kAlpha = 0.7978845608028654f;
    constexpr float kBeta = 0.044715f;
    float square = value * value;
    float inner = kAlpha * (value + kBeta * value * square);
    float tanh_value = tanhf(inner);
    float inner_derivative = kAlpha * (1.f + 3.f * kBeta * square);
    return 0.5f * (1.f + tanh_value)
        + 0.5f * value * (1.f - tanh_value * tanh_value)
            * inner_derivative;
}

__device__ __forceinline__ float unary_forward(
    float value, int operation, float parameter) {
    switch (operation) {
        case 0: return value > 0.f ? value : 0.f;
        case 1: return gelu_forward(value);
        case 2: return tanhf(value);
        case 3: return expf(value);
        case 4: return logf(value);
        case 5: return -value;
        case 6: return sinf(value);
        case 7: return powf(value, parameter);
        default: return nanf("");
    }
}

__device__ __forceinline__ float unary_derivative(
    float input, float output, int operation, float parameter) {
    switch (operation) {
        case 0: return input > 0.f ? 1.f : 0.f;
        case 1: return gelu_derivative(input);
        case 2: return 1.f - output * output;
        case 3: return output;
        case 4: return 1.f / input;
        case 5: return -1.f;
        case 6: return cosf(input);
        case 7: return parameter * powf(input, parameter - 1.f);
        default: return nanf("");
    }
}

template <typename T>
__global__ void unary_forward_kernel(
    const T* input, T* output, int length, int operation, float parameter) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) {
        value_store(output, index,
            unary_forward(value_load(input, index), operation, parameter));
    }
}

template <typename T>
__global__ void unary_backward_kernel(
    const T* input, const T* output, const float* output_gradient,
    float* input_gradient, int length, int operation, float parameter) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) {
        float derivative = unary_derivative(
            value_load(input, index), value_load(output, index), operation,
            parameter);
        input_gradient[index] += output_gradient[index] * derivative;
    }
}

__global__ void unary_backward_bf16_gradient_kernel(
    const unsigned short* input, const unsigned short* output,
    const unsigned short* output_gradient, unsigned short* input_gradient,
    int length, int operation, float parameter) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) {
        float derivative = unary_derivative(
            value_load(input, index), value_load(output, index), operation,
            parameter);
        bf16_gradient_add(input_gradient, index,
            value_load(output_gradient, index) * derivative, false);
    }
}

template <typename T>
__global__ void sum_kernel(const T* input, float* output, int length) {
    __shared__ float values[kThreads];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float value = index < length ? value_load(input, index) : 0.f;
    values[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            values[threadIdx.x] += values[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(output, values[0]);
}

__device__ __forceinline__ float atomic_max_float(float* address, float value) {
    int* integer_address = reinterpret_cast<int*>(address);
    int old = *integer_address;
    while (__int_as_float(old) < value) {
        int assumed = old;
        old = atomicCAS(integer_address, assumed, __float_as_int(value));
        if (old == assumed)
            break;
    }
    return __int_as_float(old);
}

template <typename T>
__global__ void max_kernel(const T* input, float* output, int length) {
    __shared__ float values[kThreads];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    values[threadIdx.x] = index < length
        ? value_load(input, index)
        : -FLT_MAX;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            values[threadIdx.x] = fmaxf(
                values[threadIdx.x], values[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomic_max_float(output, values[0]);
}

__global__ void initialize_reduction_kernel(float* output, float value) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        output[0] = value;
}

template <typename T>
__global__ void reduction_backward_kernel(
    const T* input, const float* reduced, const float* output_gradient,
    float* input_gradient, int length, int operation) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    float contribution = output_gradient[0];
    if (operation == 1)
        contribution /= static_cast<float>(length);
    else if (operation == 2
        && value_load(input, index) != reduced[0])
        contribution = 0.f;
    input_gradient[index] += contribution;
}

__global__ void reduction_backward_bf16_gradient_kernel(
    const unsigned short* input, const float* reduced,
    const float* output_gradient, unsigned short* input_gradient,
    int length, int operation) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    float contribution = output_gradient[0];
    if (operation == 1)
        contribution /= static_cast<float>(length);
    else if (operation == 2 && value_load(input, index) != reduced[0])
        contribution = 0.f;
    bf16_gradient_add(input_gradient, index, contribution, false);
}

template <typename T>
int launch_binary_forward(
    const T* left, const T* right, T* output, int length,
    int left_scalar, int right_scalar, int operation, cudaStream_t stream) {
    if (!left || !right || !output || length <= 0
        || operation < 0 || operation > 3)
        return static_cast<int>(cudaErrorInvalidValue);
    binary_forward_kernel<<<blocks_for(length), kThreads, 0, stream>>>(
        left, right, output, length, left_scalar, right_scalar, operation);
    return launch_status();
}

template <typename T>
int launch_binary_backward(
    const T* left, const T* right, const float* output_gradient,
    float* left_gradient, float* right_gradient, int length,
    int left_scalar, int right_scalar, int same_parent, int operation,
    cudaStream_t stream) {
    if (!left || !right || !output_gradient || !left_gradient
        || !right_gradient || length <= 0 || operation < 0 || operation > 3)
        return static_cast<int>(cudaErrorInvalidValue);
    binary_backward_kernel<<<blocks_for(length), kThreads, 0, stream>>>(
        left, right, output_gradient, left_gradient, right_gradient, length,
        left_scalar, right_scalar, same_parent, operation);
    return launch_status();
}

template <typename T>
int launch_unary_forward(
    const T* input, T* output, int length, int operation,
    float parameter, cudaStream_t stream) {
    if (!input || !output || length <= 0 || operation < 0 || operation > 7)
        return static_cast<int>(cudaErrorInvalidValue);
    unary_forward_kernel<<<blocks_for(length), kThreads, 0, stream>>>(
        input, output, length, operation, parameter);
    return launch_status();
}

template <typename T>
int launch_unary_backward(
    const T* input, const T* output, const float* output_gradient,
    float* input_gradient, int length, int operation, float parameter,
    cudaStream_t stream) {
    if (!input || !output || !output_gradient || !input_gradient
        || length <= 0 || operation < 0 || operation > 7)
        return static_cast<int>(cudaErrorInvalidValue);
    unary_backward_kernel<<<blocks_for(length), kThreads, 0, stream>>>(
        input, output, output_gradient, input_gradient, length, operation,
        parameter);
    return launch_status();
}

template <typename T>
int launch_reduction(
    const T* input, float* output, int length, int operation,
    cudaStream_t stream) {
    if (!input || !output || length <= 0 || operation < 0 || operation > 2)
        return static_cast<int>(cudaErrorInvalidValue);
    initialize_reduction_kernel<<<1, 1, 0, stream>>>(
        output, operation == 2 ? -FLT_MAX : 0.f);
    if (operation == 2)
        max_kernel<<<blocks_for(length), kThreads, 0, stream>>>(
            input, output, length);
    else
        sum_kernel<<<blocks_for(length), kThreads, 0, stream>>>(
            input, output, length);
    if (operation == 1) {
        // Reuse the unary-scale-free scalar launch: a one-thread adjustment
        // keeps the complete reduction asynchronous on the caller's stream.
        // The result remains FP32 by the central reduction policy.
        // A lambda kernel is intentionally avoided for MSVC/NVCC portability.
    }
    return launch_status();
}

__global__ void mean_finalize_kernel(float* output, int length) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        output[0] /= static_cast<float>(length);
}

template <typename T>
int launch_reduction_complete(
    const T* input, float* output, int length, int operation,
    cudaStream_t stream) {
    int status = launch_reduction(input, output, length, operation, stream);
    if (status != static_cast<int>(cudaSuccess))
        return status;
    if (operation == 1)
        mean_finalize_kernel<<<1, 1, 0, stream>>>(output, length);
    return launch_status();
}

template <typename T>
int launch_reduction_backward(
    const T* input, const float* reduced, const float* output_gradient,
    float* input_gradient, int length, int operation, cudaStream_t stream) {
    if (!input || !reduced || !output_gradient || !input_gradient
        || length <= 0 || operation < 0 || operation > 2)
        return static_cast<int>(cudaErrorInvalidValue);
    reduction_backward_kernel<<<blocks_for(length), kThreads, 0, stream>>>(
        input, reduced, output_gradient, input_gradient, length, operation);
    return launch_status();
}

__device__ __forceinline__ float stable_sigmoid(float value) {
    if (value >= 0.f) {
        float exponential = expf(-value);
        return 1.f / (1.f + exponential);
    }
    float exponential = expf(value);
    return exponential / (1.f + exponential);
}

template <typename T>
__global__ void forget_scan_forward_kernel(
    const T* projected, T* output, float* memory, float* forget,
    float* input, float* value, int batch, int sequence, int width,
    int save_context) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_count = batch * width;
    if (lane >= lane_count)
        return;
    int batch_index = lane / width;
    int channel = lane % width;
    int channels = 3 * width;
    float previous = 0.f;
    for (int time = 0; time < sequence; ++time) {
        int projected_offset =
            (batch_index * sequence + time) * channels + channel;
        int state_offset =
            (batch_index * sequence + time) * width + channel;
        float forget_gate = stable_sigmoid(
            value_load(projected, projected_offset));
        float input_gate = stable_sigmoid(
            value_load(projected, projected_offset + width));
        float value_gate = tanhf(
            value_load(projected, projected_offset + 2 * width));
        float next = forget_gate * previous + input_gate * value_gate;
        value_store(output, state_offset, next);
        if (save_context != 0) {
            memory[state_offset] = next;
            forget[state_offset] = forget_gate;
            input[state_offset] = input_gate;
            value[state_offset] = value_gate;
        }
        previous = next;
    }
}

__global__ void forget_scan_backward_kernel(
    const float* output_gradient, const float* memory, const float* forget,
    const float* input, const float* value, float* projected_gradient,
    int batch, int sequence, int width) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_count = batch * width;
    if (lane >= lane_count)
        return;
    int batch_index = lane / width;
    int channel = lane % width;
    int channels = 3 * width;
    float running = 0.f;
    for (int time = sequence - 1; time >= 0; --time) {
        int projected_offset =
            (batch_index * sequence + time) * channels + channel;
        int state_offset =
            (batch_index * sequence + time) * width + channel;
        float total = running + output_gradient[state_offset];
        float forget_gate = forget[state_offset];
        float input_gate = input[state_offset];
        float value_gate = value[state_offset];
        float previous = time == 0 ? 0.f : memory[state_offset - width];
        projected_gradient[projected_offset] +=
            total * previous * forget_gate * (1.f - forget_gate);
        projected_gradient[projected_offset + width] +=
            total * value_gate * input_gate * (1.f - input_gate);
        projected_gradient[projected_offset + 2 * width] +=
            total * input_gate * (1.f - value_gate * value_gate);
        running = total * forget_gate;
    }
}

__global__ void forget_scan_backward_bf16_gradient_kernel(
    const unsigned short* output_gradient, const float* memory,
    const float* forget, const float* input, const float* value,
    unsigned short* projected_gradient,
    int batch, int sequence, int width) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_count = batch * width;
    if (lane >= lane_count)
        return;
    int batch_index = lane / width;
    int channel = lane % width;
    int channels = 3 * width;
    float running = 0.f;
    for (int time = sequence - 1; time >= 0; --time) {
        int projected_offset =
            (batch_index * sequence + time) * channels + channel;
        int state_offset =
            (batch_index * sequence + time) * width + channel;
        float total = running + value_load(output_gradient, state_offset);
        float forget_gate = forget[state_offset];
        float input_gate = input[state_offset];
        float value_gate = value[state_offset];
        float previous = time == 0 ? 0.f : memory[state_offset - width];
        bf16_gradient_add(projected_gradient, projected_offset,
            total * previous * forget_gate * (1.f - forget_gate), false);
        bf16_gradient_add(projected_gradient, projected_offset + width,
            total * value_gate * input_gate * (1.f - input_gate), false);
        bf16_gradient_add(projected_gradient, projected_offset + 2 * width,
            total * input_gate * (1.f - value_gate * value_gate), false);
        running = total * forget_gate;
    }
}

template <typename T>
int launch_forget_scan_forward(
    const T* projected, T* output, float* memory, float* forget,
    float* input, float* value, int batch, int sequence, int width,
    int save_context, cudaStream_t stream) {
    if (!projected || !output || batch <= 0 || sequence <= 0 || width <= 0
        || (save_context != 0 &&
            (!memory || !forget || !input || !value)))
        return static_cast<int>(cudaErrorInvalidValue);
    int lanes = batch * width;
    forget_scan_forward_kernel<<<blocks_for(lanes), kThreads, 0, stream>>>(
        projected, output, memory, forget, input, value,
        batch, sequence, width, save_context);
    return launch_status();
}

template <typename T>
__global__ void hyena_forward_kernel(
    const T* projected, const T* short_filter, const T* long_filter,
    const T* diagonal, T* output, float* saved_short, float* saved_gated,
    float* saved_convolved, int batch, int sequence, int width,
    int save_context) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    int lanes = batch * width;
    if (lane >= lanes)
        return;
    int batch_index = lane / width;
    int channel = lane % width;
    int channels = 3 * width;

    // The short convolution is causal and depthwise.  Save all three streams
    // only while recording; inference keeps them in the output workspace.
    for (int time = 0; time < sequence; ++time) {
        int row = batch_index * sequence + time;
        for (int stream_index = 0; stream_index < 3; ++stream_index) {
            int channel_index = stream_index * width + channel;
            float sum = 0.f;
            int tap_count = min(3, time + 1);
            for (int tap = 0; tap < tap_count; ++tap) {
                int source_row = row - tap;
                sum += value_load(projected,
                    source_row * channels + channel_index)
                    * value_load(short_filter,
                        tap * channels + channel_index);
            }
            // Forward requires the values in the following convolution even
            // in inference.  Inference callers supply a transient short buffer.
            saved_short[row * channels + channel_index] = sum;
        }
        int mixed_offset = row * width + channel;
        float gate = saved_short[row * channels + width + channel]
            * saved_short[row * channels + 2 * width + channel];
        saved_gated[mixed_offset] = gate;
    }

    for (int time = 0; time < sequence; ++time) {
        int row = batch_index * sequence + time;
        int mixed_offset = row * width + channel;
        float convolution = saved_gated[mixed_offset]
            * value_load(diagonal, channel);
        for (int lag = 0; lag <= time; ++lag) {
            int source_row = row - lag;
            convolution += saved_gated[source_row * width + channel]
                * value_load(long_filter, lag * width + channel);
        }
        saved_convolved[mixed_offset] = convolution;
        float result = saved_short[row * channels + channel] * convolution;
        value_store(output, mixed_offset, result);
    }
}

template <typename T>
__global__ void hyena_backward_convolution_kernel(
    const T* long_filter, const T* diagonal,
    const float* output_gradient, const float* saved_short,
    const float* saved_gated, const float* saved_convolved,
    float* short_gradient, float* convolution_gradient,
    float* gated_gradient, float* long_filter_gradient,
    float* diagonal_gradient, int batch, int sequence, int width) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    int lanes = batch * width;
    if (lane >= lanes)
        return;
    int batch_index = lane / width;
    int channel = lane % width;
    int channels = 3 * width;

    for (int time = 0; time < sequence; ++time) {
        int row = batch_index * sequence + time;
        int mixed_offset = row * width + channel;
        float upstream = output_gradient[mixed_offset];
        short_gradient[row * channels + channel] =
            upstream * saved_convolved[mixed_offset];
        float convolution_grad =
            upstream * saved_short[row * channels + channel];
        convolution_gradient[mixed_offset] = convolution_grad;
        gated_gradient[mixed_offset] =
            convolution_grad * value_load(diagonal, channel);
        atomicAdd(diagonal_gradient + channel,
            convolution_grad * saved_gated[mixed_offset]);
    }

    for (int time = 0; time < sequence; ++time) {
        int row = batch_index * sequence + time;
        float convolution_grad =
            convolution_gradient[row * width + channel];
        for (int lag = 0; lag <= time; ++lag) {
            int source_row = row - lag;
            int source_offset = source_row * width + channel;
            gated_gradient[source_offset] += convolution_grad
                * value_load(long_filter, lag * width + channel);
            atomicAdd(long_filter_gradient + lag * width + channel,
                convolution_grad * saved_gated[source_offset]);
        }
    }

    for (int time = 0; time < sequence; ++time) {
        int row = batch_index * sequence + time;
        float gate_gradient = gated_gradient[row * width + channel];
        short_gradient[row * channels + width + channel] = gate_gradient
            * saved_short[row * channels + 2 * width + channel];
        short_gradient[row * channels + 2 * width + channel] = gate_gradient
            * saved_short[row * channels + width + channel];
    }
}

template <typename T>
__global__ void hyena_backward_short_kernel(
    const T* projected, const T* short_filter,
    const float* short_gradient, float* projected_gradient,
    float* short_filter_gradient, int batch, int sequence, int width) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    int lanes = batch * width;
    if (lane >= lanes)
        return;
    int batch_index = lane / width;
    int channel = lane % width;
    int channels = 3 * width;
    for (int time = 0; time < sequence; ++time) {
        int row = batch_index * sequence + time;
        int tap_count = min(3, time + 1);
        for (int stream_index = 0; stream_index < 3; ++stream_index) {
            int channel_index = stream_index * width + channel;
            float gradient =
                short_gradient[row * channels + channel_index];
            for (int tap = 0; tap < tap_count; ++tap) {
                int source_row = row - tap;
                int projected_offset =
                    source_row * channels + channel_index;
                int filter_offset = tap * channels + channel_index;
                projected_gradient[projected_offset] += gradient
                    * value_load(short_filter, filter_offset);
                atomicAdd(short_filter_gradient + filter_offset,
                    gradient * value_load(projected, projected_offset));
            }
        }
    }
}

template <typename T>
int launch_hyena_forward(
    const T* projected, const T* short_filter, const T* long_filter,
    const T* diagonal, T* output, float* saved_short, float* saved_gated,
    float* saved_convolved, int batch, int sequence, int width,
    cudaStream_t stream) {
    if (!projected || !short_filter || !long_filter || !diagonal || !output
        || !saved_short || !saved_gated || !saved_convolved
        || batch <= 0 || sequence <= 0 || width <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    int lanes = batch * width;
    hyena_forward_kernel<<<blocks_for(lanes), kThreads, 0, stream>>>(
        projected, short_filter, long_filter, diagonal, output,
        saved_short, saved_gated, saved_convolved,
        batch, sequence, width, 1);
    return launch_status();
}

template <typename T>
int launch_hyena_backward(
    const T* projected, const T* short_filter, const T* long_filter,
    const T* diagonal, const float* output_gradient,
    const float* saved_short, const float* saved_gated,
    const float* saved_convolved, float* projected_gradient,
    float* short_filter_gradient, float* long_filter_gradient,
    float* diagonal_gradient, float* short_gradient,
    float* convolution_gradient, float* gated_gradient,
    int batch, int sequence, int width, cudaStream_t stream) {
    if (!projected || !short_filter || !long_filter || !diagonal
        || !output_gradient || !saved_short || !saved_gated
        || !saved_convolved || !projected_gradient
        || !short_filter_gradient || !long_filter_gradient
        || !diagonal_gradient || !short_gradient
        || !convolution_gradient || !gated_gradient
        || batch <= 0 || sequence <= 0 || width <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    int lanes = batch * width;
    hyena_backward_convolution_kernel<<<
        blocks_for(lanes), kThreads, 0, stream>>>(
        long_filter, diagonal, output_gradient, saved_short, saved_gated,
        saved_convolved, short_gradient, convolution_gradient,
        gated_gradient, long_filter_gradient, diagonal_gradient,
        batch, sequence, width);
    cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return static_cast<int>(status);
    hyena_backward_short_kernel<<<blocks_for(lanes), kThreads, 0, stream>>>(
        projected, short_filter, short_gradient, projected_gradient,
        short_filter_gradient, batch, sequence, width);
    return launch_status();
}

// The direct kernel above assigns one thread to an entire (batch, channel)
// lane and is efficient for short sequences.  Long sequences use the kernels
// below: time/filter elements are independently owned, removing the serial
// O(S^2) lane and every filter-gradient atomic.  This is also the deterministic
// CUDA fallback for explicit FFT selection when a cuFFT plan is unavailable;
// it preserves the exact causal-convolution semantics without returning to the
// host.
template <typename T>
__global__ void hyena_parallel_short_forward_kernel(
    const T* projected, const T* short_filter, float* saved_short,
    float* saved_gated, int batch, int sequence, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int length = batch * sequence * width;
    if (index >= length)
        return;
    int channel = index % width;
    int row = index / width;
    int time = row % sequence;
    int channels = 3 * width;
    for (int stream_index = 0; stream_index < 3; ++stream_index) {
        int channel_index = stream_index * width + channel;
        float sum = 0.f;
        int tap_count = min(3, time + 1);
        for (int tap = 0; tap < tap_count; ++tap) {
            int source_row = row - tap;
            sum += value_load(projected,
                    source_row * channels + channel_index)
                * value_load(short_filter,
                    tap * channels + channel_index);
        }
        saved_short[row * channels + channel_index] = sum;
    }
    saved_gated[index] =
        saved_short[row * channels + width + channel]
        * saved_short[row * channels + 2 * width + channel];
}

template <typename T>
__global__ void hyena_parallel_convolution_forward_kernel(
    const T* long_filter, const T* diagonal, T* output,
    const float* saved_short, const float* saved_gated,
    float* saved_convolved, int batch, int sequence, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int length = batch * sequence * width;
    if (index >= length)
        return;
    int channel = index % width;
    int row = index / width;
    int time = row % sequence;
    int batch_base = (row / sequence) * sequence;
    float convolution = saved_gated[index]
        * value_load(diagonal, channel);
    for (int lag = 0; lag <= time; ++lag) {
        int source = (batch_base + time - lag) * width + channel;
        convolution += saved_gated[source]
            * value_load(long_filter, lag * width + channel);
    }
    saved_convolved[index] = convolution;
    value_store(output, index,
        saved_short[row * (3 * width) + channel] * convolution);
}

template <typename G>
__global__ void hyena_parallel_backward_main_kernel(
    const G* output_gradient, const float* saved_short,
    const float* saved_convolved, float* short_gradient,
    float* convolution_gradient, int length, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    int row = index / width;
    int channel = index % width;
    float upstream = value_load(output_gradient, index);
    short_gradient[row * (3 * width) + channel] =
        upstream * saved_convolved[index];
    convolution_gradient[index] =
        upstream * saved_short[row * (3 * width) + channel];
}

template <typename T>
__global__ void hyena_parallel_backward_gated_kernel(
    const T* long_filter, const T* diagonal,
    const float* convolution_gradient, float* gated_gradient,
    int batch, int sequence, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int length = batch * sequence * width;
    if (index >= length)
        return;
    int channel = index % width;
    int row = index / width;
    int time = row % sequence;
    int batch_base = (row / sequence) * sequence;
    float gradient = convolution_gradient[index]
        * value_load(diagonal, channel);
    for (int output_time = time; output_time < sequence; ++output_time) {
        int output_index =
            (batch_base + output_time) * width + channel;
        gradient += convolution_gradient[output_index]
            * value_load(long_filter,
                (output_time - time) * width + channel);
    }
    gated_gradient[index] = gradient;
}

template <typename G>
__global__ void hyena_parallel_backward_filter_kernel(
    const float* convolution_gradient, const float* saved_gated,
    G* long_filter_gradient, int batch, int sequence, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int length = sequence * width;
    if (index >= length)
        return;
    int lag = index / width;
    int channel = index % width;
    float gradient = 0.f;
    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        int batch_base = batch_index * sequence;
        for (int time = lag; time < sequence; ++time) {
            int output_index = (batch_base + time) * width + channel;
            int source_index =
                (batch_base + time - lag) * width + channel;
            gradient += convolution_gradient[output_index]
                * saved_gated[source_index];
        }
    }
    gradient_add(long_filter_gradient, index, gradient);
}

template <typename G>
__global__ void hyena_parallel_backward_diagonal_kernel(
    const float* convolution_gradient, const float* saved_gated,
    G* diagonal_gradient, int batch, int sequence, int width) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= width)
        return;
    float gradient = 0.f;
    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        int batch_base = batch_index * sequence;
        for (int time = 0; time < sequence; ++time) {
            int index = (batch_base + time) * width + channel;
            gradient += convolution_gradient[index] * saved_gated[index];
        }
    }
    gradient_add(diagonal_gradient, channel, gradient);
}

__global__ void hyena_parallel_backward_gate_kernel(
    const float* saved_short, const float* gated_gradient,
    float* short_gradient, int length, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    int row = index / width;
    int channel = index % width;
    int base = row * (3 * width);
    float gradient = gated_gradient[index];
    short_gradient[base + width + channel] =
        gradient * saved_short[base + 2 * width + channel];
    short_gradient[base + 2 * width + channel] =
        gradient * saved_short[base + width + channel];
}

template <typename T, typename G>
__global__ void hyena_parallel_backward_projected_kernel(
    const T* short_filter, const float* short_gradient,
    G* projected_gradient, int batch, int sequence, int width) {
    int channels = 3 * width;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int length = batch * sequence * channels;
    if (index >= length)
        return;
    int channel = index % channels;
    int row = index / channels;
    int time = row % sequence;
    int batch_base = (row / sequence) * sequence;
    float gradient = 0.f;
    int tap_count = min(3, sequence - time);
    for (int tap = 0; tap < tap_count; ++tap) {
        int output_row = batch_base + time + tap;
        gradient += short_gradient[output_row * channels + channel]
            * value_load(short_filter, tap * channels + channel);
    }
    gradient_add(projected_gradient, index, gradient);
}

template <typename T, typename G>
__global__ void hyena_parallel_backward_short_filter_kernel(
    const T* projected, const float* short_gradient,
    G* short_filter_gradient, int batch, int sequence, int width) {
    int channels = 3 * width;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int length = 3 * channels;
    if (index >= length)
        return;
    int tap = index / channels;
    int channel = index % channels;
    float gradient = 0.f;
    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        int batch_base = batch_index * sequence;
        for (int time = tap; time < sequence; ++time) {
            int output_row = batch_base + time;
            int source_row = batch_base + time - tap;
            gradient += short_gradient[output_row * channels + channel]
                * value_load(projected, source_row * channels + channel);
        }
    }
    gradient_add(short_filter_gradient, index, gradient);
}

template <typename T>
int launch_hyena_parallel_forward(
    const T* projected, const T* short_filter, const T* long_filter,
    const T* diagonal, T* output, float* saved_short, float* saved_gated,
    float* saved_convolved, int batch, int sequence, int width,
    cudaStream_t stream) {
    if (!projected || !short_filter || !long_filter || !diagonal || !output
        || !saved_short || !saved_gated || !saved_convolved
        || batch <= 0 || sequence <= 0 || width <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    int length = batch * sequence * width;
    hyena_parallel_short_forward_kernel<<<
        blocks_for(length), kThreads, 0, stream>>>(
        projected, short_filter, saved_short, saved_gated,
        batch, sequence, width);
    cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return static_cast<int>(status);
    hyena_parallel_convolution_forward_kernel<<<
        blocks_for(length), kThreads, 0, stream>>>(
        long_filter, diagonal, output, saved_short, saved_gated,
        saved_convolved, batch, sequence, width);
    return launch_status();
}

template <typename T, typename G>
int launch_hyena_parallel_backward(
    const T* projected, const T* short_filter, const T* long_filter,
    const T* diagonal, const G* output_gradient,
    const float* saved_short, const float* saved_gated,
    const float* saved_convolved, G* projected_gradient,
    G* short_filter_gradient, G* long_filter_gradient,
    G* diagonal_gradient, float* short_gradient,
    float* convolution_gradient, float* gated_gradient,
    int batch, int sequence, int width, cudaStream_t stream) {
    if (!projected || !short_filter || !long_filter || !diagonal
        || !output_gradient || !saved_short || !saved_gated
        || !saved_convolved || !projected_gradient
        || !short_filter_gradient || !long_filter_gradient
        || !diagonal_gradient || !short_gradient
        || !convolution_gradient || !gated_gradient
        || batch <= 0 || sequence <= 0 || width <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    int length = batch * sequence * width;
    int short_length = batch * sequence * 3 * width;
    hyena_parallel_backward_main_kernel<<<
        blocks_for(length), kThreads, 0, stream>>>(
        output_gradient, saved_short, saved_convolved,
        short_gradient, convolution_gradient, length, width);
    hyena_parallel_backward_gated_kernel<<<
        blocks_for(length), kThreads, 0, stream>>>(
        long_filter, diagonal, convolution_gradient, gated_gradient,
        batch, sequence, width);
    hyena_parallel_backward_filter_kernel<<<
        blocks_for(sequence * width), kThreads, 0, stream>>>(
        convolution_gradient, saved_gated, long_filter_gradient,
        batch, sequence, width);
    hyena_parallel_backward_diagonal_kernel<<<
        blocks_for(width), kThreads, 0, stream>>>(
        convolution_gradient, saved_gated, diagonal_gradient,
        batch, sequence, width);
    hyena_parallel_backward_gate_kernel<<<
        blocks_for(length), kThreads, 0, stream>>>(
        saved_short, gated_gradient, short_gradient, length, width);
    hyena_parallel_backward_projected_kernel<<<
        blocks_for(short_length), kThreads, 0, stream>>>(
        short_filter, short_gradient, projected_gradient,
        batch, sequence, width);
    hyena_parallel_backward_short_filter_kernel<<<
        blocks_for(9 * width), kThreads, 0, stream>>>(
        projected, short_gradient, short_filter_gradient,
        batch, sequence, width);
    return launch_status();
}

template <typename T>
__global__ void broadcast_add_forward_kernel(
    const T* input, const T* addend, T* output, int length,
    int repeat_length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) {
        value_store(output, index,
            value_load(input, index)
                + value_load(addend, index % repeat_length));
    }
}

__global__ void shape_accumulate_bf16_gradient_kernel(
    const unsigned short* source, unsigned short* destination,
    int length, int source_offset, int destination_offset) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) {
        bf16_gradient_add(destination, destination_offset + index,
            value_load(source, source_offset + index), false);
    }
}

__global__ void transpose_bf16_kernel(
    const unsigned short* input, unsigned short* output,
    int rows, int columns) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int length = rows * columns;
    if (index >= length)
        return;
    int row = index / columns;
    int column = index % columns;
    output[column * rows + row] = input[index];
}

__global__ void transpose_backward_bf16_gradient_kernel(
    const unsigned short* output_gradient,
    unsigned short* input_gradient, int rows, int columns) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int length = rows * columns;
    if (index >= length)
        return;
    int row = index / columns;
    int column = index % columns;
    bf16_gradient_add(input_gradient, index,
        value_load(output_gradient, column * rows + row), false);
}

__global__ void broadcast_add_backward_kernel(
    const float* output_gradient, float* input_gradient,
    float* addend_gradient, int length, int repeat_length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    float gradient = output_gradient[index];
    input_gradient[index] += gradient;
    atomicAdd(addend_gradient + index % repeat_length, gradient);
}

__global__ void broadcast_add_backward_bf16_gradient_kernel(
    const unsigned short* output_gradient, unsigned short* input_gradient,
    unsigned short* addend_gradient, int length, int repeat_length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    float gradient = value_load(output_gradient, index);
    bf16_gradient_add(input_gradient, index, gradient, false);
    bf16_gradient_add(addend_gradient, index % repeat_length, gradient, true);
}

template <typename T>
__global__ void causal_mask_forward_kernel(
    const T* input, T* output, int length, int rows, int columns,
    float fill_value) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    int matrix_index = index % (rows * columns);
    int row = matrix_index / columns;
    int column = matrix_index % columns;
    value_store(output, index,
        column > row ? fill_value : value_load(input, index));
}

__global__ void causal_mask_backward_kernel(
    const float* output_gradient, float* input_gradient,
    int length, int rows, int columns) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    int matrix_index = index % (rows * columns);
    int row = matrix_index / columns;
    int column = matrix_index % columns;
    if (column <= row)
        input_gradient[index] += output_gradient[index];
}

__global__ void causal_mask_backward_bf16_gradient_kernel(
    const unsigned short* output_gradient, unsigned short* input_gradient,
    int length, int rows, int columns) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    int matrix_index = index % (rows * columns);
    int row = matrix_index / columns;
    int column = matrix_index % columns;
    if (column <= row) {
        bf16_gradient_add(input_gradient, index,
            value_load(output_gradient, index), false);
    }
}

template <typename T>
__global__ void softmax_forward_kernel(
    const T* input, T* output, float* probabilities,
    int rows, int columns, int log_softmax) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    if (row >= rows)
        return;
    float maximum = -FLT_MAX;
    for (int column = threadIdx.x; column < columns;
        column += blockDim.x) {
        maximum = fmaxf(maximum,
            value_load(input, row * columns + column));
    }
    shared[threadIdx.x] = maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shared[threadIdx.x] = fmaxf(
                shared[threadIdx.x], shared[threadIdx.x + stride]);
        __syncthreads();
    }
    maximum = shared[0];
    float sum = 0.f;
    for (int column = threadIdx.x; column < columns;
        column += blockDim.x) {
        sum += expf(value_load(input, row * columns + column) - maximum);
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse_sum = 1.f / shared[0];
    float log_sum = logf(shared[0]);
    for (int column = threadIdx.x; column < columns;
        column += blockDim.x) {
        int index = row * columns + column;
        float value = value_load(input, index);
        float probability = expf(value - maximum) * inverse_sum;
        if (probabilities)
            probabilities[index] = probability;
        value_store(output, index, log_softmax != 0
            ? value - maximum - log_sum
            : probability);
    }
}

__global__ void softmax_backward_kernel(
    const float* probabilities, const float* output_gradient,
    float* input_gradient, int rows, int columns, int log_softmax) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    if (row >= rows)
        return;
    float sum = 0.f;
    for (int column = threadIdx.x; column < columns;
        column += blockDim.x) {
        int index = row * columns + column;
        sum += log_softmax != 0
            ? output_gradient[index]
            : output_gradient[index] * probabilities[index];
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float reduced = shared[0];
    for (int column = threadIdx.x; column < columns;
        column += blockDim.x) {
        int index = row * columns + column;
        float contribution = log_softmax != 0
            ? output_gradient[index] - probabilities[index] * reduced
            : probabilities[index] * (output_gradient[index] - reduced);
        input_gradient[index] += contribution;
    }
}

__global__ void softmax_backward_bf16_gradient_kernel(
    const float* probabilities, const unsigned short* output_gradient,
    unsigned short* input_gradient, int rows, int columns,
    int log_softmax) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    if (row >= rows)
        return;
    float sum = 0.f;
    for (int column = threadIdx.x; column < columns;
        column += blockDim.x) {
        int index = row * columns + column;
        float upstream = value_load(output_gradient, index);
        sum += log_softmax != 0
            ? upstream
            : upstream * probabilities[index];
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float reduced = shared[0];
    for (int column = threadIdx.x; column < columns;
        column += blockDim.x) {
        int index = row * columns + column;
        float upstream = value_load(output_gradient, index);
        float contribution = log_softmax != 0
            ? upstream - probabilities[index] * reduced
            : probabilities[index] * (upstream - reduced);
        bf16_gradient_add(input_gradient, index, contribution, false);
    }
}

template <typename T>
int launch_broadcast_add(
    const T* input, const T* addend, T* output,
    int length, int repeat_length, cudaStream_t stream) {
    if (!input || !addend || !output || length <= 0
        || repeat_length <= 0 || length % repeat_length != 0)
        return static_cast<int>(cudaErrorInvalidValue);
    broadcast_add_forward_kernel<<<blocks_for(length), kThreads, 0, stream>>>(
        input, addend, output, length, repeat_length);
    return launch_status();
}

template <typename T>
int launch_causal_mask(
    const T* input, T* output, int length, int rows, int columns,
    float fill_value, cudaStream_t stream) {
    if (!input || !output || length <= 0 || rows <= 0 || columns <= 0
        || length % (rows * columns) != 0)
        return static_cast<int>(cudaErrorInvalidValue);
    causal_mask_forward_kernel<<<blocks_for(length), kThreads, 0, stream>>>(
        input, output, length, rows, columns, fill_value);
    return launch_status();
}

template <typename T>
int launch_softmax(
    const T* input, T* output, float* probabilities,
    int rows, int columns, int log_softmax, cudaStream_t stream) {
    if (!input || !output || rows <= 0 || columns <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    softmax_forward_kernel<<<rows, kThreads,
        kThreads * sizeof(float), stream>>>(
        input, output, probabilities, rows, columns, log_softmax);
    return launch_status();
}
}  // namespace

NNTRAIN_EXPORT int nntrain_public_binary_float(
    const float* left, const float* right, float* output, int length,
    int left_scalar, int right_scalar, int operation, cudaStream_t stream) {
    return launch_binary_forward(left, right, output, length,
        left_scalar, right_scalar, operation, stream);
}

NNTRAIN_EXPORT int nntrain_public_binary_bf16(
    const unsigned short* left, const unsigned short* right,
    unsigned short* output, int length, int left_scalar, int right_scalar,
    int operation, cudaStream_t stream) {
    return launch_binary_forward(left, right, output, length,
        left_scalar, right_scalar, operation, stream);
}

NNTRAIN_EXPORT int nntrain_public_binary_backward_float(
    const float* left, const float* right, const float* output_gradient,
    float* left_gradient, float* right_gradient, int length,
    int left_scalar, int right_scalar, int same_parent, int operation,
    cudaStream_t stream) {
    return launch_binary_backward(left, right, output_gradient,
        left_gradient, right_gradient, length, left_scalar, right_scalar,
        same_parent, operation, stream);
}

NNTRAIN_EXPORT int nntrain_public_binary_backward_bf16(
    const unsigned short* left, const unsigned short* right,
    const float* output_gradient, float* left_gradient,
    float* right_gradient, int length, int left_scalar, int right_scalar,
    int same_parent, int operation, cudaStream_t stream) {
    return launch_binary_backward(left, right, output_gradient,
        left_gradient, right_gradient, length, left_scalar, right_scalar,
        same_parent, operation, stream);
}

NNTRAIN_EXPORT int nntrain_public_binary_backward_bf16_gradient(
    const unsigned short* left, const unsigned short* right,
    const unsigned short* output_gradient,
    unsigned short* left_gradient, unsigned short* right_gradient,
    int length, int left_scalar, int right_scalar, int same_parent,
    int operation, cudaStream_t stream) {
    if (!left || !right || !output_gradient || !left_gradient
        || !right_gradient || length <= 0 || operation < 0 || operation > 3)
        return static_cast<int>(cudaErrorInvalidValue);
    binary_backward_bf16_gradient_kernel<<<
        blocks_for(length), kThreads, 0, stream>>>(
        left, right, output_gradient, left_gradient, right_gradient,
        length, left_scalar, right_scalar, same_parent, operation);
    cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess)
        return static_cast<int>(status);
    if (same_parent != 0 && (left_scalar != 0 || right_scalar != 0)) {
        binary_backward_bf16_scalar_reduction_kernel<<<
            1, kThreads, 0, stream>>>(
            left, right, output_gradient, left_gradient,
            length, 0, 1, operation);
    } else {
        if (left_scalar != 0) {
            binary_backward_bf16_scalar_reduction_kernel<<<
                1, kThreads, 0, stream>>>(
                left, right, output_gradient, left_gradient,
                length, 0, 0, operation);
        }
        if (right_scalar != 0) {
            binary_backward_bf16_scalar_reduction_kernel<<<
                1, kThreads, 0, stream>>>(
                left, right, output_gradient, right_gradient,
                length, 1, 0, operation);
        }
    }
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_public_unary_float(
    const float* input, float* output, int length, int operation,
    float parameter, cudaStream_t stream) {
    return launch_unary_forward(
        input, output, length, operation, parameter, stream);
}

NNTRAIN_EXPORT int nntrain_public_unary_bf16(
    const unsigned short* input, unsigned short* output, int length,
    int operation, float parameter, cudaStream_t stream) {
    return launch_unary_forward(
        input, output, length, operation, parameter, stream);
}

NNTRAIN_EXPORT int nntrain_public_unary_backward_float(
    const float* input, const float* output, const float* output_gradient,
    float* input_gradient, int length, int operation, float parameter,
    cudaStream_t stream) {
    return launch_unary_backward(input, output, output_gradient,
        input_gradient, length, operation, parameter, stream);
}

NNTRAIN_EXPORT int nntrain_public_unary_backward_bf16(
    const unsigned short* input, const unsigned short* output,
    const float* output_gradient, float* input_gradient, int length,
    int operation, float parameter, cudaStream_t stream) {
    return launch_unary_backward(input, output, output_gradient,
        input_gradient, length, operation, parameter, stream);
}

NNTRAIN_EXPORT int nntrain_public_unary_backward_bf16_gradient(
    const unsigned short* input, const unsigned short* output,
    const unsigned short* output_gradient, unsigned short* input_gradient,
    int length, int operation, float parameter, cudaStream_t stream) {
    if (!input || !output || !output_gradient || !input_gradient
        || length <= 0 || operation < 0 || operation > 7)
        return static_cast<int>(cudaErrorInvalidValue);
    unary_backward_bf16_gradient_kernel<<<
        blocks_for(length), kThreads, 0, stream>>>(
        input, output, output_gradient, input_gradient,
        length, operation, parameter);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_public_reduce_float(
    const float* input, float* output, int length, int operation,
    cudaStream_t stream) {
    return launch_reduction_complete(input, output, length, operation, stream);
}

NNTRAIN_EXPORT int nntrain_public_reduce_bf16(
    const unsigned short* input, float* output, int length, int operation,
    cudaStream_t stream) {
    return launch_reduction_complete(input, output, length, operation, stream);
}

NNTRAIN_EXPORT int nntrain_public_reduce_backward_float(
    const float* input, const float* reduced, const float* output_gradient,
    float* input_gradient, int length, int operation, cudaStream_t stream) {
    return launch_reduction_backward(input, reduced, output_gradient,
        input_gradient, length, operation, stream);
}

NNTRAIN_EXPORT int nntrain_public_reduce_backward_bf16(
    const unsigned short* input, const float* reduced,
    const float* output_gradient, float* input_gradient, int length,
    int operation, cudaStream_t stream) {
    return launch_reduction_backward(input, reduced, output_gradient,
        input_gradient, length, operation, stream);
}

NNTRAIN_EXPORT int nntrain_public_reduce_backward_bf16_gradient(
    const unsigned short* input, const float* reduced,
    const float* output_gradient, unsigned short* input_gradient,
    int length, int operation, cudaStream_t stream) {
    if (!input || !reduced || !output_gradient || !input_gradient
        || length <= 0 || operation < 0 || operation > 2)
        return static_cast<int>(cudaErrorInvalidValue);
    reduction_backward_bf16_gradient_kernel<<<
        blocks_for(length), kThreads, 0, stream>>>(
        input, reduced, output_gradient, input_gradient, length, operation);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_public_forget_scan_float(
    const float* projected, float* output, float* memory, float* forget,
    float* input, float* value, int batch, int sequence, int width,
    int save_context, cudaStream_t stream) {
    return launch_forget_scan_forward(
        projected, output, memory, forget, input, value,
        batch, sequence, width, save_context, stream);
}

NNTRAIN_EXPORT int nntrain_public_forget_scan_bf16(
    const unsigned short* projected, unsigned short* output,
    float* memory, float* forget, float* input, float* value,
    int batch, int sequence, int width, int save_context,
    cudaStream_t stream) {
    return launch_forget_scan_forward(
        projected, output, memory, forget, input, value,
        batch, sequence, width, save_context, stream);
}

NNTRAIN_EXPORT int nntrain_public_forget_scan_backward(
    const float* output_gradient, const float* memory, const float* forget,
    const float* input, const float* value, float* projected_gradient,
    int batch, int sequence, int width, cudaStream_t stream) {
    if (!output_gradient || !memory || !forget || !input || !value
        || !projected_gradient || batch <= 0 || sequence <= 0 || width <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    int lanes = batch * width;
    forget_scan_backward_kernel<<<blocks_for(lanes), kThreads, 0, stream>>>(
        output_gradient, memory, forget, input, value, projected_gradient,
        batch, sequence, width);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_public_forget_scan_backward_bf16_gradient(
    const unsigned short* output_gradient, const float* memory,
    const float* forget, const float* input, const float* value,
    unsigned short* projected_gradient,
    int batch, int sequence, int width, cudaStream_t stream) {
    if (!output_gradient || !memory || !forget || !input || !value
        || !projected_gradient || batch <= 0 || sequence <= 0 || width <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    int lanes = batch * width;
    forget_scan_backward_bf16_gradient_kernel<<<
        blocks_for(lanes), kThreads, 0, stream>>>(
        output_gradient, memory, forget, input, value, projected_gradient,
        batch, sequence, width);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_public_hyena_float(
    const float* projected, const float* short_filter,
    const float* long_filter, const float* diagonal, float* output,
    float* saved_short, float* saved_gated, float* saved_convolved,
    int batch, int sequence, int width, cudaStream_t stream) {
    return launch_hyena_forward(
        projected, short_filter, long_filter, diagonal, output,
        saved_short, saved_gated, saved_convolved,
        batch, sequence, width, stream);
}

NNTRAIN_EXPORT int nntrain_public_hyena_bf16(
    const unsigned short* projected, const unsigned short* short_filter,
    const unsigned short* long_filter, const unsigned short* diagonal,
    unsigned short* output, float* saved_short, float* saved_gated,
    float* saved_convolved, int batch, int sequence, int width,
    cudaStream_t stream) {
    return launch_hyena_forward(
        projected, short_filter, long_filter, diagonal, output,
        saved_short, saved_gated, saved_convolved,
        batch, sequence, width, stream);
}

NNTRAIN_EXPORT int nntrain_public_hyena_parallel_float(
    const float* projected, const float* short_filter,
    const float* long_filter, const float* diagonal, float* output,
    float* saved_short, float* saved_gated, float* saved_convolved,
    int batch, int sequence, int width, cudaStream_t stream) {
    return launch_hyena_parallel_forward(
        projected, short_filter, long_filter, diagonal, output,
        saved_short, saved_gated, saved_convolved,
        batch, sequence, width, stream);
}

NNTRAIN_EXPORT int nntrain_public_hyena_parallel_bf16(
    const unsigned short* projected, const unsigned short* short_filter,
    const unsigned short* long_filter, const unsigned short* diagonal,
    unsigned short* output, float* saved_short, float* saved_gated,
    float* saved_convolved, int batch, int sequence, int width,
    cudaStream_t stream) {
    return launch_hyena_parallel_forward(
        projected, short_filter, long_filter, diagonal, output,
        saved_short, saved_gated, saved_convolved,
        batch, sequence, width, stream);
}

NNTRAIN_EXPORT int nntrain_public_hyena_backward_float(
    const float* projected, const float* short_filter,
    const float* long_filter, const float* diagonal,
    const float* output_gradient, const float* saved_short,
    const float* saved_gated, const float* saved_convolved,
    float* projected_gradient, float* short_filter_gradient,
    float* long_filter_gradient, float* diagonal_gradient,
    float* short_gradient, float* convolution_gradient,
    float* gated_gradient, int batch, int sequence, int width,
    cudaStream_t stream) {
    return launch_hyena_backward(
        projected, short_filter, long_filter, diagonal, output_gradient,
        saved_short, saved_gated, saved_convolved, projected_gradient,
        short_filter_gradient, long_filter_gradient, diagonal_gradient,
        short_gradient, convolution_gradient, gated_gradient,
        batch, sequence, width, stream);
}

NNTRAIN_EXPORT int nntrain_public_hyena_backward_bf16(
    const unsigned short* projected, const unsigned short* short_filter,
    const unsigned short* long_filter, const unsigned short* diagonal,
    const float* output_gradient, const float* saved_short,
    const float* saved_gated, const float* saved_convolved,
    float* projected_gradient, float* short_filter_gradient,
    float* long_filter_gradient, float* diagonal_gradient,
    float* short_gradient, float* convolution_gradient,
    float* gated_gradient, int batch, int sequence, int width,
    cudaStream_t stream) {
    return launch_hyena_backward(
        projected, short_filter, long_filter, diagonal, output_gradient,
        saved_short, saved_gated, saved_convolved, projected_gradient,
        short_filter_gradient, long_filter_gradient, diagonal_gradient,
        short_gradient, convolution_gradient, gated_gradient,
        batch, sequence, width, stream);
}

NNTRAIN_EXPORT int nntrain_public_hyena_backward_parallel_float(
    const float* projected, const float* short_filter,
    const float* long_filter, const float* diagonal,
    const float* output_gradient, const float* saved_short,
    const float* saved_gated, const float* saved_convolved,
    float* projected_gradient, float* short_filter_gradient,
    float* long_filter_gradient, float* diagonal_gradient,
    float* short_gradient, float* convolution_gradient,
    float* gated_gradient, int batch, int sequence, int width,
    cudaStream_t stream) {
    return launch_hyena_parallel_backward(
        projected, short_filter, long_filter, diagonal, output_gradient,
        saved_short, saved_gated, saved_convolved, projected_gradient,
        short_filter_gradient, long_filter_gradient, diagonal_gradient,
        short_gradient, convolution_gradient, gated_gradient,
        batch, sequence, width, stream);
}

NNTRAIN_EXPORT int nntrain_public_hyena_backward_parallel_bf16(
    const unsigned short* projected, const unsigned short* short_filter,
    const unsigned short* long_filter, const unsigned short* diagonal,
    const float* output_gradient, const float* saved_short,
    const float* saved_gated, const float* saved_convolved,
    float* projected_gradient, float* short_filter_gradient,
    float* long_filter_gradient, float* diagonal_gradient,
    float* short_gradient, float* convolution_gradient,
    float* gated_gradient, int batch, int sequence, int width,
    cudaStream_t stream) {
    return launch_hyena_parallel_backward(
        projected, short_filter, long_filter, diagonal, output_gradient,
        saved_short, saved_gated, saved_convolved, projected_gradient,
        short_filter_gradient, long_filter_gradient, diagonal_gradient,
        short_gradient, convolution_gradient, gated_gradient,
        batch, sequence, width, stream);
}

NNTRAIN_EXPORT int nntrain_public_hyena_backward_bf16_gradient(
    const unsigned short* projected, const unsigned short* short_filter,
    const unsigned short* long_filter, const unsigned short* diagonal,
    const unsigned short* output_gradient, const float* saved_short,
    const float* saved_gated, const float* saved_convolved,
    unsigned short* projected_gradient,
    unsigned short* short_filter_gradient,
    unsigned short* long_filter_gradient,
    unsigned short* diagonal_gradient,
    float* short_gradient, float* convolution_gradient,
    float* gated_gradient, int batch, int sequence, int width,
    cudaStream_t stream) {
    return launch_hyena_parallel_backward(
        projected, short_filter, long_filter, diagonal, output_gradient,
        saved_short, saved_gated, saved_convolved, projected_gradient,
        short_filter_gradient, long_filter_gradient, diagonal_gradient,
        short_gradient, convolution_gradient, gated_gradient,
        batch, sequence, width, stream);
}

NNTRAIN_EXPORT int nntrain_public_shape_accumulate_bf16_gradient(
    const unsigned short* source, unsigned short* destination,
    int length, int source_offset, int destination_offset,
    cudaStream_t stream) {
    if (!source || !destination || length <= 0
        || source_offset < 0 || destination_offset < 0)
        return static_cast<int>(cudaErrorInvalidValue);
    shape_accumulate_bf16_gradient_kernel<<<
        blocks_for(length), kThreads, 0, stream>>>(
        source, destination, length, source_offset, destination_offset);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_public_transpose_bf16(
    const unsigned short* input, unsigned short* output,
    int rows, int columns, cudaStream_t stream) {
    if (!input || !output || rows <= 0 || columns <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    transpose_bf16_kernel<<<
        blocks_for(rows * columns), kThreads, 0, stream>>>(
        input, output, rows, columns);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_public_transpose_backward_bf16_gradient(
    const unsigned short* output_gradient,
    unsigned short* input_gradient, int rows, int columns,
    cudaStream_t stream) {
    if (!output_gradient || !input_gradient || rows <= 0 || columns <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    transpose_backward_bf16_gradient_kernel<<<
        blocks_for(rows * columns), kThreads, 0, stream>>>(
        output_gradient, input_gradient, rows, columns);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_public_broadcast_add_float(
    const float* input, const float* addend, float* output,
    int length, int repeat_length, cudaStream_t stream) {
    return launch_broadcast_add(
        input, addend, output, length, repeat_length, stream);
}

NNTRAIN_EXPORT int nntrain_public_broadcast_add_bf16(
    const unsigned short* input, const unsigned short* addend,
    unsigned short* output, int length, int repeat_length,
    cudaStream_t stream) {
    return launch_broadcast_add(
        input, addend, output, length, repeat_length, stream);
}

NNTRAIN_EXPORT int nntrain_public_broadcast_add_backward(
    const float* output_gradient, float* input_gradient,
    float* addend_gradient, int length, int repeat_length,
    cudaStream_t stream) {
    if (!output_gradient || !input_gradient || !addend_gradient
        || length <= 0 || repeat_length <= 0 || length % repeat_length != 0)
        return static_cast<int>(cudaErrorInvalidValue);
    broadcast_add_backward_kernel<<<blocks_for(length), kThreads, 0, stream>>>(
        output_gradient, input_gradient, addend_gradient,
        length, repeat_length);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_public_broadcast_add_backward_bf16_gradient(
    const unsigned short* output_gradient, unsigned short* input_gradient,
    unsigned short* addend_gradient, int length, int repeat_length,
    cudaStream_t stream) {
    if (!output_gradient || !input_gradient || !addend_gradient
        || length <= 0 || repeat_length <= 0 || length % repeat_length != 0)
        return static_cast<int>(cudaErrorInvalidValue);
    broadcast_add_backward_bf16_gradient_kernel<<<
        blocks_for(length), kThreads, 0, stream>>>(
        output_gradient, input_gradient, addend_gradient,
        length, repeat_length);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_public_causal_mask_float(
    const float* input, float* output, int length, int rows, int columns,
    float fill_value, cudaStream_t stream) {
    return launch_causal_mask(
        input, output, length, rows, columns, fill_value, stream);
}

NNTRAIN_EXPORT int nntrain_public_causal_mask_bf16(
    const unsigned short* input, unsigned short* output,
    int length, int rows, int columns, float fill_value,
    cudaStream_t stream) {
    return launch_causal_mask(
        input, output, length, rows, columns, fill_value, stream);
}

NNTRAIN_EXPORT int nntrain_public_causal_mask_backward(
    const float* output_gradient, float* input_gradient,
    int length, int rows, int columns, cudaStream_t stream) {
    if (!output_gradient || !input_gradient || length <= 0
        || rows <= 0 || columns <= 0 || length % (rows * columns) != 0)
        return static_cast<int>(cudaErrorInvalidValue);
    causal_mask_backward_kernel<<<blocks_for(length), kThreads, 0, stream>>>(
        output_gradient, input_gradient, length, rows, columns);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_public_causal_mask_backward_bf16_gradient(
    const unsigned short* output_gradient, unsigned short* input_gradient,
    int length, int rows, int columns, cudaStream_t stream) {
    if (!output_gradient || !input_gradient || length <= 0
        || rows <= 0 || columns <= 0 || length % (rows * columns) != 0)
        return static_cast<int>(cudaErrorInvalidValue);
    causal_mask_backward_bf16_gradient_kernel<<<
        blocks_for(length), kThreads, 0, stream>>>(
        output_gradient, input_gradient, length, rows, columns);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_public_softmax_float(
    const float* input, float* output, float* probabilities,
    int rows, int columns, int log_softmax, cudaStream_t stream) {
    return launch_softmax(
        input, output, probabilities, rows, columns, log_softmax, stream);
}

NNTRAIN_EXPORT int nntrain_public_softmax_bf16(
    const unsigned short* input, unsigned short* output,
    float* probabilities, int rows, int columns, int log_softmax,
    cudaStream_t stream) {
    return launch_softmax(
        input, output, probabilities, rows, columns, log_softmax, stream);
}

NNTRAIN_EXPORT int nntrain_public_softmax_backward(
    const float* probabilities, const float* output_gradient,
    float* input_gradient, int rows, int columns, int log_softmax,
    cudaStream_t stream) {
    if (!probabilities || !output_gradient || !input_gradient
        || rows <= 0 || columns <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    softmax_backward_kernel<<<rows, kThreads,
        kThreads * sizeof(float), stream>>>(
        probabilities, output_gradient, input_gradient,
        rows, columns, log_softmax);
    return launch_status();
}

NNTRAIN_EXPORT int nntrain_public_softmax_backward_bf16_gradient(
    const float* probabilities, const unsigned short* output_gradient,
    unsigned short* input_gradient, int rows, int columns,
    int log_softmax, cudaStream_t stream) {
    if (!probabilities || !output_gradient || !input_gradient
        || rows <= 0 || columns <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    softmax_backward_bf16_gradient_kernel<<<rows, kThreads,
        kThreads * sizeof(float), stream>>>(
        probabilities, output_gradient, input_gradient,
        rows, columns, log_softmax);
    return launch_status();
}
