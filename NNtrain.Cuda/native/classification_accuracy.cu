#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <climits>
#include <cmath>

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

constexpr int kThreads = 256;

template <typename T>
struct dense_logit_reader {
    const T* values;

    __device__ __forceinline__ float load(int index) const {
        return static_cast<float>(values[index]);
    }
};

template <>
__device__ __forceinline__ float
dense_logit_reader<unsigned short>::load(int index) const {
    return __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(values)[index]);
}

struct bfp8_logit_reader {
    const signed char* payload;
    const float* scales;
    int block_size;

    __device__ __forceinline__ float load(int index) const {
        return static_cast<float>(payload[index]) *
            scales[index / block_size];
    }
};

struct argmax_candidate {
    float value;
    int index;
};

__device__ __forceinline__ argmax_candidate prefer_argmax(
    argmax_candidate left,
    argmax_candidate right) {
    if (right.value > left.value ||
        (right.value == left.value && right.index < left.index)) {
        return right;
    }
    return left;
}

template <typename Reader>
__global__ void classification_correct_kernel(
    Reader logits,
    const int* targets,
    int* correct_count,
    int sample_count,
    int class_count) {
    int sample = blockIdx.x;
    if (sample >= sample_count)
        return;

    int row_offset = sample * class_count;
    // The managed CPU reference initializes its running maximum from class 0.
    // A NaN in that exact position therefore pins argmax to zero; later NaNs
    // are ignored by the strict `candidate > best` comparison. Preserve that
    // edge-case while still allowing an associative block reduction.
    float first = logits.load(row_offset);
    if (isnan(first)) {
        if (threadIdx.x == 0 && targets[sample] == 0)
            atomicAdd(correct_count, 1);
        return;
    }

    argmax_candidate local{-CUDART_INF_F, INT_MAX};
    for (int class_index = threadIdx.x;
        class_index < class_count;
        class_index += blockDim.x) {
        float value = logits.load(row_offset + class_index);
        if (isnan(value))
            continue;
        local = prefer_argmax(local, {value, class_index});
    }

    __shared__ float shared_values[kThreads];
    __shared__ int shared_indices[kThreads];
    shared_values[threadIdx.x] = local.value;
    shared_indices[threadIdx.x] = local.index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            argmax_candidate left{
                shared_values[threadIdx.x],
                shared_indices[threadIdx.x],
            };
            argmax_candidate right{
                shared_values[threadIdx.x + stride],
                shared_indices[threadIdx.x + stride],
            };
            argmax_candidate selected = prefer_argmax(left, right);
            shared_values[threadIdx.x] = selected.value;
            shared_indices[threadIdx.x] = selected.index;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && shared_indices[0] == targets[sample])
        atomicAdd(correct_count, 1);
}

template <typename Reader>
int launch_correct_count(
    int device,
    Reader reader,
    const int* targets,
    int* correct_count,
    int sample_count,
    int class_count,
    cudaStream_t stream) {
    if (device < 0 || targets == nullptr || correct_count == nullptr ||
        sample_count <= 0 || class_count <= 0 ||
        static_cast<long long>(sample_count) * class_count > INT_MAX) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = static_cast<cudaError_t>(
        nntrain_cuda_set_device(device));
    if (status != cudaSuccess)
        return static_cast<int>(status);
    status = cudaMemsetAsync(correct_count, 0, sizeof(int), stream);
    if (status != cudaSuccess)
        return static_cast<int>(status);
    int threads = class_count <= 32
        ? 32
        : class_count <= 64
            ? 64
            : class_count <= 128
                ? 128
                : kThreads;
    classification_correct_kernel<<<sample_count, threads, 0, stream>>>(
        reader,
        targets,
        correct_count,
        sample_count,
        class_count);
    return static_cast<int>(cudaPeekAtLastError());
}

} // namespace

NNTRAIN_EXPORT int nntrain_classification_correct_f32(
    int device,
    const float* logits,
    const int* targets,
    int* correct_count,
    int sample_count,
    int class_count,
    void* stream) {
    if (logits == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    return launch_correct_count(
        device,
        dense_logit_reader<float>{logits},
        targets,
        correct_count,
        sample_count,
        class_count,
        static_cast<cudaStream_t>(stream));
}

NNTRAIN_EXPORT int nntrain_classification_correct_bf16(
    int device,
    const unsigned short* logits,
    const int* targets,
    int* correct_count,
    int sample_count,
    int class_count,
    void* stream) {
    if (logits == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    return launch_correct_count(
        device,
        dense_logit_reader<unsigned short>{logits},
        targets,
        correct_count,
        sample_count,
        class_count,
        static_cast<cudaStream_t>(stream));
}

NNTRAIN_EXPORT int nntrain_classification_correct_bfp8(
    int device,
    const signed char* logits_payload,
    const float* logits_scales,
    int logits_block_size,
    const int* targets,
    int* correct_count,
    int sample_count,
    int class_count,
    void* stream) {
    if (logits_payload == nullptr || logits_scales == nullptr ||
        logits_block_size <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    return launch_correct_count(
        device,
        bfp8_logit_reader{
            logits_payload,
            logits_scales,
            logits_block_size,
        },
        targets,
        correct_count,
        sample_count,
        class_count,
        static_cast<cudaStream_t>(stream));
}
