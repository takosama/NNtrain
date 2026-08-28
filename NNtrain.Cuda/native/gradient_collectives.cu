#include "cuda_internal.cuh"

#include <cuda_bf16.h>
#include <new>

namespace {

constexpr int kGradientBucketThreads = 256;

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

int gradient_blocks(int length) {
    return min(
        (length + kGradientBucketThreads - 1) / kGradientBucketThreads,
        512);
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
        retain(nntrain::cuda::internal::select_device(
            pipeline->source_device));
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
        retain(nntrain::cuda::internal::select_device(
            pipeline->destination_device));
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

}  // namespace

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

// A captured external-record node publishes readiness to a communication
// stream outside the graph. Keep the legacy export above unchanged for eager
// execution and ABI compatibility; graph capture opts into this entry point.
NNTRAIN_EXPORT int nntrain_gradient_record_ready_external(
    int device, cudaEvent_t event, cudaStream_t compute_stream) {
    if (device < 0 || !event)
        return (int)cudaErrorInvalidValue;
    (void)device;
    return (int)cudaEventRecordWithFlags(
        event, compute_stream, cudaEventRecordExternal);
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
    cudaError_t status = nntrain::cuda::internal::select_device(
        source_device);
    if (status == cudaSuccess) {
        status = cudaStreamCreateWithFlags(
            &pipeline->download_stream, cudaStreamNonBlocking);
    }
    for (int slot = 0; status == cudaSuccess && slot < 2; ++slot) {
        status = cudaEventCreateWithFlags(
            &pipeline->download_done[slot], cudaEventDisableTiming);
    }
    if (status == cudaSuccess) {
        status = nntrain::cuda::internal::select_device(
            destination_device);
    }
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

    cudaError_t status = nntrain::cuda::internal::select_device(
        pipeline->source_device);
    if (status != cudaSuccess)
        return (int)status;
    status = cudaEventSynchronize(remote_ready);
    if (status != cudaSuccess)
        return (int)status;
    status = nntrain::cuda::internal::select_device(
        pipeline->destination_device);
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
        cudaError_t queue_status =
            nntrain::cuda::internal::select_device(
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
        status = nntrain::cuda::internal::select_device(
            pipeline->source_device);
        if (status == cudaSuccess) {
            status = cudaEventSynchronize(
                pipeline->download_done[slot]);
        }
        if (status != cudaSuccess)
            break;
        status = nntrain::cuda::internal::select_device(
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
        status = nntrain::cuda::internal::select_device(
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
