#include <cuda_runtime.h>
#include <cstddef>
#include <cstring>

#if defined(_WIN32)
#define NNTRAIN_EXPORT extern "C" __declspec(dllexport)
#else
#define NNTRAIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
thread_local int selected_device = -1;

int select_device(int device) {
    if (selected_device == device)
        return static_cast<int>(cudaSuccess);
    cudaError_t status = cudaSetDevice(device);
    if (status == cudaSuccess)
        selected_device = device;
    return static_cast<int>(status);
}
}

NNTRAIN_EXPORT int nntrain_cuda_device_count(int* count) {
    if (count == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    return static_cast<int>(cudaGetDeviceCount(count));
}

NNTRAIN_EXPORT int nntrain_cuda_device_name(
    int device,
    char* destination,
    int capacity) {
    if (destination == nullptr || capacity <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess)
        return static_cast<int>(status);
#if defined(_WIN32)
    strncpy_s(destination, static_cast<size_t>(capacity), properties.name,
        _TRUNCATE);
#else
    std::strncpy(destination, properties.name, static_cast<size_t>(capacity));
    destination[capacity - 1] = '\0';
#endif
    return static_cast<int>(cudaSuccess);
}

NNTRAIN_EXPORT int nntrain_cuda_set_device(int device) {
    return select_device(device);
}

NNTRAIN_EXPORT int nntrain_cuda_synchronize(int device) {
    int status = select_device(device);
    if (status != cudaSuccess)
        return status;
    return static_cast<int>(cudaDeviceSynchronize());
}

NNTRAIN_EXPORT int nntrain_cuda_mem_info(
    int device,
    size_t* free_bytes,
    size_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    int status = select_device(device);
    if (status != cudaSuccess)
        return status;
    return static_cast<int>(cudaMemGetInfo(free_bytes, total_bytes));
}

NNTRAIN_EXPORT int nntrain_cuda_malloc(
    int device,
    size_t bytes,
    void** pointer) {
    if (pointer == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    int status = select_device(device);
    if (status != cudaSuccess)
        return status;
    cudaError_t allocation_status = cudaMalloc(pointer, bytes);
    if (allocation_status != cudaSuccess) {
        // Managed pools may release cached buffers and retry. Clear CUDA's
        // per-thread last-error slot so a successful retry is not reported by
        // the next kernel's cudaPeekAtLastError call.
        (void)cudaGetLastError();
    }
    return static_cast<int>(allocation_status);
}

NNTRAIN_EXPORT int nntrain_cuda_free(int device, void* pointer) {
    if (pointer == nullptr)
        return static_cast<int>(cudaSuccess);
    int status = select_device(device);
    if (status != cudaSuccess)
        return status;
    return static_cast<int>(cudaFree(pointer));
}

NNTRAIN_EXPORT int nntrain_cuda_memset(
    int device,
    void* destination,
    int value,
    size_t bytes) {
    int status = select_device(device);
    if (status != cudaSuccess)
        return status;
    return static_cast<int>(cudaMemset(destination, value, bytes));
}

NNTRAIN_EXPORT int nntrain_cuda_copy_h2d(
    int device,
    void* destination,
    const void* source,
    size_t bytes) {
    int status = select_device(device);
    if (status != cudaSuccess)
        return status;
    return static_cast<int>(cudaMemcpy(
        destination, source, bytes, cudaMemcpyHostToDevice));
}

NNTRAIN_EXPORT int nntrain_cuda_copy_d2h(
    int device,
    void* destination,
    const void* source,
    size_t bytes) {
    int status = select_device(device);
    if (status != cudaSuccess)
        return status;
    return static_cast<int>(cudaMemcpy(
        destination, source, bytes, cudaMemcpyDeviceToHost));
}

NNTRAIN_EXPORT int nntrain_cuda_host_alloc(size_t bytes, void** pointer) {
    if (pointer == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    return static_cast<int>(cudaMallocHost(pointer, bytes));
}

NNTRAIN_EXPORT int nntrain_cuda_host_free(void* pointer) {
    if (pointer == nullptr)
        return static_cast<int>(cudaSuccess);
    return static_cast<int>(cudaFreeHost(pointer));
}

NNTRAIN_EXPORT int nntrain_cuda_event_create(int device, void** event) {
    if (event == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    int status = select_device(device);
    if (status != cudaSuccess)
        return status;
    return static_cast<int>(cudaEventCreateWithFlags(
        reinterpret_cast<cudaEvent_t*>(event), cudaEventDisableTiming));
}

NNTRAIN_EXPORT int nntrain_cuda_event_destroy(int device, void* event) {
    if (event == nullptr)
        return static_cast<int>(cudaSuccess);
    int status = select_device(device);
    if (status != cudaSuccess)
        return status;
    return static_cast<int>(cudaEventDestroy(
        reinterpret_cast<cudaEvent_t>(event)));
}

NNTRAIN_EXPORT int nntrain_cuda_copy_d2h_async_record(
    int device,
    void* destination,
    const void* source,
    size_t bytes,
    void* stream,
    void* event) {
    if (destination == nullptr || source == nullptr || event == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    int status = select_device(device);
    if (status != cudaSuccess)
        return status;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t copy_status = cudaMemcpyAsync(
        destination, source, bytes, cudaMemcpyDeviceToHost, cuda_stream);
    if (copy_status != cudaSuccess)
        return static_cast<int>(copy_status);
    return static_cast<int>(cudaEventRecord(
        reinterpret_cast<cudaEvent_t>(event), cuda_stream));
}

NNTRAIN_EXPORT int nntrain_cuda_copy_h2d_async_record(
    int device,
    void* destination,
    const void* source,
    size_t bytes,
    void* stream,
    void* event) {
    if (destination == nullptr || source == nullptr || event == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    int status = select_device(device);
    if (status != cudaSuccess)
        return status;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t copy_status = cudaMemcpyAsync(
        destination, source, bytes, cudaMemcpyHostToDevice, cuda_stream);
    if (copy_status != cudaSuccess)
        return static_cast<int>(copy_status);
    return static_cast<int>(cudaEventRecord(
        reinterpret_cast<cudaEvent_t>(event), cuda_stream));
}

NNTRAIN_EXPORT int nntrain_cuda_event_synchronize(int device, void* event) {
    if (event == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    int status = select_device(device);
    if (status != cudaSuccess)
        return status;
    return static_cast<int>(cudaEventSynchronize(
        reinterpret_cast<cudaEvent_t>(event)));
}

NNTRAIN_EXPORT int nntrain_cuda_copy_d2d(
    int destination_device,
    void* destination,
    int source_device,
    const void* source,
    size_t bytes) {
    if (destination_device == source_device) {
        int status = select_device(destination_device);
        if (status != cudaSuccess)
            return status;
        return static_cast<int>(cudaMemcpy(
            destination, source, bytes, cudaMemcpyDeviceToDevice));
    }
    return static_cast<int>(cudaMemcpyPeer(
        destination, destination_device, source, source_device, bytes));
}

NNTRAIN_EXPORT int nntrain_cuda_can_access_peer(
    int device,
    int peer_device,
    int* can_access) {
    if (can_access == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    return static_cast<int>(cudaDeviceCanAccessPeer(
        can_access, device, peer_device));
}

NNTRAIN_EXPORT const char* nntrain_cuda_error_string(int status) {
    return cudaGetErrorString(static_cast<cudaError_t>(status));
}
